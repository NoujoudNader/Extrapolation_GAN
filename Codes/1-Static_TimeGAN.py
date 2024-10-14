import warnings
warnings.filterwarnings('ignore')

import time
start_time = time.time()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, Input, Flatten, Dropout, Concatenate, Bidirectional, Layer, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.utils import plot_model
from matplotlib.ticker import MaxNLocator

# Check if GPU is being utilized
print(tf.config.list_physical_devices('GPU'))

# OS
if not os.path.exists("loss_plots"):
    os.makedirs("loss_plots")

if not os.path.exists("generated_data"):
    os.makedirs("generated_data")

# Training dataframe
train_df= pd.read_csv("/work/hmajed/train_df_filtered.csv")
train_df['time_UTC']= pd.to_datetime(train_df['time_UTC'])

train_df.drop(['index','Unnamed: 0','forecast_data','observed_data','time_UTC_dt'], axis=1, inplace= True)
pivot_df = train_df.pivot_table(values='offset', index='time_UTC', columns='station_id', aggfunc='first')
pivot_df = pivot_df.iloc[:, :180]

new_row = pivot_df.iloc[-1]
pivot_df.loc['Filler'] = new_row

# Testing dataframe
testing_df= pd.read_csv("/work/hmajed/test_df_filtered.csv")
testing_df['time_UTC']= pd.to_datetime(testing_df['time_UTC'])

testing_df.drop(['index','Unnamed: 0','forecast_data','observed_data','time_UTC_dt'], axis=1, inplace= True)
testing_pivot = testing_df.pivot_table(values='offset', index='time_UTC', columns='station_id', aggfunc='first')

# Training coords
coords_df = train_df[['station_id', 'x', 'y']].drop_duplicates()

# Testing coords
testing_coords = testing_df[['station_id', 'x', 'y']].drop_duplicates()

coords_scaler = MinMaxScaler()
coords_scaler.fit(pd.concat([coords_df[['x', 'y']], testing_coords[['x', 'y']]]))

coords_df[['x', 'y']] = coords_scaler.transform(coords_df[['x', 'y']])
testing_coords[['x', 'y']] = coords_scaler.transform(testing_coords[['x', 'y']])

coords_df['coords'] = list(zip(coords_df['x'], coords_df['y']))
testing_coords['coords'] = list(zip(testing_coords['x'], testing_coords['y']))

# Training coords array
coords_array = coords_df[['x', 'y']].values
# Testing coords array
testing_coords_array = testing_coords[['x', 'y']].values

## Prepare Data

seq_len = 7
n_seq = 24

# Normalize the dataframe & create the equivalent dictionary
scaler_ts = MinMaxScaler()

scaled_df = scaler_ts.fit_transform(pivot_df)
scaled_df = pd.DataFrame(scaled_df, columns=pivot_df.columns, index=pivot_df.index)

reshaped_data = {}

for station_id in scaled_df.columns:
	observations = scaled_df[station_id].values
	
	observations = observations.reshape(7,24)

	reshaped_data[station_id] = observations

batch_size = 10

# Data Creation
def create_dataset(reshaped_data, coords_array, batch_size):
    X_list = []
    S_list = []
    for i, key in enumerate(reshaped_data.keys()):
        X_list.append(reshaped_data[key])
        S_list.append(coords_array[i])
    
    X_data = np.array(X_list)
    S_data = np.array(S_list)
    
    dataset = tf.data.Dataset.from_tensor_slices((X_data, S_data))
    dataset = dataset.shuffle(buffer_size=len(X_data)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE) # PREFETCH?
    
    return dataset

train_dataset = create_dataset(reshaped_data, coords_array, batch_size)

# Making the data from which the Generator takes input
def make_temp_random_data():
	while True:
		yield np.random.uniform(low=0, high=1, size=(seq_len, n_seq))
		
random_temp_series = iter(tf.data.Dataset
					 .from_generator(make_temp_random_data, output_types=tf.float32)
					 .batch(batch_size)
					 .repeat())

# Making the data from which the Testing Generator takes input		
test_temp_series = iter(tf.data.Dataset
					 .from_generator(make_temp_random_data, output_types=tf.float32)
					 .batch(1)
					 .repeat())

## TimeGAN Components

## Model Architectures & Inputs

autoencoder_optimizer= Adam()
supervisor_optimizer= Adam()
generator_optimizer= Adam()
discriminator_optimizer= Adam()

mse = MeanSquaredError()
bce = BinaryCrossentropy()

neurons_count= 128

# Input Placeholders
input_shape = (seq_len, n_seq)
static_dim = 2

X = Input(shape=(seq_len, n_seq), name='RealData')
S = Input(shape=(2,), name='StaticData')

ht = Input(shape=(seq_len, n_seq), name="SupervisorTempInput")
hs = Input(shape=(4, ), name= "SupervisorStaticInput")

Z = Input(shape=(seq_len, n_seq), name='RandomTemp')

Dt_input = Input(shape=(seq_len, n_seq), name='DiscTempInput')
Ds_input = Input(shape=(4, ), name='DiscStaticInput')

## AutoEncoder

class EmbedderLayer(Layer):
    def __init__(self, n_seq, **kwargs):
        super(EmbedderLayer, self).__init__(**kwargs)
        self.n_seq = n_seq
        self.temp_embedder = Sequential([
            GRU(units= neurons_count, return_sequences=True),
            GRU(units= neurons_count, return_sequences=True),
            GRU(units= neurons_count, return_sequences=True),
            GRU(units= neurons_count, return_sequences=True),
        ])
        self.static_embedder = Dense(4, activation='sigmoid', name='StaticEmbedding')
        self.embedder_output = Dense(n_seq, activation='sigmoid', name='EmbedderOutput')

    def call(self, inputs):
        X, S = inputs
        HS = self.static_embedder(S)
        HS_reshaped = tf.reshape(HS, (-1, 1, 4))
        HS_reshaped = tf.tile(HS_reshaped, [1, tf.shape(X)[1], 1])
        h_inputs = tf.concat([X, HS_reshaped], axis=-1)
        e_outputs = self.temp_embedder(h_inputs)
        HT = self.embedder_output(e_outputs)
        return HT, HS

class RecoveryLayer(Layer):
    def __init__(self, n_seq, **kwargs):
        super(RecoveryLayer, self).__init__(**kwargs)
        self.n_seq = n_seq
        self.recovery_temp = Sequential([
            GRU(units= neurons_count, return_sequences=True),
            GRU(units= neurons_count, return_sequences=True),
            GRU(units= neurons_count, return_sequences=True),
            GRU(units= neurons_count, return_sequences=True),
        ])
        self.recovery_static = Dense(2, activation='sigmoid', name='StaticRecovery')
        self.recovery_output = Dense(n_seq, activation='sigmoid', name='RecoveryOutput')

    def call(self, inputs):
        H, S = inputs
        S_reshaped = tf.reshape(S, (-1, 1, 4))
        S_reshaped = tf.tile(S_reshaped, [1, tf.shape(H)[1], 1])
        H_inputs = tf.concat([H, S_reshaped], axis=-1)
        r_outputs = self.recovery_temp(H_inputs)
        XT_tilde = self.recovery_output(r_outputs)
        XS_tilde = self.recovery_static(S)

        return XT_tilde, XS_tilde

embedder_layer = EmbedderLayer(n_seq)
recovery_layer = RecoveryLayer(n_seq)

H, HS = embedder_layer([X, S])
X_tilde, S_tilde = recovery_layer([H, HS])

autoencoder = Model(inputs=[X, S],
					outputs=[X_tilde, S_tilde],
					name='Autoencoder')

autoencoder.compile(optimizer = autoencoder_optimizer, loss=mse)

## Supervisor

class SupervisorLayer(Layer):
    def __init__(self, n_seq, **kwargs):
        super(SupervisorLayer, self).__init__(**kwargs)
        self.n_seq = n_seq
        self.temp_supervisor = Sequential([
            GRU(units= neurons_count, return_sequences=True),
            GRU(units= neurons_count, return_sequences=True),
            GRU(units= neurons_count, return_sequences=True),
        ])
        self.static_supervisor = Dense(4, activation='sigmoid')
        self.supervisor_output = Dense(n_seq, activation='sigmoid', name='SupervisorOutput')
    
    def call(self, inputs):
        ht, hs = inputs
        static_embedded = self.static_supervisor(hs)
        reshaped_static = tf.reshape(static_embedded, (-1, 1, 4))
        reshaped_static = tf.tile(reshaped_static, [1, tf.shape(ht)[1], 1])
        concatenated_input = Concatenate(axis=-1)([ht, reshaped_static])
        output_sequence = self.temp_supervisor(concatenated_input)
        HT_sup = self.supervisor_output(output_sequence)

        return HT_sup, static_embedded

supervisor_layer = SupervisorLayer(n_seq)
sup_temp_emb, sup_static_emb = supervisor_layer([ht, hs])

supervisor_model = Model(inputs=[ht, hs],
					outputs=[sup_temp_emb, sup_static_emb],
					name='Supervisor')

supervisor_model.compile(optimizer = supervisor_optimizer, loss=mse)

## Adversarial Net

class GeneratorLayer(Layer):
    def __init__(self, n_seq, **kwargs):
        super(GeneratorLayer, self).__init__(**kwargs)
        self.n_seq = n_seq
        self.temp_generator = Sequential([
            GRU(units= neurons_count, return_sequences=True),
            GRU(units= neurons_count, return_sequences=True),
            GRU(units= neurons_count, return_sequences=True),
            GRU(units= neurons_count, return_sequences=True),
        ])
        self.static_generator = Dense(4, activation='sigmoid')
        self.generator_output = Dense(n_seq, activation= 'sigmoid', name= 'GeneratorOutput')

    def call(self, inputs):
        ZT, S = inputs
        static_embedded = self.static_generator(S)
        reshaped_static = tf.reshape(static_embedded, (-1, 1, 4))
        reshaped_static = tf.tile(reshaped_static, [1, tf.shape(ZT)[1], 1])
        concatenated_input = Concatenate(axis=-1)([ZT, reshaped_static])
        output_sequence = self.temp_generator(concatenated_input)
        g_outputs = self.generator_output(output_sequence)

        return g_outputs, static_embedded

generator_layer = GeneratorLayer(n_seq)
g_outputs, g_static = generator_layer([Z, S])

generator = Model(inputs=[Z, S],
					outputs=[g_outputs, g_static],
					name='Generator')

class DiscriminatorLayer(Layer):
    def __init__(self, n_seq, **kwargs):
        super(DiscriminatorLayer, self).__init__(**kwargs)
        self.n_seq = n_seq
        self.discriminator_network = Sequential([
            Bidirectional(GRU( neurons_count, return_sequences=True)),
            Bidirectional(GRU( neurons_count, return_sequences=False)),
            Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        T_input, S_input = inputs
        T_input_reshaped = tf.reshape(T_input, (-1, 7, 24))  # Reshape T_input if necessary
        Y = self.discriminator_network(T_input_reshaped)
        return Y

discriminator_layer = DiscriminatorLayer(n_seq)
Y_fake = discriminator_layer([Dt_input, Ds_input])

discriminator = Model(inputs=[Dt_input, Ds_input],
					  outputs= Y_fake,
					  name='Discriminator'
					 )

discriminator.compile(optimizer= Adam(), loss= mse)

# Adversarial Supervised Model
# Generator takes random input - Supervisor reshapes - Discriminator classifies

E_t_hat, E_s_hat = generator([Z, S])
H_t_hat, H_s_hat = supervisor_model([E_t_hat, E_s_hat])
Y_fake = discriminator([H_t_hat, H_s_hat])

adversarial_supervised = Model(inputs=[Z, S],
							   outputs=Y_fake,
							   name='AdversarialNetSupervised')

adversarial_supervised.compile(optimizer= Adam(), loss= mse)

# Adversarial Unsupervised Model
# Generator takes random input - Discriminator classifies

Y_fake_e = discriminator([E_t_hat, E_s_hat])

adversarial_emb = Model(inputs=[Z, S],
					outputs=Y_fake_e,
					name='AdversarialNetUnsupervised')

adversarial_emb.compile(optimizer= Adam(), loss= mse)

# Real Data Discriminator
# Embedder takes real data - Discriminator takes embeddings

Y_real = discriminator([H, HS])

discriminator_real = Model(inputs=[X,S],
							outputs=Y_real,
							name='DiscriminatorReal')

# Synthetic Data Generator
# Generator - Supervisor - Recovery

Xt_hat, Xs_hat = recovery_layer([H_t_hat, H_s_hat])

synthetic_data = Model(inputs=[Z, S],
					   outputs=[Xt_hat, Xs_hat],
					   name='SyntheticData')

synthetic_data.compile(optimizer= Adam(), loss= mse)

## TimeGAN Training

# Train steps and weights
train_steps_1 = 500
train_steps_2 = 3000

temp_embedding_weight= 10
static_embedding_weight= 15
gamma = 1
supervised_gen_loss_weight= 100
unsupervised_gen_loss_weight = 1
distribution_loss_weight = 1
embedding_loss_weight = 10 
supervised_loss_weight = 0.1

## Phase 1: Autoencoder Training

@tf.function
def train_autoencoder_init(x, s):
	with tf.GradientTape() as tape:
		x_tilde, s_tilde = autoencoder([x, s])
		embedding_loss_t0 = mse(x, x_tilde)
		static_loss_t0 = mse(s, s_tilde)
		e_loss_0 = temp_embedding_weight * tf.sqrt(embedding_loss_t0) + static_embedding_weight * tf.sqrt(static_loss_t0)
		
	var_list = autoencoder.trainable_variables
	gradients = tape.gradient(e_loss_0, var_list)
	autoencoder_optimizer.apply_gradients(zip(gradients, var_list))
	
	return tf.sqrt(embedding_loss_t0), tf.sqrt(static_loss_t0)

autoencoder_static_loss_hist = []
autoencoder_training_history = []

for step in tqdm(range(train_steps_1)):
    for x_batch, s_batch in train_dataset:
        step_e_loss_t0, step_stat_loss = train_autoencoder_init(x_batch, s_batch)
    if step % 5 == 0:
        autoencoder_training_history.append(step_e_loss_t0.numpy())
        autoencoder_static_loss_hist.append(step_stat_loss.numpy())

plt.figure(figsize=(20, 10))
plt.plot(range(0, len(autoencoder_training_history) * 5, 5), autoencoder_training_history, label='Embedding Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Temporal AutoEnc. Loss Evolution')
plt.legend()
plt.grid(True)
plt.savefig("loss_plots/Temp_AutoEnc_Loss.png")
plt.close()

plt.figure(figsize=(20, 10))
plt.plot(range(0, len(autoencoder_static_loss_hist) * 5, 5), autoencoder_static_loss_hist, label='Embedding Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Static AutoEnc. Loss Evolution')
plt.legend()
plt.grid(True)
plt.savefig("loss_plots/Static_AutoEnc_Loss.png")
plt.close()

print("Finished AutoEncoder Training")

# Phase 2: Supervisor Training
@tf.function
def train_supervisor(x, s):
	with tf.GradientTape() as tape:
		ht_hat_supervised, hs_hat_supervised = supervisor_model([x, s])
		gt_loss_s = mse(x, ht_hat_supervised)
		gs_loss_s = mse(s, hs_hat_supervised)
		g_loss_s = (gt_loss_s + gs_loss_s)

	var_list = supervisor_model.trainable_variables
	gradients = tape.gradient(g_loss_s, var_list)
	supervisor_optimizer.apply_gradients(zip(gradients, var_list))
	
	return tf.sqrt(gt_loss_s), tf.sqrt(gs_loss_s)

supervised_temp_training_history = []
supervised_static_training_history = []

for step in tqdm(range(train_steps_1)):
    for x_batch, s_batch in train_dataset:
        
        ht_batch, hs_batch = embedder_layer([x_batch, s_batch])
        step_gt_loss_s, step_gs_loss_s = train_supervisor(ht_batch, hs_batch)
    if step % 5 == 0: # Change
        supervised_temp_training_history.append(step_gt_loss_s.numpy())
        supervised_static_training_history.append(step_gs_loss_s.numpy())

plt.figure(figsize=(20, 10))
plt.plot(range(0, len(supervised_temp_training_history) * 5, 5), supervised_temp_training_history, label='Supervised Temporal Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Supervised Temporal Loss Evolution')
plt.legend()
plt.grid(True)
plt.savefig("loss_plots/Temp_Supervisor_Loss.png")
plt.close()

plt.figure(figsize=(20, 10))
plt.plot(range(0, len(supervised_static_training_history) * 5, 5), supervised_static_training_history, label='Supervised Static Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Supervised Static Loss Evolution')
plt.legend()
plt.grid(True)
plt.savefig("loss_plots/Static_Supervisor_Loss.png")
plt.close()

print("Finished Supervisor Training.")

## Phase 3: Joint Training
## Losses

def get_generator_moment_loss(y_true, y_pred):
	
	y_true = tf.cast(y_true, tf.float32)
	y_pred = tf.cast(y_pred, tf.float32)
	
	y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
	y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
	g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
	g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
	
	return g_loss_mean + g_loss_var

@tf.function
def get_discriminator_loss(x, s, z):

	y_real = discriminator_real([x, s]) # real loss
	discriminator_loss_real = bce(y_true=tf.ones_like(y_real),
								  y_pred=y_real)
	
	y_fake = adversarial_supervised([z, s]) # supervised loss
	discriminator_loss_fake = bce(y_true=tf.zeros_like(y_fake),
								  y_pred=y_fake)
	
	y_fake_e = adversarial_emb([z, s]) # unsupervised loss
	discriminator_loss_fake_e = bce(y_true=tf.zeros_like(y_fake_e),
									y_pred=y_fake_e)
	
	return (discriminator_loss_real +
			discriminator_loss_fake +
			gamma * discriminator_loss_fake_e)

@tf.function
def train_generator(x, ht_1, hs, z, s):
	with tf.GradientTape() as tape:

		y_fake = adversarial_supervised([z, s])
		generator_loss_unsupervised = bce(y_true=tf.ones_like(y_fake),
										  y_pred=y_fake)
		
		y_fake_e = adversarial_emb([z, s])
		generator_loss_unsupervised_e = bce(y_true=tf.ones_like(y_fake_e),
											y_pred=y_fake_e)
				
		ht_hat_supervised, hs_hat_supervised = supervisor_model([ht_1, hs])
		generator_temp_loss_supervised = mse(ht_1, ht_hat_supervised)
		generator_static_loss_supervised = mse(hs, hs_hat_supervised)

		xt_hat, xs_hat = synthetic_data([z, s])
		
		generator_moment_loss = get_generator_moment_loss(x, xt_hat)

		generator_loss = (generator_loss_unsupervised +
						  unsupervised_gen_loss_weight * generator_loss_unsupervised_e +
						  supervised_gen_loss_weight * tf.sqrt(generator_temp_loss_supervised + generator_static_loss_supervised) +
						  distribution_loss_weight * generator_moment_loss)
	
	var_list = adversarial_emb.trainable_variables + adversarial_supervised.trainable_variables + synthetic_data.trainable_variables
	gradients = tape.gradient(generator_loss, var_list)
	generator_optimizer.apply_gradients(zip(gradients, var_list))
	
	return xt_hat, generator_loss_unsupervised, generator_temp_loss_supervised, generator_static_loss_supervised, generator_moment_loss

@tf.function
def train_embedder(x, s, ht_1, hs):
	with tf.GradientTape() as tape:
		
		ht_hat_supervised, hs_hat_supervised = supervisor_model([ht_1, hs])
		generator_loss_supervised = mse(x, ht_hat_supervised)
	
		x_tilde, s_tilde = autoencoder([x, s])
		embedding_loss_t0 = mse(x, x_tilde)
		embedding_static_loss_t0 = mse(s, s_tilde)
		
		e_loss = (embedding_loss_weight * tf.sqrt(embedding_loss_t0 + embedding_static_loss_t0) + supervised_loss_weight * generator_loss_supervised)
	
	var_list = autoencoder.trainable_variables
	gradients = tape.gradient(e_loss, var_list)
	autoencoder_optimizer.apply_gradients(zip(gradients, var_list))
	
	return tf.sqrt(embedding_loss_t0), tf.sqrt(embedding_static_loss_t0)

@tf.function
def train_discriminator(x, s, z):
	with tf.GradientTape() as tape:
		discriminator_loss = get_discriminator_loss(x, s, z)
	
	var_list = discriminator.trainable_variables
	gradients = tape.gradient(discriminator_loss, var_list)
	discriminator_optimizer.apply_gradients(zip(gradients, var_list))
	
	return discriminator_loss

## Training Loop

step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0

d_loss_history = []
g_loss_u_history = []
g_loss_s_temp_history = []
g_loss_s_static_history = []
g_loss_v_history = []

syn_gen_0 = []
syn_gen_1 = []
syn_gen_2 = []
syn_gen_3 = []
syn_gen_4 = []
syn_gen_5 = []
		
for step in tqdm(range(train_steps_2)):
    for x_batch, s_batch in train_dataset:
        Z_ = next(random_temp_series)

        for i in range(2):
            # generator
            ht_batch, hs_batch = embedder_layer([x_batch, s_batch])
            x_hat, step_g_loss_u, step_g_t_loss_s, step_g_s_loss_s, step_g_loss_v = train_generator(x_batch, ht_batch, hs_batch, Z_, s_batch)
            
            # embedder
            step_e_t_loss_t0, step_e_s_loss_t0 = train_embedder(x_batch, s_batch, ht_batch, hs_batch)
            
        step_d_loss = get_discriminator_loss(x_batch, s_batch, Z_) 
        if step_d_loss > 0.15:
            step_d_loss = train_discriminator(x_batch, s_batch, Z_)

        if step == 0:
            syn_gen_0.append(x_hat)
        if step == 400:
            syn_gen_1.append(x_hat)
        if step == 800:
            syn_gen_2.append(x_hat)
        if step == 1200:
            syn_gen_3.append(x_hat)
        if step == 1600:
            syn_gen_4.append(x_hat)
        if step == 1999:
            syn_gen_5.append(x_hat)
            
    if step % 5 == 0:  
        d_loss_history.append(step_d_loss.numpy())
        g_loss_u_history.append(step_g_loss_u.numpy())
        g_loss_s_temp_history.append(step_g_t_loss_s.numpy()) 
        g_loss_s_static_history.append(step_g_s_loss_s.numpy())
        g_loss_v_history.append(step_g_loss_v.numpy())

print("Finished Joint Training")

list_0 = []
list_1 = []
list_2 = []
list_3 = []
list_4 = []
list_5 = []

for i in range(len(syn_gen_0)):
	list_0.append(syn_gen_0[i].numpy().flatten().tolist())
	
for i in range(len(syn_gen_1)):
	list_1.append(syn_gen_1[i].numpy().flatten().tolist())
	
for i in range(len(syn_gen_2)):
	list_2.append(syn_gen_2[i].numpy().flatten().tolist())
	
for i in range(len(syn_gen_3)):
	list_3.append(syn_gen_3[i].numpy().flatten().tolist())
	
for i in range(len(syn_gen_4)):
	list_4.append(syn_gen_4[i].numpy().flatten().tolist())
	
for i in range(len(syn_gen_5)):
	list_5.append(syn_gen_5[i].numpy().flatten().tolist())

def plot_syn(gen_list, epoch_numb):
	plt.figure(figsize=(20, 4))

	plt.subplot(1, 2, 2)

	for i in range(len(gen_list)):    
		plt.plot(gen_list[i])

	epoch= epoch_numb
	plt.xlabel('Time')
	plt.ylabel('Generated offset')
	plt.title(f'Training Synthetic Data {epoch}')
	plt.grid(True)

	plt.tight_layout()
	plt.savefig(f"generated_data/Training_Generations_{epoch}.png")
	plt.close()

plot_syn(list_0, 0)
plot_syn(list_1, 400)
plot_syn(list_2, 800)
plot_syn(list_3, 1200)
plot_syn(list_4, 1600)
plot_syn(list_5, 2000)

plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.plot(range(0, len(d_loss_history) * 5, 5), d_loss_history, label='Discriminator Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Discriminator Loss Over Training Steps')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("loss_plots/Discriminator_Loss.png")
plt.close()

plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 2)
plt.plot(range(0, len(g_loss_u_history) * 5, 5), g_loss_u_history, label='Generator Loss (U)')
plt.plot(range(0, len(g_loss_s_temp_history) * 5, 5), g_loss_s_temp_history, label='Generator Temporal Loss (S)')
plt.plot(range(0, len(g_loss_s_static_history) * 5, 5), g_loss_s_static_history, label='Generator Static Loss (S)')
plt.plot(range(0, len(g_loss_v_history) * 5, 5), g_loss_v_history, label='Generator Loss (V)')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Generator Losses Over Training Steps')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("loss_plots/Generator_Losses.png")
plt.close()

# Making the data from which the Testing Generation takes input
		
test_temp_series = iter(tf.data.Dataset
					 .from_generator(make_temp_random_data, output_types=tf.float32)
					 .batch(1) # 1 as we are generating for each station at a time
					 .repeat())

## Generate Synthetic Data
scaler_list= {}

testing_pivot_scaled = testing_pivot.copy()

for column in testing_pivot_scaled.columns:
	test_scaler = MinMaxScaler()
	testing_pivot_scaled[[column]]= test_scaler.fit_transform(testing_pivot_scaled[[column]])
	scaler_list[column]= test_scaler
	
generated_data= {}

j=0

for column in testing_pivot_scaled.columns:
	Z_ = next(test_temp_series)
	S_ = testing_coords_array[j].reshape(1, 2)
	S_ = tf.cast(S_, tf.float32)

	generated_temporal, generated_static = synthetic_data([Z_, S_])
	generated_temporal_np = generated_temporal.numpy()
		
	generated_data[column] = generated_temporal_np
	j+=1
	
unscaled_data= {}

for key in generated_data:
	temporary_list= []
	for i in range(seq_len): # 7
		for j in range(n_seq): # 24
			temporary_list.append(generated_data[key][0][i][j])
			
	temporary_array = np.array(temporary_list).reshape(-1, 1)
	temporary_array = scaler_list[key].inverse_transform(temporary_array)
	temporary_list = temporary_array.flatten().tolist()
	
	unscaled_data[key]= temporary_list
	
unscaled_df = pd.DataFrame(unscaled_data)
unscaled_df = unscaled_df.drop(unscaled_df.index[-1])
unscaled_df.to_csv("generated_data/generated_data.csv", index= False)

list_real= []
list_gen = []

for column in testing_pivot.columns:
	temp_list_real= testing_pivot[column].tolist()
	temp_list_gen= unscaled_df[column].tolist()
	list_real.append(temp_list_real)
	list_gen.append(temp_list_gen)
	
array1 = np.array(list_real)
array2 = np.array(list_gen)

mse = np.mean((array1 - array2) ** 2)
mae = np.mean((array1 - array2))

print("MSE:", mse)
print("MAE:", mae)

i= 0

for column in unscaled_df.columns:
	
	fig, ax = plt.subplots()
	ax.plot(list_real[i], color='darkcyan', label='Real Data')
	ax.plot(list_gen[i], color='orange', label='Synthetic Data')
	ax.set_ylabel('Values')
	ax.legend()
	ax.set_title(column)

	plt.savefig(f"generated_data/{column}_Generated_Data.png")
	plt.close()
	i+=1

end_time= time.time()

elapsed_time= end_time - start_time

print(f"The code took {elapsed_time} seconds to execute.")

