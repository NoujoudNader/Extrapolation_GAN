This folder was created to organize the actual codes used in the project.

The project consisted of two phases:

  1- The first phase was centered around Hurricane Ian, with the gauge stations being divided into train-test split by utilizing KMeans clustering algorithm. This approach was selected to have a wide variety of gauge stations       according to the locations, instead of having the testing stations clustered in some areas while ignoring others.
  2- The second phase emcompassed 7 hurricanes. Their data is merged into one dataframe, even if common stations exist. The same train-test split procedure is taken. In this case, we have a far larger dataset with a more   
     diverse storm surge distributions. A for loop is used to perform grid search hyperparameter tuning according to the following parameters: neurons count, number of GRU layers & number of training epochs.

In both phases, the model trains on the training data in batches, and generates the temporal data of the testing stations by recieving the coordinates only as input.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Python version: 3.9

matplotlib: 3.9.1
numpy: 1.23.5
pandas: 2.2.0
scikit-learn: 1.2.1
scipy: 1.9.0
tensorflow: 2.16.2
