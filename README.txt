HurriGAN

The project consisted of two phases:

1- Processing:
  - Storms_Preprocessing: Includes adjusting the stations of each storm to have consistent number of values and removing the stations with outlier values.
  - Stations-train-test-split: Cluster the stations by coordinates, then from each cluster take 1 station to form the testing set.

2- Generative AI code: Contains the model architecture, building, hyperparameter tuning approach and training.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Python version: 3.9

matplotlib: 3.9.1
numpy: 1.23.5
pandas: 2.2.0
scikit-learn: 1.2.1
scipy: 1.9.0
tensorflow: 2.16.2
