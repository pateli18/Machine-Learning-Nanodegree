# Helper Functions

## Tune Models
Tunes hyperparameters for chosen models using grid search cross validation and returns the chosen hyperparameters, feature importance, and predicitions on the training and testing set

Run the function from the terminal using the following command:
```
python tune_models.py <train_dataset_filepath> <test_dataset_filepath> <parameters_dataset_filepath> <feature_importance_dataset_filepath> <train_predictions_dataset_filepath> <test_predictions_dataset_filepath> <model_names>
```
The inputs correspond to the following:
* `train_dataset_filepath` is the training dataset's filepath
* `test_dataset_filepath` is the testing dataset's filepath
* `parameters_dataset_filepath` is a dataset storing the hyperparameters for all models run. If this is the first model, enter the desired filepath to save down the new file
* `feature_importance_dataset_filepath` is a dataset storing the feature importance for all models run. If this is the first model, enter the desired filepath to save down the new file
* `train_predictions_dataset_filepath` is a dataset storing the raw predictions and predicted class probabilities on the training set for all models run. If this is the first model, enter the desired filepath to save down the new file
* `test_predictions_dataset_filepath` is a dataset storing the raw predictions and predicted class probabilities on the testing set for all models run. If this is the first model, enter the desired filepath to save down the new file
* `model_names` can take on the following values (for multiple values, separate each by a comma with *no* spaces):
  * `bgn` for guessing `False` for all records
  * `bga` for guessing `True` for all records
  * `lr` for Logistic Regression
  * `rf` for a Random Forest Classifier
  * `xgb` for an XGBoost Classifier
  * `svm` for a Support Vector Machine Classifier
  
## Select Top Features
Creates new dataset by paring features based on feature importance in order to reduce dimensionality

Run the function from the terminal using the following command:
```
python select_top_features.py <train_dataset_filepath> <test_dataset_filepath> <feature_importance_dataset_filepath> <num_features> <model_name>
```
The inputs correspond to the following:
* `train_dataset_filepath` is the training dataset's filepath
* `test_dataset_filepath` is the testing dataset's filepath
* `feature_importance_dataset_filepath` is a dataset storing the feature importance for all models run
* `num_features` is the number of features to keep (i.e.`10` would correspond to keeping the top 10 features)
* `model_name` can take on the following values:
  * `lr` for Logistic Regression
  * `rf` for a Random Forest Classifier
  * `xgb` for an XGBoost Classifier
  * `svm` for a Support Vector Machine Classifier

## Merge Datasets
Merges model outputs from different runs together

Run the function from the terminal using the following command:
```
python merge_datasets.py <parameters_dataset_filepath> <feature_importance_dataset_filepath> <train_predictions_dataset_filepath> <test_predictions_dataset_filepath> <suffix> <model_name>
```
The inputs correspond to the following:
* `parameters_dataset_filepath` is the primary dataset storing the hyperparameters for all models run
* `feature_importance_dataset_filepath` is the primary dataset storing the feature importance for all models run
* `train_predictions_dataset_filepath` is the primary dataset storing the raw predictions and predicted class probabilities on the training set for all models run
* `test_predictions_dataset_filepath` is the primary dataset storing the raw predictions and predicted class probabilities on the testing set for all models run
* `suffix` is the additional signature on the files from the model you want to merge (i.e. if the primary model was named `train_dataset.csv` and the merge model named `train_dataset_xgboost10.csv`, then suffix would be `_xgboost10`)
* `model_name` is the name that the model should appear as in the primary datasets

## Create Ensemble Model
Creates datasets for use in an ensemble model

Run the function from the terminal using the following command:
```
python create_ensemble_model.py <parameters_dataset_filepath> <train_dataset_filepath> <test_dataset_filepath> <feature_importance_dataset_filepath>
```
The inputs correspond to the following:
* `parameters_dataset_filepath` is a dataset storing the hyperparameters for the models to include in the ensemble
* `train_dataset_filepath` is the training dataset's filepath
* `test_dataset_filepath` is the testing dataset's filepath
* `feature_importance_dataset_filepath` is a dataset storing the feature importance for all models run

## Calculate Costs
Creates dataset storing interventional program costs for models in a dataset by probability threshold and probability of readmission

Run the function from the terminal using the following command:
```
python create_ensemble_model.py <predictions_dataset_filepath> <costs_dataset_filepath> <effectiveness_min> <effectiveness_max> <effectiveness_increment>
```
The inputs correspond to the following:
* `predictions_dataset_filepath` is a dataset storing predictions for either a training or test set
* `costs_dataset_filepath` is a dataset storing the interventional program costs for all models run. If this file does not exist, enter the desired filepath to save down the new file
* `effectiveness_min` is an `integer` of the minimum probability of readmission to calculate costs for (i.e. if the interventional program is maximally effective, the probability of readmission would be 0, while if it was completely ineffective the probability of readmission would be 1)
* `effectiveness_max` is an `integer` of the maximum probability of readmission to calculate costs for. Must be greater than `effectiveness_min`
* `effectiveness_increment` is an `integer` that represents the amount to increment between `effectiveness_min` and `effectiveness_max`


