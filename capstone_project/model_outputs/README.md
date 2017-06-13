# Model Outputs

* `ensemble_models.csv` stores the model names and parameters of those models used in the ensemble models
* `model_feature_importance_dataset.csv` stores feature importance (or coefficients in the case of logistic regression) for each model
* `model_parameters_dataset.csv` stores the hyper-parameters for each model chosen through grid search cross-validation
* `model_train_predictions.csv` stores the acutal predictions and probabilities of the models on the training feature set
* `model_test_predictions.csv` stores the acutal predictions and probabilities of the models on the testing feature set
* `model_train_costs.csv` stores the program costs of implementing each model by probability threshold and probability of readmission on the training set
* `model_test_costs.csv` stores the program costs of implementing each model by probability threshold and probability of readmission on the testing set
