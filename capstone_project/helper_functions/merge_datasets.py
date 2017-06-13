import pandas as pd
import sys

def merge_datasets(parameters_dataset, feature_importance_dataset, train_predictions_dataset, test_predictions_dataset, suffix, model_name):
	parameters_df = pd.read_csv(parameters_dataset)
	feature_importance_df = pd.read_csv(feature_importance_dataset)
	train_predictions_df = pd.read_csv(train_predictions_dataset)
	test_predictions_df = pd.read_csv(test_predictions_dataset)

	print('Parameters Shape: {0}'.format(parameters_df.shape))
	print('Feature Importance Shape: {0}'.format(feature_importance_df.shape))
	print('Train Predictions Shape: {0}'.format(train_predictions_df.shape))
	print('Test Predictions Shape: {0}'.format(test_predictions_df.shape))

	parameters_new_df = pd.read_csv(parameters_dataset.replace('.csv', suffix + '.csv'))
	feature_importance_new_df = pd.read_csv(feature_importance_dataset.replace('.csv', suffix + '.csv'))
	train_predictions_new_df = pd.read_csv(train_predictions_dataset.replace('.csv', suffix + '.csv'))
	test_predictions_new_df = pd.read_csv(test_predictions_dataset.replace('.csv', suffix + '.csv'))

	parameters_new_df['Name'] = parameters_new_df['Name'].apply(lambda x: model_name)
	parameters_df = pd.concat([parameters_df, parameters_new_df], axis = 0)

	new_feature_importance_columns = ['Features', model_name]
	feature_importance_columns = {column_old:column_new for column_old, column_new in zip(feature_importance_new_df.columns, new_feature_importance_columns)}
	feature_importance_new_df = feature_importance_new_df.rename(columns = feature_importance_columns)
	feature_importance_df = feature_importance_df.merge(feature_importance_new_df, how = 'left', on = 'Features')

	train_predictions_new_df = train_predictions_new_df.drop('READMISSION', axis = 1)
	new_train_columns = [model_name + '_Train_Predictions', model_name + '_Train_Probabilities']
	train_predictions_columns = {column_old:column_new for column_old, column_new in zip(train_predictions_new_df.columns, new_train_columns)}
	train_predictions_new_df = train_predictions_new_df.rename(columns = train_predictions_columns)
	train_predictions_df = pd.concat([train_predictions_df, train_predictions_new_df], axis = 1)

	test_predictions_new_df = test_predictions_new_df.drop('READMISSION', axis = 1)
	new_test_columns = [model_name + '_Test_Predictions', model_name + '_Test_Probabilities']
	test_predictions_columns = {column_old:column_new for column_old, column_new in zip(test_predictions_new_df.columns, new_test_columns)}
	test_predictions_new_df = test_predictions_new_df.rename(columns = test_predictions_columns)
	test_predictions_df = pd.concat([test_predictions_df, test_predictions_new_df], axis = 1)

	print('Parameters Shape: {0}'.format(parameters_df.shape))
	print('Feature Importance Shape: {0}'.format(feature_importance_df.shape))
	print('Train Predictions Shape: {0}'.format(train_predictions_df.shape))
	print('Test Predictions Shape: {0}'.format(test_predictions_df.shape))

	parameters_df.to_csv(parameters_dataset, index = False)
	feature_importance_df.to_csv(feature_importance_dataset, index = False)
	train_predictions_df.to_csv(train_predictions_dataset, index = False)
	test_predictions_df.to_csv(test_predictions_dataset, index = False)

parameters_dataset = sys.argv[1]
feature_importance_dataset = sys.argv[2]
train_predictions_dataset = sys.argv[3]
test_predictions_dataset = sys.argv[4]
suffix = sys.argv[5]
model_name = sys.argv[6]
merge_datasets(parameters_dataset, feature_importance_dataset, train_predictions_dataset, test_predictions_dataset, suffix, model_name)