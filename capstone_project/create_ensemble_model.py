import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import sys


def select_top_features(train_df, test_df, feature_importance_df, num_features, model_name):

	feature_importance_df = feature_importance_df.sort_values(model_name, ascending = False)
	features = feature_importance_df.iloc[:num_features]['Features'].tolist()

	train_df = train_df[features]
	test_df = test_df[features]

	return train_df, test_df

def ensemble_model(parameters_dataset, train_dataset, test_dataset, feature_importance_dataset):
	parameters_df = pd.read_csv(parameters_dataset)
	train_df = pd.read_csv(train_dataset)
	test_df = pd.read_csv(test_dataset)
	feature_importance_df = pd.read_csv(feature_importance_dataset)

	print(train_df.shape)
	print(test_df.shape)

	train_df['record_id'] = train_df.index

	y_train = train_df['READMISSION']

	k_folds = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

	for i in range(parameters_df.shape[0]):
		model_name = parameters_df.iloc[i]['Name']

		print(model_name)

		model_params = parameters_df.iloc[i]['Best Parameters']
		model_params = literal_eval(model_params)

		if 'Logistic Regression' in model_name:
			model_params['n_jobs'] = -1
			model_type_name = 'Logistic Regression'
			model = LogisticRegression(**model_params)
		elif 'Random Forest' in model_name:
			model_params['n_jobs'] = -1
			model_type_name = 'Random Forest All'
			model = RandomForestClassifier(**model_params)
		else:
			model_type_name = 'XGBoost'
			model = XGBClassifier(**model_params)

		if 'Top' in model_name:
			num_features = int(model_name.split(' ')[-1])
			X_train, X_test = select_top_features(train_df, test_df, feature_importance_df,
				num_features, model_type_name)
		else:
			X_train = train_df.drop(['READMISSION', 'record_id'], axis = 1)
			X_test = test_df.drop('READMISSION', axis = 1)			

		individual_model_pred = pd.DataFrame()

		for train_index, test_index in k_folds.split(X_train, y_train):
			model.fit(X_train.iloc[train_index], y_train[train_index])
			predictions = model.predict(X_train.iloc[test_index]).reshape(-1, 1)
			data = np.concatenate([test_index.reshape(-1, 1), predictions], axis = 1)
			individual_model_pred_sub = pd.DataFrame(data, columns=['record_id', model_name + '_Prediction'])
			individual_model_pred = pd.concat([individual_model_pred, individual_model_pred_sub])

		train_df = train_df.merge(individual_model_pred, how = 'left', on = 'record_id')
		
		model.fit(X_train, y_train)
		test_df[model_name + '_Prediction'] = model.predict(X_test).reshape(-1, 1)

	train_df = train_df.drop('record_id', axis = 1)

	print(train_df.shape)
	print(test_df.shape)

	train_df.to_csv(train_dataset.replace('.csv', '_ensemble.csv'), index = False)
	test_df.to_csv(test_dataset.replace('.csv', '_ensemble.csv'), index = False)

parameters_dataset = sys.argv[1]
train_dataset = sys.argv[2]
test_dataset = sys.argv[3]
feature_importance_dataset = sys.argv[4]
ensemble_model(parameters_dataset, train_dataset, test_dataset, feature_importance_dataset)



