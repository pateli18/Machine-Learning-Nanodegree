import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
	from xgboost.sklearn import XGBClassifier
except:
	pass
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import sys

RANDOM_STATE = 42
MODEL_INFO = {
	'bgn':{
		'name':'Blind Guess None'
	},
	'bga':{
		'name':'Blind Guess All'
	},	
	'lr':{
		'name':'Logistic Regression',
		'parameters_to_tune': {'penalty':['l1'], 'C':[10.0**i for i in range(-5,6)],
		'class_weight':['balanced'],'random_state':[RANDOM_STATE]}
	},
	'rf':{
		'name':'Random Forest',
		'parameters_to_tune': {'n_estimators': [1000],
		'max_features' : ['sqrt', 'log2', .20, .30], 'min_samples_leaf' : [1, 10, 100],
		'random_state': [RANDOM_STATE], 'class_weight' : ['balanced'],
		'max_depth': [10, 100, None]}
	},
	'xgb':{
		'name':'XGBoost',
		'parameters_to_tune': {'learning_rate':[.01], 'min_child_weight':[1],
		'max_depth':[3], 'gamma':[0, .1, .2, .3], 'subsample':[1], 'n_estimators':[1000],
		'colsample_bytree':[0.5], 'objective':['binary:logistic'], 'reg_lambda':[1, 10**1, 10**2],
		'seed':[RANDOM_STATE]}
	},
	'svm':{
		'name':'SVM',
		'parameters_to_tune': {'C':[10.0**-7, 10.0**-3, 1.0, 10.0**3, 10.0**7],'kernel':['rbf'],
		'gamma':[10.0**-7, 10.0**-3, 1.0, 10.0**3, 10.0**7], 'class_weight':['balanced'],
		'random_state':[RANDOM_STATE]}
	},
}

def run_model(model_name, model, X_train, X_test, y_train):
	model_cv = GridSearchCV(model, MODEL_INFO[model_name]['parameters_to_tune'], 
		scoring = "f1", cv = 5, verbose = 2, n_jobs = -1)
	model_cv.fit(X_train, y_train)
	best_params = str(model_cv.best_params_)
	model_best = model_cv.best_estimator_
	train_predictions = model_best.predict(X_train)
	test_predictions = model_best.predict(X_test)
	if model_name != 'svm':
		train_probabilities = model_best.predict_proba(X_train)
		test_probabilities = model_best.predict_proba(X_test)
		train_probabilities = train_probabilities[:, 1]
		test_probabilities = test_probabilities[:, 1]
		if model_name == 'lr':
			feature_importance = model_best.coef_.reshape(-1, 1)
		else:
			feature_importance = model_best.feature_importances_.reshape(-1, 1)
	else:
		train_probabilities = None
		test_probabilities = None
		feature_importance = None
	return best_params, (train_predictions, test_predictions), (train_probabilities,
	test_probabilities), feature_importance

def tune_models(train_dataset, test_dataset, parameters_dataset, 
	feature_importance_dataset, train_predictions_dataset, test_predictions_dataset,
	model_names):
	available_models = [key for key in MODEL_INFO.keys()]
	model_names = [model_name for model_name in model_names if model_name in available_models]
	print('Running Models: {0}'.format(str(model_names)))

	train_df = pd.read_csv(train_dataset)
	test_df = pd.read_csv(test_dataset)

	X_train = train_df.drop('READMISSION', axis = 1)
	y_train = train_df['READMISSION']
	X_test = test_df.drop('READMISSION', axis = 1)
	y_test = test_df['READMISSION']

	try:
		parameters_df = pd.read_csv(parameters_dataset)
	except IOError:
		parameters_df = pd.DataFrame()

	try:
		feature_importance_df = pd.read_csv(feature_importance_dataset)
	except IOError:
		feature_importance_df = pd.DataFrame(list(X_train.columns), columns = ['Features'])

	try:
		train_predictions_df = pd.read_csv(train_predictions_dataset)
	except IOError:
		train_predictions_df = train_df[['READMISSION']]

	try:
		test_predictions_df = pd.read_csv(test_predictions_dataset)
	except IOError:
		test_predictions_df = test_df[['READMISSION']]

	for model_name in model_names:
		model_full_name = MODEL_INFO[model_name]['name']
		print('Running {0}'.format(model_full_name))
		if model_name == 'bgn' or model_name == 'bga':
			predictions = True if model_name == 'bga' else False
			probabilities = 1.0 if model_name == 'bga' else 0.0
			train_predictions = y_train.apply(lambda x: predictions)
			train_probabilities = y_train.apply(lambda x: probabilities)
			test_predictions = y_test.apply(lambda x: predictions)
			test_probabilities = y_test.apply(lambda x: probabilities)
		else:
			if model_name == 'lr':
				model = LogisticRegression()
			elif model_name == 'rf':
				model = RandomForestClassifier()
			elif model_name == 'xgb':
				model = XGBClassifier()
			else:
				model = SVC()
			best_params, predictions, probabilities, feature_importance = run_model(model_name, model, X_train, X_test, y_train)
			train_predictions = predictions[0]
			test_predictions = predictions[1]
			train_probabilities = probabilities[0]
			test_probabilities = probabilities[1]

			parameters_df_add = pd.DataFrame([[model_full_name,
				str(best_params)]], columns = ['Name', 'Best Parameters'])
			parameters_df = pd.concat([parameters_df, parameters_df_add])
			parameters_df.to_csv(parameters_dataset, index = False)

			if model_name != 'svm':
				feature_importance_df[model_full_name] = feature_importance
				feature_importance_df.to_csv(feature_importance_dataset, index = False)

		train_predictions_df[model_full_name + '_Train_Predictions'] = train_predictions
		test_predictions_df[model_full_name + '_Test_Predictions'] = test_predictions

		if model_name != 'svm':
			train_predictions_df[model_full_name + '_Train_Probabilities'] = train_probabilities
			test_predictions_df[model_full_name + '_Test_Probabilities'] = test_probabilities

		train_predictions_df.to_csv(train_predictions_dataset, index = False)
		test_predictions_df.to_csv(test_predictions_dataset, index = False)
	print('Complete')

train_dataset = sys.argv[1]
test_dataset = sys.argv[2]
parameters_dataset = sys.argv[3]
feature_importance_dataset = sys.argv[4]
train_predictions_dataset = sys.argv[5]
test_predictions_dataset = sys.argv[6]
model_names = sys.argv[7].split(',')
tune_models(train_dataset, test_dataset, parameters_dataset, feature_importance_dataset,
	train_predictions_dataset, test_predictions_dataset, model_names)








