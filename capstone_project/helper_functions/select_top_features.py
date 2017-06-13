import pandas as pd
import sys

def select_top_features(train_filepath, test_filepath, feature_importance_filepath, num_features, model_name):
	model_names = {'lr':'Logistic Regression', 'rf':'Random Forest', 'xgb':'XGBoost', 'svm':'SVM'}
	train_df = pd.read_csv(train_filepath)
	test_df = pd.read_csv(test_filepath)
	feature_importance_df = pd.read_csv(feature_importance_filepath)

	feature_importance_df = feature_importance_df.sort_values(model_names[model_name], ascending = False)
	features = feature_importance_df.iloc[:num_features]['Features'].tolist()
	features.append('READMISSION')

	train_df = train_df[features]
	test_df = test_df[features]

	train_df.to_csv(train_filepath.replace('.csv', '_{0}{1}.csv'.format(model_name, num_features)), index = False)
	test_df.to_csv(test_filepath.replace('.csv', '_{0}{1}.csv'.format(model_name, num_features)), index = False)

train_filepath = sys.argv[1]
test_filepath = sys.argv[2]
feature_importance_filepath = sys.argv[3]
num_features = int(sys.argv[4])
model_name = sys.argv[5]
select_top_features(train_filepath, test_filepath, feature_importance_filepath, num_features, model_name)