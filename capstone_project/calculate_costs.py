import pandas as pd
import sys

def calculate_costs(predictions_dataset, costs_dataset, effectiveness_min, effectiveness_max, effectiveness_increment):
	predictions_df = pd.read_csv(predictions_dataset)

	try:
		all_costs_df = pd.read_csv(costs_dataset)
		models = [column for column in predictions_df.columns if column not in all_costs_df['Model'].tolist()]
	except IOError:
		all_costs_df = pd.DataFrame()
		models = predictions_df.columns

	print(all_costs_df.shape)

	probability_thresholds = xrange(0, 100, 1)

	C_INTERVENTION = 1300.0 
	C_READMISSION = 13679.0

	cost_df = predictions_df[['READMISSION']]
	cost_df['Intervention'] = cost_df['READMISSION'].apply(lambda x: 0.0)
	cost_df['Readmission'] = cost_df['READMISSION'].apply(lambda x: C_READMISSION if x else 0.0)

	all_costs_columns = ['Model', 'Probability of Readmission']
	all_costs_columns_prob = [str(i) for i in probability_thresholds]
	all_costs_columns.extend(all_costs_columns_prob)

	for model_num, model in enumerate(models):
		for p_readmission in xrange(effectiveness_min, effectiveness_max, effectiveness_increment):
			p_readmission = p_readmission * 1.0 / 100.0
			costs = []
			for i in probability_thresholds:
				i = i * 1.0 / 100.0
				cost_df['Predicted Readmission'] = predictions_df[model].apply(lambda x: x > i)
				cost_df['Intervention'] = cost_df['Predicted Readmission'].apply(lambda x:C_INTERVENTION if x else 0.0)
				cost_df['P_Readmission'] = cost_df['Predicted Readmission'].apply(lambda x:p_readmission if x else 1.0)
				cost_df['Total Cost'] = cost_df['Intervention'] + cost_df['P_Readmission'] * cost_df['Readmission']
				costs.append(cost_df['Total Cost'].sum())
			costs_data = [model, p_readmission]
			costs_data.extend(costs)
			all_costs_sub_df = pd.DataFrame([costs_data], columns = all_costs_columns)
			all_costs_df = pd.concat([all_costs_df, all_costs_sub_df])
		print('{0} out of {1} models cost data added'.format(model_num, len(models)))

	print(all_costs_df.shape)
	all_costs_df.to_csv(costs_dataset, index = False)

predictions_dataset = sys.argv[1]
costs_dataset = sys.argv[2]
effectiveness_min = int(sys.argv[3])
effectiveness_max = int(sys.argv[4])
effectiveness_increment = int(sys.argv[5])
calculate_costs(predictions_dataset, costs_dataset, effectiveness_min, effectiveness_max, effectiveness_increment)
