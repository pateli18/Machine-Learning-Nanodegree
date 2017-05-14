import pandas as pd
import quandl
import time
import sys

def format_zipcode(postal_code):
    postal_code = str(int(postal_code))
    if len(postal_code) < 5:
        zeros_to_add = 5 - len(postal_code)
        for zero in range(zeros_to_add):
            postal_code = '0' + postal_code
    return postal_code

def get_zipcode_file(zipcode_filepath):
	zipcodes_df = pd.read_csv(zipcode_filepath)
	zipcodes_df = zipcodes_df[~zipcodes_df['Postal Code'].isnull()]
	zipcodes_df = zipcodes_df.drop('Unnamed: 7', axis = 1)
	zipcodes_df['Postal Code'] = zipcodes_df['Postal Code'].apply(lambda x: format_zipcode(x))
	return zipcodes_df

def get_quandl_data(zipcode_filepath, dataset_filepath):
	zipcodes_df = get_zipcode_file(zipcode_filepath)
	all_zipcodes = set(zipcodes_df['Postal Code'].tolist())
	try:
		df = pd.read_csv(dataset_filepath)
		df['Postal Code'] = df['Postal Code'].apply(lambda x: format_zipcode(x))
		pulled_zipcodes = set(df['Postal Code'].tolist())
		zipcodes_to_pull = list(all_zipcodes - pulled_zipcodes)
	except IOError:
		df = pd.DataFrame()
		zipcodes_to_pull = list(all_zipcodes)
	dataset_code = (dataset_filepath.split('.')[0]).split('_')[-1]
	total = len(zipcodes_to_pull)
	for index, value in enumerate(zipcodes_to_pull):
		columns_to_keep = [column for column in df.columns if '.1' not in str(column)]
		df = df[columns_to_keep]
		try:
			data = quandl.get("ZILL/Z{0}_{1}".format(value, dataset_code)).transpose()
		except quandl.NotFoundError:
			data = pd.DataFrame()
			print("{0} Not Found".format(value))
		data['Postal Code'] = value
		df = pd.concat([df, data])
		df.to_csv(dataset_filepath, index = False)
		print("{0} out of {1} Complete".format((index + 1), total))
		time.sleep(.3)
	print('Complete')

quandl.ApiConfig.api_key = "o3JsTtndey7CzUf1z6v6"
zipcode_filepath = sys.argv[1]
dataset_filepath = sys.argv[2]
get_quandl_data(zipcode_filepath, dataset_filepath)
