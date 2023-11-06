# Convert data into csv. This script will also map the Type and Material columns to numerical values
import pandas as pd 
import geopandas as gd

def map_col_to_numeric(dataframe, column_name):
	col = dataframe[column_name].unique()
	output_mapping = {}
	for index, ele in enumerate(col):
		output_mapping[ele] = index
	dataframe = dataframe.replace({column_name : output_mapping})
	return dataframe, output_mapping

# replace path with your own data path
gdf_2019_survey = gd.read_file('[Your_own_path]/data/2019 Survey')
data_2019 = pd.DataFrame(data=gdf_2019_survey)
# delete unused columns
data_2019 = data_2019.drop(columns=['Descriptio', 'GlobalID', 'CreationDa', 'Creator', 'EditDate', 'Editor'])

# map columns to numeric values
data_2019, type_mapping = map_col_to_numeric(data_2019, "Type")
data_2019, material_mapping = map_col_to_numeric(data_2019, "Material")

# create columns for x, y, and z
data_2019['X'] = data_2019.apply(lambda row: row.geometry.x, axis = 1)
data_2019['Y'] = data_2019.apply(lambda row: row.geometry.y, axis = 1)
data_2019['Z'] = data_2019.apply(lambda row: row.geometry.z, axis = 1)
# drop goometry column
data_2019 = data_2019.drop(columns=['geometry'])

print(data_2019)
print(data_2019.dtypes)

data_2019.to_csv('data_training.csv', index = False)
