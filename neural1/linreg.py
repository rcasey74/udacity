# TODO: Add import statements
import pandas as pd
#import numpy as np
from sklearn import linear_model

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = linear_model.LinearRegression()

x_values = bmi_life_data[['BMI']]
#x_values.values.reshape(-1,1)
#print('Num x values: %d' % x_values.size)


y_values = bmi_life_data[['Life expectancy']]
#y_values.values.reshape(-1,1)
#print('Num y values: %d' % y_values.size)


bmi_life_model.fit( x_values, y_values )

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
test_BMI = 21.07931
laos_life_exp = bmi_life_model.predict(test_BMI)
print('Laos life expectancy = %f' % laos_life_exp )
