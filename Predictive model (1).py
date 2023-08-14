# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:31:12 2023

@author: Madhu
"""

import numpy as np
import pandas as pd
import pickle

# Load the model from the file
loaded_model = pickle.load(open('C:/Users/Madhu/OneDrive/Documents/Python Scripts/rf_model.pkl', 'rb'))


input_data= pd.DataFrame({'day.mins': [129.100], 
                          'voice.messages': [0], 
                          'day.charge': [21.95], 
                          'eve.mins': [228.500], 
                          'intl.plan_yes': [0], 
                          'customer.calls': [4], 
                          'night.mins': [208.80], 
                          'voice.plan_yes': [0],
                          'eve.charge': [19.4200], 
                          'intl.plan_no': [1], 
                          'account.length': [65.0]})

# Use the predict() method of the loaded model to make predictions for the input data
prediction = loaded_model.predict(input_data)

# Print the predicted value
print(prediction)

if prediction[0] == 0:
    print("This customer is not likely to churn.")
else:
    print("This customer is likely to churn.")