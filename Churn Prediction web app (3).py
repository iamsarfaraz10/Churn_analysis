# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:42:24 2023

@author: Madhu
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the model from the file
loaded_model = pickle.load(open('C:/Users/Madhu/OneDrive/Documents/Python Scripts/rf_model.pkl', 'rb'))

def Churn(input_data):
    
    # Use the predict() method of the loaded model to make predictions for the input data
    prediction = loaded_model.predict(input_data)

    # Print the predicted value
    print(prediction)

    if prediction[0] == 0:
        return("This customer is not likely to churn.")
    else:
        return("This customer is likely to churn.")
    

    
def main():
    
    # Creating the user interface
    st.title("Customer Churn Prediction App")
    st.sidebar.header("Enter Customer Details")
    
    # Creating input fields for customer information
    day_mins = st.sidebar.number_input("Day Mins")
    voice_messages = st.sidebar.number_input("Voice Messages")
    day_charge = st.sidebar.number_input("Day Charge")
    eve_mins = st.sidebar.number_input("Eve Mins")
    intl_plan_yes = st.sidebar.radio("International Plan (Yes/No)", ["Yes", "No"])
    customer_calls = st.sidebar.number_input("Customer Calls")
    night_mins = st.sidebar.number_input("Night Mins")
    voice_plan_yes = st.sidebar.radio("Voice Plan (Yes/No)", ["Yes", "No"])
    eve_charge = st.sidebar.number_input("Eve Charge")
    intl_plan_no = 1 if intl_plan_yes == "No" else 0
    account_length = st.sidebar.number_input("Account Length")
    
    # Creating a dictionary for the input data
    data = {"day.mins": day_mins, "voice.messages": voice_messages, "day.charge": day_charge, "eve.mins": eve_mins,
        "intl.plan_yes": 1 if intl_plan_yes == "Yes" else 0, "customer.calls": customer_calls, "night.mins": night_mins,
        "voice.plan_yes": 1 if voice_plan_yes == "Yes" else 0, "eve.charge": eve_charge, "intl.plan_no": intl_plan_no,
        "account.length": account_length}
    
    # define the variable intl_plan
    intl_plan = 'Yes'
    voice_plan = 'Yes'
    
    input_data = pd.DataFrame({
    'day.mins': [day_mins], 
    'voice.messages': [voice_messages], 
    'day.charge': [day_charge], 
    'eve.mins': [eve_mins], 
    'intl.plan_yes': [1 if intl_plan.lower() == 'yes' else 0], 
    'customer.calls': [customer_calls], 
    'night.mins': [night_mins], 
    'voice.plan_yes': [1 if voice_plan.lower() == 'yes' else 0],
    'eve.charge': [eve_charge], 
    'intl.plan_no': [0 if intl_plan.lower() == 'yes' else 1], 
    'account.length': [account_length]
})
  
    # Selecting only the required features from the input data
    model_input = input_data[['day.mins', 'voice.messages', 'day.charge', 'eve.mins', 'intl.plan_yes', 'customer.calls',
                          'night.mins', 'voice.plan_yes', 'eve.charge', 'intl.plan_no', 'account.length']]
   
   
    
    prediction = ""
    
    # Loading the trained model using pickle
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
       
    #Creating a button for prediction
    if st.sidebar.button("Predict"):
        prediction = rf_model.predict(model_input)
        if prediction[0] == 0:
            st.success("This customer is not likely to churn.")
        else:
            st.error("This customer is likely to churn.")
        
    st.success(prediction)



if __name__ == "__main__":
    main()


    
    















