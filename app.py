import streamlit as st
import pickle
import numpy as np

# Load the trained KNN model
with open('best_knn_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Credit Score Prediction')

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    outstanding_debt = st.number_input('Outstanding Debt')
    interest_rate = st.number_input('Interest Rate')
    credit_history_age = st.number_input('Credit History Age')
    changed_credit_limit = st.number_input('Changed Credit Limit')

with col2:
    delay_from_due_date = st.number_input('Delay from Due Date')
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio')
    monthly_balance = st.number_input('Monthly Balance')
    annual_income = st.number_input('Annual Income')

# When the button is pressed
if st.button('Predict'):
    # Create an array of the input features
    features = np.array([
        outstanding_debt, 
        interest_rate, 
        credit_history_age, 
        changed_credit_limit, 
        delay_from_due_date, 
        credit_utilization_ratio, 
        monthly_balance, 
        annual_income
    ]).reshape(1, -1)
    
    # Make a prediction
    prediction = model.predict(features)
    
    # Display the result in a highlighted format
    st.markdown(f'### Predicted Credit Score: `{prediction[0]}`')



