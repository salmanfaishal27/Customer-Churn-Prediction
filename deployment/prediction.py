import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json


# Load All Files
from tensorflow.keras.models import load_model

with open('pipeline.pkl', 'rb') as file_1:
  model_pipeline = pickle.load(file_1)

model_ann = load_model('seqimp_model.h5')

def run():
    st.title('Job Placement Prediction')
    # Membuat form
    with st.form(key='form_parameters'):
        
        age= st.number_input('Age', min_value=0, max_value=100, value=25, step =1, help= 'Age.')
        gender= st.selectbox('Gender', ('Male','Female'), index=1, help='Gender of the Customers.')
        region_category= st.selectbox('Region Category', ('City','Village','Town'), index=1, help='Region that a customer belongs to.')
        membership_category= st.selectbox('Membership', (1,2,3,4,5,6), index=1, help='1. No Membership, 2. Basic Membership, 3.Silver Membership, 4. Gold Membership, 5. Premium Membership, 6. Platinum Membership .')
        joined_through_referral= st.selectbox('Refferal', ('Yes','No'), index=1, help='Whether a customer joined using any referral code or ID.')
        preferred_offer_types= st.selectbox('Preffered offer types', ('Without Offers','Credit/Debit Card Offers','Gift Vouchers/Coupons'), index=1, help='Type of offer that a customer prefers.')
        avg_transaction_value= st.number_input('Average Transaction Value', min_value=0, value=25, step =1, help= 'Average transaction value of a customer.')
        avg_frequency_login_days= st.number_input('Average frequency login days ', min_value=0, value=25, step =1, help= 'Number of times a customer has logged in to the website.')
        points_in_wallet= st.number_input('Reward Point', min_value=0, value=900, step =1, help= '	Points awarded to a customer on each transaction.')
        used_special_discount= st.selectbox('Discount', ('Yes','No'), index=1, help='Whether a customer uses special discounts offered.')
        complaint_status= st.selectbox('Complaint Status', ('Not Applicable','Unsolved','Solved','Solved in Follow-up','No Information Available'), index=1, help='	Whether the complaints raised by a customer was resolved.')
        feedback= st.selectbox('Feedback', ('Poor Product Quality','No reason specified','Too many ads','Poor Website','Poor Customer Service','Reasonable Price','User Friendly Website','Products always in Stock','Quality Customer Care'), index=1, help='Feedback provided by a customer.')
        

        submitted = st.form_submit_button('Predict')

    data_inf = {
    'age': age,
    'gender': gender,
    'region_category': region_category,
    'membership_category': membership_category,
    'joined_through_referral': joined_through_referral,
    'preferred_offer_types': preferred_offer_types,
    'avg_transaction_value': avg_transaction_value,
    'avg_frequency_login_days': avg_frequency_login_days,
    'points_in_wallet': points_in_wallet,
    'used_special_discount': used_special_discount,
    'complaint_status': complaint_status,
    'feedback': feedback,
    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    data_inf_transform = model_pipeline.transform(data_inf)
    

    if submitted:       
        y_pred_inf = model_ann.predict(data_inf_transform)
        if y_pred_inf >= 0.5 :
            st.write('## Churn ? Yes')
        else:
            st.write('## Churn ? No')

      
if __name__ == '__main__':
    run()