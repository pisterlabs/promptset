import os
import openai
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('predictions.csv')
# Load your API key from an environment variable or secret management service
openai.api_key = os.environ['OPENAI_API_KEY']
st.title('Customer Retention Email Generation')
'\n'
customerid = st.selectbox('Select a customer',df['customerid'].unique())
df['gender']=df['gender'].map({'Male':0,'Female':1})
df[['engagementscore','proba']]=(MinMaxScaler().fit_transform(df[['engagementscore','proba']])*100).astype(int)
df = df[df['customerid']==customerid]

st.header('Customer Profile')
st.write(df.T)
st.header('Configuration options')
st.write('Email 1')
email1_threshold = st.slider('Select a range for churn probability',0,100,(25,50))
st.write('Email 2')
email2_threshold = st.slider('Select a range for churn probability',0,100,(50,75))
st.write('Email 3')
email3_threshold = st.slider('Select a range for churn probability',0,100,(75,100))

proba = int(df['proba'].values[0])

discount = st.slider('What discount are we willing to offer?', 0, 100, 5)
emails = ['''Subject: Get Ready to Experience Our New Product Promotion

Dear [Name],

We hope you are doing well!

As one of our valued customers, we wanted to make sure you were aware of our new product promotion. We have launched a special promotion that provides our customers with an exclusive discount on our new products.

This promotion is only available for a limited time and gives you the chance to save big on our new products. You will also receive access to exclusive offers and discounts that are only available to our customers.

We hope that you take advantage of this great offer and experience all the benefits of being one of our valued customers. If you have any questions or would like more information, please do not hesitate to contact us.

We look forward to hearing from you soon.

Sincerely,
[Your Name]''',
'''Subject: Enjoy [Discount]% Off Our New Products 

Dear [Name],

We are excited to announce that we have launched a new promotional campaign featuring a 20% discount on our new products! 

As one of our valued customers, you can take advantage of this exclusive offer and save big on our newest products. This promotion is only available for a limited time, so don't miss out on this great opportunity! 

Not only will you save [Discount]% on our new products, but you will also gain access to exclusive offers and discounts available only to our customers. 

We hope that you take advantage of this great offer and experience all the benefits of being one of our valued customers. If you have any questions or would like more information, please do not hesitate to contact us.

We look forward to hearing from you soon! 

Sincerely,
[Your Name]''',
'''Subject: Get Ready to Experience Our Exclusive Discount Offer

Dear [Name],

We hope you are doing well!

As one of our valued customers, we wanted to make sure you were aware of our exclusive discount offer. We have launched a special offer that provides our customers with an exclusive discount on our products.

This offer is only available for a limited time and gives you the chance to save big on our products. You will also receive access to exclusive offers and discounts that are only available to our customers.

Don't miss out on this amazing offer and experience all the benefits of being one of our valued customers. Click on the link below to take advantage of this offer now!

[Link Here]

We look forward to hearing from you soon.

Sincerely,
[Your Name]''']


if st.button('Generate Email'):
	text = emails[0] if proba in range(*email1_threshold) else emails[1] if proba in range(*email2_threshold) else emails[2] if  proba in range(*email3_threshold) else ''
	if text!='':
		prompt='''Instruction: Write an email in minimum 400 words similar to the example email. Strictly adhere to minimum word limits. Address the email to {}.discount percentage to offer is {}%.\n'''.format(customerid,discount,)
		prompt+="Example Email: "+text
		prompt+='\n400 word Email:\n'
		response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0.4, max_tokens=3500)
		mail = response['choices'][0]['text']
		st.header('Email')
		st.write(mail)
		print(len(mail.split()))
	else:
		st.write('No Action Needed')

