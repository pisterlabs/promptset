#!/usr/bin/env python
# coding: utf-8

# In[31]:


import openai
import pandas as pd
import re
import math
import time
from tenacity import (retry,stop_after_attempt,wait_random_exponential)


# In[32]:


output_data = pd.read_excel('rpa pricing extraction.xlsx')


# In[33]:


#output_data = input_data.copy()


# In[34]:


output_data["Max Discount"]=None
output_data["IBD Bank Name/UPI"]=None
output_data["Min Swipe"]=None
output_data["Bump up"]=None
output_data["VPC(Amazon)"]=None
output_data["SuperCoins(Flipkart)"]=None
output_data.drop('BUP', axis=1, inplace=True)
output_data.drop('VPC', axis=1, inplace=True)
output_data.drop('SuperCoins', axis=1, inplace=True)
output_data


# In[35]:


def create_content(price, offers):
    content=f"The Price of the SKU is {price}\n\nOffers:\n\n{offers}\n\nWhich is the best Offer?"
    return content


# In[36]:


def extract_discount_details(offer_details):
    # Split the input string into lines
    lines = offer_details.split('\n')
    
    max_discount = 0
    bank_name = ""
    min_swipe = 0
    
    for line in lines:
        if "maximum discount applicable =" in line:
            discount_str = line.split("maximum discount applicable =")[1]
            discount_str = discount_str[:10]
            if(pd.isna(discount_str)==False):
                discount_value = re.sub(r'[^\d]', '', discount_str)
                discount = float(discount_value)
                max_discount = max(max_discount, discount)
        
        if "Bank Name =" in line:
            bank_name = line.split("Bank Name =")[1].strip()
            if "Minimum Swipe for availing maximum Discount" in bank_name:
                bank_name = bank_name.replace("Minimum Swipe for availing maximum Discount","")
        
        if "Minimum Swipe for availing maximum Discount =" in line:
            min_swipe_str = line.split("Minimum Swipe for availing maximum Discount = ")[1]
            min_swipe_str = min_swipe_str[:10]
            if(pd.isna(min_swipe_str)==False and min_swipe_str != "Not mentioned"):
                min_swipe_value = re.sub(r'[^\d]', '', min_swipe_str)
                min_swipe = float(min_swipe_value)
            #min_swipe = int(min_swipe_str.replace(',', ''))
    
    return max_discount, bank_name, min_swipe


# In[37]:


def extract_bup_details(bup_details):
    bup=0
    bup = re.sub(r'[^\d.]', '', bup_details)
    return bup
    


# In[38]:


def extract_vpc_details(vpc_details):
    vpc=0
    vpc = re.sub(r'[^\d.]', '', vpc_details)
    return vpc


# In[39]:


def extract_supercoins_details(supercoins_details):
    supercoins=0
    #supercoins = re.sub(r'[^\d.]', '', supercoins_details)
    supercoins=re.findall(r'\d+', supercoins_details)[0] if re.findall(r'\d+', supercoins_details) else None
    return supercoins


# In[40]:


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(model, messages):
    #print("MODEL", model)
    output = openai.ChatCompletion.create(model=model, messages=messages)
    return output


# In[41]:


for i in range(len(input_data)):
    #time.sleep(1)
    print("value of i", i)
    sku_price = input_data['Current Price'][i]
    sku_offers = input_data['Bank Offers'][i]
    content = create_content(sku_price, sku_offers)
    #print("CONTENT", content)
    messages_2 = [
    {"role": "system", "content": "You are a helpful assistant which helps in the identification of the best offer. If any offer have more than one discount % always take the higher one. Summarize the best offer at the end by clearly mentioning Maximum discount/cashback, Bank Name & minimum swipe amount on separate lines"},
    {"role": "user", "content": "The Price of SKU is 17999\n\nOffers:\n\n1. Get 5% back with Amazon Pay ICICI Bank credit card for Prime members. 3% back for others. Not applicable on Amazon business transactions.\n2. ₹100 cashback & ₹500 welcome rewards on Amazon Pay Later. Activate now. Pay next month at Zero interest or in EMIs! T&C apply.\n3. Flat INR 500 Instant Discount on ICICI Bank Credit Cards (excluding Amazon Pay ICICI Credit Card) Credit Card Txn. Minimum purchase 4. value INR 15999\n4. Flat INR 750 Instant Discount on OneCard Credit Card Non EMI Txn. Minimum purchase value INR 15999\n5. Flat ₹3,000 off on HDFC Bank Credit Card EMI Trxns on orders priced ₹50,000 and above\n\nWhich is the best offer?"},
    {"role": "assistant", "content": "Assuming prime member from the first offer the maximum discount applicable is 5%, since 5% is greater than 3%. So the discount amount is 5% of price of sku, which is 17999. This equals 900. From the second offer the maximum discount is 100 cashback + 500 welcome rewards= 600. so the discount from this offer is 600. From the third offer since the purchase value 17999 is greater than Minimum purchase value of ₹15999, hence the discount amount is applicable & it is equal to ₹500. From the fourth offer the discount = ₹750 since the Minimum pucrhase value of ₹15999 is less than than ₹17999. From the fifth offer the discount is not applicable since sku price (17999) is less than 50000. Comparing the discounts from all offers the maximum discount is ₹900. So maximum discount = ₹900; Bank Name = Amazon Pay ICICI Bank, Minimum Swipe for availing maximum discount = 0"},
    {"role": "user", "content": "The Price of the SKU is 18999\n\nOffers:\n\n1. Flat INR 2000 Instant Discount on HDFC Bank Credit CardTxn. Minimum purchase value INR 17999\n\n2. Flat INR 2100 Instant Discount on ICICI Bank Credit Cards (excluding Amazon Pay ICICI Credit Card) Credit CardTxn. Minimum purchase value INR 17999\n\n3. Flat INR 2000 Instant Discount on HDFC Bank Debit Card EMI Txn. Minimum purchase value INR 19999\n\n4. 5% Instant Discount up to INR 250 on HSBC Cashback Card Credit Card Transactions. Minimum purchase value INR 1000\n\n5. Get 5% back with Amazon Pay ICICI Bank credit card for Prime members. 3% back for others. Not applicable on Amazon business transactions.\n\n6. ₹100 cashback & ₹2150 welcome rewards on Amazon Pay Later.\n\nWhich is the best Offer"},
    {"role": "assistant", "content": "From the first offer since the price value of 18999 is greater than 17999 the discount amount is applicable & it is equal to 2000 . From the second offer since the price value of 18999 is greater than 17999 the discount amount is 2100. From the third offer since the price value of 18999 is less than minimum purchase value of 19999, the discount value is not applicable. From the fourth offer, the discount amount is 5% of the price of the SKU, up to a maximum of 250. Since the price of the SKU is 18999, the discount amount will be 250. Assuming prime member, from the fifth offer, the maximum discount applicable is 5%, since 5% is greater than 3%. So the discount amount is 5% of the price of the SKU, which is 18999. This equals 950.From the sixth offer, the discount amount is 100 cashback + 2150 welcome rewards = 2250. Comparing the discounts from the offers, the maximum discount is 2250. So, \nthe maximum discount applicable = 2250\nBank Name = Amazon Pay Later\nMinimum Swipe for availing maximum Discount = 0"},
    {"role": "user", "content": "The Price of the SKU is 18999\n\nOffers:\n\n1. Flat INR 2000 Instant Discount on HDFC Bank Credit CardTxn. Minimum purchase value INR 17999\n\n2. Flat INR 2100 Instant Discount on ICICI Bank Credit Cards (excluding Amazon Pay ICICI Credit Card) Credit CardTxn. Minimum purchase value INR 17999\n\n3. Flat INR 2000 Instant Discount on HDFC Bank Debit Card EMI Txn. Minimum purchase value INR 19999\n\n4. 5% Instant Discount up to INR 250 on HSBC Cashback Card Credit Card Transactions. Minimum purchase value INR 1000\n\n5. Get 5% back with Amazon Pay ICICI Bank credit card for Prime members. 3% back for others. Not applicable on Amazon business transactions.\n\n6. ₹100 cashback & ₹500 welcome rewards on Amazon Pay Later.\n\nWhich is the best Offer?"},
    {"role": "assistant", "content": "From the first offer since the price value of 18999 is greater than 17999 the discount amount is applicable & it is equal to 2000 . From the second offer since the price value of 18999 is greater than 17999 the discount amount is applicable & it is equal to 2100. From the third offer since the price value of 18999 is less than minimum purchase value of 19999, the discount value is not applicable. From the fourth offer, the discount amount is 5% of the price of the SKU, up to a maximum of 250. Since the price of the SKU is 18999, the discount amount will be 250. Assuming prime member, from the fifth offer, the maximum discount applicable is 5%, since 5% is greater than 3%. So the discount amount is 5% of the price of the SKU, which is 18999. This equals 950.From the sixth offer, the discount amount is 100 cashback + 500 welcome rewards = 600. Comparing the discounts from the offers, the maximum discount is 2000. So, \nthe maximum discount applicable = 2000\nBank Name = HDFC Bank Credit Card\nMinimum Swipe for availing maximum Discount = 17999"},
    {"role": "user", "content": content}
    ]
    output= completion_with_backoff(model="gpt-3.5-turbo",messages=messages_2)
    print(f"FEW SHOT INTERFERENCE-OPEN AI: \n{output['choices'][0]['message']['content']}")
    response = output['choices'][0]['message']['content']
    max_discount, bank_name, min_swipe = extract_discount_details(response)
    output_data["Max Discount"][i] = max_discount
    output_data['IBD Bank Name/UPI'][i] = bank_name
    output_data['Min Swipe'][i]= min_swipe

    
    if input_data["BUP"][i] != None and pd.isna(input_data["BUP"][i])==False:
        bup= extract_bup_details(input_data["BUP"][i])
        print(bup)
        output_data['Bump up'][i]=bup

    if input_data["VPC"][i] != None and pd.isna(input_data["VPC"][i])==False:
        print("VPC value from input file", input_data["VPC"][i])
        vpc= extract_vpc_details(input_data["VPC"][i])
        output_data['VPC(Amazon)'][i]=vpc

    if input_data["SuperCoins"][i] != None and pd.isna(input_data["SuperCoins"][i])==False:
        print('supercoins', input_data["SuperCoins"][i])
        supercoins= extract_supercoins_details(input_data["SuperCoins"][i])
        output_data['SuperCoins(Flipkart)'][i]=supercoins

output_data.to_excel('output_data.xlsx')


# In[ ]:





# In[ ]:




