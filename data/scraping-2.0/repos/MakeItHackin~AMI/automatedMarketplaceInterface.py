print('======================================================')
print('STARTING... ')

import streamlit as st
import pandas as pd
import json
import etsyGetJson as etsy
import tindieGetJson as tindie
import lectronzGetJson as lectronz
import shopifyGetJson as shopify
import streamlit.components.v1 as components
import streamlit as st
import os

import uploadCode
import printReceipt
import platformFunctions
import shipEngineFunctions as shipEngine
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import math
import shlex
import warnings
import openai
import random
# Use the filterwarnings() function to suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import requests

load_dotenv()  # Load environment variables from .env file

shopify_shipping_location_id = os.getenv('shopify_shipping_location_id')
open_ai_api_key = os.getenv('open_ai_api_key')

st.set_page_config(layout='wide', initial_sidebar_state='collapsed')

try:
    if (st.session_state["label_tracking_number"] != ""):
        st.session_state['disabled_ship_package'] = True
    else:
        order_selected = st.session_state['shipping_order_dropdown'][0]
        if (order_selected != 'Select an order'):
            st.session_state['disabled_ship_package'] = False 
except:
    pass


        
def cleanUpText(text):
    
    updated_string = text.replace('“', '"')
    updated_string = updated_string.replace('”', '"')
    updated_string = updated_string.replace('’', "'")
    updated_string = updated_string.replace('‘', "'")
    updated_string = updated_string.replace('–', "-")
    updated_string = updated_string.replace('—', "-")
    updated_string = updated_string.replace('…', "...")
    updated_string = updated_string.replace('™', "(TM)")
    updated_string = updated_string.replace('®', "(R)")
    updated_string = updated_string.replace('©', "(C)")
    updated_string = updated_string.replace('°', "deg")
    updated_string = updated_string.replace('•', "*")
    updated_string = updated_string.replace('²', "2")
    updated_string = updated_string.replace('³', "3")
    updated_string = updated_string.replace('¼', "1/4")
    updated_string = updated_string.replace('½', "1/2")
    updated_string = updated_string.replace('¾', "3/4")
    updated_string = updated_string.replace('×', "x")
    updated_string = updated_string.replace('÷', "/")
    updated_string = updated_string.replace('≤', "<=")
    updated_string = updated_string.replace('≥', ">=")
    updated_string = updated_string.replace('≠', "!=")
    updated_string = updated_string.replace('≈', "~")
    updated_string = updated_string.replace('±', "+/-")
    updated_string = updated_string.replace('µ', "u")
    updated_string = updated_string.replace('∞', "oo")
    updated_string = updated_string.replace('√', "sqrt")
    updated_string = updated_string.replace('∆', "delta")
    updated_string = updated_string.replace('∑', "sum")
    updated_string = updated_string.replace('∏', "prod")
    updated_string = updated_string.replace('∫', "int")
    updated_string = updated_string.replace('∮', "oint")
    updated_string = updated_string.replace('∴', "therefore")
    updated_string = updated_string.replace('∵', "because")
    updated_string = updated_string.replace('∼', "~")
    updated_string = updated_string.replace('≡', "==")
    updated_string = updated_string.replace('≅', "~")
    updated_string = updated_string.replace('≈', "~")
    updated_string = updated_string.replace('≠', "!=")
    updated_string = updated_string.replace('≡', "==")
    updated_string = updated_string.replace('≤', "<=")
    updated_string = updated_string.replace('≥', ">=")
    updated_string = updated_string.replace('≪', "<<")
    updated_string = updated_string.replace('≫', ">>")
    updated_string = updated_string.replace('⊂', "subset of")
    updated_string = updated_string.replace('⊃', "superset of")
    updated_string = updated_string.replace('⊆', "subset of or equal to")
    updated_string = updated_string.replace('⊇', "superset of or equal to")
    updated_string = updated_string.replace('⊕', "+")
    updated_string = updated_string.replace('⊗', "x")
    updated_string = updated_string.replace('⊥', "perpendicular")
    updated_string = updated_string.replace('⋅', ".")
    updated_string = updated_string.replace('⌈', "[")
    updated_string = updated_string.replace('⌉', "]")
    updated_string = updated_string.replace('⌊', "[")
    updated_string = updated_string.replace('⌋', "]")
    updated_string = updated_string.replace('〈', "<")
    updated_string = updated_string.replace('〉', ">")

    updated_string = updated_string.replace('č', "c")
    updated_string = updated_string.replace('&quot;', "\"")
    
    updated_string = updated_string.replace('&lt;', "<")
    updated_string = updated_string.replace('&#39;', "\'")
    
    return(updated_string)
            

firstClassInternationalWordList = ['Standard International (Global Postal Shipping or USPS First Class International)', 'Standard International', 'USPS First-Class Mail International', 'first class international', 'first-class international', 'first class international mail', 'first-class international mail', 'first class international package', 'first-class international package', 'standard international rate', 'first class package international']
firstClassWordList = ['USPS First-Class Mail', 'first class', 'first-class', 'first class mail', 'first-class mail', 'first class package', 'first-class package']
priorityMailExpressInternationalWordList = ['USPS Priority Mail Express International', 'priority mail express international', 'priority-mail express international', 'priority mail express international flat rate', 'priority-mail express international flat rate', 'priority mail express international large flat rate', 'priority-mail express international large flat rate', 'priority mail express international small flat rate', 'priority-mail express international small flat rate', 'priority mail express international padded flat rate', 'priority-mail express international padded flat rate', 'priority mail express international legal flat rate', 'priority-mail express international legal flat rate', 'priority mail express international gift card flat rate', 'priority-mail express international gift card flat rate', 'priority mail express international window flat rate', 'priority-mail express international window flat rate']
priorityMailInternationalWordList = ['USPS Priority Mail International', 'priority mail international', 'priority-mail international', 'priority mail international flat rate', 'priority-mail international flat rate', 'priority mail international large flat rate', 'priority-mail international large flat rate', 'priority mail international small flat rate', 'priority-mail international small flat rate', 'priority mail international padded flat rate', 'priority-mail international padded flat rate', 'priority mail international legal flat rate', 'priority-mail international legal flat rate', 'priority mail international gift card flat rate', 'priority-mail international gift card flat rate', 'priority mail international window flat rate', 'priority-mail international window flat rate']
priorityMailExpressWordList = ['USPS Priority Mail Express', 'priority mail express', 'priority-mail express', 'priority mail express flat rate', 'priority-mail express flat rate', 'priority mail express large flat rate', 'priority-mail express large flat rate', 'priority mail express small flat rate', 'priority-mail express small flat rate', 'priority mail express padded flat rate', 'priority-mail express padded flat rate', 'priority mail express legal flat rate', 'priority-mail express legal flat rate', 'priority mail express gift card flat rate', 'priority-mail express gift card flat rate', 'priority mail express window flat rate', 'priority-mail express window flat rate']
priorityMailWordList = ['USPS Priority Mail', 'priority mail', 'priority-mail', 'priority mail flat rate', 'priority-mail flat rate', 'priority mail large flat rate', 'priority-mail large flat rate', 'priority mail small flat rate', 'priority-mail small flat rate', 'priority mail padded flat rate', 'priority-mail padded flat rate', 'priority mail legal flat rate', 'priority-mail legal flat rate', 'priority mail gift card flat rate', 'priority-mail gift card flat rate', 'priority mail window flat rate', 'priority-mail window flat rate']
groundAdvantageWordList = ['standard ground rate', 'USPS Ground Advantage', 'ground advantage', 'ground-advantage', 'ground advantage flat rate', 'ground-advantage flat rate', 'ground advantage large flat rate', 'ground-advantage large flat rate', 'ground advantage small flat rate', 'ground-advantage small flat rate', 'ground advantage padded flat rate', 'ground-advantage padded flat rate', 'ground advantage legal flat rate', 'ground-advantage legal flat rate', 'ground advantage gift card flat rate', 'ground-advantage gift card flat rate', 'ground advantage window flat rate', 'ground-advantage window flat rate']
mediaMailWordList = ['USPS Media Mail', 'media mail', 'media-mail', 'media mail flat rate', 'media-mail flat rate', 'media mail large flat rate', 'media-mail large flat rate', 'media mail small flat rate', 'media-mail small flat rate', 'media mail padded flat rate', 'media-mail padded flat rate', 'media mail legal flat rate', 'media-mail legal flat rate', 'media mail gift card flat rate', 'media-mail gift card flat rate', 'media mail window flat rate', 'media-mail window flat rate']

def convert_to_lowercase(input_list):
    lowercase_list = [item.lower() if isinstance(item, str) else item for item in input_list]
    return lowercase_list

firstClassInternationalWordList = convert_to_lowercase(firstClassInternationalWordList)
firstClassWordList = convert_to_lowercase(firstClassWordList)
priorityMailExpressInternationalWordList = convert_to_lowercase(priorityMailExpressInternationalWordList)
priorityMailInternationalWordList = convert_to_lowercase(priorityMailInternationalWordList)
priorityMailExpressWordList = convert_to_lowercase(priorityMailExpressWordList)
priorityMailWordList = convert_to_lowercase(priorityMailWordList)
groundAdvantageWordList = convert_to_lowercase(groundAdvantageWordList)
mediaMailWordList = convert_to_lowercase(mediaMailWordList)

def shipping_order_change():
    order_selected = st.session_state['shipping_order_dropdown'][0]
    if (order_selected != 'Select an order'):
        print('ORDER HAS CHANGED:', order_selected) #prints the index of the option selected
        
        st.session_state["ground_advantage_price"] = '  $' + "{:.2f}".format(0.00)
        st.session_state["media_mail_price"] = '  $' + "{:.2f}".format(0.00)
        st.session_state["first_class_price"] = '  $' + "{:.2f}".format(0.00)
        st.session_state["priority_mail_price"] = '  $' + "{:.2f}".format(0.00)
        st.session_state["priority_mail_express_price"] = '  $' + "{:.2f}".format(0.00)    
        st.session_state["priority_mail_express_international_price"] = '  $' + "{:.2f}".format(0.00)
        st.session_state["priority_mail_international_price"] = '  $' + "{:.2f}".format(0.00)
        st.session_state["first_class_international_price"] = '  $' + "{:.2f}".format(0.00)
        st.session_state["priority_mail_small_box_price"] = '  $' + "{:.2f}".format(0.00)
        st.session_state['address_verification'] = ""
        #st.session_state['disabled_fulfill_order'] = 
        st.session_state['disabled_ship_package'] = False
        st.session_state["label_tracking_number"] = ""
        
        comboOrderID = order_selected.split('--')[0].strip()
        shipping_name = df.loc[df['order id'] == comboOrderID, 'recipient_name'].values[0]
        #print('shipping_name------:', shipping_name)
        shipping_addressline1 = df.loc[df['order id'] == comboOrderID, 'buyer_address1'].values[0]
        shipping_addressline2 = df.loc[df['order id'] == comboOrderID, 'buyer_address2'].values[0]
        shipping_addressline3 = df.loc[df['order id'] == comboOrderID, 'buyer_address3'].values[0]
        shipping_city = df.loc[df['order id'] == comboOrderID, 'buyer_city'].values[0]
        shipping_state = df.loc[df['order id'] == comboOrderID, 'buyer_state_province'].values[0]
        shipping_zip = df.loc[df['order id'] == comboOrderID, 'buyer_postal_code'].values[0]
        shipping_country_code = df.loc[df['order id'] == comboOrderID, 'buyer_country_code'].values[0]
        shipping_method = df.loc[df['order id'] == comboOrderID, 'order_shipping_method'].values[0]
        shipping_total = df.loc[df['order id'] == comboOrderID, 'order_shipping_total'].values[0]
        order_value = df.loc[df['order id'] == comboOrderID, 'order_items_total'].values[0]
        dfwp = df.loc[df['order id'] == comboOrderID, 'DFWP'].values[0]
        prni = df.loc[df['order id'] == comboOrderID, 'PRNI'].values[0]
        
        st.session_state['addressName'] = shipping_name
        st.session_state['addressLine1'] = shipping_addressline1
        st.session_state['addressLine2'] = shipping_addressline2
        st.session_state['addressLine3'] = shipping_addressline3
        st.session_state['addressCity'] = shipping_city
        st.session_state['addressState'] = shipping_state
        st.session_state['addressZip'] = shipping_zip
        st.session_state['addressCountry'] = shipping_country_code
        st.session_state['shippingPaid'] = shipping_total
        #print(order_value)
        order_value = order_value.replace('$','')
        #st.session_state['packageValue'] = math.ceil(float(order_value[1:]))
        st.session_state['packageValue'] = math.ceil(float(order_value))
        
        #if any(shipping_method.lower() == item.lower() for item in firstClassInternationalWordList if isinstance(item, str)):
        if (shipping_method.lower() in firstClassInternationalWordList):
            #shipping_method = ('First Class International',)
            shipping_method = ('First Class International' + str(st.session_state.get("first_class_international_price", " 0.00")),6)
        #elif any(shipping_method.lower() == item.lower() for item in firstClassWordList if isinstance(item, str)):
        elif (shipping_method.lower() in firstClassWordList):
            #shipping_method = ('First Class',)
            shipping_method = ('First Class' + str(st.session_state.get("first_class_price", " 0.00")),2)
        #elif any(shipping_method.lower() == item.lower() for item in priorityMailExpressInternationalWordList if isinstance(item, str)):
        elif (shipping_method.lower() in priorityMailExpressInternationalWordList):
            #shipping_method = ('Priority Mail Express International',)
            shipping_method = ('Priority Mail Express International' + str(st.session_state.get("priority_mail_express_international_price", " 0.00")),8)
        #elif any(shipping_method.lower() == item.lower() for item in priorityMailExpressWordList if isinstance(item, str)):
        elif (shipping_method.lower() in priorityMailExpressWordList):
            #shipping_method = ('Priority Mail Express',)
            shipping_method = ('Priority Mail Express' + str(st.session_state.get("priority_mail_express_price", " 0.00")),5)
        #elif any(shipping_method.lower() == item.lower() for item in priorityMailInternationalWordList if isinstance(item, str)):
        elif (shipping_method.lower() in priorityMailInternationalWordList):
            #shipping_method = ('Priority Mail International',)
            shipping_method = ('Priority Mail International' + str(st.session_state.get("priority_mail_international_price", " 0.00")),7)
        #elif any(shipping_method.lower() == item.lower() for item in priorityMailWordList if isinstance(item, str)):
        elif (shipping_method.lower() in priorityMailWordList):
            #shipping_method = ('Priority Mail',)
            shipping_method = ('Priority Mail' + str(st.session_state.get("priority_mail_price", " 0.00")),3)
        elif (shipping_method.lower() in groundAdvantageWordList):
            #shipping_method = ('Ground Advantage',)
            shipping_method = ('Ground Advantage' + str(st.session_state.get("ground_advantage_price", " 0.00")),1)
        elif (shipping_method.lower() in mediaMailWordList):
            #shipping_method = ('Media Mail',)
            shipping_method = ('Media Mail' + str(st.session_state.get("media_mail_price", " 0.00")),9)
        else:
            shipping_method = ('Unknown',10)
        
        st.session_state['shipping_method_dropdown'] = shipping_method
        
        if (shipping_country_code != 'US'):
            st.session_state['insuranceCheckBox'] = True
        else:
            st.session_state['insuranceCheckBox'] = False

    else:
        resetTab1Fields()
        
    
def update_manifest_count():
    try:
        date = st.session_state['shippingDate']
        date = date.strftime("%Y-%m-%d")
        count = shipEngine.getManifestCount(date)
        st.session_state['manifest_packages'] = str(count) 
    except:
        st.session_state['manifest_packages'] = 'Needs refresh' 
    return    

def load_data(etsyOrFile = True):
    
    if (st.session_state['data_source_radio'] != 'Fake'):
        if (etsyOrFile == True):
            
            if (st.session_state["exchange_rate_checkbox"] == True):
                eurToUsdExchangeRate = lectronz.getExchangeRate()
                if (eurToUsdExchangeRate != 0):
                    st.session_state["eur_to_usd_exchange_rate"] = eurToUsdExchangeRate
                    print('eurToUsdExchangeRate:', eurToUsdExchangeRate)
                else:
                    print('ERROR: eurToUsdExchangeRate is 0')
            else:
                st.session_state["eur_to_usd_exchange_rate"] = 1
                print('Checkbox is not checked, don\'t update')
                
            exchangeRateForLectronz = st.session_state["eur_to_usd_exchange_rate"]

            include_completed = st.session_state['completedOrderCheckBox']
            amount_completed = st.session_state['amount_of_completed_orders']
            amount_offset = st.session_state['amount_of_completed_orders_offset']
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            try:
                newEtsyData = etsy.getNewOrders()#include_completed, amount_completed, amount_offset)
            except:
                print('ERROR: etsy.getNewOrders()')
                newEtsyData = []
            try:            
                newTindieData = tindie.getNewOrders()
            except:
                print('ERROR: tindie.getNewOrders()')
                newTindieData = []
            try:
                newLectronzData = lectronz.getNewOrders(exchangeRateForLectronz)
            except:
                print('ERROR: lectronz.getNewOrders()')
                newLectronzData = []
            try:
                newShopifyData = shopify.getNewOrders()
            except:
                print('ERROR: shopify.getNewOrders()')
                newShopifyData = []
                    
            combinedJSON = {'data': [], 'date': ''}
            combinedJSON['data'] = newEtsyData + newTindieData + newShopifyData + newLectronzData
            combinedJSON['date'] = timestamp
                    
            for order in combinedJSON['data']:
                combinedJSON['data'][combinedJSON['data'].index(order)]['personalization'] = cleanUpText(combinedJSON['data'][combinedJSON['data'].index(order)]['personalization'])
            with open("combined_orders_in_json.json", "w") as file:
                json.dump(combinedJSON, file, indent=4)
            
            if (st.session_state['data_source_radio'] == 'Sanitized'):
                print('Sanitized')
                sanitized_first_names = ['John', 'Jane', 'Joe', 'Mary', 'Sally', 'Bob', 'Bill', 'Sue', 'Tom', 'Tim', 'Tina', 'Tanya', 'Terry', 'Trevor']
                sanitized_last_names = ['Smith', 'Jones', 'Johnson', 'Williams', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia', 'Martinez', 'Robinson', 'Clark', 'Rodriguez', 'Lewis', 'Lee', 'Walker', 'Hall', 'Allen', 'Young', 'Hernandez', 'King', 'Wright', 'Lopez', 'Hill', 'Scott', 'Green', 'Adams', 'Baker', 'Gonzalez', 'Nelson', 'Carter', 'Mitchell', 'Perez', 'Roberts', 'Turner', 'Phillips', 'Campbell', 'Parker', 'Evans', 'Edwards', 'Collins', 'Stewart', 'Sanchez', 'Morris', 'Rogers', 'Reed', 'Cook', 'Morgan', 'Bell', 'Murphy', 'Bailey', 'Rivera', 'Cooper', 'Richardson', 'Cox', 'Howard', 'Ward', 'Torres', 'Peterson', 'Gray', 'Ramirez', 'James', 'Watson', 'Brooks', 'Kelly', 'Sanders', 'Price', 'Bennett', 'Wood', 'Barnes', 'Ross', 'Henderson', 'Coleman', 'Jenkins', 'Perry', 'Powell', 'Long', 'Patterson', 'Hughes', 'Flores', 'Washington', 'Butler', 'Simmons', 'Foster', 'Gonzales', 'Bryant', 'Alexander', 'Russell', 'Griffin', 'Diaz', 'Hayes'] 
                email_suffix_examples = ['@gmail.com', '@yahoo.com', '@hotmail.com', '@aol.com', '@outlook.com', '@icloud.com', '@protonmail.com', '@zoho.com', '@mail.com', '@gmx.com', '@yandex.com', '@inbox.com', '@fastmail.com', '@tutanota.com', '@hushmail.com', '@runbox.com', '@mailfence.com', '@ctemplar.com', '@pm.me', '@disroot.org', '@elude.in', '@keemail.me', '@onionmail.org', '@secmail.pro']
                two_character_country_codes = ['CA', 'MX', 'GB', 'DE', 'FR', 'IT', 'ES', 'NL', 'SE', 'CH', 'AU', 'JP', 'CN', 'IN', 'BR', 'RU', 'ZA', 'EG', 'NG', 'KE', 'MA', 'DZ', 'TN', 'SA', 'AE', 'TR', 'IR', 'PK', 'AF', 'IQ', 'KW', 'QA', 'OM', 'YE', 'SY', 'IL', 'LB', 'JO', 'CY', 'GR', 'UA', 'PL', 'CZ', 'HU', 'AT', 'DK', 'FI', 'NO', 'IE', 'IS', 'PT', 'AR', 'CL', 'CO', 'PE', 'VE', 'EC', 'CR', 'GT', 'CU', 'BO', 'DO', 'HT', 'JM', 'BS', 'BB', 'TT', 'BZ', 'GY', 'SR', 'PY', 'UY', 'FK', 'GL', 'BM', 'KY', 'VG', 'VI', 'AS', 'GU', 'MP', 'PR']
                country_name_examples = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central African Republic (CAR)', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Democratic Republic of the Congo', 'Republic of the Congo', 'Costa Rica', 'Cote d\'Ivoire', 'Croatia', 'Cuba', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia (FYROM)', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar (Burma)', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates (UAE)', 'United Kingdom (UK)', 'United States of America (USA)', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican City (Holy See)', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe']
                two_character_state_codes = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA','HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD','MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ','NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC','SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
                sanitized_city_examples = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Washington', 'Boston', 'El Paso', 'Nashville', 'Detroit', 'Portland', 'Memphis', 'Oklahoma City', 'Las Vegas', 'Louisville', 'Baltimore', 'Milwaukee', 'Albuquerque', 'Tucson', 'Fresno', 'Sacramento', 'Mesa', 'Kansas City', 'Atlanta', 'Long Beach', 'Colorado Springs', 'Raleigh', 'Miami', 'Virginia Beach', 'Omaha', 'Oakland', 'Minneapolis', 'Tulsa', 'Arlington', 'New Orleans', 'Wichita', 'Cleveland', 'Tampa', 'Bakersfield', 'Aurora', 'Honolulu', 'Anaheim', 'Santa Ana', 'Corpus Christi', 'Riverside', 'St. Louis', 'Lexington', 'Pittsburgh', 'Stockton', 'Anchorage', 'Cincinnati', 'Saint Paul', 'Greensboro', 'Toledo', 'Newark', 'Plano', 'Henderson', 'Lincoln', 'Orlando', 'Jersey City', 'Chula Vista', 'Buffalo', 'Fort Wayne', 'Chandler', 'St. Petersburg', 'Laredo', 'Durham', 'Irvine', 'Madison', 'Norfolk', 'Lubbock', 'Gilbert', 'Winston–Salem', 'Glendale', 'Reno', 'Hialeah', 'Garland', 'Chesapeake', 'Irving', 'North Las Vegas', 'Scottsdale', 'Baton Rouge', 'Fremont', 'Richmond', 'Boise', 'San Bernardino']
                sanitized_street_address_examples = ['123 Main St', '456 Elm St', '789 Oak St', '1011 Pine St', '1213 Maple St', '1415 Cedar St', '1617 Walnut St', '1819 Ash St', '2021 Spruce St', '2223 Birch St', '2425 Chestnut St', '2627 Poplar St', '2829 Fir St', '3031 Locust St', '3233 Mulberry St', '3435 Sycamore St', '3637 Willow St', '3839 Magnolia St', '4041 Cedar St', '4243 Pine St', '4445 Oak St', '4647 Elm St', '4849 Maple St', '5051 Walnut St', '5253 Ash St', '5455 Spruce St', '5657 Birch St', '5859 Chestnut St', '6061 Poplar St', '6263 Fir St', '6465 Locust St', '6667 Mulberry St', '6869 Sycamore St', '7071 Willow St', '7273 Magnolia St', '7475 Cedar St', '7677 Pine St', '7879 Oak St', '8081 Elm St', '8283 Maple St', '8485 Walnut St', '8687 Ash St', '8889 Spruce St', '9091 Birch St', '9293 Chestnut St', '9495 Poplar St', '9697 Fir St', '9899 Locust St'] 
                
                for order in combinedJSON['data']:
                    
                    sanitized_buyer_name = random.choice(sanitized_first_names) + ' ' + random.choice(sanitized_last_names)
                    sanitized_recipient_name = random.choice(sanitized_first_names) + ' ' + random.choice(sanitized_last_names)
                    sanitized_email_address = sanitized_buyer_name.replace(' ', '') + random.choice(email_suffix_examples)
                    sanitized_country_code = random.choice(two_character_country_codes)
                    sanitized_country_name = random.choice(country_name_examples)
                    sanitized_state_code = random.choice(two_character_state_codes)
                    sanitized_city_name = random.choice(sanitized_city_examples)
                    sanitized_street_address = random.choice(sanitized_street_address_examples)
                    sanitized_zip_code = random.randint(10000, 99999)
                    sanitized_zip_code = str(sanitized_zip_code)
                    sanitized_phone_number = random.randint(1000000000, 9999999999)
                    sanitized_phone_number = str(sanitized_phone_number)
                    sanitized_phone_number = sanitized_phone_number[:3] + '-' + sanitized_phone_number[3:6] + '-' + sanitized_phone_number[6:]
                    sanitized_order_id = random.randint(10000000, 99999999)
                    sanitized_order_id = str(sanitized_order_id)
                    combinedJSON['data'][combinedJSON['data'].index(order)]['order_id'] = sanitized_order_id
                    combinedJSON['data'][combinedJSON['data'].index(order)]['buyer_name'] = sanitized_buyer_name
                    combinedJSON['data'][combinedJSON['data'].index(order)]['recipient_name'] = sanitized_recipient_name
                    combinedJSON['data'][combinedJSON['data'].index(order)]['buyer_email'] = sanitized_email_address
                    combinedJSON['data'][combinedJSON['data'].index(order)]['buyer_phone'] = sanitized_phone_number
                    combinedJSON['data'][combinedJSON['data'].index(order)]['buyer_address1'] = sanitized_street_address
                    combinedJSON['data'][combinedJSON['data'].index(order)]['buyer_address2'] = ""
                    combinedJSON['data'][combinedJSON['data'].index(order)]['buyer_address3'] = ""
                    combinedJSON['data'][combinedJSON['data'].index(order)]['buyer_city'] = sanitized_city_name
                    combinedJSON['data'][combinedJSON['data'].index(order)]['buyer_state_province'] = sanitized_state_code
                    combinedJSON['data'][combinedJSON['data'].index(order)]['buyer_country_code'] = sanitized_country_code
                    combinedJSON['data'][combinedJSON['data'].index(order)]['buyer_postal_code'] = sanitized_zip_code
                    combinedJSON['data'][combinedJSON['data'].index(order)]['buyer_country_name'] = sanitized_country_name
                    #print('sanitized_order_id', sanitized_order_id)
                    with open("sanitized_data.json", "w") as file:
                        json.dump(combinedJSON, file, indent=4)
            
            newData = combinedJSON['data']
            resetTab1Fields()
            st.session_state['refresh_datetime'] = timestamp
        else:
            if (st.session_state['data_source_radio'] == 'Real'):
                with open("combined_orders_in_json.json", "r") as f:
                    combinedJSON = json.load(f)
                    #newData = json.load(f)
                newData = combinedJSON['data']
                for order in combinedJSON['data']:
                    combinedJSON['data'][combinedJSON['data'].index(order)]['personalization'] = cleanUpText(combinedJSON['data'][combinedJSON['data'].index(order)]['personalization'])
                    
                if (len(newData) == 0):
                    with open("empty_orders_in_json.json", "r") as f:
                        combinedJSON = json.load(f)
                    newData = combinedJSON['data']
                    
                st.session_state['refresh_datetime'] = combinedJSON['date']
                
            elif (st.session_state['data_source_radio'] == 'Sanitized'):
                with open("sanitized_data.json", "r") as f:
                    combinedJSON = json.load(f)
                    #newData = json.load(f)
                newData = combinedJSON['data']
                for order in combinedJSON['data']:
                    combinedJSON['data'][combinedJSON['data'].index(order)]['personalization'] = cleanUpText(combinedJSON['data'][combinedJSON['data'].index(order)]['personalization'])
                    
                if (len(newData) == 0):
                    with open("empty_orders_in_json.json", "r") as f:
                        combinedJSON = json.load(f)
                    newData = combinedJSON['data']
                    
                st.session_state['refresh_datetime'] = combinedJSON['date']
                

    else:
        with open("fake_data.json", "r") as f:
            combinedJSON = json.load(f)
        newData = combinedJSON['data']
        for order in combinedJSON['data']:
            combinedJSON['data'][combinedJSON['data'].index(order)]['personalization'] = cleanUpText(combinedJSON['data'][combinedJSON['data'].index(order)]['personalization'])
            
        if (len(newData) == 0):
            with open("empty_orders_in_json.json", "r") as f:
                combinedJSON = json.load(f)
            newData = combinedJSON['data']
            
        st.session_state['refresh_datetime'] = combinedJSON['date']
        
    return(pd.json_normalize(newData))



def resetTab1Fields():
    st.session_state['addressName'] = ""
    st.session_state['addressLine1'] = ""
    st.session_state['addressLine2'] = ""
    st.session_state['addressLine3'] = ""
    st.session_state['addressCity'] = ""
    st.session_state['addressState'] = ""
    st.session_state['addressZip'] = ""
    st.session_state['addressCountry'] = ""
    st.session_state['shippingPaid'] = ""
    #st.session_state['shipping_method_dropdown'] = shipping_method
    st.session_state['packageValue'] = 0
    st.session_state['insuranceCheckBox'] = False
    st.session_state['address_verification'] = ""
    st.session_state["disabled_ship_package"] = True
    st.session_state['disabled_fulfill_order'] = True
    st.session_state["label_tracking_number"] = ""
    st.session_state['shipping_method_dropdown'] = ('Select a shipping method',0)
    ###st.session_state['shippingDate'] = st.session_state["date_for_shipment"]
    st.session_state['shipping_order_dropdown'] = ('Select an order',)

    st.session_state["first_class_price"] = '  $' + "{:.2f}".format(0.00)
    st.session_state["priority_mail_price"] = '  $' + "{:.2f}".format(0.00)
    st.session_state["priority_mail_express_price"] = '  $' + "{:.2f}".format(0.00)    
    st.session_state["priority_mail_express_international_price"] = '  $' + "{:.2f}".format(0.00)
    st.session_state["priority_mail_international_price"] = '  $' + "{:.2f}".format(0.00)
    st.session_state["first_class_international_price"] = '  $' + "{:.2f}".format(0.00)
    st.session_state["priority_mail_small_box_price"] = '  $' + "{:.2f}".format(0.00)
    st.session_state["ground_advantage_price"] = '  $' + "{:.2f}".format(0.00)
    st.session_state["media_mail_price"] = '  $' + "{:.2f}".format(0.00)    
    st.session_state["disabled_void_label_button"] = True
    st.session_state["void_label_button_checkbox"] = False
    st.session_state["shipping_text_results"] = "Results will appear here"
    st.session_state["disabled_mass_ship_button"] = True
    st.session_state["mass_ship_button_checkbox"] = False
    st.session_state["inventory_and_prices_button_checkbox"] = False
    

                    
def print_receipt():
    order_selected = st.session_state['shipping_order_dropdown'][0]
    
    if (order_selected != 'Select an order'):
        comboOrderID = order_selected.split('--')[0].strip()
        actualOrderID = order_selected.split('-')[0].strip()
        platform = df.loc[df['order id'] == comboOrderID, 'platform'].values[0]
        print('printing receipt for', actualOrderID, 'from', platform)
        
        #if (platform == 'etsy'):
        with open(f"{platform}_orderDetails-{actualOrderID}.json", 'r') as f:
            orderDetails = json.load(f)
            #print('orderDetails ---', orderDetails)
            canadianTax = orderDetails['canadian_tax']
            vatTax = orderDetails['vat_tax_paid']
            vatCurrencyCode = orderDetails['vat_currency_code']
            display_canadian_tax = orderDetails['display_canadian_tax']
            display_euro_vat_tax = orderDetails['display_euro_vat_tax']
            display_uk_vat_tax = orderDetails['display_uk_vat_tax']
            display_canada_zero_rated_tax = orderDetails['display_canada_zero_rated_tax']
            #print('canadianTax', canadianTax)
            
            try:
                if (printReceipt.printReceipt(orderDetails) == True):
                    print('Receipt Printed')
                else:
                    print('Error: Print Receipt Failed')
                #PRINT TAX SLIP
                if (display_canadian_tax == True):    
                    print('Order has Canadian Tax, print a receipt with a message')
                    if (printReceipt.printMessage(canadianTax + ' Provincial\nSales Tax Paid') == True):
                        print('Canadian Tax Slip Printed')
                    else:
                        print('Error: Canadian Tax Slip Not Printed')
                elif (display_canada_zero_rated_tax == True):    
                    print('Order has Canadian Tax, print a receipt with a message')
                    if (printReceipt.printMessage('Contents have been rated for zero tax') == True):
                        print('Canadian Tax Slip Printed')
                    else:
                        print('Error: Canadian Tax Slip Not Printed')
                elif (display_uk_vat_tax == True):
                    print('Order has UK VAT Tax, print a receipt with a message')
                    if (printReceipt.printMessage(vatCurrencyCode + ' ' + str(vatTax) + ' VAT Paid.\nEtsy UK VAT# 370 6004 28') == True):
                        print('UK VAT Tax Slip Printed')
                    else:
                        print('Error: UK VAT Tax Slip Not Printed')
                elif (display_euro_vat_tax == True):
                    print('Order has EURO VAT Tax, print a receipt with a message')
                    if (printReceipt.printMessage(vatCurrencyCode + ' ' + str(vatTax) + ' VAT Paid.\nEtsy IOSS# IM3720000224,') == True):
                        print('EURO VAT Tax Slip Printed')
                    else:
                        print('Error: EURO VAT Tax Slip Not Printed')
            except Exception as e:
                print('ERROR printing receipt', e)
       
            

def ounces_to_pounds(ounces):
    pounds = ounces / 16
    pounds_rounded = round(pounds, 1)  # Round up to the nearest tenth of a pound
    return pounds_rounded

def get_rates_and_verify_address():
    order_selected = st.session_state['shipping_order_dropdown'][0]
    if (order_selected != 'Select an order'):
        comboOrderID = order_selected.split('--')[0].strip()
        shipping_method = df.loc[df['order id'] == comboOrderID, 'order_shipping_method'].values[0]
        shippingDate = st.session_state["shippingDate"]
        
        if (st.session_state["packageWeightLB"] > 0):
            print('packageWeightLB more than 0 LBs', st.session_state["packageWeightLB"], 'pounds', st.session_state["packageWeightOZ"], 'ounces')
            packageWeight = ounces_to_pounds(st.session_state["packageWeightOZ"]) + st.session_state["packageWeightLB"]
            packageWeightType = "pound"
            print('package weight is ', packageWeight)
        else:
            print('packageWeightLB less than 0 LBs', st.session_state["packageWeightOZ"], 'ounces')
            packageWeight = st.session_state["packageWeightOZ"]
            packageWeightType = "ounce"
            
        addressErrorOverride = st.session_state["address_error_override"]

        packageLength = st.session_state["packageSizeLength"]
        packageWidth = st.session_state["packageSizeWidth"]
        packageHeight = st.session_state["packageSizeHeight"]
        packageDimensionUnit = "inch"
        insuredTotalValue = st.session_state["packageValue"]
        
        shippingPaidAmount = st.session_state['shippingPaid']
        shippingPaidAmount = shippingPaidAmount[1:]
        
        insuredGrandTotal = insuredTotalValue + math.ceil(float(shippingPaidAmount))
        
        name = st.session_state["addressName"]
        addressLine1 = st.session_state["addressLine1"]
        addressLine2 = st.session_state["addressLine2"]
        addressLine3 = st.session_state["addressLine3"]
        city = st.session_state["addressCity"]
        state = st.session_state["addressState"]
        postalCode = st.session_state["addressZip"]
        countryCode = st.session_state["addressCountry"]
        shippingMethodDropDown = st.session_state["shipping_method_dropdown"]
        insurance = st.session_state["insuranceCheckBox"]
        itemQuantity = 1
        
        addressJSON = {
            "ship_date": str(shippingDate),
            "package_weight": packageWeight,
            "package_weight_type": packageWeightType,
            "package_length": packageLength,
            "package_width": packageWidth,
            "package_height": packageHeight,
            "package_dimension_unit": packageDimensionUnit,
            "insured_total_value": insuredGrandTotal,
            "name": name,
            "address_line1": addressLine1,
            "address_line2": addressLine2,
            "address_line3": addressLine3,
            "city_locality": city,
            "state_province": state,
            "postal_code": postalCode,
            "country_code": countryCode,
            "shipping_method": shippingMethodDropDown,
            "insurance": insurance,
            "item_quantity": itemQuantity
        }
        
        addressJSON = json.dumps(addressJSON)
        rates = shipEngine.getShippingRates(addressJSON)
        
        if ('ERROR' in rates[0]):
            print("There was an Error getting rates")
            print(rates[1])
            try:
                st.session_state["address_verification"] = rates[1]['errors'][0]['message'].upper()
            except:
                st.session_state["address_verification"] = 'ERROR WHEN GETTING RATES'
            st.session_state["disabled_ship_package"] = True
            st.session_state['disabled_fulfill_order'] = True
            st.session_state["sidebar_text"] = str(rates[1])
        else:
            
            ratePrices = rates[0]

            if ratePrices['first_class_price'] is not None:
                st.session_state["first_class_price"] = '  $' + "{:.2f}".format(ratePrices['first_class_price'])
            else:
                st.session_state["first_class_price"] = '  $' + "{:.2f}".format(0.00)
            if ratePrices['priority_mail_price'] is not None:
                st.session_state["priority_mail_price"] = '  $' + "{:.2f}".format(ratePrices['priority_mail_price'])
            else:
                st.session_state["priority_mail_price"] = '  $' + "{:.2f}".format(0.00)
            if ratePrices['priority_mail_express_price'] is not None:
                st.session_state["priority_mail_express_price"] = '  $' + "{:.2f}".format(ratePrices['priority_mail_express_price'])
            else:
                st.session_state["priority_mail_express_price"] = '  $' + "{:.2f}".format(0.00)    
            if ratePrices['priority_mail_express_international_price'] is not None:
                st.session_state["priority_mail_express_international_price"] = '  $' + "{:.2f}".format(ratePrices['priority_mail_express_international_price'])
            else:
                st.session_state["priority_mail_express_international_price"] = '  $' + "{:.2f}".format(0.00)
            if ratePrices['priority_mail_international_price'] is not None:
                st.session_state["priority_mail_international_price"] = '  $' + "{:.2f}".format(ratePrices['priority_mail_international_price'])
            else:
                st.session_state["priority_mail_international_price"] = '  $' + "{:.2f}".format(0.00)
            if ratePrices['first_class_international_price'] is not None:
                st.session_state["first_class_international_price"] = '  $' + "{:.2f}".format(ratePrices['first_class_international_price'])
            else:
                st.session_state["first_class_international_price"] = '  $' + "{:.2f}".format(0.00)
            if ratePrices['priority_mail_small_box_price'] is not None:
                st.session_state["priority_mail_small_box_price"] = '  $' + "{:.2f}".format(ratePrices['priority_mail_small_box_price'])
            else:
                st.session_state["priority_mail_small_box_price"] = '  $' + "{:.2f}".format(0.00)
            if ratePrices['ground_advantage_price'] is not None:
                st.session_state["ground_advantage_price"] = '  $' + "{:.2f}".format(ratePrices['ground_advantage_price'])
            else:
                st.session_state["ground_advantage_price"] = '  $' + "{:.2f}".format(0.00)
            if ratePrices['media_mail_price'] is not None:
                st.session_state["media_mail_price"] = '  $' + "{:.2f}".format(ratePrices['media_mail_price'])
            else:
                st.session_state["media_mail_price"] = '  $' + "{:.2f}".format(0.00)
                
            st.session_state["address_verification"] = 'ADDRESS VERIFIED/UPDATED'
            st.session_state["disabled_ship_package"] = False

            st.session_state["addressName"] = rates[1]['name']
            st.session_state["addressLine1"] = rates[1]['address_line1']
            st.session_state["addressLine2"] = rates[1]['address_line2']
            st.session_state["addressLine3"] = rates[1]['address_line3']
            st.session_state["addressCity"] = rates[1]['city_locality']
            st.session_state["addressState"] = rates[1]['state_province']
            st.session_state["addressZip"] = rates[1]['postal_code']
            st.session_state["addressCountry"] = rates[1]['country_code']
            
            if (shipping_method.lower() in firstClassInternationalWordList):
                #shipping_method = ('First Class International',)
                shipping_method = ('First Class International' + str(st.session_state.get("first_class_international_price", " 0.00")),6)
            #elif any(shipping_method.lower() == item.lower() for item in firstClassWordList if isinstance(item, str)):
            elif (shipping_method.lower() in firstClassWordList):
                #shipping_method = ('First Class',)
                shipping_method = ('First Class' + str(st.session_state.get("first_class_price", " 0.00")),2)
            #elif any(shipping_method.lower() == item.lower() for item in priorityMailExpressInternationalWordList if isinstance(item, str)):
            elif (shipping_method.lower() in priorityMailExpressInternationalWordList):
                #shipping_method = ('Priority Mail Express International',)
                shipping_method = ('Priority Mail Express International' + str(st.session_state.get("priority_mail_express_international_price", " 0.00")),8)
            #elif any(shipping_method.lower() == item.lower() for item in priorityMailExpressWordList if isinstance(item, str)):
            elif (shipping_method.lower() in priorityMailExpressWordList):
                #shipping_method = ('Priority Mail Express',)
                shipping_method = ('Priority Mail Express' + str(st.session_state.get("priority_mail_express_price", " 0.00")),5)
            #elif any(shipping_method.lower() == item.lower() for item in priorityMailInternationalWordList if isinstance(item, str)):
            elif (shipping_method.lower() in priorityMailInternationalWordList):
                #shipping_method = ('Priority Mail International',)
                shipping_method = ('Priority Mail International' + str(st.session_state.get("priority_mail_international_price", " 0.00")),7)
            #elif any(shipping_method.lower() == item.lower() for item in priorityMailWordList if isinstance(item, str)):
            elif (shipping_method.lower() in priorityMailWordList):
                #shipping_method = ('Priority Mail',)
                shipping_method = ('Priority Mail' + str(st.session_state.get("priority_mail_price", " 0.00")),3)
            elif (shipping_method.lower() in groundAdvantageWordList):
                #shipping_method = ('Ground Advantage',)
                shipping_method = ('Ground Advantage' + str(st.session_state.get("ground_advantage_price", " 0.00")),1)
            elif (shipping_method.lower() in mediaMailWordList):
                #shipping_method = ('Media Mail',)
                shipping_method = ('Media Mail' + str(st.session_state.get("media_mail_price", " 0.00")),9)
            else:
                shipping_method = ('Unknown',10)
            
            st.session_state['shipping_method_dropdown'] = shipping_method
                
def ship_package():
    order_selected = st.session_state['shipping_order_dropdown'][0]
    if (order_selected != 'Select an order'):
        print('attempting to ship package')
        if (st.session_state["address_verification"] != 'ADDRESS VERIFIED/UPDATED'):
            print('address not verified... attempting to verify address')
            get_rates_and_verify_address()
            if (st.session_state["address_verification"] != 'ADDRESS VERIFIED/UPDATED'):
                print('address still not verified... cannot ship package')
                return
        if (st.session_state["address_verification"] == 'ADDRESS VERIFIED/UPDATED'):
            comboOrderID = order_selected.split('--')[0].strip()
            actualOrderID = order_selected.split('-')[0].strip()
            shipping_method = st.session_state['shipping_method_dropdown']
            #print('shipping_method', shipping_method, shipping_method[1])
            shipping_method = shipping_method[1]
            package_type = "package"
            if (shipping_method == 1):
                service_code = 'usps_ground_advantage'
            elif (shipping_method == 2):
                service_code = 'usps_first_class_mail'
            elif (shipping_method == 3):
                service_code = 'usps_priority_mail'
            elif (shipping_method == 4):
                service_code = 'usps_priority_mail'
                package_type = "small_flat_rate_box"
            elif (shipping_method == 5):
                service_code = 'usps_priority_mail_express'
            elif (shipping_method == 6):
                service_code = 'usps_first_class_mail_international'
            elif (shipping_method == 7):
                service_code = 'usps_priority_mail_international'
            elif (shipping_method == 8):
                service_code = 'usps_priority_mail_express_international'
            elif (shipping_method == 9):
                service_code = 'usps_media_mail'
            else:
                service_code = 'uknown'
                
            shippingDate = st.session_state["shippingDate"]
            
            if (st.session_state["packageWeightLB"] > 0):
                print('packageWeightLB more than 0 LBs', st.session_state["packageWeightLB"], 'pounds', st.session_state["packageWeightOZ"], 'ounces')
                packageWeight = ounces_to_pounds(st.session_state["packageWeightOZ"]) + st.session_state["packageWeightLB"]
                packageWeightType = "pound"
                print('package weight is ', packageWeight)
            else:
                print('packageWeightLB less than 0 LBs', st.session_state["packageWeightOZ"], 'ounces')
                packageWeight = st.session_state["packageWeightOZ"]
                packageWeightType = "ounce"
            
            packageLength = st.session_state["packageSizeLength"]
            packageWidth = st.session_state["packageSizeWidth"]
            packageHeight = st.session_state["packageSizeHeight"]
            packageDimensionUnit = "inch"
            insuredTotalValue = st.session_state["packageValue"]
            
            shippingPaidAmount = st.session_state['shippingPaid']
            shippingPaidAmount = shippingPaidAmount[1:]
            
            insuredGrandTotal = insuredTotalValue + math.ceil(float(shippingPaidAmount))
            
            name = st.session_state["addressName"]
            addressLine1 = st.session_state["addressLine1"]
            addressLine2 = st.session_state["addressLine2"]
            addressLine3 = st.session_state["addressLine3"]
            city = st.session_state["addressCity"]
            state = st.session_state["addressState"]
            postalCode = st.session_state["addressZip"]
            countryCode = st.session_state["addressCountry"]
            insurance = st.session_state["insuranceCheckBox"]
            itemQuantity = df.loc[df['order id'] == comboOrderID, 'total'].values[0]
            platform = df.loc[df['order id'] == comboOrderID, 'platform'].values[0]
            orderID = df.loc[df['order id'] == comboOrderID, 'order id'].values[0]
            
            notes_for_shipment = st.session_state["notesForShipment"]
            
            if (notes_for_shipment == ""):
                notes_for_shipment = "none"

            ship_date_override = st.session_state["ship_date_override"]
            verified_address_JSON = {
                "verified_ship_date": str(shippingDate),
                "verified_package_weight": packageWeight,
                "verified_package_weight_type": packageWeightType,
                "verified_package_length": packageLength,
                "verified_package_width": packageWidth,
                "verified_package_height": packageHeight,
                "verified_package_dimension_unit": packageDimensionUnit,
                "verified_insured_total_value": insuredGrandTotal,
                "verified_name": name,
                "verified_address_line1": addressLine1,
                "verified_address_line2": addressLine2,
                "verified_address_line3": addressLine3,
                "verified_city_locality": city,
                "verified_state_province": state,
                "verified_postal_code": postalCode,
                "verified_country_code": countryCode,
                "verified_service_code": service_code,
                "verified_insurance": insurance,
                "verified_item_quantity": itemQuantity,
                "verified_package_type": package_type,
                "verified_platform": platform,
                "verified_order_id": comboOrderID,
                "verified_ship_date_override": ship_date_override, 
                "verified_notes_for_shipment": notes_for_shipment
            }
                        
            with open(f"{platform}_orderDetails-{actualOrderID}.json", 'r') as f:
                originalJSON = json.load(f)
                newJSON = {**originalJSON, **verified_address_JSON}

            
            if (originalJSON['order_id'] == newJSON['order_id']):
                print('order id matches.  good to ship.')
                customerShipment = shipEngine.shipPackage(newJSON)
                shippingErrorMessage = customerShipment[4]
                shippingEnvironmentIsProd = customerShipment[3]
                customerShipmentShipDate = customerShipment[2]
                customerShipmentStatusCode = customerShipment[1]
                customerShipment = customerShipment[0]

                if (customerShipmentStatusCode != 200):
                    print('ERROR CREATING SHIPPING LABEL. STATUS CODE', customerShipmentStatusCode, shippingErrorMessage)
                    st.session_state["address_verification"] = shippingErrorMessage
                    st.session_state["disabled_ship_package"] = True
                    st.session_state['disabled_fulfill_order'] = True
                    st.session_state["sidebar_text"] = shippingErrorMessage
            
                else:
                    labelID = customerShipment['label_id']
                    labelStatus = customerShipment['status']
                    labelShipmentID = customerShipment['shipment_id']
                    labelTrackingNumber = customerShipment['tracking_number']
                    labelPDFUrl = customerShipment['label_download']['pdf']
                    labelZPLUrl = customerShipment['label_download']['zpl']
                    shipped_date = customerShipment['ship_date']

                    st.session_state["label_tracking_number"] = labelTrackingNumber 
                        
                    shipped_date = datetime.strptime(shipped_date, '%Y-%m-%dT%H:%M:%SZ')
                    
                    st.session_state['ship_date_for_customer'] = customerShipmentShipDate
                    st.session_state['actual_shipped_environment'] = shippingEnvironmentIsProd
                    st.session_state['disabled_fulfill_order'] = False
                    printShippingLabelBoolean = True
                    try:
                        if (printShippingLabelBoolean == True):
                            print('Printing Shipping Label...')
                            shippingData = shipEngine.printLabel(platform, orderID, labelPDFUrl)  #comboOrderID
                            shippingDownloadStatusCode = shippingData[0]
                            shippingPrintStatus = shippingData[1]
                            
                            print(shippingDownloadStatusCode, shippingPrintStatus)
                        else:
                            print('Not printing shipping label due to global boolean')
                    except:
                        print('error printing shipping label')
                    
                    time.sleep(2)
                    print_receipt()
                    time.sleep(1)
                        
                    if (shippingEnvironmentIsProd == True):
                        print('Shipping label was created in production environment, fulfilling order...')
                        fulfill_order()
                    else:
                        print('Not fulfilling order because shipping label was created in test environment.')
            else:
                print('order id does not match')

        else:
            print('address not verified... cannot ship package')

def fulfill_order():
    if (st.session_state.get('actual_shipped_environment') == True or st.session_state.get('fulfill_button_checkbox') == True):
        print('Shipping label was created in production environment.', st.session_state.get('actual_shipped_environment'))
        if (st.session_state["label_tracking_number"] != ""):
            #'''
            order_selected = st.session_state['shipping_order_dropdown'][0]
            if (order_selected != 'Select an order'):
                comboOrderID = order_selected.split('--')[0].strip()
                actualOrderID = order_selected.split('-')[0].strip()
                platform = df.loc[df['order id'] == comboOrderID, 'platform'].values[0]
                buyerFirstName = df.loc[df['order id'] == comboOrderID, 'name'].values[0]
                buyerFirstName = buyerFirstName.split(' ')[0]
                buyerFirstName = buyerFirstName.capitalize()
                try:
                    customerShipmentShipDate = st.session_state['ship_date_for_customer']
                except:
                    customerShipmentShipDate = st.session_state['shippingDate'].strftime('%B %d')
                trackingNumber = st.session_state["label_tracking_number"]
                with open(f"{platform}_orderDetails-{actualOrderID}.json", 'r') as f:
                    orderJson = json.load(f)
                
                if (platform == 'etsy'):
                    is_buyer_repeat = df.loc[df['order id'] == comboOrderID, 'buyer_is_repeat'].values[0]
                    
                    trackingCompany = "usps"
                                        
                    fulfillmentObject = platformFunctions.fulfillEtsyOrder(actualOrderID, trackingCompany, trackingNumber, customerShipmentShipDate)
                    fulfillmentShipped = fulfillmentObject[0]
                    fulfillmentStatus = fulfillmentObject[1]
                    print('fulfillmentShipped', fulfillmentShipped)
                    print('fulfillmentStatus', fulfillmentStatus)
                    if (fulfillmentStatus == 'Completed'):
                        print('Fulfillment was successful')
                        try: 
                            if (platformFunctions.sendEmailToCustomer(orderJson, customerShipmentShipDate) == True):
                                print('Message Sent')
                            else:
                                print('Error: Message Not Sent')
                        except Exception as e:
                            print('ERROR sending message', e)
                    else:
                        print('Fulfillment was not successful')
                elif (platform == 'tindie'):
                    print('Fulling on Tindie')  
                    trackingCompany = 'USPS'
                    fulfillmentStatus = platformFunctions.fulfillTindieOrder(actualOrderID, trackingCompany, trackingNumber, customerShipmentShipDate)
                    print('fulfillmentStatus', fulfillmentStatus)
                    if (fulfillmentStatus == 200):
                        print('Fulfillment was successful')
                        try: 
                            if (platformFunctions.sendEmailToCustomer(orderJson, customerShipmentShipDate) == True):
                                print('Message Sent')
                            else:
                                print('Error: Message Not Sent')
                        except Exception as e:
                            print('ERROR sending message', e)
                    else:
                        print('Fulfillment was not successful')

                elif (platform == 'mih'):
                    print('Fulling on Shopify')
                    
                    locationID = shopify_shipping_location_id
                    trackingCompany = "USPS"
                    trackingURL = "https://tools.usps.com/go/TrackConfirmAction.action?tLabels=" + trackingNumber
                    
                    fulfillmentObject = platformFunctions.fulfillShopifyOrder(orderJson['shopify_order_id'], locationID, trackingCompany, trackingNumber, trackingURL)
                    fulfillmentID = fulfillmentObject[0]
                    fulfillmentStatus = fulfillmentObject[1]

                    if (fulfillmentStatus == 'success'):
                        print('Fulfillment was successful')
                        try: 
                            if (platformFunctions.sendEmailToCustomer(orderJson, customerShipmentShipDate) == True):
                                print('Message Sent')
                            else:
                                print('Error: Message Not Sent')
                        except Exception as e:
                            print('ERROR sending message', e)
                    else:
                        print('Fulfillment was not successful')    
                
                elif (platform == 'lectronz'):
                    print('Fulling on Lectronz')
                    fulfillmentStatus = platformFunctions.fulfillLectronzOrder(actualOrderID, trackingNumber)
                    if (fulfillmentStatus[0] == 'fulfilled'):
                        print('Fulfillment was successful')
                        try: 
                            if (platformFunctions.sendEmailToCustomer(orderJson, customerShipmentShipDate) == True):
                                print('Message Sent')
                            else:
                                print('Error: Message Not Sent')
                        except Exception as e:
                            print('ERROR sending message', e)
                    else:
                        print('Fulfillment was not successful')
                else:
                    print('Platform not supported yet')
            #'''
        else:
            print('Not fulfilling order due to no tracking number.')
    else:
        print('Not fulfilling order due to Shipping label was created in test environment.', st.session_state.get('actual_shipped_environment'))
    print('DONE')
    print('=======================================================================')                        
    return
    
    

def on_fulfill_button_change():
    st.session_state["disabled_fulfill_order"] = not st.session_state['fulfill_button_checkbox']
    
def on_void_label_button_change():
    st.session_state["disabled_void_label_button"] = not st.session_state['void_label_button_checkbox']
    
def on_inventory_and_prices_button_change():
    st.session_state["disabled_inventory_and_prices_button"] = not st.session_state['inventory_and_prices_button_checkbox']



def mass_ship():
    return

def on_mass_ship_button_change():
    st.session_state["disabled_mass_ship_button"] = not st.session_state['mass_ship_button_checkbox']
    return
    
def on_address_button_change():
    st.session_state["disabled_ship_package"] = not st.session_state['address_error_override']


def get_date_for_shipment():
    st.session_state["date_for_shipment"] = datetime.today()
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    monday = today + timedelta(days=(7 - today.weekday()))
    current_hour = today.hour
    
    max_date = today + timedelta(days=6)
    
    if (current_hour < 10 and today.weekday() < 5):
        st.session_state["date_for_shipment"] = datetime.today()#.strftime('%Y-%m-%d')
    else:

        if tomorrow.weekday() < 5:  # Monday is 0, Friday is 4
            # If tomorrow is a weekday, print tomorrow's date
            st.session_state["date_for_shipment"] = tomorrow # ().strftime('%Y-%m-%d')
        else:
            # If tomorrow is a weekend, print Monday's date
            st.session_state["date_for_shipment"] = monday
    st.session_state['max_date_for_shipment'] = max_date
    return

def update_package_sizes():
    packageSize = st.session_state["package_size_dropdown"][0]
    packageSize = packageSize.split('x')
    if (packageSize[0] == 'Manual'):
        st.session_state["packageSizeLength"] = 1
        st.session_state["packageSizeWidth"] = 1
        st.session_state["packageSizeHeight"] = 1
    else:
        packageLength = packageSize[0]
        packageWidth = packageSize[1]
        packageHeight = packageSize[2]
        st.session_state["packageSizeLength"] = int(packageLength)
        st.session_state["packageSizeWidth"] = int(packageWidth)
        st.session_state["packageSizeHeight"] = int(packageHeight)
    return


def create_manifest():
    date_for_manifest = st.session_state['shippingDate']
    date_for_manifest = date_for_manifest.strftime("%Y-%m-%d")
    if (shipEngine.createManifest(date_for_manifest) == True):
        print('Manifest Created')
    else:
        print('Manifest Not Created')
    return

def load_labels():
    with open("./labels/labels.json", "r") as f:
        labelsJSON = json.load(f)
    labelsData = labelsJSON['labels']
    return(pd.json_normalize(labelsData))

def load_manifests():
    with open("./manifests/manifests.json", "r") as f:
        manifestsJSON = json.load(f)
    manifestsData = manifestsJSON['manifests']
    for i, manifest in enumerate(manifestsData):
        manifestsData[i]['manifest_download'] = manifest['manifest_download']['href']        
    return(pd.json_normalize(manifestsData))

def print_label():
    label_selected = st.session_state['label_order_dropdown'][0]
    if (label_selected != 'Select a Label'):
        label_id_from_dropdown = label_selected.split('--')[0].strip()
        orderID = df_labels.loc[df_labels['label_id'] == label_id_from_dropdown, 'order_id'].values[0]
        platform = df_labels.loc[df_labels['label_id'] == label_id_from_dropdown, 'order_platform'].values[0]
        labelPDFUrl = df_labels.loc[df_labels['label_id'] == label_id_from_dropdown, 'label_download_pdf'].values[0]
        try:
            print('Printing Shipping Label...')
            shippingData = shipEngine.printLabel(platform, orderID, labelPDFUrl)
            shippingDownloadStatusCode = shippingData[0]
            shippingPrintStatus = shippingData[1]
            
            print(shippingDownloadStatusCode, shippingPrintStatus)
        except:
            print('error printing shipping label')
    return

def void_label():
    label_selected = st.session_state['label_order_dropdown'][0]
    if (label_selected != 'Select a Label'):
        label_id_from_dropdown = label_selected.split('--')[0].strip()
        already_voided = df_labels.loc[df_labels['label_id'] == label_id_from_dropdown, 'is_voided'].values[0]
        if (already_voided == True):
            print('Label Already Voided')
        else:
            print('Voiding Label', label_id_from_dropdown)
            void_results = shipEngine.voidLabel(label_id_from_dropdown)
            
            pretty_json = json.dumps(void_results[1], indent=4)
            
            if (void_results[1]['approved'] == True):
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with open("./labels/labels.json", 'r') as label_json_file:
                    label_data = json.load(label_json_file)
                    label_data['date_time_updated'] = timestamp
                
                label_index = 0
                for label in label_data['labels']:
                    if label['label_id'] == label_id_from_dropdown:
                        #label = True
                        label_data['labels'][label_index]['is_voided'] = True
                        label_data['labels'][label_index]['voided_at'] = timestamp
                    label_index += 1
                
                with open("./labels/labels.json", 'w') as json_file:
                    json.dump(label_data, json_file, indent=4)
                    
                print('Label Voided')
                
            else:
                print('LABEL NOT VOIDED')
                print(pretty_json)
    
            st.session_state["shipping_text_results"] = pretty_json
        
        
    st.session_state["disabled_void_label_button"] = True
    st.session_state["void_label_button_checkbox"] = False
    return

def reset_combined_orders_file():
    with open("combined_orders_in_json.json", 'w') as json_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        blankdata = {"data": [], "date": timestamp}
        json.dump(blankdata, json_file, indent=4)
    return

def curl_to_requests(curl_command):
    # Split the curl command into individual components
    curl_parts = shlex.split(curl_command)
    
    #print('curl_parts', curl_parts)
    # Extract the URL from the curl command
    url = curl_parts[1]
    
    # Extract headers from the curl command
    headers = {}
    for i in range(len(curl_parts)):
        if curl_parts[i] == "-H":
            header_key_value = curl_parts[i + 1].split(':', maxsplit=1)
            header_key = header_key_value[0].strip()
            header_value = header_key_value[1].strip()
            headers[header_key] = header_value
    
    # Extract data from the curl command
    data = None
    if "--data" in curl_parts:
        data_index = curl_parts.index("--data") + 1
        data = curl_parts[data_index]
        #print('data was found')
    elif "--data-raw" in curl_parts:
        data_index = curl_parts.index("--data-raw") + 1
        data = curl_parts[data_index]
        #print('data raw was found')
    elif "--data-binary" in curl_parts:
        data_index = curl_parts.index("--data-binary") + 1
        data = curl_parts[data_index]
        #print('data binary was found')
    else:
        #print('data was not found')
        pass
    # Construct the equivalent Python requests code as a string
    python_code = f'''
import requests

url = "{url}"
headers = {headers}
    '''
    
    python_code = f'import requests\n\nurl = "{url}"\nheaders = {headers}\n'
    
    
    if data:
        python_code += f'data = "{data}"\n'
        python_code += 'response = requests.post(url, headers=headers, data=data)\n\n'
        call = 'post'
    else:
        python_code += 'response = requests.get(url, headers=headers)\n\n'
        call = 'get'

    python_code += 'print(response.status_code)\n'
    python_code += 'print(response.text)\n'
    
    return (python_code, url, headers, data, call)



def run_python_requests(url_arg, headers_arg, data_arg, call_arg):
    #print('the args', url_arg, headers_arg, data_arg, call_arg)
    url_arg = url_arg[0]
    #print('url_arg', url_arg)
    headers_arg = headers_arg[0]
    #print('headers_arg', headers_arg)
    data_arg = data_arg[0]
    #print('data_arg', data_arg)
    call_arg = call_arg[0]

    if (call_arg == 'post'):
        response = requests.post(url_arg, headers=headers_arg, data=data_arg)
    elif (call_arg == 'get'):
        response = requests.get(url_arg, headers=headers_arg)

    response_to_print = ''
    response_is_json = False
    
    try:
        json_response = response.json()
        response_is_json = True
        response_to_print = str(json.dumps(json_response, indent=4))
    except:
        response_to_print = response.text
        response_is_json = False

    st.session_state["curl_python_results"] = ('STATUS CODE: ' + str(response.status_code) +'\nRESPONSE:\n\n' + response_to_print, response_is_json)
    return(response.status_code, response_to_print)


def load_inventory():
    with open("./products/products.json", "r") as f:
        productsJSON = json.load(f)
    productsData = productsJSON['products']
    flattened_products = []
    for product in productsData:
        if len(product['product_options']) == 0:
            flattened_products.append(product)
        else:
            for option in product['product_options']:
                flattened_product = product.copy()
                flattened_product.update(option)
                flattened_product['option_display_name'] = option['option_display_name']
                flattened_product['option_additional_cost'] = option['option_additional_cost']
                flattened_product['option_in_stock'] = option['option_in_stock']
                flattened_products.append(flattened_product)
                
    df = pd.json_normalize(flattened_products)
    return df

def reverse_inventory(df):
    products = []
    for _, row in df.iterrows():
        product = {
            'product_id': row['product_id'],
            'product_base_price': row['product_base_price'],
            'product_in_stock': row['product_in_stock'],
            'product_id_lectronz': row['product_id_lectronz']
        }
        if row['product_options']:
            option = {
                'option_id': row['option_id'],
                'option_additional_cost': row['option_additional_cost'],
                'option_in_stock': row['option_in_stock']
            }
            product['product_options'] = [option]
        else:
            product['product_options'] = []
        products.append(product)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    inventory = {'date_time_updated': timestamp, 'products': products}
    return inventory




def update_inventory_and_prices(reverse_inventory_text):
    
    with open("./products/edited_inventory.json", "w") as f:
        json.dump(reverse_inventory_text, f, indent=4)
    
    with open("./products/products.json", "r") as f:
        productsJSON = json.load(f)
        productsData = productsJSON['products']
    
    with open("./products/edited_inventory.json", "r") as f:
        inventoryJSON = json.load(f)
        inventoryData = inventoryJSON['products']
    
    for product in productsData:
        for inventory_product in inventoryData:
            if product['product_id'] == inventory_product['product_id']:
                if (product['product_base_price'] != inventory_product['product_base_price']):
                    print('product price updated!')
                    product['product_base_price'] = inventory_product['product_base_price']
                if (product['product_in_stock'] != inventory_product['product_in_stock']):
                    print('product in stock updated!')
                    product['product_in_stock'] = inventory_product['product_in_stock']
                    #print('------', int(product['product_id_lectronz']), int(inventory_product['product_in_stock']))
                    if (product['product_id_lectronz'] != ""):
                        lectronz_response = platformFunctions.updateLectronzProductInventory(int(product['product_id_lectronz']), int(inventory_product['product_in_stock']))
                        if (lectronz_response[1] == 200):
                            print('Lectronz Inventory Updated Successfully to:', lectronz_response[0])
                        else:
                            print('Lectronz Inventory Not Updated Successfully:', lectronz_response[0])
                if len(product['product_options']) > 0:
                    for option in product['product_options']:
                        for inventory_option in inventory_product['product_options']:
                            if option['option_id'] == inventory_option['option_id']:
                                if(option['option_additional_cost'] != inventory_option['option_additional_cost']):
                                    print('option additional cost updated!')
                                    option['option_additional_cost'] = inventory_option['option_additional_cost']
                                if(option['option_in_stock'] != inventory_option['option_in_stock']):
                                    print('option in stock updated!')
                                    option['option_in_stock'] = inventory_option['option_in_stock']
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    productsJSON['date_time_updated'] = timestamp
    productsJSON['products'] = productsData
    
    with open("./products/products.json", "w") as f:  #change this to rewrite the original file
        json.dump(productsJSON, f, indent=4)
            
    st.session_state["disabled_inventory_and_prices_button"] = True
    st.session_state["inventory_and_prices_button_checkbox"] = False
    
    return



def comms_order_change():
    order_selected = st.session_state['comms_order_dropdown'][0]
    if (order_selected != 'Select an order'):
        print('COMMS ORDER HAS CHANGED:', order_selected) #prints the index of the option selected
        
        comboOrderID = order_selected.split('--')[0].strip()

        buyer_email_address = df_comms.loc[df_comms['order id'] == comboOrderID, 'buyer_email'].values[0]
        
        buyerFirstName = df_comms.loc[df_comms['order id'] == comboOrderID, 'name'].values[0].split(' ')[0]
        
        platform = df_comms.loc[df_comms['order id'] == comboOrderID, 'platform'].values[0]
        
        order_id = df_comms.loc[df_comms['order id'] == comboOrderID, 'order id'].values[0]
        
        email_subject = platform.capitalize() + ' Order #' + order_id + ' - Order Update'
        
        email_body = 'Hello ' + buyerFirstName + ',\n\nThank you so much for your order!  \n\nBest,\n\nAndrew/MakeItHackin'
        
        st.session_state['comms_email_address'] = buyer_email_address
        st.session_state['comms_email_subject'] = email_subject
        st.session_state['comms_email_box'] = email_body

    return        
        
        

def comms_send_email():
    return         
         
   
   
   
         
def generate_ai_email():
    print('Generating AI Email...')
    order_comms_selected = st.session_state['comms_order_dropdown'][0]
    if (order_comms_selected != 'Select an order'):

        comboOrderID = order_comms_selected.split('--')[0].strip()
        actualOrderID = order_comms_selected.split('-')[0].strip()
        platform = df_comms.loc[df_comms['order id'] == comboOrderID, 'platform'].values[0]
        orderID = df_comms.loc[df_comms['order id'] == comboOrderID, 'order id'].values[0]
            
        with open(f"{platform}_orderDetails-{actualOrderID}.json", 'r') as f:
            originalJSON = json.load(f)
            
        order_number = originalJSON['order_id']
        #platform = originalJSON['platform']
        products_ordered = originalJSON['products']
        customer_city = originalJSON['buyer_city']
        customer_state = originalJSON['buyer_state_province']
        is_custom_order = originalJSON['is_custom_order']
        is_customer_repeat = originalJSON['buyer_is_repeat']
        order_notes = originalJSON['order_notes']
        customer_name = originalJSON['buyer_name']
        customer_name = customer_name.split(' ')[0]
        is_order_gift = originalJSON['order_is_gift']
        is_order_international = originalJSON['is_international']
        
        company_name = 'Make It Hackin'
        
        product_list = []
        
        for product in products_ordered:
            product_list.append(product['product_name'])    
        
        
        if len(product_list) > 1:
            product_list_string = ', '.join(product_list[:-1]) + ', and ' + product_list[-1]
        else:
            product_list_string = product_list[0]
        
        
        openai.api_key = open_ai_api_key

        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"I am an email generating bot that will help write emails to customers.  I keep emails short and sweet.  I do not include a subject for the email.  I only generate the body of the email.  If the customer is from an international country, I will let them know that their shipment may take 1-3 weeks for delivery."},
            {"role": "user", "content": "Please write a very welcoming email to the customer.  My name is Andrew / MakeItHackin.  My company name is" + company_name + "The customer's name is " + customer_name + ".  The customer's order number is " + order_number + ".  The customer's order is " + product_list_string + ".  The customer's location is " + customer_city + ", " + customer_state + ".  The customer's order notes are " + order_notes + ".  The customer's order is a custom order: " + str(is_custom_order) + ".  The customer is a repeat customer: " + str(is_customer_repeat) + ".  The customer's order is a gift: " + str(is_order_gift) + ".  The customer's order is international: " + str(is_order_international) + "."},
            ]
        )
        
        generated_response = response['choices'][0]['message']['content']
        
        email_body = generated_response
        st.session_state['comms_email_box'] = email_body
    
    return 







def load_packaging():
    with open("./shipping/packaging.json", "r") as f:
        packagingJSON = json.load(f)
    packagingData = packagingJSON['packaging']
    return(pd.json_normalize(packagingData))




def load_carriers():
    with open("./shipping/shipping_rates.json", "r") as f:
        carriersJSON = json.load(f)
    carriersData = carriersJSON['carriers']

    flattened_carriers = []
    for carrier in carriersData:
        if (len(carrier['carrier_services']) == 0):
            flattened_carriers.append(carrier)
        else:
            for service in carrier['carrier_services']:
                flattened_carrier = carrier.copy()
                flattened_carrier.update(service)
                flattened_carrier['carrier_service_name'] = service['carrier_service_name']
                flattened_carrier['carrier_service_code'] = service['carrier_service_code']
                flattened_carrier['carrier_service_is_enabled'] = service['carrier_service_is_enabled']
                flattened_carrier['carrier_service_is_international'] = service['carrier_service_is_international']
                flattened_carrier['carrier_service_is_domestic'] = service['carrier_service_is_domestic']
                flattened_carrier['carrier_service_delivery_time'] = service['carrier_service_delivery_time']
                
                flattened_carriers.append(flattened_carrier)
                
    df = pd.json_normalize(flattened_carriers)
    return df


def update_packaging(updated_packaging_df):
    
    return

def update_carriers(updated_carrier_df):
    
    return


def on_order_change_code_uploader():
              
    order_selected_code_uploader = st.session_state['order_dropdown_code_uploader'][0]
    st.session_state['notes_code_uploader'] = ""
    if (order_selected_code_uploader != 'Select an order'):
        print('ORDER HAS CHANGED:', order_selected_code_uploader) #prints the index of the option selected
        if ('example product 1' in order_selected_code_uploader.lower()):
            st.session_state['product_options_code_uploader'] = ("Example Product 1", 0)
        elif ('example product 2' in order_selected_code_uploader.lower()):
            st.session_state['product_options_code_uploader'] = ("Example Product 2", 1)
        elif ('example product 3' in order_selected_code_uploader.lower()):
            st.session_state['product_options_code_uploader'] = ("Example Product 3", 2)
        else:
            st.session_state['product_options_code_uploader'] = ("Select a Product", 4)
        
        comboOrderID = order_selected_code_uploader.split('--')[0].strip()
        platform = df.loc[df['order id'] == comboOrderID, 'platform'].values[0]
        
        personal_and_order_notes = ''
        
        if (platform == 'etsy'):
            personalization_field = df.loc[df['order id'] == comboOrderID, 'personalization'].values[0]
            order_notes = df.loc[df['order id'] == comboOrderID, 'order notes'].values[0]
            private_notes = df.loc[df['order id'] == comboOrderID, 'private notes'].values[0]
            personal_and_order_notes = 'PERSONALIZATION:\n' + personalization_field 
            if (order_notes == ''):
                personal_and_order_notes = personal_and_order_notes + '\n\nORDER NOTES:\n' + 'NONE'
            else:
                personal_and_order_notes = personal_and_order_notes + '\n\nORDER NOTES\n' + order_notes
            if (private_notes == ''):
                personal_and_order_notes = personal_and_order_notes + '\n\nPRIVATE NOTES:\n' + 'NONE'
            else:
                personal_and_order_notes = personal_and_order_notes + '\n\nPRIVATE NOTES:\n' + private_notes
        else:
            personalization_field = df.loc[df['order id'] == comboOrderID, 'order notes'].values[0]
            personal_and_order_notes = 'ORDER NOTES\n' + personalization_field
        
        st.session_state["personalization_text_code_uploader"] = personal_and_order_notes
            
        product_type = df.loc[df['order id'] == comboOrderID, 'type'].values[0]

        st.session_state['product_type_code_uploader'] = product_type
            
        if ('DEFAULT' in order_selected_code_uploader):
            st.session_state['custom_type_code_uploader'] = 'Default'
            default_variables()
        elif ('CUSTOM' in order_selected_code_uploader):
            st.session_state['custom_type_code_uploader'] = 'Custom'
            custom_variables = formatPersonalizationForCode(personalization_field, product_type)
            if (custom_variables[0] == True):
                set_custom_variables_fields(custom_variables)
            else:
                print('CUSTOM VARIABLES NOT FORMATTED CORRECTLY')
    return



def default_variables():
    #print('OPTION SELECTED:', st.session_state["product_options_code_uploader"])
    if st.session_state["product_options_code_uploader"][1] == 0:
        st.session_state['variable_1'] = 'This'
        st.session_state['variable_2'] = 'is'
        st.session_state['variable_3'] = 'An'
        st.session_state['variable_4'] = 'Example 1'
        st.session_state['variable_5'] = 'Product'
        st.session_state['variable_6'] = 'Yay!'
    elif st.session_state["product_options_code_uploader"][1] == 1:
        st.session_state['variable_1'] = 'Example 2'
        st.session_state['variable_2'] = 'Product'
        st.session_state['variable_3'] = 'Is Awesome!'
        st.session_state['variable_4'] = 'Thanks'
        st.session_state['variable_5'] = 'For'
        st.session_state['variable_6'] = 'Purchasing!'
    elif st.session_state["product_options_code_uploader"][1] == 2:
        st.session_state['variable_1'] = 'Thanks'
        st.session_state['variable_2'] = 'For Buying'
        st.session_state['variable_3'] = 'Example 3'
        st.session_state['variable_4'] = 'You Made'
        st.session_state['variable_5'] = 'A Good'
        st.session_state['variable_6'] = 'Choice!'
    elif st.session_state["product_options_code_uploader"][1] == 4:
        clear_variables()
    return



def formatPersonalizationForCode(input_string, product_type):
    input_string = cleanUpText(input_string)
    
    #the product type can be used to do more custom formatting if desired
    if (product_type == 'Option 1'):
        product_type = 'option_1'
    elif (product_type == 'Option 2'):
        product_type = 'option_2'
    elif (product_type == 'Option 3'):
        product_type = 'option_3'
        
    successful = False
        
    try:
        
        if ("variable 1:" not in input_string):
            input_string = 'variable 1: ' + input_string
        if ("variable 2:" not in input_string):
            input_string = input_string + '\nvariable 2:'
        if ("variable 3:" not in input_string):
            input_string = input_string + '\nvariable 3:'    
        if ("variable 4:" not in input_string):
            input_string = input_string + '\nvariable 4:'    
        if ("variable 5:" not in input_string):
            input_string = input_string + '\nvariable 5:'
        if ("variable 6:" not in input_string):
            input_string = input_string + '\nvariable 6:'
        
        variable_1_custom = input_string.split("variable 1:")[1].split("variable 2:")[0].strip()
        variable_2_custom = input_string.split("variable 2:")[1].split("variable 3:")[0].strip()
        variable_3_custom = input_string.split("variable 3:")[1].split("variable 4:")[0].strip()
        variable_4_custom = input_string.split("variable 4:")[1].split("variable 5:")[0].strip()
        variable_5_custom = input_string.split("variable 5:")[1].split("variable 6:")[0].strip()
        variable_6_custom = input_string.split("variable 6:")[1].strip()
        
        successful = True

    except Exception as e:
        print('ERROR in formatPersonalizationForCode:', e)
        variable_6_custom = 'custom variables not formatted correctly'
        variable_5_custom = 'custom variables not formatted correctly'
        variable_4_custom = 'custom variables not formatted correctly'
        variable_3_custom = 'custom variables not formatted correctly'
        variable_2_custom = 'custom variables not formatted correctly'
        variable_1_custom = 'custom variables not formatted correctly'
        successful = False
    
    custom_variables = [successful, variable_1_custom, variable_2_custom, variable_3_custom, variable_4_custom, variable_5_custom, variable_6_custom]
    
    return(custom_variables)

if 'custom_type_code_uploader' not in st.session_state:
    st.session_state['custom_type_code_uploader'] = 'Default'
    
if 'notes_warning_text' not in st.session_state:
    st.session_state['notes_warning_text'] = ''    
    
if 'data_source_radio' not in st.session_state:
    st.session_state['data_source_radio'] = 'Fake'
    
def set_custom_variables_fields(custom_variables):
    if st.session_state["product_options_code_uploader"][1] != 4:
        st.session_state['variable_1'] = custom_variables[1]
        st.session_state['variable_2'] = custom_variables[2]
        st.session_state['variable_3'] = custom_variables[3]
        st.session_state['variable_4'] = custom_variables[4]
        st.session_state['variable_5'] = custom_variables[5]
        st.session_state['variable_6'] = custom_variables[6]
    
    return

def on_product_type_change_code_uploader():
    if st.session_state['custom_type_code_uploader'] == 'Default':
        default_variables()
    elif st.session_state['custom_type_code_uploader'] == 'Custom':
        custom_variables()
    return


def custom_variables():
    if st.session_state["product_options_code_uploader"][1] == 0:
        st.session_state['variable_1'] = 'This'
        st.session_state['variable_2'] = 'is a'
        st.session_state['variable_3'] = 'Custom'
        st.session_state['variable_4'] = 'Example 1'
        st.session_state['variable_5'] = 'Product'
        st.session_state['variable_6'] = 'Yay!'
    elif st.session_state["product_options_code_uploader"][1] == 1:
        st.session_state['variable_1'] = 'Example 2'
        st.session_state['variable_2'] = 'Custom Product'
        st.session_state['variable_3'] = 'Is Awesome!'
        st.session_state['variable_4'] = 'Thanks'
        st.session_state['variable_5'] = 'For'
        st.session_state['variable_6'] = 'Purchasing!'
    elif st.session_state["product_options_code_uploader"][1] == 2:
        st.session_state['variable_1'] = 'Thanks For'
        st.session_state['variable_2'] = 'Buying a Custom'
        st.session_state['variable_3'] = 'Example 3'
        st.session_state['variable_4'] = 'You Made'
        st.session_state['variable_5'] = 'A Good'
        st.session_state['variable_6'] = 'Choice!'
    elif st.session_state["product_options_code_uploader"][1] == 4:
        clear_variables()
    return
        
def set_default_variables():
    default_variables()
    return


def clear_variables():
    st.session_state['variable_1'] = ''
    st.session_state['variable_2'] = ''
    st.session_state['variable_3'] = ''
    st.session_state['variable_4'] = ''
    st.session_state['variable_5'] = ''
    st.session_state['variable_6'] = ''
    return


def upload_code_to_board():
    print('UPLOADING CODE TO BOARD')
    product_name = st.session_state["product_options_code_uploader"][0]
    product_index = st.session_state["product_options_code_uploader"][1]
    is_product_custom = st.session_state["custom_type_code_uploader"]
    product_option = st.session_state["product_type_code_uploader"]
    
    if (is_product_custom == 'Default'):
        is_product_custom = False
    elif (is_product_custom == 'Custom'):
        is_product_custom = True
        
    upload_configs = {
        "product_name": product_name,
        "product_index": product_index,
        "is_product_custom": is_product_custom,
        "product_option": product_option,
        "variable_1": st.session_state['variable_1'],
        "variable_2": st.session_state['variable_2'],
        "variable_3": st.session_state['variable_3'],
        "variable_4": st.session_state['variable_4'],
        "variable_5": st.session_state['variable_5'],
        "variable_6": st.session_state['variable_6']
    }
    
    uploadStatus = uploadCode.upload(upload_configs)

    if (uploadStatus[0] == 'Arduino Upload Success!'):
        with open(uploadStatus[1], "r") as f:       
            st.session_state["code_uploader_code_results"] = f.read()
        
    st.session_state["sidebar_text"] = uploadStatus[0]    
    
    return  

def save_notes():
    with open("./settings/notes.json", 'w') as json_file:
        # Save the updated JSON data back to the file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        notesData = {"date": timestamp, "notes": st.session_state["notes_text"]}
        json.dump(notesData, json_file, indent=4)        
    st.session_state['notes_warning_text'] = ''   
    return                   

def notes_warning():
    #notes_status_button_name = "Press to Save Updates"
    st.session_state["notes_warning_text"] = "WARNING: You have unsaved changes.  Press the save button to save your changes."
    return

def analyze_issue(problem):
    return

def main():
    global df, placeholderImage, selected_type, shipping_method_list, df_labels, df_comms
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(["Orders", "Code Uploader", "Mass Order Fulfillment", "Shipping Labels", "Packaging", "Inventory", "Communication", "Promotions", "Reverse Engineering", "Settings", "Notes/Updates"])

        
    with tab1: #orders
        df = load_data(False)

        columns = list(df.columns)

        new_order = ['order_id', 'buyer_name', 'product_name', 'product_quantity', 'product_type', 'personalization', 'order_date', 'ship_by_date', 'order_state', 'order_notes', 'private_notes', 'platform', 'order_has_unread_message', 'is_international', 'is_custom_order', 'default_ordered_with_personalization', 'personalized_ordered_with_no_instructions', 'total_item_quantity']
        columns_reordered = [c for c in new_order if c in columns] + [c for c in columns if c not in new_order]
        
        df_reordered = df[columns_reordered]
        df = df_reordered        
        df.set_index('order_id', inplace=True)
        df.reset_index(inplace=True)
        
        df = df.rename(columns={'order_id': 'order id', 'buyer_name': 'name', 'order_date': 'order date', 'ship_by_date': 'ship by', 'order_state': 'order state', 'order_notes': 'order notes', 'private_notes': 'private notes', 'order_has_unread_message': 'unread', 'total_item_quantity': 'total', 'is_international': 'INT', 'is_custom_order': 'CUS', 'default_ordered_with_personalization': 'DFWP', 'personalized_ordered_with_no_instructions': 'PRNI', 'product_name': 'product', 'product_quantity': 'QTY', 'product_type': 'type', })
        
        st.dataframe(df.set_index(df.columns[0]), height=175) #ORIGINAL
        
        st.sidebar.text_area("Print Statements", placeholder = "print statments will go here", height = 300, key="sidebar_text", disabled = False)

        
        orderIDIndex = df.columns.get_loc("order id")
        order_id_values = df.iloc[:, orderIDIndex].values
        orderNameIndex = df.columns.get_loc("name")
        order_name_values = df.iloc[:, orderNameIndex].values
        orderProductIndex = df.columns.get_loc("product")
        order_product_values = df.iloc[:, orderProductIndex].values
        orderCustomIndex = df.columns.get_loc("CUS")
        order_custom_values = df.iloc[:, orderCustomIndex].values
        orderTypeIndex = df.columns.get_loc("type")
        order_type_values = df.iloc[:, orderTypeIndex].values
        orderNumberAndNamesList = []
        for i in range(len(order_id_values)):
            concatenated_value = str(order_id_values[i]) + ' -- ' + str(order_name_values[i])
            orderNumberAndNamesList.append((concatenated_value,))
        
        new_list = [('Select an order',)]
        code_uploader_list = [('Select an order',)]
        shipping_method_list = [('Select a shipping method',0),('Ground Advantage' + str(st.session_state.get("ground_advantage_price", " 0.00")),1),('First Class' + str(st.session_state.get("first_class_price", " 0.00")),2),('Priority Mail' + str(st.session_state.get("priority_mail_price", " 0.00")),3),('Priority Mail (small box)' + str(st.session_state.get("priority_mail_small_box_price", " 0.00")),4),('Priority Mail Express' + str(st.session_state.get("priority_mail_express_price", " 0.00")),5),('First Class International' + str(st.session_state.get("first_class_international_price", " 0.00")),6),('Priority Mail International' + str(st.session_state.get("priority_mail_international_price", " 0.00")),7),('Priority Mail Express International' + str(st.session_state.get("priority_mail_express_international_price", " 0.00")),8),('Media Mail' + str(st.session_state.get("media_mail_price", " 0.00")),9),('Unknown',10)]
        for i in range(len(order_id_values)):
            customValue = 'DEFAULT'
            if ('default' in order_type_values[i].lower()):
                customValue = 'DEFAULT'
            elif ('custom' in order_type_values[i].lower()):
                customValue = 'CUSTOM'
            concatenated_value = str(order_id_values[i]) + ' -- ' + str(order_name_values[i]) + ' -- ' + str(order_product_values[i]) + ' -- ' + customValue
            code_uploader_list.append((concatenated_value,))

        
        for element in orderNumberAndNamesList:
            if element not in new_list:
                new_list.append(element)
        
        order_list = new_list
                
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            selected_order = st.selectbox("Select an order", order_list, index=0, format_func=lambda option: option[0], key="shipping_order_dropdown", on_change=shipping_order_change)

        with col4:
            st.text_input("Data Last Updated:", key="refresh_datetime")


        with col2:
            col_left, col_mid, col_right = st.columns(3)
            
            with col_left:
                st.button('Refresh', key="refresh_button", on_click=load_data)
            with col_mid:
                st.button("Print Receipt", on_click=print_receipt, disabled=st.session_state.get("disabled_tab1_buttons", True))    
            with col_right:
                st.button("Get Rates/Verify", on_click=get_rates_and_verify_address, disabled=st.session_state.get("disabled_tab1_buttons", True))    
            
        with col3:
            col_left, col_mid, col_right = st.columns(3)
            
            with col_left:
                st.button("Ship Package", disabled=st.session_state.get("disabled_ship_package", True), on_click=ship_package)
            with col_mid:
                st.button('Fulfill Order', key="fulfill_order_button", on_click=fulfill_order, disabled=st.session_state.get("disabled_fulfill_order", True))
            with col_right:
                st.button('Print Manifest', key="manifest_button", on_click=create_manifest)
                 
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            name = st.text_input("Name", key="addressName")    
            address1 = st.text_input("Address Line 1", key="addressLine1")
            address2 = st.text_input("Address Line 2", key="addressLine2")
            address3 = st.text_input("Address Line 3", key="addressLine3")
            st.text_input("Customs Info", key="customsInfo")
            
        with col2:
            city = st.text_input("City", key="addressCity")
            state = st.text_input("State", key="addressState")
            zip = st.text_input("Zip", key="addressZip")
            country_code = st.text_input("Country Code", key="addressCountry")
            st.text_input("Notes For Shipment", key="notesForShipment")
            
        with col3:
            shipping_verification = st.text_input("Address Verification", key="address_verification")
            shipping_total = st.text_input("Shipping Paid", key="shippingPaid")

            shipping_method = st.selectbox("Shipping Method", shipping_method_list, index=0, format_func=lambda option: option[0], key="shipping_method_dropdown")#, on_change=on_order_change)
            get_date_for_shipment() 
            
            left_col, right_col = st.columns([1, 1])
            with left_col:
                shipping_date = st.date_input("Shipping Date", key="shippingDate", min_value=datetime.today(), max_value=st.session_state.get("max_date_for_shipment", None), value=st.session_state.get("date_for_shipment", datetime.today()), on_change=update_manifest_count)
            update_manifest_count()
            with right_col:
                st.text_input("Manifest Packages", key="manifest_packages", value=st.session_state.get("manifest_count", None))

            st.text_input("Tracking Number", key="label_tracking_number")
        with col4:
            package_size = st.selectbox("Package Size - Presets", [("Manual",),("7x4x2",),("11x7x3",)], index=1, format_func=lambda option: option[0], key="package_size_dropdown", on_change=update_package_sizes)
            
            left_col, mid_col, right_col = st.columns([1, 1, 1])
            
            with left_col:
                st.number_input("Length", min_value=1, max_value=99, step=1, key="packageSizeLength", value=7)
            with mid_col:
                st.number_input("Width", min_value=1, max_value=99, step=1, key="packageSizeWidth", value=4)
            with right_col:
                st.number_input("Height", min_value=1, max_value=99, step=1, key="packageSizeHeight", value=2)
                
            left_col, right_col = st.columns([1, 1])
            with left_col:
                st.number_input("Package Weight - Pounds", min_value=0, max_value=15, step=1, key="packageWeightLB")
            with right_col:
                st.number_input("Package Weight - Ounces", value=8, min_value=0, max_value=15, step=1, key="packageWeightOZ")
            package_value = st.number_input("Package Value", min_value=0, max_value=999, step=1, key="packageValue")
            
            left_col, right_col = st.columns([1, 1])
            with left_col:
                st.checkbox('Override Ship Date', key='ship_date_override', value=True)
                st.checkbox('Enable Fulfill Button', value=False, on_change=on_fulfill_button_change, key="fulfill_button_checkbox")
                st.checkbox('Completed Orders', key='completedOrderCheckBox', value=False, on_change=on_address_button_change)
                st.number_input("Amount of Completed Orders", value=10, min_value=0, max_value=1000, step=10, key="amount_of_completed_orders")
            with right_col:
                st.checkbox('Insurance', key='insuranceCheckBox', value=False)
                st.checkbox('Override Address Errors', key='address_error_override', value=False)
                st.checkbox('placeholder_checkbox', key='placeholder_checkbox', value=False)
                st.number_input("Offset", value=0, min_value=0, max_value=1000, step=10, key="amount_of_completed_orders_offset")
        


            
    with tab3: #mass order fulfillment
        st.dataframe(df.set_index(df.columns[0]), height=175) 
        st.radio(
                "Fulfill Orders By",
                ["None", "Product Type", "Ship By Date", "Domestic", "International", "Custom", "All"],
                key="mass_order_fulfillment_type",
                horizontal=True
            )
        col1, col2, col3 = st.columns([1, 1, 1]) 
        with col1:
            st.multiselect(
                'Products',
                ['Flipper Multiboard', 'PCB Badge', 'Wireless Board', 'RFID Reader', 'All Products'],
                )
        with col2:
            st.date_input("Ship All Orders That Have Been Scheduled to Ship By", min_value=datetime.today())
        with col3:
            st.date_input("Date to Actually Ship", min_value=datetime.today())
        
        col1, col2, col3 = st.columns([1, 1, 1])    
        with col1:
            st.checkbox('Enable Mass Ship Button', value=False, on_change=on_mass_ship_button_change, key="mass_ship_button_checkbox")
        with col2:
            st.button("Ship 'Em", on_click=mass_ship,  key="mass_ship_button", disabled=st.session_state.get("disabled_mass_ship_button", True))
    
        
    with tab4: #shipping labels
        
        df_labels = load_labels()
        
        label_columns = list(df_labels.columns)
        
        st.write("SHIPPING LABELS")
        st.dataframe(df_labels.set_index('label_id'), height=175, column_order=("label_id", "order_id",  "is_voided", "customer_name","tracking_number", "ship_date", "notes", "is_manifested", "manifest_name", "manifest_id", "label_download_pdf", "order_date",  "order_platform", "box_size", "weight", "weight_unit", "is_insured", "label_creation_date", "label_status", "service_code", "tracking_status", "shipping_address", "voided_at"))
        
        label_orderIDIndex = df_labels.columns.get_loc("order_id")
        label_order_id_values = df_labels.iloc[:, label_orderIDIndex].values
        label_labelID_Index = df_labels.columns.get_loc("label_id")
        label_label_id_values = df_labels.iloc[:, label_labelID_Index].values
        label_order_customer_name_index = df_labels.columns.get_loc("customer_name")
        label_order_name_values = df_labels.iloc[:, label_order_customer_name_index].values
        label_orderNumberAndNamesList = []
        for i in range(len(label_order_id_values)):
            concatenated_value = str(label_label_id_values[i]) + ' -- ' + str(label_order_id_values[i]) + ' -- ' + str(label_order_name_values[i])
            label_orderNumberAndNamesList.append((concatenated_value,))
        
        new_list = [('Select a Label',)]
        
        for element in label_orderNumberAndNamesList:
            if element not in new_list:
                new_list.append(element)

        label_order_list = new_list

        labels_col1, labels_col2, labels_col3, labels_col4 = st.columns([1, 1, 1, 1])
        
        with labels_col1:
            selected_label = st.selectbox("Select a Label", label_order_list, index=0, format_func=lambda option: option[0], key="label_order_dropdown")
        with labels_col2:
            st.checkbox('Enable Void Label Button', value=False, on_change=on_void_label_button_change, key="void_label_button_checkbox")
        with labels_col3:
            st.button("Void Label", on_click=void_label,  key="void_label_button", disabled=st.session_state.get("disabled_void_label_button", True))
        with labels_col4:
            st.button("Print Label", on_click=print_label)

            
        df_manifests = load_manifests()    
        
        st.write("MANIFESTS")
        st.dataframe(df_manifests.set_index('manifest_id'), height=175, column_order=("manifest_id", "form_id",  "created_at", "ship_date", "shipments", "label_ids", "warehouse_id", "submission_id", "carrier_id", "manifest_download"))

        manifest_id_Index = df_manifests.columns.get_loc("manifest_id")
        manifest_id_values = df_manifests.iloc[:, manifest_id_Index].values
        
        manifest_id_list = []
        for i in range(len(manifest_id_values)):
            concatenated_value = str(manifest_id_values[i])
            manifest_id_list.append((concatenated_value,))
            
        manifest_list = [('Select a Manifest',)]
        
        for manifest_element in manifest_id_list:
            if manifest_element not in manifest_list:
                manifest_list.append(manifest_element)
                
        manifest_col1, manifest_col2, manifest_col3 = st.columns([1, 1, 1])
        
        with manifest_col1:
            selected_manifest = st.selectbox("Select a Manifest", manifest_list, index=0, format_func=lambda option: option[0], key="manifest_list_dropdown", on_change=shipping_order_change)
        with manifest_col2:
            st.button("Edit Manifest")
        with manifest_col3:
            st.button("Print Manifest")
        
        st.text_area("Results", placeholder = "Results will go here", height = 200, key="shipping_text_results", disabled = True)

    
    with tab5: #packaging
        df_packaging = load_packaging()
        
        packaging_columns = list(df_packaging.columns)

        st.write("PACKAGING")
        updated_packaging = st.data_editor(df_packaging.set_index('package_name'), num_rows = "dynamic", height=300, column_order=("package_name", "package_description",  "package_status", "package_length", "package_width", "package_height", "package_weight_lb", "package_weight_oz", "package_id"))

        df_carriers = load_carriers()
        
        carriers_columns = list(df_carriers.columns)

        st.write("CARRIERS")
        updated_carriers = st.data_editor(df_carriers.set_index('carrier_name'), num_rows = "dynamic", height=300, column_order=("carrier_name", "carrier_is_default",  "carrier_is_enabled", "carrier_service_name", "carrier_service_code", "carrier_service_is_enabled", "carrier_service_is_international", "carrier_service_is_domestic", "carrier_service_delivery_time"))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.button("Update Packages", on_click=update_packaging,  args=(updated_packaging,), key="update_packaging_button")
        with col2:    
            st.button("Update Carriers", on_click=update_carriers,  args=(updated_carriers,), key="update_carriers_button")
            
                
        
    with tab6: #inventory
        df_inventory = load_inventory()
        edited_inventory_df = st.data_editor(df_inventory.set_index('product_name'), height = 400, disabled=("product_name", "product_sku", "option_display_name", "product_weight_lb", "product_weight_oz", "product_length", "product_width", "product_height", "product_shipping_rates", "product_tags"), column_order=("product_name", "option_display_name", "option_in_stock", "option_additional_cost", "product_in_stock", "product_base_price", "product_sku", "product_weight_lb", "product_weight_oz", "product_length", "product_width", "product_height", "product_shipping_rates", "product_tags"))
        inventory_col1, inventory_col2, inventory_col3 = st.columns([2, 1, 1])
        with inventory_col1:

            st.radio(
                "Update Platform",
                ["All Platforms", "Etsy", "Tindie", "Shopify", "Lectronz"],
                key="inventory_platform",
                horizontal=True
            )
        
        reverse_inventory_text = reverse_inventory(edited_inventory_df)

        with inventory_col2:
            st.checkbox('Enable Update Button', value=False, on_change=on_inventory_and_prices_button_change, key="inventory_and_prices_button_checkbox")
        with inventory_col3:
            st.button("Update Inventory and Prices", on_click=update_inventory_and_prices,  args=(reverse_inventory_text,), key="update_inventory_and_prices_button", disabled=st.session_state.get("disabled_inventory_and_prices_button", True))

    
    
    
    
    
           
    with tab7: #communication

        df_comms = df
        comms_order_list = order_list          
        
        st.dataframe(df_comms.set_index(df_comms.columns[0]), height=200) #ORIGINAL 
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:   
            selected_comms_order = st.selectbox("Select an order", comms_order_list, index=0, format_func=lambda option: option[0], key="comms_order_dropdown", on_change=comms_order_change)     
        with col2:
            st.text_input("Customer E-mail Address", key="comms_email_address")
        with col3:
            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.button("Generate AI Email", on_click=generate_ai_email)#, disabled=st.session_state.get("disabled_tab1_buttons", True))
            with col_right:
                st.button("Send E-Mail to Customer", on_click=comms_send_email)#, disabled=st.session_state.get("disabled_tab1_buttons", True))
        st.text_input("E-mail Subject", key="comms_email_subject")
        st.text_area("Type your e-email here", "", key= "comms_email_box", height = 300, disabled = False)#, label_visibility='hidden')
        
         
    
    
        
    with tab8: #promotions
        col1, col2, col3 = st.columns([1, 1, 1]) 
        with col1:
            col_left, col_right= st.columns([1, 1])
            with col_left:
                st.radio(
                    "Type of Sale",
                    ["Site-sale", "Discount Code"],
                    key="sale_type",
                    horizontal=True
                )
            with col_right:
                st.button("Generate Code")
            st.text_input("Promotion Code")#, key="notes")
            st.date_input("Start Date", min_value=datetime.today())
        with col2:
            st.radio(
                "Type of Sale",
                ["Percentage", "Dollar Amount"],
                key="promotion_type",
                horizontal=True
            )
            st.number_input("Discount Amount", step=1,)
            st.date_input("End Date", min_value=datetime.today())
        with col3:
            st.radio(
                "Platform",
                ["Etsy", "Tindie", "Shopify", "Lectronz", "All Platforms"],
                key="promotion_platform",
                horizontal=True
            )
            st.multiselect(
            'Apply to Products',
            ['Flipper Multiboard', 'PCB Badge', 'Wireless Board', 'RFID Reader', 'All Products'],
            )
            col1, col2= st.columns([1, 1]) 
            with col1:
                st.checkbox('Promote on Social Media', value=False)
                st.checkbox('Platform Can Promote Code', value=False)
            with col2:
                st.button("Upload Promotion")
        
        st.text_area("Current Promotion Codes", placeholder = "Platform: Etsy\t\t\t\tCode: NOV2023\t\t\t\tStart Date: November 1, 2023 \t\t\tEnd Date: November 30, 2023\t\t\tEmails Sent: 4\t\t\tUses: 2\t\t\t\tRevenue: $75 \n" + "Platform: Tindie\t\t\tCode: WN83EDNW9\t\t\tStart Date: January 1, 2023 \t\t\tEnd Date: December 31, 2023\t\t\tUses: 54\t\t\t\tProducts: 2\t\t\tSale/Code: Discount \n" + "Platform: Lectronz\t\t\tCode: SALE2023\t\t\t\tStart Date: October 31, 2023 \t\t\tEnd Date: December 12, 2023\t\t\tUses: 1\t\t\t\t\tProducts: 1\t\t\tReduction: $4.00\t\t\t\tDescription: Sale\n" + "Platform: Shopify\t\t\tCode: BLACKFRIDAY2023\t\tStart Date: November 24, 2023 \t\tEnd Date: November 25, 2023\t\t\tUses: 0\t\t\t\t\tProducts: 5\t\t\tReduction: 50% off \n" , height = 130, key="current_promos", disabled = True)
        st.text_area("Expired Promotion Codes", placeholder = "Platform: Etsy\t\t\t\tCode: ABCDEF\t\t\t\tStart Date: January 1, 2023 \t\t\tEnd Date: August 7, 2023\t\t\tEmails Sent: 0\t\t\tUses: 22\t\t\tRevenue: $875 \n" + "Platform: Etsy\t\t\t\tCode: TRYNOW\t\t\t\tStart Date: May 22, 2023 \t\t\t\tEnd Date: July 3, 2023\t\t\t\tEmails Sent: 5\t\t\tUses: 92\t\t\tRevenue: $1,790 \n" + "Platform: Tindie\t\t\tCode: E9E659D1\t\t\tStart Date: February 6, 2023 \t\t\tEnd Date: March 6, 2023\t\t\tUses: 5\t\t\t\t\tProducts: 3\t\t\tSale/Code: Discount \n" + "Platform: Tindie\t\t\tCode: OIE32EKJ\t\t\tStart Date: September 4, 2022 \t\t\tEnd Date: December 3, 2022\t\tUses: 64\t\t\t\tProducts: 1\t\t\tSale/Code: Discount \n" + "Platform: Lectronz\t\t\tCode: MAKE50\t\t\t\tStart Date: August 1, 2023 \t\t\t\tEnd Date: August 31, 2023\t\t\tUses: 11\t\t\t\tProducts: 1\t\t\tReduction: $5.00\t\t\t\tDescription: Loyalty \n" + "Platform: Shopify\t\t\tCode: MOTHERSDAY23\t\tStart Date: May 1, 2023 \t\t\t\tEnd Date: May 31, 2023\t\t\tUses: 55\t\t\t\tProducts: 1\t\t\tReduction: 10% off \n" , height = 200, key="expired_promos", disabled = True)
    
                            
        
      
    
    with tab9: # reverse engineering tab
        curl_command_example = 'curl http://example.com/api -X POST -H "Content-Type: application/json" --data \'{"key": "value"}\' '
        curl_command_example = "curl 'https://shop.flipperzero.one/products/flipper-zero.json'"
        
        col1, col2 = st.columns(2)
        with col1:
            curl_command = st.text_area("Paste your curl command here. In Chrome: Right Click, Inspect, Go to Network Tab, Right Click on Entry, Copy as cURL (bash))", curl_command_example, height = 200, disabled = False)#, label_visibility='hidden')
        python_code = curl_to_requests(curl_command)
        python_code_text = python_code[0]
        python_code_url = python_code[1]
        python_code_headers = python_code[2]
        python_code_data = python_code[3]
        python_code_call = python_code[4]
        
        with col2:
            st.code(python_code_text, language='python')
        
        st.button("Run Python Script", on_click=run_python_requests, args=[(python_code_url,),(python_code_headers,),(python_code_data,),(python_code_call,)])#, disabled=True)
        
        curl_python_results_text = ''
        curl_python_results = ''
        
        try:
            curl_python_results = st.session_state["curl_python_results"]
            curl_python_results_text = curl_python_results[0]
            curl_python_results_is_json = curl_python_results[1]
        except:
            curl_python_results_text = 'Results will go here'
        
        result_language = 'text'
        
        try:
            if (curl_python_results_is_json == True):
                result_language = 'json'
            elif (curl_python_results_is_json == False):
                result_language = 'html'
        except:
            pass
        st.code(curl_python_results_text, language=result_language)
    
     
    
        
    with tab10: #settings
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        with col1:
            st.radio(
                    "Order Data Source",
                    ["Real", "Sanitized", "Fake"],
                    key="data_source_radio",
                    index=2,
                    horizontal=True
                )
        with col2:    
            st.radio(
                    "Shipping Environment",
                    ["Sandbox", "Production"],
                    key="shipping_environment_radio",
                    horizontal=True
                )
        with col3:
            st.radio(
                    "Device Type",
                    ["Not a Pi", "Raspberry Pi"],
                    key="rapsberry_pi_radio",
                    horizontal=True
                )
        with col4:
            st.radio(
                    "Measurement System",
                    ["Imperial", "Metric"],
                    key="measurement_system_radio",
                    horizontal=True
                )
        with col5:
            st.button("Reset Combined Orders File", on_click=reset_combined_orders_file)

        with col1:
            st.text_input("Shopify Access Token", "sdlkf392nslkdgl-38nsjngjks", type="password", key="shopify_access_token_input")
            st.text_input("Shopify API Key", "alskjdflkj32342309", type="password", key="shopify_api_key_input")
            st.text_input("Shopify Secret Key", "ifg39ghsdg", type="password", key="shopify_secret_key_input")
            st.text_input("Etsy Shop ID", "469857421", key="etsy_shop_id_input")
            st.text_input("Etsy Client ID", "aghgwh32409u9vrnvivn", type="password", key="etsy_client_id_input")
            st.text_input("Etsy Client Secret", "AHFIO978EDHFFWFO", type="password", key="etsy_client_secret_input")
        
        with col2:
            st.text_input("Tindie Username", "admin", key="tindie_username_input")
            st.text_input("Tinide Password", "this15mypa$$word", type="password", key="tindie_password_input")
            st.text_input("Tindie API Key", "sgfhj8w98039lslwie", type="password", key="tindie_api_key_input")
            st.number_input("Elasped Days Before Refreshing Tindie Cookie", value = 5, min_value=1, max_value=14, step=1, key="tindie_cookie_refresh_days_input")
        
        with col3:
            st.text_input("Lectronz Bearer Token", "asdjghasd9g8798sd7689", type="password", key="lectronz_bearer_token_input")
            st.number_input("EUR to USD Exchange Rate (Lectronz Order)", min_value=0.0000, max_value=255.0000, step=.01, key="eur_to_usd_exchange_rate")  
            st.checkbox('Update Exchange Rate Automatically', key='exchange_rate_checkbox', value=True)
            st.text_input("Exchange Rate API Key", "saldkhfkusajdhg98s6dg5656", type="password", key="exchange_rate_api_key_input")
        
        with col4:
            st.text_input("ShipEngine API Key Sandbox", "kalsjdhf98f9687f6", type="password", key="shipengine_api_key_sandbox_input")
            st.text_input("ShipEngine API Key Production", "d78f6g87df6578g", type="password", key="shipengine_api_key_production_input")
            st.text_input("ShipEngine Warehouse ID Sandbox", "se-6546545464", key="shipengine_warehouse_id_sandbox_input")
            st.text_input("ShipEngine Warehouse ID Production", "se-5432121", key="shipengine_warehouse_id_production_input")
            st.text_input("ShipEngine USPS ID Sandbox", "se-54245878", key="shipengine_carrier_id_sandbox_input")
            st.text_input("ShipEngine USPS ID Production", "se-5544657465", key="shipengine_carrier_id_production_input")
        
        with col5:
            st.text_input("From E-mail Address", "sales@example.com", key="from_email_address_input")
            st.text_input("From E-mail Password", "this15myEmailpa$$word", type="password", key="from_email_password_input")
            st.text_input("Email Server", "server.mail.com", key="email_server_input")
            st.text_input("Email Server Port", "587", key="email_server_port_input")
            st.text_input("Open AI API Key", "askjdhgkjashd398234329439fn33", type="password", key="open_ai_api_key_input")

    with open("./settings/notes.json", "r") as f:
        notesJSON = json.load(f)
        notesData = notesJSON['notes']

    with tab11: # note/updates tab
        st.text_area("Notes", notesData, placeholder = "Use this area for notes", height = 600, key="notes_text", disabled = False, on_change=notes_warning)
        st.button("Save Notes", key = "save_notes_button", on_click=save_notes)
        st.write(st.session_state["notes_warning_text"])

    
    with tab2: #code uploader
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            selected_order = st.selectbox("Select an order", code_uploader_list, index=0, format_func=lambda option: option[0], key="order_dropdown_code_uploader", on_change=on_order_change_code_uploader)
        with col2:
            product_code_option_list = [("Select a Product",4), ("Example Product 1", 0), ("Example Product 2", 1), ("Example Product 3", 2)]
            st.selectbox("Pick a product to upload", product_code_option_list, index=0, format_func=lambda option: option[0], key="product_options_code_uploader", on_change=on_product_type_change_code_uploader)
        with col3:
            st.text_input("Product Type", key="product_type_code_uploader")
        
        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            input1 = st.text_input("Variable 1", key="variable_1")
            input2 = st.text_input("Variable 2", key="variable_2")
            input3 = st.text_input("Variable 3", key="variable_3")
        
        with col2:
            input4 = st.text_input("Variable 4", key="variable_4")
            input5 = st.text_input("Variable 5", key="variable_5")
            input6 = st.text_input("Variable 6", key="variable_6")
        
        with col3:
            st.text_area("Personalization/Notes", placeholder = "Personalization and Order Notes will go here", height = 215, key="personalization_text_code_uploader", disabled = False)

        with col1:
            col11, col22, col33 = st.columns(3)
            with col11:
                st.button("Default Variables", key = "dtb_code_uploader", on_click=set_default_variables)
        
            with col22:
                st.button("Clear Variables", key = "cfb_code_uploader", on_click=clear_variables)
            
            with col33:
                st.button("Reset Variables", on_click=on_order_change_code_uploader)
            
        with col2:        
            st.button("Upload", key = "upb_code_uploader", on_click=upload_code_to_board)
            
 

        st.text_area("Custom Code", height = 300, placeholder = "Your custom code will be displayed here", key="code_uploader_code_results", disabled = True)    
        

    return
    


if __name__ == '__main__':
    main()

    
 