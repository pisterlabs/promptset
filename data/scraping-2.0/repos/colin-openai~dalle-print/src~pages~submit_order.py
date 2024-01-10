import streamlit as st
import openai
import os
import requests

from printful import get_product_templates, get_variants, get_post_headers

def submit_order(image_urls,selected_shirt,pf_key,customer_details):
    customer_details.update({'items': [ 
                        {
                            'variant_id': image_urls[selected_shirt][0],
                            "quantity": 1
                        }
                     ]})

    # TODO: Refactor this horrible line getting path of the URL file we want
    with open(os.path.join(os.curdir,'urls',str(urls[int(selected_image[-1])-1])),'r') as file:
                    selected_url = file.read()
    # TODO: Add UI to choose the placement
    placement = {
        'files': [ {
            "type": "front",
            "url": selected_url
                }
        ]
    }    

    customer_details['items'][0].update(placement)

    #st.write(customer_details)
    
    try:
        order_response = requests.post(url='https://api.printful.com/orders'
                                    ,headers = get_post_headers(pf_key)
                                    ,json=customer_details)
        st.write(f'''Order created successfully. ID is {order_response.json()['result']['id']}''')

    except Exception as e:
        st.write(f'Sad times, order failed with code {order_response.json()}')

### CONFIG
# set API key
openai.api_key = os.environ.get("OPENAI_API_KEY")
# Get API Key and create headers
pf_key = os.environ.get("PRINTFUL_API_KEY")
# Set fake customer details for later
# TODO: Add screen for them to input these
customer_details = {
    "recipient": {
        "name": "C Fresh",
        "address1": "6 Boomtown Street",
        "city": "Glasgow",
        "country_code": "GB",
        "zip": "G71 7RY"
    }
}
# Read in URLs
urls = sorted(os.listdir('urls'))
# Read in selection from other page
with open(os.path.join(os.curdir,'selection.txt'),'r') as file:
            selected_image = file.read()

st.title('Choose your image')


st.subheader('Immortalise your genius')

st.text("")


product_templates = get_product_templates(pf_key)
#st.write(product_templates)

# Here we get available variants for the product templates
# TODO: Offer them a bigger library of items
# TODO: Offer them size choice
# TODO: Offer them color choice
variant_list = []

with st.spinner('Selecting clothing options...'):

    for template in product_templates:
        variants = template['available_variant_ids']
        #st.write(variants)
        for variant in variants[:1]:
            print(f'Getting variant {variant}')
            variant_list.append(get_variants(pf_key,variant_id=variant).json()['result']['variant'])

        
# Makes a list of IDs and images for the products selected
#variant_list_trimmed = variant_list

image_urls = [(x['id'],x['image']) for x in variant_list]

#st.write(image_urls)

col1, col2 = st.columns(2)

col1.image(image_urls[0][1])
col2.image(image_urls[1][1])



selected_shirt = st.radio('Select the option you choose',[f'Option {image_urls.index(x)+1}' for x in image_urls ],index=0)

if st.button('Submit', key='orderSubmit'):
    with st.spinner('Submitting order'):
        
        try:
            st.button('Submit order', key='submitOrder',on_click=submit_order(image_urls,int(selected_shirt[-1])-1,pf_key,customer_details))

            st.write('Check out your creation at https://www.printful.com/uk/dashboard/default/orders')

        except Exception as e:

            st.write(f'Sorry, your submission failed with error {e}. Please start from the beginning.')