import streamlit as st
import langchainhelper as lch
st.title("Restaurant Name and Food Recommender System")



#name=st.sidebar.selectbox("Select a country",["Bangladesh","India","Japan","USA","Russia", "Thailand"])
user_input = st.text_input("Enter a country name")



if user_input:
    response= lch.generate_restaurant_name_and_foodItems(user_input)
    st.write("Restaurant Name according to",user_input)
    st.header(response['restaurant_name'].strip())
    st.write("**Food Items**")
    food_items= response['food_item'].strip().split(",")
    for food_item in food_items:
        st.write(food_item)
 
