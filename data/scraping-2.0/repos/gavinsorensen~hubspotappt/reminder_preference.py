import openai
import streamlit as st
import hubspot

def send_reminder(appointment_id):
    # Retrieve customer information from Hubspot using appointment_id
    customer = hubspot.get_customer(appointment_id)
    
    # Use OpenAI to generate personalized content for reminder message
    openai.api_key = "INSERT YOUR OPENAI API KEY HERE"
    content = openai.generate_reminder_content(customer)
    
    # Use Streamlit to gather customer's preferred method of contact
    contact_method = st.selectbox("How would you like to receive your reminder?", ["Email", "Text"])
    
    # Send reminder using customer's preferred method of contact
    if contact_method == "Email":
        hubspot.send_email(customer.email, content, subject="Appointment Reminder")
    elif contact_method == "Text":
        hubspot.send_text(customer.phone_number, content)
