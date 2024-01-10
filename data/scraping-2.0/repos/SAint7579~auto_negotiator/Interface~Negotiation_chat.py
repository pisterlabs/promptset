import streamlit as st
from openai import OpenAI
import json
import sys
sys.path.append('C:/VS code projects/Road to Hack/auto_negotiator/Utilities/')

st.title("Negotiation Bot")
if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# # Display initial response
# initialsiation = True
# if initialsiation:
#     response = 'Subject: Request for Negotiation: Reduced Cost and Time of Delivery\n\nDear ABC Company,\n\nI hope this email finds you well. My name is Vishwa Mohan Singh, and I am writing to discuss a potential negotiation regarding the cost and time of delivery for our recent orders with ABC Corporation.\n\nFirstly, I would like to express my appreciation for the quality of products and services that ABC Corporation has consistently provided us in the past. Our business relationship has been mutually beneficial, and I am confident that we can continue to strengthen it further.\n\nHowever, I would like to bring to your attention a concern regarding the current cost and time of delivery for our orders. The previous cost per unit was set at 5 Euros, and the delivery time was 20 days. While we understand the value of your products, we believe that there may be room for negotiation in these areas.\n\nIn light of this, I would like to propose a revised offer for our future orders. We kindly request that the cost per unit be reduced to 3 Euros, while also aiming to significantly reduce the time of delivery to 4 days. We believe that these changes will not only benefit our organization by reducing costs and improving efficiency but will also be advantageous for ABC Corporation by increasing the volume of our orders.\n\nIt is important to note that we highly value the quality and reliability of your products, and any cost reduction should not compromise the standards we have come to expect from ABC Corporation. Furthermore, we understand that reducing the time of delivery may require adjustments to your internal processes, and we are open to discussing any feasible solutions that can help achieve this goal.\n\nI would appreciate if you could review our proposal and provide your feedback at the earliest. If you require any additional information or would like to discuss this matter further, please do not hesitate to contact me directly at [Your Contact Details].\n\nThank you for your attention to this matter, and I look forward to your prompt response. I am optimistic that we can reach a mutually beneficial agreement that will further strengthen our business partnership.\n\nBest regards,\n\nVishwa Mohan Singh'
#     st.session_state.initialised = True
#     # Display assistant response
    
#     with st.chat_message("assistant"):
#         # print(response)
#         st.markdown(response)



# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    
    
    string = 'Dear ABC Company, Thank you for getting back to us with your counteroffer. We appreciate your willingness to accommodate the new cost of 3 euros, and we understand your position regarding the delivery timeline. Your offer has been noted, and we will review it internally to determine if the suggested 18-day timeline will be suitable for our needs. Thank you once again for your cooperation and flexibility during this negotiation process. We will be in touch soon with our decision. Best Regards, Vishwa Singh'
    # Display assistant response in chat message container
    jh = {"final_cost":3,"final_days":18}
    with st.chat_message("assistant"):
        st.json(jh)
        
    st.session_state.messages.append({"role": "assistant", "content": prompt})