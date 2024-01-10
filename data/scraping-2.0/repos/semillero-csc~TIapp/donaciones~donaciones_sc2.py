# importamos las librerias
import streamlit as st
import datetime
import openai 
import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
import datetime
from decouple import config
from dotenv import load_dotenv
from email.message import EmailMessage
import ssl
import smtplib



def donaciones():
    st.markdown("### Donaciones:")

    col1, col2, col3 = st.columns(3)

    with col1:
        #st.subheader(':green[Invitame a un café]')
        boton_bmc = st.button('Link - buy me a coffe', use_container_width = True)
        if boton_bmc:
            st.markdown('https://www.buymeacoffee.com/cristianmoz') 
        bmec = 'complementos/bmc2.png'
        st.image(bmec)

    with col2:
        #st.subheader(':green[Paypall]')
        boton_pp = st.button('link - Paypal', use_container_width = True)
        if boton_pp:
            st.markdown('https://paypal.me/CristianMontoya158?country.x=CO&locale.x=es_XC') 
        paypal = 'complementos/paypal1.png'
        st.image(paypal)

    with col3:
        boto_bcol = st.button('QR - Bancolombia', use_container_width = True)
        bancolomia = 'complementos/bancolombia.png'
        st.image(bancolomia)