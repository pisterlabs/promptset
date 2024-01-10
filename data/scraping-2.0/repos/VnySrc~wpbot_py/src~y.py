y = "y"

import pywhatkit
import openai

openai.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

pywhatkit.

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

# Definir as informações da conta do WhatsApp
whatsapp_phone_number = '5581994035159'
message = 'Olá, sou um bot de respostas automáticas. O que você precisa?'
driver = webdriver.Chrome()

driver.get('https://web.whatsapp.com/send?phone=' + whatsapp_phone_number)
time.sleep(15) # Espera 15 segundos para carregar a página

# Enviar a mensagem
input_box = driver.find_element_by_xpath('//*[@id="main"]/footer/div[1]/div[2]/div/div[2]')
input_box.send_keys(message + Keys.ENTER)

y = "y"