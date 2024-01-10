import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import openai



def form_data():
#    website_url = #Figure out how to input current browser url / html
#    url = 'https://momentive.wd1.myworkdayjobs.com/en-US/MC/job/US-NY-Niskayuna/Finance---Product-Management-Summer-Intern---Basics-Segment_R6552/apply/applyManually?source=LinkedIn_corporate_page'
#    response = requests.get(url)
    url = input("Please enter the URL: ")
    
    browser = webdriver.Chrome()
    browser.get(url)
    html_content = browser.page_source
    browser.quit
    
    with open('C:/Users/Sam/Desktop/test.html', 'r', encoding='utf-8') as file:
        raw_html = file.read()

    soup = BeautifulSoup(raw_html, 'html.parser')
    text_boxes = soup.find_all('input')
    text_areas = soup.find_all('textarea')
    soup_text = soup.prettify()

    labels = []

    for label in soup.find_all('label'):
        for_tag = label.get('for')
        if for_tag and soup.find(id=for_tag):
            labels.append(label.get_text())
    print(labels)
    
    with open ('C:/Users/Sam/Desktop/Generative AI User Interface/my_info.txt', 'r') as file:
        personal_info = file.readlines()
        
    
    labels_string = ', '.join(labels)

    print(labels_string)
    print(f"Here are the labels: {labels_string} you need output the values associated with each label from this text separated by commas: {personal_info} \n\n, if information is unknown, enter NA and continue on")
    
    response = openai.ChatCompletion.create(                                                
        model="gpt-3.5-turbo",                
        messages=[
            {"role": "system", "content": "You need to match a list of labels with their associated values "},
            {"role": "user", "content": "Here are the labels separated by commas: f{labels_string} you need output the values associated with each label from this text separated by commas: f{personal_info} /n/n if information is unknown, enter NA and continue on"},
            ],                                                         
            max_tokens=500,  # Adjust the max tokens as needed                              
            temperature=0.1,  # Adjust the temperature as needed                                  
        )
    
    user_input = response['choices'][0]['message']['content'].strip()
    print(user_input)
