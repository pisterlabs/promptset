import smtplib
import imaplib
import re
import openai

def add_prompt_output(user_input):

    prompt_text = """
    The following is a user question. Answer it concisely.
    
    {}
    """
    
    messages = [{"role": "system", "content": prompt_text.format(user_input)}]
    
    response = openai.chat.completions.create(
        # "gpt-4-1106-preview"
        model = "gpt-3.5-turbo", messages = messages, max_tokens = 1000, temperature = 0
    )  
    
    prompt_output = response.choices[0].message.content
    
    return(prompt_output)

def send_email(body, to_email):

    print(to_email, body)

    smtpObj = smtplib.SMTP('smtp.gmail.com', 587)
    
    smtpObj.ehlo()
    smtpObj.starttls()
    
    smtpObj.login('FROM', 'PW')
    
    smtpObj.sendmail('FROM', to_email, body)
    
    smtpObj.quit()

def read_email():
    
    imap_host = 'imap.gmail.com'
    imap_user = ''
    imap_pass = ''
    
    # connect to host using SSL
    imap = imaplib.IMAP4_SSL(imap_host)
    ## login to server
    imap.login(imap_user, imap_pass)
    imap.select('Inbox')
    tmp, data = imap.search(None, 'ALL')

    full_data = [imap.fetch(num, '(RFC822)')[1][0][1] for num in data[0].split()]    
    imap.close()
    
    payload = [process_each_email(data) for data in full_data]
    
    return(payload)
    
def process_each_email(email):
    
    output_str = email.decode()
    lines = output_str.splitlines()
    
    payload = {}
    
    for line in lines: 
      if "From:" in line:
        payload["from"] = re.findall('\S+@\S+', line)[0]
      if "To:" in line:
        payload["to"] = re.findall('\S+@\S+', line)[0]
      if "Date:" in line:
        payload["date"] = line
      if "Subject:" in line:
        payload["subject"] = line
        
    payload['body'] = output_str.split("Content-Type:")[1].split("wrote:")[0]

    if "!oai" in payload['body']: 
        answer = add_prompt_output(payload['body'])
        send_email(answer, payload['from'])
    
    return(payload)
    
read_email()
