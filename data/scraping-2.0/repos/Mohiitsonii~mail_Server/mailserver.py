import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import mailbody
from mailbody import set_attachment
import openai

fromaddr = "mohitsoni004488@gmail.com"
toaddr = "Mohitsonims04@gmail.com"

openai_enabled = True

if openai_enabled:
    import openai
    openai.api_key = ""
    from openai import generate_email_body
else:
    from mailbody import body as email_body

msg = MIMEMultipart()
msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "all okk"

if openai_enabled:
    email_body = generate_email_body()

msg.attach(MIMEText(email_body, 'plain'))

if set_attachment:
    # Attach the file
    filename = "Mohit_soni_resume.pdf"
    attachment = open("./mohit_soni_resume.pdf", "rb")

    p = MIMEBase('application', 'octet-stream')
    p.set_payload((attachment).read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    msg.attach(p)

s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
s.login(fromaddr, "")

text = msg.as_string()

s.sendmail(fromaddr, toaddr, text)
print("sent")

s.quit()
