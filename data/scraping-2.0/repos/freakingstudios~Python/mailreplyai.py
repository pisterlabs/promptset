import openpyxl
import openai
import imaplib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def generate_email_body(student_name, marks):
    # Provide your OpenAI API key
    openai.api_key = "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"


    # Generate the email body using OpenAI
    prompt = f"Dear {student_name},\n\nCongratulations on scoring {marks} marks in the exam! You did an excellent job.\n\nBest regards,\nYour School"
    completion = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=100,
        temperature=0.6,
        top_p=1.0,
        n=1,
        stop=None,
        timeout=60,
    )
    email_body = completion.choices[0].text.strip()

    return email_body

def send_email(sender_email, sender_password, receiver_email, subject, body, in_reply_to=None):
    # Set up the email details
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to

    # Attach the body of the email
    msg.attach(MIMEText(body, "plain"))

    # Connect to the SMTP server and send the email
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()

def check_email_replies(sender_email, sender_password):
    # Connect to the IMAP server
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(sender_email, sender_password)
    mail.select("inbox")

    # Search for recent emails
    _, data = mail.search(None, "UNSEEN")

    # Get the list of email IDs
    email_ids = data[0].split()

    # Iterate through the email IDs and send a reply
    for email_id in email_ids:
        _, response = mail.fetch(email_id, "(BODY.PEEK[HEADER.FIELDS (IN-REPLY-TO)])")
        in_reply_to_header = response[0][1].decode("utf-8").strip()
        _, message_data = mail.fetch(email_id, "(RFC822)")

        if in_reply_to_header == original_message_id:
            # Send a reply with thanks
            send_email(sender_email, sender_password, sender_email, "Re: Your Exam Results", "Thank you for your response!", in_reply_to=in_reply_to_header)

    # Close the connection
    mail.close()
    mail.logout()

# Provide the details for the email

sender_email = "freakingstudios496@gmail.com"
sender_password = "xamzxrwtiayqsqar"
subject = "AI Generated Marks - By Freaking Studios"

# Provide the path to the Excel file
excel_file_path = "stdata.xlsx"

# Read the student details from the Excel file
wb = openpyxl.load_workbook(excel_file_path)
sheet = wb.active

# Iterate through the rows in the Excel sheet
for row in sheet.iter_rows(min_row=2, values_only=True):
    student_name = row[0]
    student_email = row[1]
    marks = row[2]

    # Generate the email body using OpenAI
    email_body = generate_email_body(student_name, marks)

    # Send the email to the student
    send_email(sender_email, sender_password, student_email, subject, email_body)

    # Print a confirmation message
    print(f"Email sent to {student_name} ({student_email})")

    # Store the original message ID for comparison
    original_message_id = msg["Message-ID"]

# Check for email replies and send a reply with thanks
check_email_replies(sender_email, sender_password)











