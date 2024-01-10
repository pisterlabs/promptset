import win32com.client
import bs4
import openai
import time

openai.api_key = "" # obtained via https://platform.openai.com/account/api-keys

outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
inbox = outlook.GetDefaultFolder(6)
messages = inbox.Items
signature = "" # your signature, use \n for newline

while True:
    messages = inbox.Items
    found_unread = False
    for message in messages:
        if message.UnRead:
            soup = bs4.BeautifulSoup(message.Body, "html.parser")
            content = soup.get_text().split("From:")[0]
            
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt="Respond to this email with a positive attitude, absolutely NEVER use any names. End with kind regards without a name: " + content,
                max_tokens=2000,
                n=1,
                stop=None,
                temperature=0.55,
            ).get("choices")[0].get("text")

            reply = message.Reply()
            reply.Body = response + "\n\n\n" + signature
            reply.Subject = "Sv: " + message.Subject
            reply.Send()
            message.UnRead = False
            found_unread = True

    if not found_unread:
        print("Waiting for incoming emails..")

    time.sleep(1200)
