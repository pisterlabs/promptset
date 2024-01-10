#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://ausopenai.azure-api.net"
openai.api_version = "2023-05-15"
openai.api_key = "12b555ada1374acb866c89bb2cacf93c"
#print(os.getenv("AZURE_OPENAI_KEY"))
#print(os.getenv("AZURE_OPENAI_ENDPOINT") )


email_thread_summary = """
Please summarize the following email thread based on the following requirements:
Analyze the sentiment of each email or message within the thread based on the text content.
Identify positive, negative, or neutral sentiments based on the words used.
Action Items:

Extract action items or to-do tasks mentioned within the text content of the emails.
Note deadlines or due dates associated with action items mentioned in the text.
Completed Actions:

Identify and mark actions that have been completed or resolved based on text content.
Note when and by whom they were completed as mentioned in the text.
Pending Actions:

Identify actions that are still pending or require further attention based on text content. Note deadlines or expected completion dates for pending actions mentioned in the text. Determine who is responsible for each pending action based on text content.
Identify the owner of the next steps mentioned in the text.

Key Decisions:

Identify any significant decisions or conclusions reached in the email thread based on the text content.
Note when and how these decisions were made as described in the text.

Follow-up Required:

Determine if there are any requests for follow-up or additional information based on the text content.
Note any commitments made for future actions or discussions as mentioned in the text.
Thread Length:

Thread Summary:

Generate a concise summary of the entire email thread based on the text content, highlighting the most critical information, decisions, actions, and pending items.
Key Quotes:

Extract and highlight key quotes or snippets from the email thread's text content that encapsulate important information or opinions.
Please provide a comprehensive summary that covers all of these aspects. Following is the email text: 
"""

# You can use the `email_thread_summary` variable in your function or code as needed.

import email
from email.header import decode_header
import tkinter as tk
from tkinter import filedialog
import os

def read_email_file(email_file_path):
    try:
        with open(email_file_path, 'r', encoding='utf-8') as file:
            # Parse the email using the email module
            msg = email.message_from_file(file)
            
            # Initialize a list to store email thread content
            email_thread = []
            
            # Extract information from the email
            subject = decode_header(msg['Subject'])[0][0]
            sender = decode_header(msg['From'])[0][0]
            date_sent = msg['Date']
            
            # Append email information to the email thread list
            email_thread.append(f"Subject: {subject}")
            email_thread.append(f"From: {sender}")
            email_thread.append(f"Date: {date_sent}")
            email_thread.append("\nMessage:")
            
            # Extract the email content
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode('utf-8')
                        email_thread.append(body)
            else:
                body = msg.get_payload(decode=True).decode('utf-8')
                email_thread.append(body)
            
            # Return the email thread as a string
            return "\n".join(email_thread)
    except Exception as e:
        return str(e)

def get_email_file_from_gui():
    root = tk.Tk()
    root.withdraw()  # Hide the main GUI window

    file_path = filedialog.askopenfilename(title="Select an email file",
                                           filetypes=[("Email files", "*.eml *.msg"), ("All files", "*.*")])

    if not file_path:
        print("No file selected. Exiting.")
        return None

    email_thread = read_email_file(file_path)
    if email_thread:
        return email_thread
    else:
        print("The email file could not be processed. Please ensure it is in a readable format (e.g., .eml).")
        return None

if __name__ == "__main__":
    email_thread = get_email_file_from_gui()
    if email_thread:
        #print("\n Following is a summary of the email:")
        concatenated_text = email_thread_summary + email_thread
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo-16k", # engine = "deployment_name".
            messages=[
                {"role": "system", "content": concatenated_text },

            ]
        )

        # print(response)
        # print(response['choices'][0]['message']['content'])      
        # Your long string output
        # Your multiline string with various sections
        multiline_string = response['choices'][0]['message']['content']
       # Ask the user to choose a save location
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        # Create a GUI window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        if file_path:
            # Save the multiline string to the chosen file path
            with open(file_path, "w") as txt_file:
                txt_file.write(multiline_string)

            # Display a success message
            tk.messagebox.showinfo("Success", f"Text file saved to:\n{file_path}")

        # Close the GUI window
        root.destroy()





