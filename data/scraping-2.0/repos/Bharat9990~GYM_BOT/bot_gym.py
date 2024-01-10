import threading
import ssl
import smtplib
import gradio as gr
import openai
from email.message import EmailMessage
from transformers import pipeline
import os
import creds
# Replace 'YOUR_OPENAI_API_KEY' with your actual API key

openai.api_key = os.environ.get("api_key")
# Send E-mail Reminder
def send_email(email_address, subject, content):
    email_sender = os.getenv("email")
    email_password = os.getenv("password")
    email_receiver = email_address

    subject = subject
    body = content
    try:
        em = EmailMessage()
        em['From'] = email_sender
        em['To'] = email_receiver
        em['Subject'] = subject
        em.set_content(body)

        context = ssl.create_default_context()

        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(email_sender, email_password)
            smtp.sendmail(email_sender, email_receiver, em.as_string())

    except Exception as e:
        print(f"Error sending email reminder to {email_address}: {e}")

# Scheduling Reminder through SMS and Email
def schedule_reminder(injury_type, reminder_type, patient_email, interval):
    while True:
        if patient_email:
            send_email(patient_email, f"Reminder: {reminder_type} for {injury_type} injury", f"Don't forget your {reminder_type} for {injury_type} injury.")
        # Wait for the specified interval before scheduling the next reminder
        threading.Event().wait(interval)

def start_reminder_scheduler(injury_type, patient_email):
    # Schedule medication reminders every 2 days
    medication_thread = threading.Thread(target=schedule_reminder, args=(injury_type, "Medication Reminder", patient_email, 2 * 86400))
    medication_thread.start()

    # Schedule follow-up reminders every 10 days
    follow_up_thread = threading.Thread(target=schedule_reminder, args=(injury_type, "Follow-up Appointment Reminder", patient_email, 10 * 86400))
    follow_up_thread.start()

    # Schedule important recovery milestone reminders every 15 days
    milestone_thread = threading.Thread(target=schedule_reminder, args=(injury_type, "Important Recovery Milestone Reminder", patient_email, 15 * 86400))
    milestone_thread.start()

def post_gym_injury_care_guide(type_of_injury, severity, symptoms, timing=None, activities_leading_to_injury=None, desired_outcome=None, patient_email=None):
    user_input = f"Type of Injury: {type_of_injury}\nSeverity of Injury: {severity}\nSymptoms: {symptoms}"
    if timing:
        user_input += f"\nTiming: {timing}"
    if activities_leading_to_injury:
        user_input += f"\nActivities Leading to Injury: {activities_leading_to_injury}"
    if desired_outcome:
        user_input += f"\nDesired Outcome: {desired_outcome}"
    if patient_email:
        start_reminder_scheduler(type_of_injury, patient_email)
    messages = [
        {"role": "system", "content": "You are a Gym Injury Care Guide ChatBot that provides comprehensive instructions \
        for post-gym injury care. You will offer advice on the following topics in order: \
        First Aid Advice, Pain Management, Rehabilitation Exercises (including rep and set guidelines), \
        Nutrition and Hydration, Preventing Future Injuries, and Alternative Workouts during recovery."}
    ]

    messages.append({"role": "user", "content": user_input})
    
    summary = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.7)
    ChatGPT_reply = summary["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

# Define the Gradio interface
input_components = [
    gr.components.Textbox(label="Type of Injury"),
    gr.components.Dropdown(["mild", "moderate", "severe"], label="Severity of Injury"),
    gr.components.Textbox(label="Symptoms"),
    gr.components.Textbox(label="Timing (optional)"),
    gr.components.Textbox(label="Activities Leading to Injury (optional)"),
    gr.components.Dropdown(["alleviate pain", "regain full mobility", "return to your regular gym routine"], label="Desired Outcome (optional)"),
    gr.components.Textbox(label="Email (optional)"),
]

output_component = gr.components.Textbox()

iface = gr.Interface(
    fn=post_gym_injury_care_guide,
    inputs=input_components,
    outputs=output_component,
    title="Post-Gym Injury Care Guide",
    description="Provide information about your injury and symptoms to receive personalized care instructions.",
)

if __name__ == "__main__":
    iface.launch()
