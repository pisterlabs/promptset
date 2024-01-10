import os
import openai
import datetime
import re

openai.api_key = os.environ['OPENAI_API_KEY']

def create_task():
    task_description = input("Task Description: ")

    generated_breakdown = generate_subtasks(task_description)
    print("Breakdown of Smaller Subtasks:")
    for i, subtask in enumerate(generated_breakdown, start=1):
        print(f"{i}. {subtask}")

    edit_task = input("Edit Task Description? (yes/no): ")
    if edit_task.lower() == "yes":
        task_description = input("Task Description: ")

    due_date = input("Due Date (yyyy-mm-dd): ")
    while not validate_date(due_date):
        print("Invalid due date format. Please enter a valid date (yyyy-mm-dd).")
        due_date = input("Due Date (yyyy-mm-dd): ")

    accountability_buddy = input("Accountability Buddy Name: ")

    commitment_amount = input("Commitment Amount: ")
    while not validate_money(commitment_amount):
        print("Invalid commitment amount format. Please include a money sign ($) and enter a valid amount.")
        commitment_amount = input("Commitment Amount: ")

    paypal_account = input("PayPal Account for Consequence: ")
    while not validate_email(paypal_account):
        print("Invalid PayPal account format. Please enter a valid email address.")
        paypal_account = input("PayPal Account for Consequence: ")

    proof_of_completion = input("Proof of Completion: ")

    print("\nHey", accountability_buddy + ",")
    print("I hope you're doing well! I have an important task ahead of me, and I'm looking for an accountability buddy to support me.")
    print("Task:", task_description)
    print("Due Date:", due_date + ". It's essential that I complete this task on time.")
    print("Committed Amount: I've committed", commitment_amount, "to this task. If I don't finish it by the deadline, the committed amount will be sent to your PayPal account as a measure of accountability.")
    print("Proof of Completion: I'll provide concrete evidence of my task completion, such as", proof_of_completion + ".")
    print("Your support as my accountability buddy would be invaluable. Your role would involve checking in on my progress, offering encouragement, and reminding me of the financial stakes involved.")
    print("Thank you for being my accountability buddy!\n")

    confirm = input("Please send this message to your accountability buddy and then press Enter to submit the task.")

    submit_task(task_description, generated_breakdown, due_date, accountability_buddy, commitment_amount, paypal_account, proof_of_completion)


def generate_subtasks(task_description):
    prompt = f"Break down the task '{task_description}' into smaller achievable subtasks:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )
    subtasks = []
    for choice in response.choices:
        subtask = choice.text.strip()
        if subtask:
            subtask = subtask.split(".", 1)[1].strip()  # Remove the numbering
            subtasks.append(subtask)
    return subtasks[:4]  # Limit to 4 subtasks


def validate_date(date_string):
    try:
        datetime.datetime.strptime(date_string, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def validate_money(money_string):
    return money_string.startswith("$")


def validate_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)


def submit_task(task_description, subtasks, due_date, accountability_buddy, commitment_amount, paypal_account, proof_of_completion):
    filename = task_description.replace(" ", "_") + ".txt"

    with open(filename, "w") as file:
        file.write("Task Description: " + task_description + "\n\n")
        file.write("Breakdown of Smaller Subtasks:\n")
        for i, subtask in enumerate(subtasks, start=1):
            file.write(f"{i}. {subtask}\n")
        file.write("\nDue Date: " + due_date + "\n")
        file.write("Accountability Buddy: " + accountability_buddy + "\n")
        file.write("Commitment Amount: " + commitment_amount + "\n")
        file.write("PayPal Account for Consequence: " + paypal_account + "\n")
        file.write("Proof of Completion: " + proof_of_completion + "\n")

    print("Task submitted. Details saved to", filename)

