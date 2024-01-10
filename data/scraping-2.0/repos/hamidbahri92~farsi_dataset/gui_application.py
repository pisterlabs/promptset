import tkinter as tk
from api_interaction import generate_teacher_content, simulate_student_response
from openai import OpenAI
import json
import datetime

# Initialize OpenAI client
client = OpenAI(api_key="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")

def run_conversation():
    teacher_prompt = teacher_prompt_entry.get()
    student_prompt = student_prompt_entry.get()
    conversation_log = []

    for _ in range(int(num_turns_entry.get())):
        teacher_content = generate_teacher_content(teacher_prompt, max_tokens=int(max_tokens_entry.get()), temperature=float(temperature_entry.get()), top_p=float(top_p_entry.get()))
        conversation_log.append({"role": "استاد", "content": teacher_content})

        student_content = simulate_student_response(teacher_content, student_prompt, max_tokens=int(max_tokens_entry.get()), temperature=float(temperature_entry.get()), top_p=float(top_p_entry.get()))
        conversation_log.append({"role": "دانش‌آموز", "content": student_content})

        teacher_prompt = student_content  # Update the teacher prompt for the next iteration

    # Save the conversation to a file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"conversation_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(conversation_log, f, ensure_ascii=False, indent=4)

    # Update the GUI
    conversation_text.delete('1.0', tk.END)
    for entry in conversation_log:
        conversation_text.insert(tk.END, f"{entry['role']}: {entry['content']}\n")
        conversation_text.see(tk.END)

    result_label.config(text=f"مکالمه ذخیره شد: {filename}")

# GUI Setup
window = tk.Tk()
window.title("Persian Language Teaching Dataset Generation")
window.geometry("800x600")

# Text box for conversation
conversation_text = tk.Text(window, width=100, height=20, wrap='word', font=('Helvetica', 12), bd=2)
conversation_text.pack()

# Input fields for prompts and settings
teacher_prompt_entry = tk.Entry(window, width=100)
teacher_prompt_entry.pack()
teacher_prompt_entry.insert(0, "استاد: شروع به صحبت کنید")

student_prompt_entry = tk.Entry(window, width=100)
student_prompt_entry.pack()
student_prompt_entry.insert(0, "دانش‌آموز: شروع به پاسخ دهید")

max_tokens_entry = tk.Entry(window, width=100)
max_tokens_entry.pack()
max_tokens_entry.insert(0, "1000")

temperature_entry = tk.Entry(window, width=100)
temperature_entry.pack()
temperature_entry.insert(0, "0.8")

top_p_entry = tk.Entry(window, width=100)
top_p_entry.pack()
top_p_entry.insert(0, "1")

num_turns_entry = tk.Entry(window, width=100)
num_turns_entry.pack()
num_turns_entry.insert(0, "5")

# Buttons and labels
run_button = tk.Button(window, text="شروع مکالمه", command=run_conversation)
run_button.pack()

result_label = tk.Label(window, text="نتایج مکالمه اینجا نمایش داده می‌شود و در فایل ذخیره می‌شود", justify='right', wraplength=750)
result_label.pack()

window.mainloop()
