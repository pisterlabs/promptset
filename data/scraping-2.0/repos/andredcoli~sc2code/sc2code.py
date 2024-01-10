import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from openai import OpenAI
import openai
import base64
import json
import requests
prompt_text = """
    You are an expert in HTML, CSS, and JavaScript. Based on the image provided, 
    please generate and output the HTML, CSS, and JavaScript code in distinct sections.
    
    First, provide the HTML code in a section labeled 'HTML Code:'. 
    Then, provide the CSS code in a section labeled 'CSS Code:'. 
    Finally, provide the JavaScript code in a section labeled 'JavaScript Code:'.
    
    Remember, your expertise is limited to HTML, CSS, and JavaScript.
"""

#b64 encoder
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

#main api logic
def send_to_gpt4_vision_api(encoded_image, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

#prompt extraction logic - to properly display the given code correctly within their respective tabs
def extract_code(content, marker_start, marker_end=None):
    try:
        start = content.index(marker_start) + len(marker_start)
        end = content.index(marker_end, start) if marker_end else len(content)
        return content[start:end].strip()
    except ValueError:
        return ""

#Image upload logic - will reset at every new uploaded image.
def upload_image():
    global uploaded_image_path, status_label, convert_btn
    filetypes = [("Image files", "*.jpg *.jpeg *.png")]
    file_path = filedialog.askopenfilename(filetypes=filetypes)
    if file_path:
        uploaded_image_path = file_path
        print(f"New image path: {uploaded_image_path}") 

        status_label.config(text="New image uploaded. Ready to convert.")
        convert_btn.config(state=tk.NORMAL)  
        # Clear the previous conversion results from the code tabs
        html_text.delete(1.0, tk.END)
        css_text.delete(1.0, tk.END)
        js_text.delete(1.0, tk.END)
        progress_bar.stop()
        progress_bar.pack_forget()

        print("Program state reset for new image conversion.")  # Debugging statement
    else:
        status_label.config(text="Image upload canceled.")

# API Key verification logic
def apply_api_key():
    global upload_btn, status_label
    api_key = api_key_entry.get()
    if not api_key.strip():
        messagebox.showerror("Error", "API key is missing.")
        status_label.config(text="Error: API key is missing.")
        return
    client = OpenAI(api_key=api_key)
    try:
        response = client.models.list()
        if response:
            upload_btn.config(state=tk.NORMAL)
            status_label.config(text="API key is valid. You can now upload an image.")
            messagebox.showinfo("Success", "API key is valid. You can now upload an image.")
        else:
            raise ValueError("The API key is not valid.")
    except openai.OpenAIError as e:
        messagebox.showerror("Error", f"Invalid API Key: {e}")
        status_label.config(text="Invalid API Key.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to verify API Key: {e}")
        status_label.config(text="Failed to verify API Key.")

#Used to update the output from the API in its perspective tabs
def update_code_tabs(html_code, css_code, js_code):
    html_text.delete(1.0, tk.END)
    css_text.delete(1.0, tk.END)
    js_text.delete(1.0, tk.END)

    html_text.insert(tk.END, html_code)
    css_text.insert(tk.END, css_code)
    js_text.insert(tk.END, js_code)

    if not code_tabs.winfo_ismapped():
        code_tabs.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


def convert_image_to_code():
    global uploaded_image_path, api_key_entry, progress_bar, code_tabs
    if not uploaded_image_path:
        messagebox.showerror("Error", "Please upload an image first.")
        return
    api_key = api_key_entry.get()
    if not api_key:
        messagebox.showerror("Error", "Please apply your API key first.")
        return
    
    encoded_image = encode_image(uploaded_image_path)
    progress_bar.pack(side=tk.TOP, fill=tk.X)
    progress_bar.start(10)
    
    response = send_to_gpt4_vision_api(encoded_image, api_key)

    try:
        if response and 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            
            # Extract HTML, CSS, and JavaScript codes
            html_code = extract_code(content, "HTML Code:", "CSS Code:")
            css_code = extract_code(content, "CSS Code:", "JavaScript Code:")
            js_code = extract_code(content, "JavaScript Code:", "")  # No ending marker

            # Print the extracted code in the terminal (used for debugging)
            print("HTML Code:\n", html_code)
            print("\nCSS Code:\n", css_code)
            print("\nJavaScript Code:\n", js_code)

            # Update the tabs with the extracted code
            update_code_tabs(html_code, css_code, js_code)
        else:
            messagebox.showerror("Error", "The API call did not return any data.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
    finally:
        progress_bar.stop()
        progress_bar.pack_forget()

#tkinter styles.
def main():
    global api_key_entry, status_label, upload_btn, convert_btn, progress_bar, code_tabs, html_text, css_text, js_text
    app = tk.Tk()
    app.title("Screenshot to Code Converter")
    app.geometry("600x600")

    style = ttk.Style()
    style.theme_use('clam')

    upload_frame = ttk.Frame(app)
    upload_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    upload_btn = ttk.Button(upload_frame, text="Upload Screenshot", command=upload_image, state=tk.DISABLED)
    upload_btn.pack(pady=10)

    api_frame = ttk.Frame(app)
    api_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

    api_key_label = ttk.Label(api_frame, text="Enter GPT-4 Vision API Key:")
    api_key_label.pack(side=tk.LEFT)

    api_key_entry = ttk.Entry(api_frame, width=30)
    api_key_entry.pack(side=tk.LEFT, padx=10)

    apply_button = ttk.Button(api_frame, text="Apply", command=apply_api_key)
    apply_button.pack(side=tk.LEFT)

    status_label = ttk.Label(app, text="Please enter your API key and click Apply.")
    status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=20)

    convert_btn = ttk.Button(app, text="Convert", command=convert_image_to_code)

    progress_bar = ttk.Progressbar(app, mode='indeterminate')
    code_tabs = ttk.Notebook(app)

    convert_btn = ttk.Button(app, text="Convert", command=convert_image_to_code)
    convert_btn.pack(side=tk.TOP, pady=10)

    html_tab = ttk.Frame(code_tabs)
    css_tab = ttk.Frame(code_tabs)
    js_tab = ttk.Frame(code_tabs)

    code_tabs.add(html_tab, text='HTML')
    code_tabs.add(css_tab, text='CSS')
    code_tabs.add(js_tab, text='JavaScript')

    html_text = tk.Text(html_tab)
    css_text = tk.Text(css_tab)
    js_text = tk.Text(js_tab)

    html_text.pack(expand=True, fill='both')
    css_text.pack(expand=True, fill='both')
    js_text.pack(expand=True, fill='both')


    app.mainloop()

if __name__ == "__main__":
    main()
