from flask import Flask, render_template, request
import openai
import csv
import speech_recognition as sr

openai.api_key = "api_key"

app = Flask(__name__)
app.static_folder = 'static'

# Define the path to your CSV file
csv_file_path = r'path'

def search_csv_data(keyword):
    # Open the CSV file
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)

        # Search for rows containing the keyword
        matched_rows = []
        for row in reader:
            for cell in row:
                if keyword.lower() in cell.lower():
                    matched_rows.append(row)
                    break

    return matched_rows

@app.route("/")
def home():
    return render_template("combo.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')

    if userText.startswith('search csv:'):
        # Extract the keyword from user input
        keyword = userText.split(':')[1].strip()

        # Search the CSV data
        matched_data = search_csv_data(keyword)

        if matched_data:
            response = '\n'.join(', '.join(row) for row in matched_data)
        else:
            response = 'No matching data found.'
    else:
        # Generate response using OpenAI
        output = openai.Completion.create(
            engine="text-davinci-003",
            prompt=userText,
            temperature=0.7,
            max_tokens=2000,
            n=1,
            stop=None,
        )
        response = output.choices[0].text

    return response

if __name__ == "__main__":
    app.run()
