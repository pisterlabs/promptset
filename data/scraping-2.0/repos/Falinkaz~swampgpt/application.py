import csv, json, os
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI


application = Flask(__name__)
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

def get_cards_from_csv(csv_filename):
    cards = []
    with open(csv_filename, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            cards.append(row['Card Name'])
    return cards

@application.route('/')
def index():
    cards = []
    return render_template('index.html', cards=cards)

@application.route('/autocomplete')
def autocomplete():
    query = request.args.get('q', '').lower()
    csv_filename = 'allcardsv3.csv'
    cards = get_cards_from_csv(csv_filename)
    suggestions = [card for card in cards if query in card.lower()]
    return jsonify(suggestions)

@application.route('/log_selected_options', methods=['POST'])
def log_selected_options():
    data = request.get_json()
    selectedOptions = data.get('selectedOptions')
    print("selectedOptions:", selectedOptions)
    return '', 204

@application.route('/store_format', methods=['POST'])
def store_format():
    data = request.get_json()
    session['format'] = data.get('format')
    return '', 204  # No content response

@application.route('/send_gpt', methods=['POST'])
def send_gpt():
    data = request.get_json()
    selectedOptions = data.get('selectedOptions', [])
    selectedFormat = session.get('format', 'Default')  # Retrieve the selected format from the session
    client = OpenAI(api_key=API_KEY)
    thread = client.beta.threads.create()

    messages = [{"role": "user", "content": "Provide recommendations for" + str(selectedFormat) + " format" + "The user has in their deck (may or may not be a complete deck):"}]
    for card in selectedOptions:
        messages[0]["content"] += f' {card},'  # Appending each card to the content

    run = client.beta.threads.create_and_run(
    assistant_id="asst_L87PSj1oIRz5XIzBYkXA0fQJ",
    thread={"messages": messages}
    )


    run_status = client.beta.threads.runs.retrieve(
    thread_id=(run.thread_id),
    run_id=(run.id)
    )


    response_json = None  # Initialize response variable outside the loop
    while run_status.status != 'completed': # Poll for completion
        time.sleep(5)  # Poll every 5 seconds
        run_status = client.beta.threads.runs.retrieve(thread_id=run.thread_id, run_id=run.id)
    try:
        thread_messages = client.beta.threads.messages.list(run.thread_id)
        response = (thread_messages.data[0].content[0].text.value)
    except Exception as e:
        return("error: Invalid response")  # Return a generic error response

    if thread_messages:
        print(str(response))
        print(type((str(response))))
        return(str(response))  # Return the cards outside the try-except block
        

if __name__ == '__main__':
    application.run(debug=True, port=8004)



