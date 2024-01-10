import openai
import json
import os
import dotenv
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import gradio as gr

# Initialize the VADER sentiment intensity analyzer
sid = SentimentIntensityAnalyzer()

# Load environment variables
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Load memory from json file
try:
    with open('memory.json', 'r') as f:
        memory = json.load(f)
except FileNotFoundError:
    memory = {"messages": []}

def save_memory(memory):
    with open('memory.json', 'w') as f:
        json.dump(memory, f)

prompts = {
    'Knowledge': {
        'prompt': 'Q: As the personification of Knowledge, how would you respond to this?\nA:',
        'system_prompt': [
            'As Knowledge, I value creative thinking.',
            'I believe in the power of curiosity.',
            'I strive for open-mindedness.',
            'I have a love of learning.',
            'I believe in gaining perspective.',
            'I exercise sound judgment.'
        ]
    },
    'Courage': {
        'prompt': 'Q: As the personification of Courage, how would you respond to this?\nA:',
        'system_prompt': [
            'As Courage, I value bravery in the face of fear or uncertainty.',
            'I believe in persisting in pursuing goals.',
            'I uphold unwavering integrity.',
            'I approach life with vitality.'
        ]
    },
    'Humanity': {
        'prompt': 'Q: As the personification of Humanity, how would you respond to this?\nA:',
        'system_prompt': [
            'As Humanity, I value love.',
            'I believe in the power of kindness.',
            'I strive for social intelligence.',
            'I believe in caring for others.'
        ]
    },
    'Justice': {
        'prompt': 'Q: As the personification of Justice, how would you respond to this?\nA:',
        'system_prompt': [
            'As Justice, I value citizenship.',
            'I believe in fairness.',
            'I strive for leadership.'
        ]
    },
    'Temperance': {
        'prompt': 'Q: As the personification of Temperance, how would you respond to this?\nA:',
        'system_prompt': [
            'As Temperance, I value forgiveness.',
            'I believe in humility.',
            'I strive for prudence.',
            'I believe in self-regulation.'
        ]
    },
    'Transcendence': {
        'prompt': 'Q: As the personification of Transcendence, how would you respond to this?\nA:',
        'system_prompt': [
            'As Transcendence, I value the appreciation of beauty and excellence.',
            'I believe in expressing gratitude.',
            'I maintain hope.',
            'I find humor in situations.',
            'I seek meaning, purpose, and connection to something greater than oneself.'
        ]
    },
}


def chatbot(messages, memory):
    # Use all previous messages for context
    messages.extend(memory.get('messages', []))

    # Convert list of system prompts to a single string
    sys_prompt_f = ""
    for message in messages:
        if message['role'] == 'system' and isinstance(message['content'], list):
            sys_prompt_f += '\n'.join(message['content']) + '\n'
        elif message['role'] == 'system' and isinstance(message['content'], str):
            sys_prompt_f += message['content'] + '\n'

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=300,
        temperature=0.8
    )

    # Extract the output content from the response
    output_content = response.choices[0].message['content']

    # Analyze sentiment of the response
    sentiment = analyze_sentiment(output_content)

    # Formulate the response with the user input, virtue, and sentiment
    response_text = f"\"{output_content}\""
    return response_text

# Define the sentiment analysis function
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def extract_emotion_keywords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w in stop_words]

    emotion_words = []
    for word in filtered_text:
        if sid.polarity_scores(word)['compound'] != 0:
            emotion_words.append(word)

    return emotion_words

def select_best_virtue(input_text, virtues, memory):
    virtue_scores = {}
    for virtue in virtues:
        instruction = f"On a scale of 1 to 10, rank the relevance of the virtue '{virtue}' to the statement: '{input_text}'. Please respond with a number only."
        messages = [{"role": "system", "content": instruction}]
        
        # Check if user input matches an entry in memory
        for message in memory.get('messages', []):
            if isinstance(message, dict):
                if message.get('role') == 'user' and message.get('content') == input_text:
                    messages.append(message)
                    break
            else:
                if message == input_text:
                    messages.append({"role": "user", "content": message})
                    break
        
        # Join messages into single string prompt
        prompt = "\n".join([m["content"] for m in messages]) 

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.2,
            max_tokens=5
        )
        try:
            score = int(response['choices'][0]['text'].strip())
            virtue_scores[virtue] = score
        except ValueError:
            print(f"Received unexpected response: {response['choices'][0]['text']}")

    return max(virtue_scores, key=virtue_scores.get)


def chat_interface(input_text):
    # Analyze sentiment and extract keywords from user input
    sentiment = analyze_sentiment(input_text)
    keyword = extract_emotion_keywords(input_text)

    # Generate a response using the chatbot function
    best_virtue = select_best_virtue(input_text, prompts.keys(), memory)
    prompt = prompts.get(best_virtue)
    if prompt:
        full_prompt = prompt['prompt'] + ' ' + input_text + '\nA:'
        system_prompt = '\n'.join(prompt['system_prompt'])
        messages = [{"role": "system", "content": full_prompt}, {"role": "system", "content": system_prompt}]

        response = chatbot(messages, memory)

        # Analyze sentiment of the response
        response_sentiment = analyze_sentiment(response)

        # Update memory with the new conversation
        memory['messages'].append({
            'role': 'user',
            'content': input_text
        })
        memory['messages'].append({
            'role': 'assistant',
            'content': response
        })

        # Save updated memory
        save_memory(memory)

        # Load existing data from the file
        try:
            with open('data.json', 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = {}

        # Save user input, sentiment, keyword, and final virtue response to json file
        data = {
            'user_input': input_text,
            'input_sentiment': sentiment,
            'keyword': keyword,
            'virtue_response': response,
            'response_sentiment': response_sentiment,
            'responding_virtue': best_virtue
        }

        # Append the new data to the existing data
        if isinstance(existing_data, list):
            existing_data.append(data)
        elif isinstance(existing_data, dict):
            existing_data = [existing_data, data]

        # Write updated data back to JSON file
        with open('data.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

        return f"Responding Virtue: {data['responding_virtue']}\nVirtue Response: {data['virtue_response']}"

    else:
        return "Invalid core virtue."


iface = gr.Interface(fn=chat_interface, inputs="text", outputs="text")
iface.launch(share=False)


