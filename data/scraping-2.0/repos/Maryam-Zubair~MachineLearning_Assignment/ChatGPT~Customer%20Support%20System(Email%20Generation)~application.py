import openai
import os
from flask import Flask, render_template, request

# Import products_description from products_data.py
from products_data_english import products_data as products_data_english
from products_data_chinese import products_data_chinese
from products_data_spanish import products_data_spanish

products_data = {**products_data_english, **products_data_chinese, **products_data_spanish}
from products_data_english import products_data

# reading environment variable
with open(".env") as env:
    for line in env:
        key, value = line.strip().split("=")
        os.environ[key] = value
openai.api_key = os.environ.get("API_KEY")
openai.organization = os.environ.get("ORG_ID")

app = Flask(__name__, static_url_path='/static')
# Function to get response from ChatGPT
def get_completion(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

# Step 1: Generation of Customer Response
def generate_customer_comment(products_data):
    # Join product data into a single string
    products_content = " ".join(products_data)

    # Construct a prompt for GPT-3.5 Turbo
    prompt = f"""
    Assume that you are a customer to an electronic product company.
    Write a 100-word only comment about the products delimited by tripe backticks in its own language. 
    Products: ```{products_content}```
    """
    # Generate a response using GPT-3.5 Turbo
    response = get_completion(prompt)

    return response  # Return the generated customer comment


# Step 2: Generate Email Subject
def generate_email_subject(comment):
    prompt=f"""
    Imagine you're providing customer support for an electronic product company. 
    Based on the customer's comment enclosed in triple backticks, propose a brief email subject to respond to the customer.
    Customer Comment:   ```{comment}```
    """
    response=get_completion(prompt)
    return response

# Step 3: Generate Customer Comment Summary
def summarize_comment(comment):
    prompt=f"""
    Asuming that you provide customer support for an electronic product company.
    Provide a concise summary in 50 words of the following customer comment delimited in triple backticks. Comment: ```{comment}```
    """
    response=get_completion(prompt)
    return response

def translate_summary_with_chatgpt(language, summary):
    prompt= f"""
    Translate the following summary delimited by triple backticks to the language delimited by <>. 
    Language:```{language}```   
    Summary:<{summary}>
    """
    response=get_completion(prompt)
    return response

# Step 4: Analyze Customer Comment Sentiment
def analyze_sentiment(comment):
    prompt=f"""
    Asuming that you provide customer support for an electronic product company.
    What is the sentiment of the comment delimited in triple backticks. Is it positive or negative? 
    Comment: ```{comment}```
    """
    max_tokens=10
    response=get_completion(prompt)
    sentiment = response.lower()
    if "positive" in sentiment:
        return "positive"
    elif "negative" in sentiment:
        return "negative"
    else:
        return "neutral"

# Step 5: Generate Customer Email
def generate_customer_email(summary, sentiment, email_subject,language):
    if sentiment == "positive":
        response_text = "We're thrilled to hear your feedback and appreciate your positive words. Your satisfaction is our top priority!"
    elif sentiment == "negative":
        response_text = "We're truly sorry to hear about your experience. Your feedback is crucial, and we'll strive to address your concerns."
    else:
        response_text = "Thank you for your feedback! We're always looking to improve and your insights are valuable."
    prompt= f"""
    Asuming that you provide customer support for an electronic product company.
    Given the specified parameters below:
    - Comment summary enclosed in backticks (`{summary}`)
    - Our response text enclosed in triple quotes (\"\"\"{response_text}\"\"\")
    - Translate the Email subject enclosed in angle brackets ({email_subject}) to language \"{language}\"
    Write a complete email responding to the customer's comment using the language \"{language}\". 
    """
    response=get_completion(prompt)
    return response


@app.route('/', methods=['GET', 'POST'])

def index():
    answer = ""
    comment = generate_customer_comment(products_data)
    print("A customer comment has been generated.")
    if request.method == 'POST':
        language = request.form.get('language')  # Fetch language input from the webpage
        print(f"Selected language: {language}")
        answer = process_comment_to_email(comment, language)
    return render_template('index.html', question=comment, answer=answer)

def process_comment_to_email(comment, language):
    # Step 2: Generate Email Subject
    email_subject = generate_email_subject(comment)
    print(f"An email subject is generated from the customer's comment.")
    # Step 3: Generate Customer Comment Summary
    summary = summarize_comment(comment)
    print("A Summary is generated from the customer comment.")
    translated_summary = translate_summary_with_chatgpt(language, summary)
    print("The summary has been translated to requested language.")
    # Step 4: Analyze Customer Comment Sentiment
    comment_sentiment = analyze_sentiment(comment)
    print(f"Sentiment of the comment is detected as: {comment_sentiment}")
    # Step 5: Generate Customer Email
    email_content = generate_customer_email(translated_summary, comment_sentiment, email_subject, language)
    print("A customer email has been generated.")
    return email_content


if __name__ == '__main__':
    app.run(debug=True)