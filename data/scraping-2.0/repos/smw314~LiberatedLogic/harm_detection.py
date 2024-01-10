import openai
from getpass import getpass
from googleapiclient import discovery
import json
import http.client
import urllib.request
import urllib.parse
import urllib.error
import base64
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

subscription_key = ""
endpoint = ""

# Set up OpenAI API key
openai.api_key = ''

# Get user's message
message = ""

API_KEY = ''

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

analyze_request = {
  'comment': { 'text': message },
  'requestedAttributes': {'TOXICITY': {}}
}

response = client.comments().analyze(body=analyze_request).execute()

comment_value = response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']

# Generate alternatives using OpenAI's GPT-3
response = openai.Completion.create(
  engine="text-davinci-002",
  prompt=f"The following message seems unclear or harmful: \"{message}\". Can you provide three alternative ways to express the same idea in a clearer and non-harmful manner?",
  temperature=0.5,
  max_tokens=200
)


def evaluate_message(message, subscription_key, endpoint):
    headers = {
        'Content-Type': 'text/plain',
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({
        'classify': 'True',
    })

    try:
        conn = http.client.HTTPSConnection(endpoint)
        conn.request("POST", "/contentmoderator/moderate/v1.0/ProcessText/Screen?%s" % params, message, headers)
        response = conn.getresponse()
        data = response.read().decode('utf-8')  # Decode from bytes to string
        conn.close()

        # Convert from JSON string to Python dictionary
        data = json.loads(data)
        
        # Extract category scores and profanity information
        category1_score = data['Classification']['Category1']['Score']
        category2_score = data['Classification']['Category2']['Score']
        category3_score = data['Classification']['Category3']['Score']
        review_recommended = data['Classification']['ReviewRecommended']
        profanity_detected = data['Terms']

        # Print category scores and review recommendation
        print('\nOffensive Score:', round(category3_score, 3))
        print('Sexually Explicit Score:', round(category1_score, 3))
        print('Sexually Suggestive Score:', round(category2_score, 3))
        print('Review Recommended:', review_recommended)
        
        # Print each detected profanity term
        if profanity_detected is not None:
            profanity_terms = ', '.join([term_info['Term'] for term_info in profanity_detected])
            print('Profanity Detected:', profanity_terms)
        else:
            print('No profanity detected.')

    except Exception as e:
        print("Exception occurred: {0}".format(e))

print("Comment:", message)
print("\nToxicity Score:", round(comment_value, 3))

evaluate_message(message, subscription_key, endpoint)

# Print the alternatives
print("\nAlternative phrases:")
print(response.choices[0].text.strip())
