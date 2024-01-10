import json
import csv
import requests
import openai
import re

# Set your OpenAI API key
openai.api_key = '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'

# Function to make a POST request and get the top articles
def get_top_articles(query, number_of_top_articles=5):
    response = requests.post('http://localhost:3001/search', json={'query': query})
    if response.status_code == 200:
        results = response.json()
        return results[:number_of_top_articles]
    return []

# Function to ask ChatGPT to choose the best matching article
def get_best_match_from_chatgpt(claim, articles):
    prompt = f"Read the claim and the summaries of three articles below. Select the article that best matches the claim in essence.\n\nClaim: {claim}\n\nArticles:\n"
    for i, article in enumerate(articles, 1):
        prompt += f"{i}. {article['data']['Headline']} - {article['data']['What_(Claim)']}\n"
    prompt += "\nWhich article number best matches the claim?"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=60
    )
    
    # Use regex to find the first number in the response, which should be the article number.
    match = re.search(r'\d+', response.choices[0].text)
    if match:
        best_match_index = int(match.group()) - 1
        return articles[best_match_index]['data']['Headline'], articles[best_match_index]['data']['What_(Claim)']
    else:
        return "No clear best match found by ChatGPT.", ""

# Load the claims from a file (the file should be prepared with the claims)
with open('test2.json', 'r') as file:
    data = json.load(file)
    claims_sets = data['claims']
    print(f"Loaded {len(claims_sets)} sets of claims.")

# Prepare a CSV file to write the results
with open('results_comparison.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header row
    csvwriter.writerow(['Type', 'Claim', 'ChatGPT Selected Article', 'Top Matched Article', 'What_(Claim)'])
    print("Processing claims...")

    # Variables to keep track of the total number of rephrased claims and the number of correct matches
    total_rephrased_claims = 0
    correct_matches = 0

    # Iterate over the sets of claims
    for claims in claims_sets:
        original_claim = claims[0]
        # Get the top 3 articles for the original claim from the server
        top_articles_original = get_top_articles(original_claim)
        # Write the top matched article for the original claim to the CSV
        if top_articles_original:
            csvwriter.writerow(['Original', original_claim, top_articles_original[0]['data']['Headline'], top_articles_original[0]['data']['Headline'], top_articles_original[0]['data']['What_(Claim)']])
        else:
            csvwriter.writerow(['Original', original_claim, 'No articles found', 'No articles found', ''])
        # Process the rephrased claims
        for rephrased_claim in claims[1:]:
            # Get the top 3 articles for the rephrased claim from the server
            top_articles_rephrased = get_top_articles(rephrased_claim)
            # Get the best match from ChatGPT
            if top_articles_rephrased:
                best_match_article, what_claim = get_best_match_from_chatgpt(rephrased_claim, top_articles_rephrased)
                # Write to CSV
                csvwriter.writerow(['Rephrased', rephrased_claim, best_match_article, top_articles_rephrased[0]['data']['Headline'], what_claim])
                # Update the total number of rephrased claims and the number of correct matches
                total_rephrased_claims += 1
                if best_match_article == top_articles_original[0]['data']['Headline']:
                    correct_matches += 1
            else:
                csvwriter.writerow(['Rephrased', rephrased_claim, 'No articles found', 'No articles found', ''])

# Calculate and output the accuracy
if total_rephrased_claims > 0:
    accuracy = correct_matches / total_rephrased_claims
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct_matches}/{total_rephrased_claims})")
else:
    print("No rephrased claims were processed.")

print("Completed processing claims.")
