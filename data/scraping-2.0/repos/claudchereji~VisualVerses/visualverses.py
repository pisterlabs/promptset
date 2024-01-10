import openai
import requests


def get_bible_verse(book, chapter, verse):
    # Build the API URL
    base_url = "https://bible-api.com/"
    book = book.replace(" ", "%20")  # Replace spaces with %20 for the URL
    url = f"{base_url}{book}+{chapter}:{verse}?translation=kjv"
    
    # Make the API request
    response = requests.get(url, verify=True)
    
    # Check for successful request
    if response.status_code != 200:
        return "Error: API request failed"
    
    # Extract the verse text from the response
    data = response.json()
    verse_text = data["text"]
    
    return verse_text

book = input("Enter the name of a book from the Bible: (example: Matthew, Mark, Luke, etc.)\n")
chapter = input("Enter the chapter number: (make sure it's a valid chapter)\n")
verse = input("Enter the verse number: (make sure it's a valid verse)\n")

verse_text = get_bible_verse(book, chapter, verse)
print("\n\n" + verse_text)

# Set your OpenAI API key

openai.api_key = "YOUR_API_KEY"

while True:
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt= f"rewrite this text into a description that would allow me to generate an image from it using OpenAI's DALL-E 2 model: {verse_text}",
        max_tokens=1024,
        temperature=.901
    )
    # Extract the reformatted text from the response
    reformatted_text = response["choices"][0]["text"]
    print("Here is the summarized text\n" + reformatted_text + "\n")
    user_input = input("Is that a good summarization? \n\n(yes/no)\n")
    if user_input.lower() == "yes":
        while True:
            # Set the API endpoint and your API key
            api_endpoint = "https://api.openai.com/v1/images/generations"
            api_key = "YOUR_API_KEY"

            # Build the API request payload
            payload = {
                "model": "image-alpha-001",
                "prompt": f"generate an image of {reformatted_text} in a Hyperrealistic/impressionistic/conceptual art style",
                "num_images": 1,
                "size": "1024x1024",
                "response_format": "url"
            }

            # Set the API request headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            # Make the API request
            response = requests.post(api_endpoint, json=payload, headers=headers, verify=True)

            # Check for successful request
            if response.status_code != 200:
                print("Error: API request failed\n")
                break
            else:
                # Print the image URL
                image_url = response.json()["data"][0]["url"]
                print("\nHere is the link to the image, click the link to view it.\n\n" + image_url)
                user_input = input("\n\nDo you like the image? \n(yes/no)\n")
                if user_input.lower() == "yes":
                	break
                elif user_input.lower() == "no":
                    continue
                else:
                    print("Invalid input, please enter 'yes' or 'no'\n")
        break
    elif user_input.lower() == "no":
        # perform the previous action again
        continue
    else:
        print("Invalid input, please enter 'yes' or 'no'\n")
print("\n\nThank You for using VisualVerses!\n\n")
referenceVerse = (book +" " + chapter+ ":" + verse)
print("this was your verse for reference \n" + referenceVerse)
