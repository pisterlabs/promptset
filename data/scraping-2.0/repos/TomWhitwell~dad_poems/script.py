from openai import OpenAI
import os
import json
import random
import requests
import re

GUARDIAN_KEY = os.getenv('GUARDIAN_API_KEY')  # Corrected variable access
api_url = f"https://content.guardianapis.com/search?section=-(football|sport|australia-news)&show-fields=body&page-size=30&api-key={GUARDIAN_KEY}"  


def read_number(file_path):
    try:
        with open(file_path, 'r') as file:
            return int(file.read().strip())
    except FileNotFoundError:
        return 0

def write_number(file_path, number):
    with open(file_path, 'w') as file:
        file.write(str(number))

def increment_number(file_path):
    number = read_number(file_path)
    number += 1
    write_number(file_path, number)
    return number



def trim_to_words(s, num_words):
    words = s.split()
    return ' '.join(words[:num_words])
    
def strip_html_tags(text):
    """Remove HTML tags from text."""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def get_news_headlines(api_url):
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            headlines = [item['webTitle'] for item in data['response']['results']]
            return headlines
        else:
            return f"Error fetching data: HTTP {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_news_articles_and_summaries(api_url):
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            articles = []
            for item in data['response']['results']:
                title = item['webTitle']
                body = item['fields']['body']  # Get just the body field
                body = strip_html_tags(body)   # Strip HTML tags from the body
                full_content = f"{title} {body}"
                articles.append({'content': full_content})
            return articles
        else:
            return f"Error fetching data: HTTP {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

modes = [
    "creative"
]

# Now, modes contains the extended list of moods


philosophical_concepts = [
    "Existentialism", "Determinism", "Dualism", "Monism", "Nihilism", 
    "Realism", "Idealism", "Empiricism", "Rationalism", "Skepticism",
    "Pragmatism", "Stoicism", "Humanism", "Absurdism", "Relativism",
    "Solipsism", "Utilitarianism", "Hedonism", "Altruism", "Egoism",
    "Materialism", "Phenomenology", "Deontology", "Aesthetics", "Objectivism",
    "Subjectivism", "Empathy", "Ethnocentrism", "Holism", "Individualism",
    "Collectivism", "Romanticism", "Enlightenment", "Metaphysics", "Epistemology",
    "Ontology", "Teleology", "Theism", "Atheism", "Agnosticism",
    "Pantheism", "Fatalism", "Anarchism", "Marxism", "Capitalism",
    "Socialism", "Libertarianism", "Nationalism", "Globalism", "Pluralism",
    "Secularism", "Dogmatism", "Relativism", "Absolutism", "Mysticism",
    "Transcendentalism", "Pacifism", "Asceticism", "Autonomy", "Causality",
    "Vitalism", "Pessimism", "Optimism", "Empiricism", "Rationality",
    "Intuitionism", "Naturalism", "Essentialism", "Perfectionism", "Nativism",
    "Progressivism", "Conservatism", "Skepticism", "Traditionalism", "Postmodernism",
    "Structuralism", "Functionalism", "Behaviorism", "Positivism", "Constructivism",
    "Ecofeminism", "Egalitarianism", "Meritocracy", "Totalitarianism", "Authoritarianism",
    "Democracy", "Aristocracy", "Oligarchy", "Platonism", "Socratic",
    "Nietzscheanism", "Kantianism", "Hegelianism", "Darwinism", "Freudianism",
    "Confucianism", "Taoism", "Buddhism", "Stoicism", "Cynicism"
]




poets = [
    "Billy Collins",
    "RS Thomas", 
    "Simon Armitage", 
    "William Carlos Williams"
]

styles = [
"T.S. Eliot", "Robert Frost", "Sylvia Plath", "Langston Hughes", "Maya Angelou", "Pablo Neruda", "Seamus Heaney", "W.H. Auden", "Ezra Pound", "Ted Hughes", "Allen Ginsberg", "Philip Larkin", "Anne Sexton", "Elizabeth Bishop", "John Ashbery", "Billy Collins", "Carol Ann Duffy", "Charles Bukowski", "Octavio Paz", "Dylan Thomas", "Wallace Stevens", "Robert Hayden", "Gwendolyn Brooks", "Seamus Heaney", "E.E. Cummings", "Robert Lowell", "Simon Armitage", "Tracy K. Smith",  "Louise Glück", "Ocean Vuong", "Yusef Komunyakaa", "Saeed Jones", "Dorianne Laux", "Natalie Diaz", "Modernism", "Postmodernism", "Surrealism", "Harlem Renaissance", "Beat Poetry", "Black Mountain Poetry", "Language Poetry", "Imagism", "Futurism", "Dadaism", "Symbolism", "Objectivism", "Digital Poetry", "Spoken Word", "Concrete Poetry", "Romanticism", "Expressionism", "Futurism", "Minimalism", "Dirty Realism", "Narrative Poetry", "Avant-Garde Poetry", "Free Verse", "Visual Poetry", "Cyberpoetry", "Fluxus",    "Free Verse", "Haiku", "Sonnet", "Villanelle", "Sestina", "Ode", "Ghazal", "Tanka", "Ballad",
    "Blank Verse", "Rondeau", "Pantoum", "Acrostic", "Cinquain",
    "Epigram", "Concrete Poetry", "Elegy", "Narrative Poetry", "Lyric Poetry",
    "Prose Poetry", "Terza Rima", "Spoken Word", "Visual Poetry"
]


poetic_structures = [
    "Free Verse", "Haiku", "Sonnet", "Villanelle", "Sestina", "Ode", "Ghazal", "Tanka", "Ballad",
    "Blank Verse", "Rondeau", "Pantoum", "Acrostic", "Cinquain",
    "Epigram", "Concrete Poetry", "Elegy", "Narrative Poetry", "Lyric Poetry",
    "Prose Poetry", "Terza Rima", "Spoken Word", "Visual Poetry"]

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv('OPENAI_API_KEY'),
    
)

def straighten_quotes(text):
    replacements = {
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark
        "\u201C": '"',  # Left double quotation mark
        "\u201D": '"',  # Right double quotation mark
        "\u2032": "'",  # Prime (often used as apostrophe/single quote)
        "\u2033": '"',  # Double prime (often used as double quote)
    }
    for find, replace in replacements.items():
        text = text.replace(find, replace)
    return text


#         save_response_to_json(response, prompt, selected_news, selected_mode, selected_poem)

def save_response_to_json(response, prompt, selected_news, selected_mode, selected_poet, number, filename='response_news.json', archive_filename='archive_news.json'):
    if response and response.choices:
        # Access the content attribute of the message object
        response_content = response.choices[0].message.content.strip()
        response_content = straighten_quotes(response_content)
        
        # Save to the individual file
        with open(filename, 'w') as json_file:
            json.dump({"poem": response_content, "prompt": prompt, "news": selected_news, "poet": selected_poet, "mode": selected_mode, "number": number}, json_file)

        # Update the archive file
        try:
            # Read existing poems from the archive
            with open(archive_filename, 'r') as archive_file:
                archive_data = json.load(archive_file)
        except FileNotFoundError:
            # If the archive file doesn't exist, start with an empty list
            archive_data = []

        # Append the new poem to the archive
        archive_data.insert(0, {"poem": response_content, "prompt": prompt, "news": selected_news, "poet": selected_poet, "mode": selected_mode, "number": number})

        # Save the updated archive
        with open(archive_filename, 'w') as archive_file:
            json.dump(archive_data, archive_file)

def fetch_chatgpt_response(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4",
        )
        return chat_completion
    except Exception as e:
        print(f"Error in fetching response: {e}")
        return None

def main():

    new_number = increment_number("num.txt")
    print(f"The new number is: {new_number}")


    articles_and_summaries = get_news_articles_and_summaries(api_url)
    selected_concept = random.choice(philosophical_concepts)
    selected_structure = random.choice(poetic_structures)
    selected_style   = random.choice(styles)
    selected_poet = random.choice(poets)
    selected_mode = random.choice(modes)
    selected_news = trim_to_words(random.choice(articles_and_summaries)['content'],75)
#     poem_prompt=["You are a successful and innovative poet. A few moments ago, you read this story in the newspaper: \"" + selected_news + "\". Inspired, you write a poem, no more than 60 words long, in the style of " + selected_style + ". You add a one line title at the top.","You are a successful and innovative poet. You are studying " + selected_concept + ". Inspired, you write a poem, no more than 60 words long, in the style of " + selected_style + ". You add a one line title at the top."] 
# 
#     prompt = random.choice(poem_prompt)

    prompt="You are the poet " + selected_poet + ". You woke up this morning feeling " + selected_mode + ". You have just read this story in the newspaper: \"" + selected_news + "\". Write a poem in YOUR OWN DISTINCTIVE STYLE, no more than 60 words long. You may add a one line title at the top if you like."

#    prompt="Write a short poem about this news story: \"" + selected_news + "\". Write no more than 60 words. Adopt a strongly  " + selected_mode + " tone. You may add a one line title at the top if you like."



    print (prompt)
    response = fetch_chatgpt_response(prompt)
    if response:
        save_response_to_json(response, prompt, selected_news, selected_mode, selected_poet, new_number)


if __name__ == "__main__":
    main()
