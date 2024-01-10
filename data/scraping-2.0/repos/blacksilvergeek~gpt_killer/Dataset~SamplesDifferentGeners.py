literary_genres = [
    "Fiction",
    "Non-fiction",
    "Poetry",
    "Drama",
    "Novel",
    "Short story",
    "Essay",
    "Biography",
    "Autobiography",
    "Memoir",
    "Historical fiction",
    "Science fiction",
    "Fantasy",
    "Mystery",
    "Thriller",
    "Romance",
    "Horror",
    "Satire",
    "Comedy",
    "Tragedy",
    "Epic",
    "Sonnet",
    "Ballad",
    "Ode",
    "Fable",
    "Fairy tale",
    "Mythology",
    "Folklore",
    "Prose",
    "Proverb",
    "Allegory",
    "Parody",
    "Satirical essay",
    "Detective fiction",
    "Western",
    "Gothic fiction",
    "Magical realism",
    "Dystopian fiction",
    "Adventure",
    "Historical non-fiction",
    "Self-help book",
    "Philosophy",
    "Travelogue",
    "Play",
    "Graphic novel",
    "Children's literature",
    "Young adult literature",
    "Bildungsroman",
    "Comedy of manners",
    "Experimental literature"
]

authors_by_era = {
    "Ancient Era": [
        "Homer",
        "Plato",
        "Aristotle",
        "Virgil",
        "Ovid"
    ],
    "Medieval Era": [
        "Geoffrey Chaucer",
        "Dante Alighieri",
        "Thomas Aquinas",
        "Marie de France",
        "Giovanni Boccaccio"
    ],
    "Renaissance Era": [
        "William Shakespeare",
        "Miguel de Cervantes",
        "Niccolò Machiavelli",
        "Michel de Montaigne",
        "John Milton"
    ],
    "Enlightenment Era": [
        "Voltaire",
        "Jean-Jacques Rousseau",
        "Thomas Paine",
        "John Locke",
        "Denis Diderot"
    ],
    "Industrial Revolution Era": [
        "Charles Dickens",
        "Jane Austen",
        "Leo Tolstoy",
        "Mark Twain",
        "Victor Hugo"
    ],
    "Modern Era": [
        "Virginia Woolf",
        "Ernest Hemingway",
        "F. Scott Fitzgerald",
        "George Orwell",
        "Gabriel García Márquez"
    ]
}
essay_topics_by_era = {
    "Ancient Era": [
        "The Role of Gods and Goddesses in Ancient Mythology",
        "The Importance of Philosophy in Ancient Greece",
        "The Influence of Roman Law on Modern Legal Systems",
        "The Rise and Fall of Ancient Civilizations",
        "The Contributions of Ancient Egyptian Civilization",
    ],
    "Medieval Era": [
        "The Feudal System and Social Hierarchy",
        "The Impact of the Crusades on European Society",
        "The Role of the Catholic Church in Medieval Europe",
        "The Code of Chivalry and Knights in Medieval Literature",
        "The Black Death: Causes, Consequences, and Responses",
    ],
    "Renaissance Era": [
        "The Humanist Movement and Its Impact on Art and Literature",
        "The Scientific Revolution: Breakthroughs and Paradigm Shifts",
        "The Influence of Renaissance Thinkers on Political Philosophy",
        "The Role of Women in Renaissance Society",
        "Exploration and Discovery: New Worlds and Cultural Exchange",
    ],
    "Enlightenment Era": [
        "The Enlightenment Thinkers and Their Views on Human Reason",
        "The Impact of the French Revolution on European Politics",
        "The Rise of Secularism and Skepticism in the Enlightenment",
        "The Social Contract Theory and the Idea of Government",
        "The Role of Women in the Enlightenment",
    ],
    "Industrial Revolution Era": [
        "The Impact of Industrialization on Society and the Environment",
        "The Labor Movement and the Fight for Workers' Rights",
        "The Rise of Capitalism and its Effects on Wealth Distribution",
        "Urbanization and its Challenges in the Industrial Age",
        "Technological Advancements and their Influence on Daily Life",
    ],
    "Modern Era": [
        "The Effects of World Wars on Global Politics and Society",
        "The Civil Rights Movement and the Fight for Equality",
        "The Influence of Mass Media on Public Opinion",
        "Globalization and its Impact on Economics and Culture",
        "The Rise of Technology: Advancements and Ethical Considerations",
    ]
}
historical_eras = [
    "Ancient Era",
    "Classical Era",
    "Medieval Era",
    "Renaissance Era",
    "Enlightenment Era",
    "Industrial Revolution Era",
    "Modern Era",
    "Contemporary Era"
]
import json
import numpy as np
# read apikey from api_key.json
with open('api_key.json') as f:
    api_key = json.load(f)["api"]
# https://github.com/openai/openai-python
import openai
# Set up the OpenAI API client
openai.api_key = api_key
def excerpts_title(author,title,model= "gpt-3.5-turbo"):
    word_count = np.random.randint(50,1000)
    message = f"Create a section around{word_count} of text given the title {title},in style of {author} \
    the writing style and gener should following original style,\
    the theme should be any kinds of themes have been written by that novelist,\
        notice that the paragraph can imitate any section of original work"
    messages=[{"role": "user", "content": message}] # choose topic, geners and author in different era, randomness here
    # we can also command chatgpt generate paper in the easiest way, see data_gen
    temperature = np.random.uniform(0,1)

    # Generate a response
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        n=2,
        stop=None,
        temperature=temperature,
    )

    response = completion.choices[0].message.content

    return response,author,title
    
def excerpts_generator(authors_by_era,word_count,literary_genres,historical_eras,essay_topics_by_era, model= "gpt-3.5-turbo"):
    #???? not sure, but if too many messages, maybe use acreate
    ### ??? find some bugs, chatgpt did not imitate them
    # Set up prompt
    era = np.random.choice(historical_eras)
    authors = authors_by_era[era]
    author = np.random.choice(authors)
    topics = essay_topics_by_era[era]
    topic = np.random.choice(topics)
    gener = np.random.choice(literary_genres)
    word_count = np.random.randint(500,1000)
    message = f"Create a {gener} in the topic {topic},in style of {author} use about {word_count} words,\
    the theme should be any kinds of themes have been written by that novelist"
    messages=[{"role": "user", "content": message}] # choose topic, geners and author in different era, randomness here
    # we can also command chatgpt generate paper in the easiest way, see data_gen
    temperature = np.random.uniform(0.8,1)

    # Generate a response
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        n=2,
        stop=None,
        temperature=temperature,
    )

    response = completion.choices[0].message.content

    return response,author,era,topic,gener,word_count,temperature
import json,time
import csv
import threading
def write_file(f,author,excerpts_title,title,model= "gpt-3.5-turbo"):
    response,author,title =  excerpts_title(author,title)
    print(response)
    dictionary = {}
    dictionary['author'] = author
    dictionary['title'] = title
    dictionary['content'] = response
    json.dump(dictionary, f)
    f.write('\n')
    
f = open('responses_title.json', 'a')
texts = []
count = 0

with open("title_list.csv", "r", encoding="utf-8") as rf:
    reader = csv.DictReader(rf)
    for i in range(10):
        for row in reader:
            row = dict(row)
            print(row)
            author,title = row['author'],row['title_full']
            try:
                print(count)
                write_file(f,author,excerpts_title,title)
            except:
                print("error")
                texts.append((author,title))

        
        



"""f = open('responses_5.json', 'a')
for i in range(2000):
    print(i)
    
    try:
        response,author,era,topic,gener,word_count ,temperature= excerpts_generator(authors_by_era,500,literary_genres,historical_eras,essay_topics_by_era, model= "gpt-3.5-turbo")
        #print(response)
        json.dump({"author":author,"era":era,"gener":gener,"topic":topic,"randomness":temperature,"content":response}, f)
        f.write('\n')
        #responses.append(response)
    except:
        print("error")
        time.sleep(3)
            # server overloaded, so wait for a few seconds and send post again
           
        
    print(i)
f.close()
"""