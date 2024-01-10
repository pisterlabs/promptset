#%%
from langchain import PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.chains import LLMChain

load_dotenv()

llm = OpenAI(temperature=0.3)


template = """I want you to act as a tweet labeler, you are given representative words
     from a topic and three representative tweets, give more attention to the words, all the tweets are related to climate change, and COP, no need to mention it, detect subtopics.
     start with "label:" and avoid hashtags,
     which is a good short label for the topic containing the words [{words}], here you are 3 tweets to help you:
     first = \"{tweet1}\", second = \"{tweet2}\", third = \"{tweet3}\""""


prompt = PromptTemplate(
    input_variables=["words", "tweet1", "tweet2", "tweet3"],
    template=template,
)


chain = LLMChain(llm=llm, prompt=prompt)





#%%
from bertopic import BERTopic

loaded_model = BERTopic.load("/Users/alessiogandelli/data/cop26/cache/model_cop26.pkl")


# %%
topics = list(loaded_model.get_topic_info()['Topic']) # get inferred topics 
topic_words = loaded_model.get_topics() # get words for each topic
labels = {}

for topic in topics:
    tweets = loaded_model.get_representative_docs(topic)
    words = [word[0] for word in topic_words[topic]]
    labels[topic] = chain.run(words=words, tweet1=tweets[0], tweet2=tweets[1], tweet3=tweets[2])

# remove \n from values of labels 

labels = {key: value.replace('\n', '') for key, value in labels.items()} 
labels = {key: value.replace('Label:', '') for key, value in labels.items()}
#strip 
labels = {key: value.strip() for key, value in labels.items()}
labels

# save labels to file 
import json

with open('/Users/alessiogandelli/data/cop26/cache/labels.json', 'w') as fp:
    json.dump(labels, fp)
    

#%%













from langchain import PromptTemplate, FewShotPromptTemplate

examples = [
    {"words": "climate change, floods, climate crisis", "tweet1": "i love climate change ", "tweet2": "my house is under the sea now", "tweet3": "my car is useless now", "label": "climate change"},
    {"words": "climate change, floods, climate crisis", "tweet1": "i love climate change ", "tweet2": "my house is under the sea now", "tweet3": "my car is useless now", "label": "climate change"},
    {"words": "climate change, floods, climate crisis", "tweet1": "i love climate change ", "tweet2": "my house is under the sea now", "tweet3": "my car is useless now", "label": "climate change"},
    {"words": "climate change, floods, climate crisis", "tweet1": "i love climate change ", "tweet2": "my house is under the sea now", "tweet3": "my car is useless now", "label": "climate change"},
    ]


example_formatter = """ words = [{words}], tweet1 = \"{tweet1}\", tweet2 = \"{tweet2}\", tweet3 = \"{tweet3}\"
    label = \"{label}\""""

example_prompt = PromptTemplate(
    input_variables=["words", "tweet1", "tweet2", "tweet3", 'label'],
    template=example_formatter,
)

prompt = FewShotPromptTemplate(
    input_variables=["words", "tweet1", "tweet2", "tweet3"],
    examples = examples,
    example_prompt=example_prompt,
    prefix = "Give the label of every input",
    suffix = " words = [{words}], tweet1 = \"{tweet1}\", tweet2 = \"{tweet2}\", tweet3 = \"{tweet3}\" \n label = ",
    example_separator = "\n"
)

prompt.format(words="climate change, floods, climate crisis", tweet1="i love climate change ", tweet2="my house is under the sea now", tweet3="my car is useless now")
# %%
chain = LLMChain(llm=llm, prompt=prompt)
chain.run(words="climate change, floods, climate crisis", tweet1="i love climate change ", tweet2="my house is under the sea now", tweet3="my car is useless now")


# %%


topics = list(p.model.get_topic_info()['Topic']) # get inferred topics 
topic_words = p.model.get_topics() # get words for each topic
labels = {}

for topic in topics:
    tweets = p.model.get_representative_docs(topic)
    words = [word[0] for word in topic_words[topic]]



# %%
