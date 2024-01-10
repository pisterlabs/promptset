import networkx as nx
import pickle
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import os
import dotenv
import nltk

# load dot env
dotenv.load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")
    
# ## Initialize messages for chat 
# messages = [SystemMessage(content=prompt)]

# messages.append(HumanMessage(content=text))

# response = llm(messages)

class ArticleGenerator:

    def __init__(self):
        self.title = ""
        self.oldtitles = []
        self.content = ""
        self.outline = []
        self.content = {}
        self.primary_kw = ""
        self.secondary_kws = []
        self.lsi_kws = []
        self.audience = ""
        self.tone = ""
        self.G = pickle.load(open('graph.pickle', "rb"))
        self.llm = ChatOpenAI(
        openai_api_key=OPENAI_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0)
        self.messages = []
        self.llm4 = ChatOpenAI(
        openai_api_key=OPENAI_KEY,
        model_name='gpt-4',
        temperature=1)

    def print_instructions(self):
        print('''
        Instructions:
        1. Set primary keyword
        2. Set secondary keywords
        3. Set audience
        4. Generate title
        5. Generate outline
        6. Generate content per outline item
        ''')

    def set_title(self, title: str):
        self.title = title
        self.oldtitles.append(title)

    def set_primary_kw(self, primary_kw: str):
        self.primary_kw = primary_kw

    def set_secondary_kws(self, secondary_kws: list):
        self.secondary_kws = secondary_kws

    def set_lsi_kws(self, lsi_kws: list):
        self.lsi_kws = lsi_kws

    def set_audience(self, audience: str):
        self.audience = audience

    def set_tone(self, tone: str):
        self.tone = tone

    def reset_messages(self):
        self.messages = []

    def gen_title(self, instructions=None):
        assert self.primary_kw != "", "Primary keyword not set, set it using set_primary_kw() method"
        assert self.audience != "", "Audience not set, set it using set_audience() method"
        # reset messages
        self.reset_messages()
        # set prompt
        prompt = "Use this formula: Number or Trigger word + Adjective + Keyword + Promise to generate a engaging headline for an article about the primary topic\n\n Headline:"
        self.messages.append(SystemMessage(content=prompt))
        userprompt = ""
        userprompt+= "Primary topic: " + self.primary_kw + "\n"
        userprompt+= "Audience: " + self.audience + "\n"
        if instructions is not None:
            if "add_inst" in instructions:
                userprompt+= "Additional instructions: " + instructions["add_inst"] + "\n"
        self.messages.append(HumanMessage(content=userprompt))
        # generate title
        response = self.llm(self.messages)
        self.title = response.content
        self.oldtitles.append(self.title)
        return self.title
    
    def gen_outline(self, instructions=None):
        assert self.title != "", "Title not set, set it using set_title() method"
        assert self.primary_kw != "", "Primary keyword not set, set it using set_primary_kw() method"
        assert self.secondary_kws != [], "Secondary keywords not set, set it using set_secondary_kws() method"
        assert self.audience != "", "Audience not set, set it using set_audience() method"
        # reset messages
        self.reset_messages()
        # set prompt
        prompt = f'''Based on the following information, generate outer outline for a technical blog:
        1. Title: {self.title}
        2. Primary Keyword: {self.primary_kw}
        3. Secondary Keywords: {' '.join(self.secondary_kws)}
        4. Audience: {self.audience}
        5. For each outline item print only the heading, no preamble'''
        if instructions is not None:
            if "add_notes" in instructions:
                prompt+= "\n6. Use the reference notes while generating outline\nReference notes: " + instructions["add_notes"]
        prompt+= "\n\nOutline:"
        self.messages.append(SystemMessage(content=prompt))
        # generate outline
        response = self.llm4(self.messages)
        self.outline = response.content.split("\n")
        return self.outline
    
    # make ngam
    def make_ngrams(self, text, n):
        tokens = nltk.word_tokenize(text)
        ngrams = nltk.ngrams(tokens, n)
        return [ ' '.join(grams) for grams in ngrams]
    
    def fill_outline_item(self, outline_item: str, instructions=None):
        assert self.title != "", "Title not set, set it using set_title() method"
        assert self.lsi_kws != [], "LSI keywords not set, set it using get_lsi_kws() method"
        assert self.audience != "", "Audience not set, set it using set_audience() method"
        assert self.outline != [], "Outline not set, set it using gen_outline() method"
        assert outline_item in self.outline, "Outline item not in outline, set it using gen_outline() method"
        # make ngrams of outline item, ranging 1-4
        ngrams = []
        for n in range(1, 5):
            ngrams+= self.make_ngrams(outline_item, n)
        top_5 = {}
        # for the ngram in ngrams reverse order of length
        for ngram in sorted(ngrams, key=lambda x: len(x), reverse=True):
            # if ngram in self.G
            if ngram in self.G:
                # check if ngram is connected to lsi_kws
                for kw in self.lsi_kws:
                    if nx.has_path(self.G, ngram, kw):
                        # if yes, add to top_5
                        top_5[kw] = self.G.nodes[kw]['pagerank']
        # sort top_5 by pagerank
        top_5 = sorted(top_5.items(), key=lambda x: x[1], reverse=True)
        # get top 5
        top_5 = [kw for kw, _ in top_5][:5]
        # reset messages
        self.reset_messages()
        # ask user for number of paragraphs
        pgs = input("How many paragraphs do you want? ")
        assert pgs!= "", "Number of paragraphs not set"
        # set prompt
        prompt = f'''Based on the following information, generate content for a section of an article:
        1. Article Title: {self.title}
        2. Keywords to include: {' '.join(top_5)}
        3. Tone: {self.tone}
        4. Audience: {self.audience}
        5. Section heading: {outline_item}'''
        if instructions is not None:
            if "add_notes" in instructions:
                prompt+= "\n6. Use the reference notes while generating content\nReference notes: " + instructions["add_notes"]
        prompt+= f"\n\n{pgs} paragraphs of content:"
        self.messages.append(SystemMessage(content=prompt))
        # generate outline
        response = self.llm4(self.messages)
        self.content[outline_item] = response.content
        return response.content
    
    def get_top_nodes(self, node_name, n=2):
        # get the node
        node = self.G.nodes[node_name]
        # get the edges
        edges = self.G.edges(node_name, data=True)
        # get the top 5 connected nodes based on the pagerank of the in nodes
        top_nodes = sorted([(u, self.G.nodes[u]['pagerank']) for _, u, _ in edges], key=lambda x: x[1], reverse=True)
        # top nodes with name.split(' ')==n
        top_nodes = [u for u, _ in top_nodes if len(u.split(' '))==n][:30]
        # return the top 5 connected nodes
        return top_nodes
    
    def get_lsi_kws(self):
        # for each topic, get the top 15 connected nodes with n=2 to n=4
        topic = self.primary_kw
        related_keywords = set()
        for n in range(2, 5):
            try:
                related_keywords.update(self.get_top_nodes(topic, n))
            except:
                pass
        # for each related keyword, prompt the user if they want it as a lsi_kw
        for kw in related_keywords:
            # print(f"Is {kw} a LSI keyword? (y/n)")
            # ans = input()
            # if ans == "y":
            self.lsi_kws.append(kw)
        return self.lsi_kws
    
    def print_article(self):
        # make sure for each outline item, there is content
        assert len(self.outline) == len(self.content.keys()), "Content not generated for all outline items"
        # print title
        print(self.title)
        # print outline
        for item in self.outline:
            print(item)
            print(self.content[item])

    def save_article(self, filename, as_html=False):
        # make sure for each outline item, there is content
        assert len(self.outline) == len(self.content.keys()), "Content not generated for all outline items"
        if as_html:
            with open(filename, 'w') as f:
                f.write('<html>\n')
                f.write('<head>\n')
                f.write(f'<title>{self.title}</title>\n')
                f.write('</head>\n')
                f.write('<body>\n')
                f.write(f'<h1>{self.title}</h1>\n')
                for item in self.outline:
                    f.write(f'<h2>{item}</h2>\n')
                    f.write(f'<p>{self.content[item]}</p>\n')
                f.write('</body>\n')
                f.write('</html>\n')
        else:
            with open(filename, 'w') as f:
                f.write(self.title + '\n')
                for item in self.outline:
                    f.write(item + '\n')
                    f.write(self.content[item] + '\n')