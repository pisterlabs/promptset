from ctransformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
from functools import partial

from langchain.schema import StrOutputParser
from langchain_core.prompts import format_document
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.llms import Ollama
from langchain.schema.document import Document
from langchain.callbacks.manager import Callbacks
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

# maximum number of onion articles to load
__ARTICLE_LIMIT__:int = 5

# model to use for searching the pgvector database
article_model = SentenceTransformer('all-MiniLM-L6-v2')

# connection to the pgvector database for finding appropriate articles
conn = psycopg2.connect(database="rag_scratch",
                        host="localhost",
                        user="postgres",
                        port="5432")
register_vector(conn)


map_template = """
                You will be given an article with a title and a body.
                The article title will be enclosed in double backticks (``)
                The article body will be enclosed in triple backticks (```)
                Extract the most relevant facts and summarise them for answering questions about the article.
                The facts should be presented as a set of bullet points.

                ```{text}```

                SUMMARY:
            """

reduce_template = """
                You will be given article summaries that contain facts that are relevant to answering questions.
                The article summaries will be enclosed in triple backticks (```).
                The question from the user will be enclosed in double backticks (``)
                Using only the information supplied in the article summaries, answer the users question.
                Your answer should be less than 300 words.
                The style of the answer should be humourous and match the hunour of the original articles.

                Summaries:
                ```{text}```

                Question:
                ``{user_query}``


                Answer:
            """


template = """
                You will be given {article_count} articles which will be  enclosed in triple backticks (```). 
                The title of the article will be enclosed by <title> and </title> tags.
                The body of the article will be enclosed by <body> and </body> tags. 
                You will also be provided a question enclosed in double backticks(``).
                Using only the article information supplied, provide an answer to the question in as much detail as possible.
                Your answer should be less than 300 words.
                Your answer humour should be considered acerbic or black humour and use a similar style of humour to the provided articles.
                Do not summarise the articles, create a new article.

                Articles:
                ```{text}```


                Question:
                ``{creative_prompt}``


                Answer:
                """


# swapping to using Ollama as it's a hell of a lot easier to get running than using
# mistral directly on my mac, running a custom Ollama model
llm = Ollama(
    model="onion"
)

print(""" 
                                                                                                                                            
                                                                                        ,----..                                             
   ,---,                        ,-.             ___      ,---,                         /   /   \                                            
  '  .' \                   ,--/ /|           ,--.'|_  ,--.' |                        /   .     :               ,--,                        
 /  ;    '.               ,--. :/ |           |  | :,' |  |  :                       .   /   ;.  \      ,---, ,--.'|    ,---.        ,---,  
:  :       \    .--.--.   :  : ' /            :  : ' : :  :  :                      .   ;   /  ` ;  ,-+-. /  ||  |,    '   ,'\   ,-+-. /  | 
:  |   /\   \  /  /    '  |  '  /           .;__,'  /  :  |  |,--.   ,---.          ;   |  ; \ ; | ,--.'|'   |`--'_   /   /   | ,--.'|'   | 
|  :  ' ;.   :|  :  /`./  '  |  :           |  |   |   |  :  '   |  /     \         |   :  | ; | '|   |  ,"' |,' ,'| .   ; ,. :|   |  ,"' | 
|  |  ;/  \   \  :  ;_    |  |   \          :__,'| :   |  |   /' : /    /  |        .   |  ' ' ' :|   | /  | |'  | | '   | |: :|   | /  | | 
'  :  | \  \ ,'\  \    `. '  : |. \           '  : |__ '  :  | | |.    ' / |        '   ;  \; /  ||   | |  | ||  | : '   | .; :|   | |  | | 
|  |  '  '--'   `----.   \|  | ' \ \          |  | '.'||  |  ' | :'   ;   /|         \   \  ',  / |   | |  |/ '  : |_|   :    ||   | |  |/  
|  :  :        /  /`--'  /'  : |--'           ;  :    ;|  :  :_:,''   |  / |          ;   :    /  |   | |--'  |  | '.'\   \  / |   | |--'   
|  | ,'       '--'.     / ;  |,'              |  ,   / |  | ,'    |   :    |           \   \ .'   |   |/      ;  :    ;`----'  |   |/       
`--''           `--'---'  '--'                 ---`-'  `--''       \   \  /             `---`     '---'       |  ,   /         '---'        
                                                                    `----'                                     ---`-'                       
                                                                                                                                            
""")

def main():
    creative_prompt = ""

    while(creative_prompt != "q!"):
        print()
        creative_prompt = input("Enter a theme (or q! to quit): ")

        if(creative_prompt is None or len(creative_prompt) == 0):
            continue

        if(creative_prompt == "q!"):
            return

        cur = conn.cursor()
        embedded = article_model.encode(creative_prompt)

        # search the vector database for themes that match
        cur.execute(f"SELECT title, body FROM onion_articles ORDER BY embedding <-> %s LIMIT %s;", (embedded,__ARTICLE_LIMIT__))
        results = cur.fetchall() 

        if(len(results) == 0):
            print("Couldn't find any matching articles for inspiration")
            continue

#       prompt = PromptTemplate(template=template, input_variables=["article_count", "text", "creative_prompt"])
#        docs = [Document(page_content=f"<title>{t[0]}</title><body>{t[1]}</body>") for t in results]
#        llm_chain = LLMChain(prompt=prompt, llm=llm)
#        answer = llm_chain.run({
#            "article_count": __ARTICLE_LIMIT__,
#            "text": docs,
#            "creative_prompt": creative_prompt,
#        })
        

#        print(answer)

        chain = load_summarize_chain(
                                        llm,
                                        chain_type="map_reduce",
                                        map_prompt=PromptTemplate(template=map_template, input_variables=["text"]),
                                        combine_prompt=PromptTemplate(template=reduce_template, input_variables=["text", "user_query"]),
                                    )
        
        docs = [Document(page_content=f"{t[1]}", metadata={"title": f"t[0]"}) for t in results]

        out = chain.run({
                            'input_documents': docs,
                            'user_query': creative_prompt,
                        })

        print(out)
        



if __name__ == '__main__':
    main()