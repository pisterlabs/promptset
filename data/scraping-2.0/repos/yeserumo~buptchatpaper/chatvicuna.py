import time
import logging
import requests
from typing import List, Dict, Any, Optional, Mapping, Union
from load_config import load_config
import langchain
import re
from langchain.llms.base import LLM
from langchain.cache import InMemoryCache  
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
logging.basicConfig(level=logging.INFO)

langchain.llm_cache = InMemoryCache()   
config = load_config()
keywords_information = config["ARXIV"]["keywords_information"]
# ToDo: 1. 修改query，增加system和user的role以及对应的content
class ChatVicuna(LLM):
    chat_type = 0
    method_aliases = {}
    url = 'http://localhost:8000/v1/chat/completions'
    def __init__(self) -> None:
        super().__init__()
    @classmethod
    def set_chat_prompt(cls, chat_type) -> None:
        cls.chat_type = chat_type
    def set_method_aliases(self):
        self.method_aliases = {
            "0": self._construct_query_for_introduction,
            "1": self._construct_query_for_method,
            "2": self._construct_query_for_conclusion,
            "3": self._construct_get_section_name,
            "4": self._construct_query_for_abstract_summary,
            "5": self._construct_query_for_introduction_summary,
            "6": self._construct_query_for_method_summary,
            "7": self._construct_query_for_generate_abstract,
        }
    @property
    def _llm_type(self) -> str:
        return 'chat_vicuna-13b'
    def _construct_naive(self, prompt: str) -> Dict:
        query = {
            "model": 'gpt-3.5-turbo',
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                    },
                ]
        }
        return query

    def _construct_query_for_introduction(self, prompt: str) -> Dict:
        # {"model": "gpt-3.5-turbo", "messages": [{"role": "system", "content": "你好！"}]}
        query = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a research in the field of computer science,who is good at summarizing papers using concise statements."
                    },
                {
                    "role": "assistant", 
                    "content": "This is the introduction of a English document. I need your help to read and summarize the following questions: {}".format(prompt.replace("\"0", ""))
                 },
                {
                    "role": "user",
                    "content": """
                    Summarize according to the following four points. Be sure to use {} answers (proper nouns need to be marked in English)
                        - (1):What is the research background of this article?
                        - (2):What are the pastmethod? What are the problems with them? Is the approach well motivated?
                        - (3):What is the research methodology proposed in this paper?
                        - (4):On what task and what performance is achieved by the methods in this paper? Can the performance support their goals?
                    Follow the format of the output that follows:
                    Summary:
                        - (1):XXX; \n
                        - (2):XXX; \n
                        - (3):XXX; \n
                        - (4):XXX; \n

                    Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not have too much repetitive information, numerical values using the original numbers, be sure to strictly follow the format, the corresponding content output to XXX, in accordance with \n line feed.
                        """.format('English', 'English')
                    }
                    
                ]
        }
        return query
    
    def _construct_query_for_method(self, prompt: str) -> Dict:
        query = {
        "model": 'gpt-3.5-turbo',
        "messages": [
            {
                "role": "system", 
                "content": "You are a researcher in the field of computer science who is good at summarizing papers using concise statements"
                },
            {
                "role": "assistant", 
                "content": "This is the <summary> and <Method> part of an English document, where <summary> you have summarized, but the <Methods> part, I need your help to read and summarize the following questions: {}".format(prompt[1:] if prompt.startswith('1') else prompt)
                },
            {
                "role": "user",
                "content": """Describe in detail the methodological idea of this article. Be sure to use {} answers (proper nouns need to be marked in English). For exampled, its steps are.
                    - (1):...
                    - (2):...
                    - (3):...
                    - .......
                Follow the format of the output that follows:
                    Method: \n\n
                    - (1):XXX; \n
                    - (2):XXX; \n
                    - (3):XXX; \n
                    ......... \n\n

                Be sure to use {} answers(proper nouns need to be marked in English), statements as concise and academic as possible, do not repeat the content of the previous <summary>, the value of the use of the original numbers, be sure to strictly follow the format, the corresponding content output to XXX, in accordance with \n line feed, ....... means fill in according to the actual requirements, if not, you can not write.
                    """.format("English", "English")
                }
            ]
    }
        return query
    
    def _construct_query_for_conclusion(self, prompt: str) -> Dict:
        query = {
        "model": 'gpt-3.5-turbo',
        "messages": [
            {
                "role": "system", 
                "content": "You are a reviewer in the field of [computer science] and you need to critically review this article." 
                },
            {
                "role": "assistant", 
                "content": "This is the <summary> and <conclusion> part of an English literature, where <summary> you have already summarized, but <conclusion> part, I need your help to summarize the following questions: {}".format(prompt[1:] if prompt.startswith('2') else prompt)
                },
            {
                "role": "user",
                "content": """Make the following summary. Be sure to use {} answers (proper nouns need to be marked in English).
                    - (1):What is the significance of this piece of work?
                    - (2):Summarize the strengths and weaknesses of this article in three dimensions: innovation point, performance, and workload.
                    .......
                Follow the format of the output later:
                    Conclusion: \n\n
                    - (1):XXX; \n
                    - (2):Innovation point: XXX; Performance: XXX; Workload: XXX; \n
                Be sure to use {} answers(proper nouns need to be marked in English), statements as concise and academic as possible, do not repeat the content of the previous <summary>, the value of the use of the original numbers, be sure to strictly follow the format, the corresponding content output to XXX, in accroding with \n line feed, ....... means fill in according to the actual requirements, if not, you can not write.
                    """.format('English', 'English')
                }
            ]
    }
        return query
    def _construct_get_section_name_2(self, prompt: str) -> Dict:
        query = {
        "model": 'gpt-3.5-turbo',
        "messages": [
            {
                "role": "user", 
                "content": """
                Please select from {} that is most relevant to Introduction, Method and Conclusion.
                Please output the following:
                    Introduction: XXX\n
                    Method: XXX\n
                    Conclusion: XXX\n
                Please replace XXX with the item you select.
                """.format(prompt[1:] if prompt.startswith('3') else prompt)
            },
            ]
    }
        return query
    def _construct_get_section_name(self, prompt: str) -> Dict:
        query = {
        "model": 'gpt-3.5-turbo',
        "messages": [
            {
                "role": "system", 
                "content": """
                You are an expert paper writer and are familiar with the various formatting specifications for paper writing. An paper usually needs to be composed of sections such as Introduction, Related Work, Method, Experiment, Conclusion, etc., but the names of the above sections vary from one essay to another, and I will provide you with the names of the chapters of an paper that, what you need to do is to extract from these names the names corresponding to the Introduction, Method and Conclusion sections.
                """
            },
            {
                "role": "assistant",
                "content": "Please find the possible name of Introduction, Method and Conlusion. This is the name of the chapters of an paper:" + format(prompt[1:] if prompt.startswith('3') else prompt)
            },
            {
                "role": "user",
                "content": """
                Please output the results in the following format:
                    Introduction: xxx\n
                    Method: xxx\n
                    Conclusion: xxx\n
                Do not output any other information than this.\n
                It is important to note that xxx needs to be replaced with one of the names of the chapters of the paper given previously. If the name of the Introduction section is Introduction or the name of the Method section is Method in the paper, then just fill it in before.
                """
            }
                ]
    }
        return query
    
    def _construct_query_for_abstract_summary(self, prompt):
        query = {
        "model": 'gpt-3.5-turbo',
        "messages": [
            {
                "role": "system", 
                "content": """
                You are a researcher in the field of computer science and specialise in summarising papers and generating reviews in concise statements.
                """
            },
            {
                "role": "assistant",
                "content": """
                Below is the abstract section of multiple papers that are in the field of {}: \n {}
                """.format(keywords_information, prompt[1:] if prompt.startswith('4') else prompt)
            },
            {
                "role": "user",
                "content": """
                Based on the above information, please help me to write a paragraph including the importance of {}, a summary of the above information, the current status of research on {}, and a prediction of the direction of development of {}(proper nouns need to be marked in English).
                """.format(keywords_information, keywords_information, keywords_information)
            }
                ]
    }
        return query
    def _construct_query_for_introduction_summary(self, prompt):
        query = {
        "model": 'gpt-3.5-turbo',
        "messages": [
            {
                "role": "system", 
                "content": """
                 You are a researcher in the field of computer science and specialise in summarising papers and generating reviews in concise statements.
                """
            },
            {
                "role": "assistent",
                "content": """
                Here are brief descriptions of several papers, with each paragraph representing a paper: \n {}
                """.format(prompt[1:] if prompt.startswith('5') else prompt)
            },
            {
                "role": "user",
                "content": """
                I need you to write a summary for me, this summary needs to include an introduction to the field of XX, an introduction to past methods, an introduction to each paper, possible directions and prospects, etc.(proper nouns need to be marked in English).
                """
            }
                ]
    }
        return query

    def _construct_query_for_method_summary(self, prompt):
        query = {
        "model": 'gpt-3.5-turbo',
        "messages": [
            {
                "role": "system", 
                "content": """
                You are a researcher in the field of computer science and specialise in summarising papers and generating reviews in concise statements.
                """
            },
            {
                "role": "user",
                "content": """
                Please summarise the following into a paragraph, it's a detailed description of an algorithm: \n {}
                I need you to help me summarise it into a paragraph(proper nouns need to be marked in English).
                """.format(prompt[1:] if prompt.startswith('6') else prompt)
            }
                ]
    }
        return query
    
    def _construct_query_for_sentence_summary(self, prompt):
        query = {
        "model": 'gpt-3.5-turbo',
        "messages": [
            {
                "role": "system", 
                "content": """
                You are a researcher in the field of computer science and specialise in summarising papers.
                """
            },
            {
                "role": "user",
                "content": """
                Please summarise the following in a short paragraph: \n {}
                I need you to help me summarise it into a short paragraph(proper nouns need to be marked in English).
                """.format(prompt[1:] if prompt.startswith('7') else prompt)
            }
                ]
    }
        return query
    
    
    
    
    @classmethod
    def _post(cls, url: str, query: Dict) -> Any:
        _headers = {"Content_Type": "application/json"}
        with requests.session() as sess:
            resp = sess.post(url, json=query, headers=_headers, timeout=600)
        return resp
    
    def _call(self, prompt:str, stop: Optional[List[str]] = None) -> str:
        self.set_method_aliases()
        # 有些并没有使用summary prompt
        if prompt.startswith("Write a concise summary of the following:\n\n\n\"0"):
            ChatVicuna.set_chat_prompt(0)
        elif prompt.startswith("Write a concise summary of the following:\n\n\n\"1"):
            ChatVicuna.set_chat_prompt(1)
        elif prompt.startswith("Write a concise summary of the following:\n\n\n\"2"):
            ChatVicuna.set_chat_prompt(2)
        elif prompt.startswith("Write a concise summary of the following:\n\n\n\"3"):
            ChatVicuna.set_chat_prompt(3)
        elif prompt.startswith("Write a concise summary of the following:\n\n\n\"4"):
            ChatVicuna.set_chat_prompt(4)
        elif prompt.startswith("Write a concise summary of the following:\n\n\n\"5"):
            ChatVicuna.set_chat_prompt(5)
        elif prompt.startswith("Write a concise summary of the following:\n\n\n\"6"):
            ChatVicuna.set_chat_prompt(6)
        elif prompt.startswith("7"):
            ChatVicuna.set_chat_prompt(7)
        else:
            query = self._construct_naive(prompt=prompt)
        if self.chat_type == 0:
            query = self._construct_query_for_introduction(prompt)
        query = self.method_aliases[self.chat_type](prompt)
        resp = self._post(url=self.url, query=query)
        if resp.status_code == 200:
            resp_json = resp.json()
            predictions = resp_json["choices"][0]['message']['content']
            return predictions
        else:
            return "请求模型"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        _param_dict = {
            "url": self.url
        }
        return _param_dict


# if __name__ == "__main__":
#     llm = ChatVicuna()
#     TEXT_URL = '/home/sjx/Common/papertool/paper/20230722102536/output/3D-SeqMOS: A Novel Sequential 3D Moving Object Segmentation in Autonomous Driving/main.tex'
#     with open(TEXT_URL, 'r') as f:
#         text = f.read()
#     text_spliter = CharacterTextSplitter(
#         chunk_size=1500,
#         chunk_overlap=0,
#         length_function=len,
#     )
#     CHAIN_TYPE = 'map_reduce'
#     summary_chain = load_summarize_chain(llm, chain_type=CHAIN_TYPE)
#     summarize_document_chain = AnalyzeDocumentChain(
#         combine_docs_chain=summary_chain,
#         text_spliter=text_spliter,
#     )   
#     res = summarize_document_chain.run(text)




    # tools = load_tools(["llm-math"], llm=llm)
    # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    # agent.run("what 2 + 2 get?")

    # while True:
    #     query = "hello, how are you?"
    #     begin_time = time.time() * 1000
    #     response = llm(query)
    #     print(response)



    