import re
from pydantic import BaseModel, Field
from constant import LLM_MODEL_4_GENERATE, EMBEDDING_FUNC
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

class LineList(BaseModel):
    lines: list[str] = Field(description="Lines of text")

class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

output_parser = LineListOutputParser()

FIRST_ROUND_KEYWORD_PROMPT = PromptTemplate(
    input_variables=["topic", "description"],
    template="""
    You are a Professor in the field of Business and Marketing, especially topics that related to applying Extended Reality (XR) in Marketing and Business.
    Your goal is to help user generate a list of keywords based on the topic: {topic} and the description of the project.
    The list of keywords must be relevant to the topic of XR in Marketing and Business and can be search for more related work on Arxiv or Google Scholar.
    The list of keywords must aim to help user to find more relevant research papers, documents, articles, journals articles or publication that related to the topic and description of the project that the user ask.
    The keyword must be relevant to the topic of XR in Marketing and Business.
    You must generate at least 10 keywords and at most 20 keywords.
    Always make sure that the keywords are relevant to the topic of XR in Marketing and Business and the {topic}.
    
    Please do your best, this is very important to the user career.
    
    Topic: {topic}
    
    Description: {description}
    """
)

FILTER_ROUND_KEYWORD_PROMPT = PromptTemplate(
    input_variables=["topic", "description", "keyword_list"],
    template="""
    Given a list of keyword: {keyword_list}
    Along with the topic of the research: {topic}
    And the description of the topic: {description}
    
    Your role is to filter out the list of keywords that are not relevant to the topic of XR in Marketing and Business based on the given topic and description.
    The list of keywords must be relevant to the topic of XR in Marketing and Business and can be search for more related work on Arxiv or Google Scholar.
    The criteria for filtering out the keywords are:
    1. The keyword must be relevant to the topic of XR in Marketing and Business.
    2. The keyword must be relevant to the topic of {topic}.
    3. The keyword must be relevant to the description of the topic.
    4. The keyword must be able to find on Arxiv or Google Scholar.
    
    You must return in the following format: [keyword1, keyword2, keyword3, ...]
    Please do your best, this is very important to the user career. 
    
    Topic: {topic}
    Description: {description}
    Keyword list: {keyword_list}
    
    <Your answer at here>

    """
)

class QuestionGenerator:
    def __init__(
        self,
        llm_model: ChatOpenAI = LLM_MODEL_4_GENERATE,
        first_round_template: PromptTemplate = FIRST_ROUND_KEYWORD_PROMPT,
        second_round_template: PromptTemplate = FILTER_ROUND_KEYWORD_PROMPT,
        output_parser: PydanticOutputParser = output_parser,
    ):
        self.llm_chain_first_round = LLMChain(llm = llm_model,
                                  prompt = first_round_template,
                                  output_parser = output_parser)
        
        self.llm_chain_second_round = LLMChain(llm = llm_model,
                                  prompt = second_round_template)
    
    def generate_question(self, topic: str, description: str):
        response = self.llm_chain_first_round.invoke({"topic": topic, "description": description})
        
        return response["text"]
    
    def filter_result(self, topic: str, description: str, keyword_list: list[str]):
        response = self.llm_chain_second_round.invoke({"topic": topic, "description": description, "keyword_list": keyword_list})
        
        return response["text"]
    
    def parsing_keyword(self, response_string: str):
        if keywords_match := re.search(r'\[(.*?)\]', response_string, re.DOTALL):
            # Extracted content within square brackets
            keywords_text = keywords_match[1]

            # Split the content into a list of keywords
            keywords_list = re.findall(r"'(.*?)'", keywords_text)
        
        return keywords_list
    
    def customize_prompt(self, customize_prompting):
        raise NotImplementedError

#query = "Unleashing the Metaverse: Extended Reality (XR) in Marketing"
#description = """
#Embark on a captivating exploration of Extended Reality (XR) in marketing through this 10-week project. In the first three weeks, you will concentrate on a guided deep dive into the literature of XR, which encompasses Augmented Reality (AR), Virtual Reality (VR), and Mixed Reality (MR). 
#This phase will focus on understanding the concepts and potential applications of XR in marketing.Weeks four and five will involve analyzing the literature to identify key trends and patterns in XR marketing. 
#This analysis will enhance your understanding of how XR has been evolving and what aspects of marketing it has been impacting.
#In weeks six to eight, you will shift your focus to examining real-world case studies. 
#Select three impactful XR marketing campaigns and investigate them in depth. Evaluate the technologies used, the response from consumers, and the overall effectiveness of the campaigns. 
#During the ninth week, you will synthesize the knowledge gained from the literature and case studies to develop insights into the current state of XR in marketing. 
#In the final week, you will compile your insights and analyses into a comprehensive report. 
#This report will document your journey through the XR literature, key trends, and real-world applications.
#"""
#question_generator = QuestionGenerator()
#output = question_generator.generate_question(query, description)
#print(output)