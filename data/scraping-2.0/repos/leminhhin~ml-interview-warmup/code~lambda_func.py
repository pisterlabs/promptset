from modal import Stub, Image, Secret, web_endpoint
from pydantic import BaseModel

stub = Stub('ml-interview-warmup-lambda-funcs')
CACHE_PATH = "/root/model_cache"

generate_qna_image = Image.debian_slim().pip_install("langchain", "arxiv", "cohere")
evaluate_answer_image = Image.debian_slim().pip_install("langchain", "cohere")

class TopicRequest(BaseModel):
    topic: str

class EvaluationRequest(BaseModel):
    question: str
    examiner_answer: str
    user_answer: str


@stub.function(image=generate_qna_image, secret=Secret.from_name("ml-interview-warmup"))
@web_endpoint(method="GET")
def generate_qna(item: TopicRequest):
    from langchain.chat_models import ChatCohere
    from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
    from langchain.prompts import PromptTemplate
    from langchain.pydantic_v1 import BaseModel, Field
    from langchain.schema import Document
    import arxiv

    # Define desired data structure for LLM output.
    class QnA(BaseModel):
        question: str = Field(description="question about a topic")
        answer: str = Field(description="answer for the question")
   
    def search_arxiv(user_topic, n=3):
        # Construct the default API client.
        arxiv_client = arxiv.Client()

        # Search for the most recent articles matching user topic.
        search = arxiv.Search(
            query = user_topic,
            max_results = n,
            sort_by = arxiv.SortCriterion.Relevance
        )

        search_results = arxiv_client.results(search)
        docs = [
            Document(
                page_content=result.summary,
                metadata={}
            )
            for result in search_results
        ]
        return docs
    
    def get_prompt_template_and_parser():
        # Set up a parser + inject instructions into the prompt template.
        parser = PydanticOutputParser(pydantic_object=QnA)
        auto_fixing_parser = OutputFixingParser.from_llm(llm=ChatCohere(), parser=parser)

        template = """Act as an interviewer that aims to assess the interviewee's knowledge and understanding of the key aspects of a topic. You want your interviewee to demonstrate strong reasoning skills and a comprehensive understanding of the topic in order to answer correctly.
        First, based on the topic's context below to prepare a question for the interviewee.
        Second, provide an exemplary answer to the question you came up with. Your answer should be comprehensive, fully address all relevant aspects of the topic with strong reasoning, and demonstrate deep understanding.
        Context: {context}
        {format_instructions}
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context"],
            partial_variables={"format_instructions": auto_fixing_parser.get_format_instructions()}
        )

        return prompt, auto_fixing_parser


    prompt, parser = get_prompt_template_and_parser()
    model = ChatCohere(temperature=0.3)
    chain = prompt | model | parser

    user_topic = item.topic
    relevant_docs = search_arxiv(user_topic)

    output = chain.invoke({"context": relevant_docs})


    return {
            "question": output.question,
            "examiner_answer": output.answer
    }


@stub.function(image=evaluate_answer_image, secret=Secret.from_name("ml-interview-warmup"))
@web_endpoint(method="GET")
def evaluate_answer(item: EvaluationRequest):
    from langchain.chat_models import ChatCohere
    from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
    from langchain.prompts import PromptTemplate
    from langchain.pydantic_v1 import BaseModel, Field

    # Define desired data structure for LLM output.
    class AnswerEvaluation(BaseModel):
        score: int = Field(description="score for user's answer")
        justification: str = Field(description="justification explaining the score")

    def get_prompt_template_and_parser():
        # Set up a parser + inject instructions into the prompt template.
        parser = PydanticOutputParser(pydantic_object=AnswerEvaluation)
        auto_fixing_parser = OutputFixingParser.from_llm(llm=ChatCohere(), parser=parser)


        template = """Act as an examiner and assess the interviewee's answer on the following 1-5 scale:
            - Score 1: Completely incorrect, irrelevant, or blank answer
            - Score 2: Major gaps in knowledge compared to interviewer's answer
            - Score 3: Significant errors or omissions vs. interviewer's answer
            - Score 4: Lacks depth and minor errors compared to interviewer's answer
            - Score 5: Demonstrates solid understanding on par with interviewer's answer
        Provide a score between 1-5. Explain your rating by:
            - Comparing the interviewee's answer to the exemplary interviewer's answer
            - Explaining how well the interviewee's answer meets the scoring criteria standards
            - Providing specific examples of gaps, errors, or lack of depth if scoring low
        
        Question: {question}
        Interviewer's answer: {examiner_answer}
        Interviewee's answer: {user_answer}
        {format_instructions}
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "examiner_answer", "user_answer"],
            partial_variables={"format_instructions": auto_fixing_parser.get_format_instructions()}
        )

        return prompt, auto_fixing_parser


    prompt, parser = get_prompt_template_and_parser()
    model = ChatCohere()
    chain = prompt | model | parser


    question = item.question
    examiner_answer = item.examiner_answer
    user_answer = item.user_answer

    output = chain.invoke({"question": question, "examiner_answer": examiner_answer, "user_answer": user_answer})

    return {
            "score": output.score,
            "justification": output.justification,
    }