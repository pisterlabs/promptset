import openai
import os
from dotenv import load_dotenv 
load_dotenv()




class GetQueryIntent():
    def __init__(self) -> None:       
        pass
    
    def get_query_intent(self,query: str)-> str:
        """For a given query it will find out intent of query whether the given query is specific information or QnA type
        and it will return "specific_search" or "QnA"
        """
        openai.api_type = os.getenv("OPENAI_API_TYPE")
        self.userid=os.getenv("USERID")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_version= os.getenv("HELICON_OPENAI_API_VERSION")
        openai.api_base=os.getenv("HELICON_OPENAI_API_BASE")
        self.engine=os.getenv("GENERATIVE_ENGINE")
        self.model=openai.ChatCompletion(engine=self.engine)
        prompt="""Given a user query, determine its underlying intent. The query could range from seeking specific information, finding documents based on attributes, or exploring related content within the documents.        
        
        Informational Intent (VECTOR): The user is looking for general information, explanations, or answers to a question.
        Document Retrieval by Attributes (KEYWORD): The user wants to retrieve documents based on specific attributes such as circular number,circular date, name, ID, reference number, or date.

        IF Query intent is Document Retrieval by Attributes return "KEYWORD".
        IF Query intent is Informational Intent return "VECTOR".

        [EXAMPLES]
        Query: Whether sourcing support services to foreign entity qualifies as export of service?
        Intent: VECTOR

        Query: Reliance Industries Ltd v. CIT
        Intent: KEYWORD

        Query: Circular no. 495 dated 22-09-1987
        Intent: KEYWORD

        Query: GST implications on user delivery fees collected from end users/customer
        Intent: VECTOR    
        """


        message_text = [{"role":"system","content":f"{prompt}"},
                        {"role":"user","content":f"Query: {query}\nIntent:"}]
        
        completion = self.model.create(
        headers={
        "User-Id": self.userid
        },    
        engine=self.engine,
        model="gpt-3.5-turbo",
        messages = message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
        )

        return completion['choices'][0]['message']['content']



