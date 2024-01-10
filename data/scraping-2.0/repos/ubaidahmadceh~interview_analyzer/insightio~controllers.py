import openai
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
persist_directory = 'docs/chroma/'
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()



OPENAI_API_KEY  = os.getenv("API_KEY")
openai.api_key = OPENAI_API_KEY

chat = ChatOpenAI(openai_api_key = OPENAI_API_KEY, temperature = 0.6, model="gpt-3.5-turbo-16k")
chat2 = ChatOpenAI(openai_api_key = OPENAI_API_KEY, temperature = 0.5, model="gpt-3.5-turbo-16k")
chat3 = ChatOpenAI(openai_api_key = OPENAI_API_KEY, temperature = 0.9, model="gpt-4")
chat4 = ChatOpenAI(openai_api_key = OPENAI_API_KEY, temperature = 0.0, model="gpt-4")


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.0, 
    )
    
    content = response.choices[0].message["content"]
    return content

def voice_transcription(voice_file):
    audio_file= open(voice_file, "rb")
    transcript = openai.Audio.transcribe("whisper-1", openai_api_key=OPENAI_API_KEY, file = audio_file)
    return transcript


def process_insights(user_input):
    insight_schema = ResponseSchema(name="insight",
                                        description="derive meaningful insights from the text output in a list format")

    insight_schemas = [insight_schema]
    output_parser_insight = StructuredOutputParser.from_response_schemas(insight_schemas)
    format_insight = output_parser_insight.get_format_instructions()

    insight_template = """\
        insight: derive meaningful insights from the text output in a list format
        text: {text}
        {format_instructions}
        """
   ##Insight
    prompt = ChatPromptTemplate.from_template(template=insight_template)
    messages = prompt.format_messages(text= user_input,
                            format_instructions= format_insight)
    insight_output = chat4(messages)
    output_dict_insight = output_parser_insight.parse(insight_output.content)
    return output_dict_insight



def process_themes(user_input):
    ##Theme Schema
    theme_schema = ResponseSchema(name="theme",
                                description="What are the major themes in the given text?\
                                Give insightful and contextual response in a form of list.")

    motivation_schema = ResponseSchema(name="motivation",
                                        description="list of all the motivations")
    painpoint_schema = ResponseSchema(name="painpoint",
                                        description="list of all the painpoints")
    need_schema = ResponseSchema(name="need",
                                        description="list of all the needs")

    theme_schemas = [theme_schema,
                        motivation_schema,
                        painpoint_schema,need_schema]

    output_parser_theme = StructuredOutputParser.from_response_schemas(theme_schemas)
    format_theme = output_parser_theme.get_format_instructions()


    theme_template = """\
    For the following text, extract the following information:

    theme: What are the major themes in the given text?\
    Give insightful and contextual response in a form of list.

    motivation: Are there any user motivations? List all the motivations mentioned in the list.

    painpoint: Are there any user painpoints? List all the painpoints mentioned in the list.

    need: infer user needs from the text and list all the needs mentioned in the list.

    text: {text}

    {format_instructions}
    """
    ##Themes
    prompt = ChatPromptTemplate.from_template(template=theme_template)
    messages = prompt.format_messages(text=user_input, format_instructions=format_theme)
    theme_output = chat(messages)
    print("/nApi hit (theme")
    output_content = theme_output.content
    output_dict_theme = output_parser_theme.parse(output_content)


    return output_dict_theme



def process_summary(user_input):

    ## Summary Schema
    summary_schema = ResponseSchema(name="summary",
                                        description="create conclusion from the give text")
    summary_schemas = [summary_schema]
    output_parser_summary = StructuredOutputParser.from_response_schemas(summary_schemas)
    format_summary = output_parser_summary.get_format_instructions()
    summary_template = """\
        summary: construct a conclusion for the given text. What does it tell about the user?
        Use personality of a user researcher to frame your reponse.

        Format your response in JSON with following keys:

        Summary

        text1: {text}
        {format_instructions}

        """

    ##Summary
    prompt = ChatPromptTemplate.from_template(template=summary_template)
    messages = prompt.format_messages(text= user_input,
                            format_instructions= format_summary)
    summary_output = chat3(messages)
    print("/nApi hit (Summary")
    output_dict_summary = output_parser_summary.parse(summary_output.content)
    #summaries.append(output_dict_summary)

    return output_dict_summary


def chatFunction(data,prompt):
    history = ChatMessageHistory()
    memory = ConversationBufferMemory()
    profiles = []
    for i in range(len(data)):
        profiles.append(data[i]['val'])

    
    def getSplits(user_data):
        r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=10,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""])
        return(r_splitter.split_text(user_data))
    splits = []

    for i in range(len(profiles)):
        split = getSplits(profiles[i])
        splits.append(split)

    class Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    docs = []
    for j in range (len(splits)):
        for i in range(len(splits[j])):
            doc = Document(splits[j][i], {"userID": data[j]['name'],"index": i})
            docs.append(doc)
    pd = 'docs/chroma/'
    embedding = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=pd
    )
    def get_completion(prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.9, 
        )
        content = response.choices[0].message["content"]
        return content
    if prompt:
        history = ChatMessageHistory()
        question = prompt
        history.add_user_message(question)
        res = vectordb.max_marginal_relevance_search(question,k=10, fetch_k=20)
        mem = memory.load_memory_variables({})
        prompt = f"""
        Generate a contexual response using the data in {res} . Discard data in {res} if not relevant to the {question}  and link with the history in {mem}. 
        Use userIDs as references. Do not add opinions or suggestions from your own. 
        text : ```{res}```
        history: ```{mem}```
        """
        response1 = get_completion(prompt)
        history.add_ai_message(response1)
        memory.save_context({"input": question}, {"output": response1})
        return response1
    

def getPatterns(input):
    themes = ResponseSchema(name="themes",
                                description="Merge themes if they are similar in context, or have similar meaning and keep a count. Format in a Decreasing order or count. Json keys: name, count")
    themeComparisonSchemas = [themes]
    output_parser_themeComparison = StructuredOutputParser.from_response_schemas(themeComparisonSchemas)
    format_themeComparison = output_parser_themeComparison.get_format_instructions()
    themeComparison_template = """/

    themes = Merge themes if they are similar in context, or have similar meaning and keep a count. Format in a Decreasing order or count. Json keys: name, count
    text: {text}
    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_template(template=themeComparison_template)
    messages = prompt.format_messages(text = input,
                            format_instructions= format_themeComparison)
    themeComparison_output = chat(messages)
    prompt1 = f"""
    Convert the {themeComparison_output} in the following JSON format: 
    "themes:[name, count]"
    """
    response =  get_completion(prompt1)    
    output_dict_themeComparison = output_parser_themeComparison.parse(response) 
    return output_dict_themeComparison


def create_temporary_file_with_format(uploaded_file):
    _, file_extension = os.path.splitext(uploaded_file.name)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    return temp_file

def get_file_format(file_name):
    _, file_extension = os.path.splitext(file_name)
    return file_extension.lower()
