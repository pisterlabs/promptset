from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage


def chat_medical_transcription(transcript):
    chat = ChatOpenAI(
      temperature=0,
      model_name="gpt-3.5-turbo-16k-0613",
    ) # type: ignore

    # print('chat created')
   
    messages = [
        SystemMessage(
            content='''
            Du bist ein Assistent, der Arztbriefe schreibt. Du bekommst eine Transkription eines Gespräches zwischen einem Arzt und einem Patienten. Du sollst den Arztbericht nach dieser Vorlage schreiben, indem du alle Information aus dem transkribierten Text in den Arztbericht schreibst.  Bleibe bitte ausschließlich bei den Informationen aus dem Text, denke dir nichts aus und stelle keine eigenen Diagnosen. Die Vorlage für den Arztbrief sieht wie folgend aus:

            Klinische Angaben: 
            Konsil Uterusmyom Erstvorstellung Myomembolisation 

            Fragestellung: 
            Therapie 

            Voraufnahmen: 

            Befund: 

            Anamnese:  

            Voruntersuchungen:  

            Therapievorschläge:  

            Beurteilung:
            '''
        ),
        HumanMessage(
            content=f"Erstelle den Bericht von folgender Transkription: {transcript}."
        ),
    ]

    result = chat(messages)

    return result.content
