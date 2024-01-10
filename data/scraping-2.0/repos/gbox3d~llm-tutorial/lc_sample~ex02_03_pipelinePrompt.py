#%%
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate

from dotenv import load_dotenv
load_dotenv('../.env')


#%%
chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ]
)
print("chat model ready")

#%%
intro = PromptTemplate.from_template(
    """
    You are a role playing assistant.
    And you are impersonating a {character}
    """
)

example = PromptTemplate.from_template(
    """
    This is an example of how you talk:
    Human: {example_question}
    You: {example_answer}
    """
)

start = PromptTemplate.from_template(
    """
    Start now!

    Human: {question}
    You:
    """
)

full_prompt = PipelinePromptTemplate(
    final_prompt=PromptTemplate.from_template(
        """
        {intro}
                                        
        {example}
                                
        {start}
        """
    ),
    pipeline_prompts=[
        ("intro", intro),
        ("example", example),
        ("start", start),
    ],
)


_p = full_prompt.format(
    character="doctor",
    example_question="What is your name?",
    example_answer="My name is Dr. Watson",
    question="What is your fav food?",
)

print(_p)

#%%

chain = full_prompt | chat

chain.invoke(
    {
        "character": "doctor",
        "example_question": "What is your name?",
        "example_answer": "My name is Dr. Watson",
        "question": "당신의 건강을 위해 무엇을 드시겠습니까?",
    }
)

# %%
