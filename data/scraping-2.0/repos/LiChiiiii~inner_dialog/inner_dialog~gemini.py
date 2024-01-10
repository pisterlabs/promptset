import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

def gemini_summary(article: str) -> str:
    """Gemini provide bullet list of summary given an article.

    Args:
        article (str): Input article to be summarized.

    Returns:
        str: The bullet list of keyphrases acting as the summary
        of the given article.
    """
    chat = ChatGoogleGenerativeAI(temperature=0.2, model="gemini-pro", convert_system_message_to_human=True)

    messages = [
        SystemMessage(content="You are a good summarizer."),
        HumanMessage(content=
            f""" 
            {article}

            provide a summary of the above text. Your summary 
            should be informative and factual, covering the most 
            important aspects of the text. Use bullet list. be concise 
            of each bullet point, only list keyphrases.
            """
        ),
    ]
    summary_msg = chat(messages)
    return summary_msg.content

def article2hiercc_gemini(article: str, question: str) -> dict:
    """Summarize the article and produce hierarchical concept as dictionary.

    Use Gemini to perform bullet list summary and extract hierarchical
    concept.

    Args:
      article (str): An article from website or system output.
    Returns:
      dict: Hierarchical concept.
    """

    chat = ChatGoogleGenerativeAI(temperature=0.2, model="gemini-pro", convert_system_message_to_human=True)

    summary_msg = chat(
        [
            SystemMessage(content="You are a good summarizer."),
            HumanMessage(content=
                f""" 
                {article}

                provide a summary of the above text. Your summary 
                should be informative and factual, covering the most 
                important aspects of the text. Use bullet list.
                """
            ),
            
        ]
    )

    extract_hiercc_msg = f"""
        {summary_msg}

        Organize the above points into a JSON format  represents the hierarchical 
        organization of the points, with each topic and sub-topic listed as keys, 
        and their corresponding sub-topics as or examples (if any) as their values, 
        if it does not have sub-topic or examples, just put null (not None). 
        Here is an example JSON format:
        
        {{
          "{question}": {{
            "topic": {{
              "sub_topic": {{
                "sub_topic1": null,
                "sub_topic2": why_we_should_follow_this_subtopic,
              }}
            }}
          }}
        }}
        
        Please start your answer with "{" and end with "}"
        """
    

    hiercc_msg = chat.invoke(extract_hiercc_msg)
    hiercc_dict: dict = json.loads(hiercc_msg.content)
    return hiercc_dict

def mult_round_gemini(question: str, round: int = 3) -> str:
    """Directly give Gemini the question. Nothing more.

    Args:
        question (str): The question to ask.

        round (int): How many round to ask Gemini.

    Returns:
        str: Gemini's response.
    """
    ans = ""
    chat =ChatGoogleGenerativeAI(temperature=0.2, model="gemini-pro", convert_system_message_to_human=True)
    messages = [
        SystemMessage(content="You are a good summarizer."),
        HumanMessage(content=f"{question}"),
    ]
    for _ in range(round):
        print("ROUND!")
        msg = chat(messages)
        ans += gemini_summary(msg.content) + "\n---------------\n"
        messages.extend(
            [
                msg,
                HumanMessage(content="Anything else?"),
            ]
        )

    return ans


def vanilla_gemini(question: str) -> str:
    """Directly give Gemini the question. Nothing more.

    Args:
        question (str): The question to ask.

    Returns:
        str: Gemini's response.
    """
    chat = ChatGoogleGenerativeAI(temperature=0.2, model="gemini-pro", convert_system_message_to_human=True)
    response = chat(
        [
            SystemMessage(content="You are a helpful assistent."),
            HumanMessage(content=f"{question}"),
        ]
    )

    return response.content


def t2cb_ask_gemini(question: str) -> str:
    """Ask T2CB question to Gemini.

    Args:
        question (str): Things to consider before ...

    Returns
        str: Gemini's response.
    """
    # response = vanilla_gemini(question)
    response = mult_round_gemini(question, round=3)
    return response


if __name__ == "__main__":
    question = "things to consider before traveling to a foreign country."
    article = t2cb_ask_gemini(question)
    # print(article)
    hierr = article2hiercc_gemini(article=article, question=question)
    hierr_str = json.dumps(hierr, indent=2)
    print(hierr_str)
