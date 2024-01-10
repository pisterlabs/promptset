import logging
from src.lib import lib_emaildb
from src.lib import lib_logging
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.lib import lib_twitterdb
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
import dotenv

dotenv.load_dotenv()
# setup_logging()
# logger = get_logger()
lib_twitterdb.set_company_environment('TWITTER')
db = lib_twitterdb.get_docdb()
llm = lib_twitterdb.get_llm()

def format_tweet_docs(docs):
    return "\n\n".join([format_tweet_doc(doc) for doc in docs])

def format_tweet_doc(doc):
    # Extracting metadata
    tweet_id = doc.metadata.get('tweet_id', 'N/A')
    tweet_text = doc.metadata.get('tweet_text', 'N/A')

    # Formatting the tweet document as a Markdown string
    tweet_details = []
    tweet_details.append("### Tweet\n")
    tweet_details.append(f"**Tweet ID:** {tweet_id}\n")
    tweet_details.append("\n**Content:**\n")
    tweet_details.append("```\n")
    tweet_details.append(tweet_text)
    tweet_details.append("```\n")

    return '\n'.join(tweet_details)


metadata_field_info = [
    AttributeInfo(
        name="tweet_id",
        description="A unique identifier for the tweet.",
        type="string",
    )
]

tweet_retriever = SelfQueryRetriever.from_llm(
    llm,
    db,
    "Tweets I wrote",
    metadata_field_info,
    search_kwargs={"k": 10}
)

def write_tweet(input):
    write_prompt = ChatPromptTemplate.from_template("""# Write Tweet

## Instructions
- Create a tweet based on the provided content.
- The tweet should mirror the style and tone of the sample tweets, emphasizing a direct and concise communication style.
- Avoid overly enthusiastic language, fluff, or filler content. The tweet should be to the point and professional.
- Use emojis and hashtags only if they align with the style of the provided samples.
- The tweet should be engaging yet succinct, avoiding unnecessary elaboration.
- Do not include any greetings, simply start the content of the message.
- Maintain a candid and casual tone, avoiding expressions of exaggerated enthusiasm (e.g., 'thrilled', 'excited').
- Minimize the use of exclamations.
- No hash tags
- Avoid statements that imply grandiosity or hype, such as 'we're onto something big.'
- Notes for content can be transcribed audio, a collection of random notes, or any combination thereof, not necessarily in a specific order.

## Section 1: Tone and Style Examples
{tweets}

## Section 2: Content for This Tweet
{content}
""")
    print(f"Writing tweet from notes: {input}")

    chain = (
        {
            "tweets": tweet_retriever | format_tweet_docs,
            "content": RunnablePassthrough()
        }
        | write_prompt
        | llm 
        | StrOutputParser()
    )

    res = chain.invoke({
        'content': input
    }, config={
    })

    print(f"Result: '{res}'")

def main():
    bindings = KeyBindings()

    while True:
        multiline = False

        while True:
            try:
                if not multiline:
                    line = prompt('Enter text (""" for multiline, "quit" to exit, Ctrl-D to end): ', key_bindings=bindings)
                    if line.strip() == '"""':
                        multiline = True
                        continue
                    elif line.strip().lower() == 'quit':
                        return  # Exit the CLI
                    else:
                        write_tweet(line)
                        break
                else:
                    line = prompt('Enter tweet content: ', multiline=True, key_bindings=bindings)
                    write_tweet(line)
                    multiline = False
            except EOFError:
                return
            
if __name__ == "__main__":
    print(f"DID YOU UPDATE THE TWITTER DB?")
    main()