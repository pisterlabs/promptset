import logging
from src.lib import lib_emaildb
from src.lib import lib_logging
import argparse
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.lib.lc_logger import LlmDebugHandler

lib_logging.setup_logging()

# lib_logger.set_console_logging_level(logger.ERROR)
logger = lib_logging.get_logger()

def format_email_docs(docs):
    logger.info(f"Formatting docs: {docs}")
    return "\n\n".join([format_email_doc(doc) for doc in docs])

def format_email_doc(doc):
    # Extracting metadata
    email_id = doc.metadata.get('email_id', 'N/A')
    from_address = doc.metadata.get('from_address', 'N/A')
    subject = doc.metadata.get('subject', 'N/A')
    thread_id = doc.metadata.get('thread_id', 'N/A')
    to_address = doc.metadata.get('to_address', 'N/A')

    # Formatting the email document as a Markdown string
    email_details = []
    email_details.append("### Email {subject}\n")
    email_details.append(f"**Email ID:** {email_id}\n")
    email_details.append(f"**Thread ID:** {thread_id}\n")
    email_details.append(f"**Subject:** {subject}\n")
    email_details.append(f"**From:** {from_address}\n")
    email_details.append(f"**To:** {to_address}\n")
    email_details.append("\n**Body:**\n")
    email_details.append("```\n")
    email_details.append(doc.page_content)
    email_details.append("```\n")

    return '\n'.join(email_details)

def print_email_doc(page_content, metadata):
    # Extracting metadata
    email_id = metadata.get('email_id', 'N/A')
    from_address = metadata.get('from_address', 'N/A')
    subject = metadata.get('subject', 'N/A')
    thread_id = metadata.get('thread_id', 'N/A')
    to_address = metadata.get('to_address', 'N/A')

    # Formatting and printing the email document
    print("Email Document Details")
    print("----------------------")
    print(f"Email ID: {email_id}")
    print(f"Thread ID: {thread_id}")
    print(f"Subject: {subject}")
    print(f"From: {from_address}")
    print(f"To: {to_address}")
    print("\nContent:")
    print("----------------------")
    print(page_content)


from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever

metadata_field_info = [
    AttributeInfo(
        name="email_id",
        description="A unique identifier for the email.",
        type="string",
    ),
    AttributeInfo(
        name="from_address",
        description="The email address of the sender.",
        type="string",
    ),
    AttributeInfo(
        name="to_address",
        description="The email addresses of the recipients.",
        type="string",
    ),
    AttributeInfo(
        name="subject",
        description="The subject line of the email.",
        type="string",
    ),
    AttributeInfo(
        name="thread_id",
        description="Identifier for the thread to which this email belongs.",
        type="string",
    )
]


def write_email(message_type, notes):
    db = lib_emaildb.get_docdb()
    llm = lib_emaildb.get_llm()

    email_retriever = SelfQueryRetriever.from_llm(
        llm,
        db,
        "E-mails sent by me",
        metadata_field_info,
        search_kwargs={"k": 10}
    )

    lmd = LlmDebugHandler()

    print(f"Message type: {message_type} notes: {notes}")
    write_prompt = ChatPromptTemplate.from_template("""# Write a {message_type}

## Instructions
- Create a {message_type} focused on the provided content.
- Use a style and tone consistent with the medium of the {message_type}.
- Maintain a candid and casual tone, avoiding expressions of exaggerated enthusiasm (e.g., 'thrilled', 'excited').
- Minimize the use of exclamations.
- Avoid statements that imply grandiosity or hype, such as 'we're onto something big.'
- Do not include motivational or team-building statements, especially in the closing.
- Notes for content can be transcribed audio, a collection of random notes, or any combination thereof, not necessarily in a specific order.

## Section 1: Tone and Style Examples
{emails}

## Section 2: Content
{notes}
""")

    chain = (
        {
            "emails": email_retriever | format_email_docs,
            "message_type": RunnablePassthrough(),
            "notes": RunnablePassthrough()
        }
        | write_prompt
        | llm 
        | StrOutputParser()
    )


    res = chain.invoke({
        'message_type': message_type,
        'notes': notes
    }, config={
    })

    print(f"Result: '{res}'")


def main():
    bindings = KeyBindings()

    while True:
        try:
            message_type = prompt('What are you writing? ("quit" to exit, Ctrl-D to end): ', key_bindings=bindings)
            # Multiline input modkwe
            notes = prompt('Enter notes: ', multiline=True, key_bindings=bindings)
            write_email(message_type.strip(), notes.strip())
        except EOFError:
            return
            
if __name__ == "__main__":
    print(f"DID YOU UPDATE THE EMAIL DB?")
    parser = argparse.ArgumentParser(description='Run the Gmail pipeline for a specific company.')
    parser.add_argument('company', choices=['cj', 'fc'], help='Specify the company environment ("cj" or "fc").')
    args = parser.parse_args()

    lib_emaildb.set_company_environment(args.company.upper())
    main()