from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain.tools.gmail import GmailSearch
from langchain.agents.agent_toolkits import GmailToolkit
from langchain.agents import AgentExecutor,StructuredChatAgent
from langchain.output_parsers.fix import OutputFixingParser
from langchain.agents.structured_chat.output_parser import StructuredChatOutputParser,StructuredChatOutputParserWithRetries
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.getenv('GOOGLE_API_KEY'),temperature=0)


# Credentials and API resources
credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

agent = StructuredChatAgent(
    llm_chain=LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate(
            input_variables=['agent_scratchpad', 'input'],
            messages=[
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=['agent_scratchpad', 'input'],
                        template='Respond to the human as helpfully and accurately as possible. '
                                 'You have access to the following tools:\n\n'
                                 '1. create_gmail_draft: Use this tool to create a draft email with the provided message fields.\n'
                                 '   Args:\n'
                                 '   - message: The message to include in the draft. (Type: string)\n'
                                 '   - to: The list of recipients. (Type: array of strings)\n'
                                 '   - subject: The subject of the message. (Type: string)\n'
                                 '   - cc: The list of CC recipients. (Type: array of strings)\n'
                                 '   - bcc: The list of BCC recipients. (Type: array of strings)\n\n'
                                 '2. send_gmail_message: Use this tool to send email messages.\n'
                                 '   Args:\n'
                                 '   - message: The message to include in the email. (Type: string)\n'
                                 '   - to: The list of recipients. (Type: array of strings)\n'
                                 '   - subject: The subject of the email. (Type: string)\n'
                                 '   - cc: The list of CC recipients. (Type: array of strings)\n'
                                 '   - bcc: The list of BCC recipients. (Type: array of strings)\n\n'
                                 '3. search_gmail: Use this tool to search for email messages or threads.\n'
                                 '   Args:\n'
                                 '   - query: The Gmail query. Example filters include from:sender, to:recipient, subject:subject, ...\n'
                                 '   - resource: Whether to search for threads or messages. (Type: string, default: "messages")\n'
                                 '   - max_results: The maximum number of results to return. (Type: integer, default: 10)\n\n'
                                 '4. get_gmail_message: Use this tool to fetch an email by message ID.\n'
                                 '   Args:\n'
                                 '   - message_id: The unique ID of the email message. (Type: string)\n\n'
                                 '5. get_gmail_thread: Use this tool to search for email messages.\n'
                                 '   Args:\n'
                                 '   - thread_id: The thread ID. (Type: string)\n\n'
                                 'Use a JSON blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\n\n'
                                 'Valid "action" values: "Final Answer" or create_gmail_draft, send_gmail_message, search_gmail, get_gmail_message, get_gmail_thread\n\n'
                                 'Provide only ONE action per $JSON_BLOB, as shown:\n\n'
                                 '{{\n  "action": $TOOL_NAME,\n  "action_input": $INPUT\n}}\n\n'
                                 'Follow this format:\n\n'
                                 'Question: input question to answer\n'
                                 'Thought: consider previous and subsequent steps\n'
                                 'Action:\n'
                                 '```\n$JSON_BLOB\n```\n'
                                 'Observation: action result\n'
                                 '... (repeat Thought/Action/Observation N times)\n'
                                 'Thought: I know what to respond\n'
                                 'Action:\n'
                                 '```\n{{\n  "action": "Final Answer",\n  "action_input": "Final response to human"\n}}\n```\n\n'
                                 'Begin! Reminder to ALWAYS respond with a valid JSON blob of a single action. '
                                 'Use tools if necessary. Respond directly if appropriate. '
                                 'Format is Action:```$JSON_BLOB```then Observation:.\nThought:'
                                 '{input}\n\n{agent_scratchpad}'
                    )
                )
            ],
            
        )
    ),
    output_parser=StructuredChatOutputParserWithRetries(
        output_fixing_parser=OutputFixingParser(
            parser=StructuredChatOutputParser(),
            retry_chain=LLMChain(
                prompt=PromptTemplate(
                    input_variables=['completion', 'error', 'instructions'],
                    template='Instructions:\n--------------\n{instructions}\n--------------\n'
                            'Completion:\n--------------\n{completion}\n--------------\n\n'
                            'Above, the Completion did not satisfy the constraints given in the Instructions.\n'
                            'Error:\n--------------\n{error}\n--------------\n\n'
                            'Please try again. Please only respond with an answer that satisfies '
                            'the constraints laid out in the Instructions:'
                ),
                llm=llm
            )
        )
    ),
    allowed_tools=[
        'create_gmail_draft',
        'send_gmail_message',
        'search_gmail',
        'get_gmail_message',
        'get_gmail_thread'
    ]
)

agent_executor = AgentExecutor(agent=agent, tools=toolkit.get_tools(),llm=llm,verbose=True,handle_parsing_errors=True)



def email(user_message):
    return agent_executor.run({"input":user_message})