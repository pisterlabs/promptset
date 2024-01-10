from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

chat_llm = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0.9, max_tokens=1000, verbose=True)

systemMessage = "Answer the request of the user. Respond in three parts formatted as JSON: \n" \
"prompt_summary: a summary of the last user message in the prompt\n" \
"full_response: the full text response to the user as normally generated\n" \
"response_summary: a summary of the full text response\n" \
"The full response will be displayed for the user, while the prompt and response summaries will be used "\
"as a compressed representation of the conversation for subsequent prompts - so they have to capture "\
"important details and facts that will give the model enough context to generate subsequent responses "\
"without access to the full conversation history."

promptSummarySchema = ResponseSchema(name = "prompt_summary", 
                description = "A brief summary of the essence of the last user message in the prompt")
fullResponseSchema = ResponseSchema(name = "full_response",
                description = "The full text response to the user as normally generated")
responseSummarySchema = ResponseSchema(name = "response_summary",
                description = "A brief summary of the essence of the response")
responseSchemas = [promptSummarySchema, fullResponseSchema, responseSummarySchema]
outputParser = StructuredOutputParser.from_response_schemas(responseSchemas)

messages = [SystemMessage(content = systemMessage)]
lastUserInputSummary = ""
userInput = ""
promptNum = 0

while userInput != "quit":
    userInput = input("Enter your input: ")
    if userInput == "quit":
            print("Bot: Goodbye")
            break
    userInput += "(remember to provide a JSON response)"
    messages.append(HumanMessage(content = userInput))

    promptNum += 1
    print("------------------")
    print(f"Prompt {promptNum}:")
    for message in messages:
        print(f"{message.type}: {message.content}")
    print("------------------")

    result = chat_llm(messages)
    print(result.content)
    responseDict = outputParser.parse(result.content)

    print("Bot: " + responseDict["full_response"])

    messages.pop() # remove the full user input
    messages.append(HumanMessage(content = responseDict["prompt_summary"]))
    messages.append(AIMessage(content = responseDict["response_summary"]))
