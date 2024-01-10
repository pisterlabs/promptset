from flask import Flask, request, jsonify
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
from langchain import PromptTemplate, HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.schema.messages import AIMessage, HumanMessage

app = Flask(__name__)

template = """The following is a chat between a person facing some mental health problems, and is comfortable talking to an AI assistant, answer informally and respectfully, 
suggest them things to do
ask them more and more questions about their mental health

Current conversation:
{history}
Human: {input}
AI Assistant:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

repo_id="google/flan-t5-xxl"
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 128}
)

# print(llm_chain.run(question))

conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm, 
    verbose=False, 
    memory=ConversationBufferMemory()
)


@app.route('/converse', methods=['POST'])
def converse():
    query = request.json.get('input')
    response = conversation.predict(input=query)
    history = conversation.memory.chat_memory.messages
    if len(history)>10:
        history = history[-10:]
    print(len(history))
    last_ai_message = next(
        reversed([message for message in history if isinstance(message, AIMessage)])
    )
    reply = last_ai_message.content
    return jsonify(response=reply)

if __name__ == '__main__':
    app.run(debug=True)

