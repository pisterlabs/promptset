import os
import textwrap
import telebot

from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
bot = telebot.TeleBot("6030967124:AAFDZUXO6VY3Udmf0BKRApW-fZ5LmGeiSLw")

loader = TextLoader("data.txt")
document = loader.load()

print(document)

#Preprocessing


def wrap_text_preserve_newlines(text, width=110):
    #Split the input text into lines based on newline characters
    lines = text.split('\n')

    #Wrap each line individually
    wrapped_lines = [textwrap. fill(line, width=width) for line in lines]

    #Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

#print(wrap_text_preserve_newlines(str(document[0])))

#Text Splitting
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(document)

#print(len(docs))
#print(docs[0])

#Embedding
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

#doc = db.similarity_search(query)
#print(wrap_text_preserve_newlines(str(doc[0].page_content)))

llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.8, "max_length": 512})

chain = load_qa_chain(llm, chain_type="stuff")

# while True:
#     query = input("Enter the query (or type 'quit' to exit): ")
#
#     if query.lower() == 'quit':
#         break
#
#     docsResult = db.similarity_search(query)
#     print(chain.run(input_documents=docsResult, question=query))


@bot.message_handler(content_types=['text'])
def handle_text(message):
    if(message.text == "/start"):
        return
    mytext = message.text;
    docsResult = db.similarity_search(mytext)
    # if("-/-" in message.text):
        # mytext = ": " + mytext;
    # response = openai.Completion.create(
    #     engine="text-davinci-003",
    #     prompt=(f"{mytext}\n"),
    #     max_tokens=600,
    #     n=1,
    #     stop=None,
    #     temperature=0.5,
    # )
    # if (message.text == "/city"):

    response = chain.run(input_documents=docsResult, question=mytext)
    bot.send_message(message.chat.id, response)
    #print(message)
    print(message.chat.id, ":@", message.from_user.username, ": ", message.from_user.first_name, " : ", message.text)

    # if(message.chat.id !=468004165):
    #     a = message.from_user.first_name
    #     a+=": "
    #     a+=message.text
    #     bot.send_message(468004165, a)
bot.polling()