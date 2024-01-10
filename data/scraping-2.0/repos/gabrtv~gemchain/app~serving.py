from langchain_google_genai import ChatGoogleGenerativeAI

def serve_model(prompt):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    return llm.invoke(prompt)