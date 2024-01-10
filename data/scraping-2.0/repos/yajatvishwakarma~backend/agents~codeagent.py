from langchain.chat_models import ChatVertexAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import DirectoryLoader
from git import Repo
from langchain.document_loaders import GitLoader
from fastapi import APIRouter
import asyncio
from bson.objectid import ObjectId

from pymongo import MongoClient
router = APIRouter()

client = MongoClient("mongodb://localhost:27017/")
db = client["judgy"]


@router.get("/code-agent")
def codeAgent_endpoint():
    return {"message": "Hello from Code Agent"}


async def invoke_code_agent(repolink: str, project_id: str):
    DIRECTORY = "./projects_source_code/"+project_id
    repo = Repo.clone_from(
        repolink, to_path=DIRECTORY
    )
    branch = repo.head.reference
    loader = GitLoader(repo_path=DIRECTORY, branch=branch)
    llm = ChatVertexAI()
    index = VectorstoreIndexCreator().from_loaders([loader])
    # Get theme from hackathon collection
    technologies = ""
    for x in db.hackathons.find():
        technologies = x["technologies"]
        break
    prompt = """
        You are a code reviewer. This is a hackathon project. You have to answer the question about the project.
        Question: {question}
        Rules for answering: 
            1. Remember to answer like a code reviewer.
            2. Answer the question as best you can. If you are unable to answer, say 'I am unsure, I need human assistance' .
            3. You must answer in one paragraph. Do not use formatting.
            4. Your paragraph must not have more than 70 words.
            5. You must analyze all the files in the project.
            6. If you don't know the answer, you must research and answer.
        """

    questionToAsk = [
        "What are the technologies and programming language used in this project?",
        "Explain the project in brief",
        "How is the code quality of this project?",
        "Does the project import and use any of the following dependencies/packages/APIs/libraries : "+technologies + "? ",
    ]
    agentAnswers = []
    for question in questionToAsk:
        response = index.query(question, llm)
        agentAnswers.append(response)

    # Save the answers to the database
    final = []
    for i in range(len(questionToAsk)):
        newval = {"question": questionToAsk[i], "answer": agentAnswers[i]}
        final.append(newval)
    query = {"_id": ObjectId(project_id)}
    newvalues = {"$set": {"codeAgentAnalysis": final}}
    db.projects.update_one(query, newvalues)
    print("Code Agent : Task Complete")
