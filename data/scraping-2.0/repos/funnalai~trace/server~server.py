import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from preprocessing import return_relevant_document_context
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sources.slack import get_slack_data
from dotenv import load_dotenv
from prisma import Prisma
from sources.linear import get_linear_data
from sources.db_utils import connect_db
from utils.classifier import get_conv_classification
from utils.s3 import upload_image_to_s3
from datetime import datetime
from utils.embeddings import get_embeddings, str_to_np_embed
from utils.summary import get_meta_summary
from views.graphs import view_time_conversations, vis_convos
import json
import os
import numpy as np
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()


app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def replace_ids_with_names(conversation_summary):
    # currently names are the ids, as 1, 2 and so on. We need to replace these with the actual names
    # regex for every number
    conversation_regex = re.compile(r'(\d+)')
    # for every regex
    for match in conversation_regex.finditer(conversation_summary):
        # get the id
        id = match.group(1)
        # get the name
        name = await get_name_for_id(id)
        # replace the id with the name
        if name is not None:
            conversation_summary = conversation_summary.replace(
                f'{id}', f'{name}')
    return str(conversation_summary)


class CompletionRequestBody(BaseModel):
    context: str


def extract_stream(file: UploadFile = File(...)):
    pdf_as_bytes = file.file.read()
    # We convert the bytes into a Streamable object of bytes
    return io.BytesIO(pdf_as_bytes)


@app.get("/")
async def root():
    try:
        db = Prisma()
        await db.connect()
        # post = await db.user.create({})

        return {"hello": "world"}
    except Exception as ex:
        print(ex)
        return {"error": "yes"}

hardcoded_names = {"1": "Will", "2": "Arushi",
                   "3": "Amir", "4": "Harrison"}


async def get_name_for_id(id):
    user = hardcoded_names.get(id, None)
    if user is None:
        return None
    return user


@app.get("/chat")
async def chat(id: str, input: str):
    try:
        db = await connect_db()
        user = await db.user.find_first(where={"id": int(id)}, include={"processedConversations": True})
        conversations = user.processedConversations
        embedding_strs = [
            conversation.embedding for conversation in conversations]
        embeddings = [str_to_np_embed(embedding_str)
                      for embedding_str in embedding_strs]

        input_embed = get_embeddings(input)
        # Find the most similar conversation
        similarities = [np.dot(input_embed, embedding)
                        for embedding in embeddings]
        top_indices = sorted(range(len(similarities)),
                             key=lambda i: similarities[i], reverse=True)[:5]
        max_similarity = max(similarities)
        max_similarity_index = similarities.index(max_similarity)
        # Get the conversation
        conversation = conversations[max_similarity_index]

        # get the conversations corresponding to the top_indices
        top_conversations = [conversations[index] for index in top_indices]

        context = ""
        for top_conv in top_conversations:
            context += top_conv.summary + "\n"

        # For every id in the conversation summary, replace with name using get_name_for_id
        conversation_summary = await replace_ids_with_names(conversation.summary)
        context_summary = await replace_ids_with_names(context)

        # make a call to gpt4 to get the answer with the context
        llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                     temperature=0.7, max_tokens=500)
        prompt = f"Using the following context, answer the provided question using only the context. If you don't know the answer, say you don't know, do not make anything up. Context: {context_summary}\n Question: {input}\n Answer: "
        # make llm call
        response = llm(prompt)

        return {"conversation": conversation, "summary": conversation_summary, "completion": response}

    except Exception as ex:
        print(ex)
        raise HTTPException(status_code=400, detail="Error getting user")


async def parse_processed_conversation(conv):
    """
    Parse a processed conversation object from the database into a dictionary
    """

    if conv.projectId is None:
        projectId = conv.projectId = -1
    else:
        projectId = conv.projectId

    # get tags from summary for prettier visualizations
    return {
        "id": conv.id,
        # await replace_ids_with_names(conv.summary),
        "summary": await replace_ids_with_names(conv.summary),
        "embedding": str_to_np_embed(conv.embedding),
        "startTime": conv.startTime.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "endTime": conv.endTime.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "projectId": projectId,
        "slackUrl": conv.slackUrl,
    }


@app.get("/user")
async def get_user(id: str):
    try:
        db = await connect_db()
        user = await db.user.find_first(where={"id": int(id)})
        raw_messages = await db.rawmessage.find_many(where={"userId": int(id)}, include={"processedConversations": True})
        # fetch the processed conversations from the raw message proccsed conversation field above

        processed_conversations = []

        for message in raw_messages:
            processed_conv = await parse_processed_conversation(message.processedConversations)
            processed_conversations.append(processed_conv)

        cluster_graph_link = vis_convos(processed_conversations, user.name)
        # print("before")
        time_graph_link = await view_time_conversations(
            processed_conversations, user.name, db)
        # print("generated time graph")
        # print("generated db scan")

        # time_graph_link = await upload_image_to_s3(time_graph, os.getenv("S3_BUCKET"), f"""{user.name}-time-{datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%fZ")}.png""")
        # clusters_graph_link = await upload_image_to_s3(cluster_graph, os.getenv("S3_BUCKET"), f"""{user.name}-clusters-{datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%fZ")}.png""")
        # # copy user into a new dictionary and add the two properties above
        userObj = user.dict()
        userObj['timeGraphHTML'] = time_graph_link
        userObj['clustersGraph'] = cluster_graph_link
        return userObj  # json.dumps(processed_conversations)
    except Exception as ex:
        print(ex)
        raise HTTPException(status_code=400, detail="Error getting user")


@app.get("/users")
async def get_users():
    try:
        db = await connect_db()
        users = await db.user.find_many()
        return users
    except Exception as ex:
        print(ex)
        raise HTTPException(status_code=400, detail="Error getting users")


@app.get("/slack")
async def slack():
    try:
        data = await get_slack_data()
        return data
    except Exception as ex:
        print(ex)
        raise HTTPException(status_code=400, detail="Slack API call failed")


@app.get("/populate-all")
async def populate():
    try:
        await linear()
        await slack()
        await map_slack_to_linear()
    except Exception as ex:
        print("Error populating all data: ", ex)
        raise ex


@app.get("/linear")
async def linear():
    """
    Fetch data from the linear API, write users and tickets.
    """
    async def add_users_to_db_from_issues(db, issues):
        """
        Given a list of issues from Linear, add all users to the database
        that are not already in the database.
        """
        for issue in issues:
            if 'assignee' in issue and issue['assignee']:
                assignee = issue['assignee']
                try:
                    # We haven't seen this linear user before
                    userByLinearId = await db.user.find_first(where={"linearId": assignee['id']})
                    # We haven't seen this slack user (same email) before
                    userByEmail = await db.user.find_first(where={"email": assignee['email']})
                    if not userByLinearId and not userByEmail:
                        user = await db.user.create({"linearId": assignee['id'], "name": assignee['name'], 'email': assignee['email']})
                except Exception as ex:
                    print(ex)
                    return {"status": 400, "error": f"User creation failed {assignee['id']}"}
        return {"status": 200, "success": True, "message": "Users added to database"}

    async def add_projects_to_db(db, issues):
        """
        Given a list of issues from Linear, add all projects to the database
        """
        for issue in issues:
            if 'project' in issue and issue['project']:
                project = issue['project']
                try:
                    # We haven't seen this project before
                    projectStr = project['name']
                    projectByStr = await db.project.find_first(where={'name': projectStr})
                    if not projectByStr:
                        project = await db.project.create({"name": projectStr})
                except Exception as ex:
                    print(ex)
                    return {"status": 400, "error": f"Project creation failed {projectStr}"}
        return {"status": 200, "success": True, "message": "Projects added to database"}

    async def get_user_from_issue(db, issue):
        """
        Given an issue from Linear, get the database user id for the linear id
        """
        # Get the database user id for the linear id
        if 'assignee' in issue and issue['assignee']:
            assignee = issue['assignee']
            user = await db.user.find_first(where={"linearId": assignee['id']})
        else:
            user = None
        return user

    async def get_project_from_issue(db, issue):
        """
        Given an issue from Linear, get the project name
        """
        if 'project' in issue and issue['project']:
            project = issue['project']
            projectStr = project['name']
            dbProject = await db.project.find_first(where={'name': projectStr})
            if dbProject:
                return dbProject
        return None

    async def add_ticket_to_user(ticket, user):
        """
        Given a ticket and a user, add the ticket to the user's tickets
        """
        if user:
            await db.user.update(where={"id": user.id}, data={"tickets": {"connect": {"id": ticket.id}}})

    async def add_ticket_to_project(ticket, project):
        """
        Given a ticket and a project, add the ticket to the project's tickets
        """
        if project:
            await db.project.update(where={"id": project.id}, data={"tickets": {"connect": {"id": ticket.id}}})

    async def add_project_to_user(project, user):
        """
        Given a project and a user, add the project to the user's projects
        """
        if user and project:
            await db.user.update(where={"id": user.id}, data={"projects": {"connect": {"id": project.id}}})

    db = await connect_db()
    if not db:
        return {"status": 400, "error": "Database connection failed"}

    data = get_linear_data()
    # do all sanity checks
    if data['status'] != 200:
        return data

    # get all issues
    issues = data['data']['issues']['nodes']

    # For every issue, check if the assignee exists in the database.
    # If not, create a new user.
    response = await add_users_to_db_from_issues(db, issues)
    if response['status'] != 200:
        return response

    response = await add_projects_to_db(db, issues)
    if response['status'] != 200:
        return response

    # add all issues to the database
    for issue in issues:
        try:
            # If ticket with linearId already exists, skip
            ticket = await db.ticket.find_first(where={"linearId": issue['id']})
            if ticket:
                continue

            user = await get_user_from_issue(db, issue)
            project = await get_project_from_issue(db, issue)
            await add_project_to_user(project, user)

            projectId = project.id if project else None

            query = {
                "linearId": issue['id'],
                "title": issue['title'],
                "description": issue['description'],
                "createdAt": datetime.strptime(issue['createdAt'], "%Y-%m-%dT%H:%M:%S.%fZ"),
            }
            if projectId is not None:
                query['projectId'] = projectId
            if user is not None:
                query['userId'] = user.id

            # Create the issue
            ticket = await db.ticket.create(query)
            await add_ticket_to_user(ticket, user)
            await add_ticket_to_project(ticket, project)

        except Exception as ex:
            print(ex)
            return {"status": 400, "error": f"Issue creation failed {issue['id']}"}

    return {"status": 200, "success": True, "message": "Issues added to database"}


@app.get("/map-slack-to-linear")
async def map_slack_to_linear():
    async def get_project_for_conv(conversation, projects):
        """
        Given a conversation, get the project it belongs to
        """
        convStr = conversation.summary
        projNames = [proj.name for proj in projects]
        projClassName = get_conv_classification(convStr, projNames).strip()
        if projClassName == "None":
            return None
        else:
            # Get the project for projects with the name projClassName
            project = None
            for proj in projects:
                if proj.name == projClassName:
                    project = proj
                    break
            return project

    # Get all users from the database
    db = await connect_db()
    if not db:
        return {"status": 400, "error": "Database connection failed"}

    # Get users with all the processed conversations relations
    users = await db.user.find_many(include={"processedConversations": True})
    projects = await db.project.find_many()
    for user in users:
        # Get all conversations for the user, that is, where user is in the users list
        conversations = user.processedConversations
        if not conversations:
            continue
        for conversation in conversations:
            classified_proj = await get_project_for_conv(conversation, projects)
            print("classified_proj", classified_proj)
            if classified_proj:
                # if a previous projectId exists on the conversation, remove it
                if conversation.projectId:
                    await db.project.update(where={"id": conversation.projectId}, data={"messages": {"disconnect": {"id": conversation.id}}})
                await db.processedconversation.update(where={"id": conversation.id}, data={"projectId": classified_proj.id})
                await db.project.update(where={"id": classified_proj.id}, data={"messages": {"connect": {"id": conversation.id}}})
