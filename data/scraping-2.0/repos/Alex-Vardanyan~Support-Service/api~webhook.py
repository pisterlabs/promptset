from flask import url_for
from api.controller import Controller
from celery import shared_task  # , chain, signature, current_app
from itsdangerous import URLSafeSerializer, BadData
from config import config
import requests
import json
import openai

url = config.get("ZENDESK_URL")
smooch_url = config.get("SMOOCH_URL")
username = config.get("ZENDESK_USERNAME")
password = config.get("ZENDESK_PASSWORD")
app_id = config.get("ZENDESK_APP_ID")
key_id = config.get("ZENDESK_KEY_ID")
secret_key = config.get("ZENDESK_SECRET_KEY")


# todo refactor code, create Helper function
@shared_task()
def assign(ticket_id, assignee=None):
    uri = f"/api/v2/tickets/{ticket_id}"

    headers = {'Content-Type': 'application/json'}
    auth = (username, password)
    assignee_id = config.get("ASSIGNEE_ID")

    payload = {"ticket": {
        "assignee_id": assignee if assignee is not None else assignee_id}}
    response = requests.put(url + uri, json=payload, auth=auth, headers=headers)

    if response.status_code == 200:
        return True
    else:
        print(f'Error updating ticket {ticket_id}: {response.status_code}', response.text)
        return False


def get_agents():
    uri = "/api/v2/users"
    headers = {'Content-Type': 'application/json'}
    auth = (username, password)
    response = requests.get(url + uri, auth=auth, headers=headers)
    agents = []
    if response.status_code == 200:
        users = json.loads(response.text)["users"]
        for user in users:
            agents.append(user["id"])
        return agents
    else:
        print(f"Couldn't get agents list: {response.status_code}", response.text)
        return False


def get_least_busy_agent():
    agents = get_agents()
    if agents:
        agent_to_be_assigned = agents[0]
        min_tickets_assigned = 0

        headers = {'Content-Type': 'application/json'}
        auth = (username, password)

        for agent in agents:
            uri = f"/api/v2/users/{agent}/related"
            response = requests.get(url + uri, auth=auth, headers=headers)
            if response.status_code == 200:
                ticket_count = json.loads(response.text)["user_related"]["assigned_tickets"]
                if agents.index(agent) == 0:
                    min_tickets_assigned = ticket_count
                    continue
                if ticket_count < min_tickets_assigned:
                    agent_to_be_assigned = agent

        return agent_to_be_assigned


def conversation_reply(conversation_id, answer):
    uri = f"/v2/apps/{app_id}/conversations/{conversation_id}/messages"
    headers = {'Content-Type': 'application/json'}
    auth = (key_id, secret_key)
    payload = {
        "author": {
            "type": "business",
            "displayName": "Atlas"
        },
        "content": {
            "type": "text",
            "text": answer if answer else "this is test response",
            "actions": [
                {
                    "type": "reply",
                    "text": "Assign to Agent",
                    "payload": "/call_support_agent"
                },
                {
                    "type": "reply",
                    "text": "Mark as Solved",
                    "payload": "/mark_solved"
                }
            ]
        }
    }
    response = requests.post(smooch_url + uri, json=payload, auth=auth, headers=headers)
    if response.status_code == 201:
        return True
    else:
        print(f'Error updating conversation {conversation_id}: {response.status_code}', response.text)
        return False


def get_conversation_id(ticket_id):
    uri = f"/api/v2/tickets/{ticket_id}/audits"
    headers = {'Content-Type': 'application/json'}
    auth = (username, password)
    response = requests.get(url + uri, auth=auth, headers=headers)
    if response.status_code == 200:
        print("testing conversation id", json.loads(response.text))
        audits = json.loads(response.text)["audits"]
        for audit in audits[::-1]:
            try:
                events = audit["events"]
                for event in events:
                    return event["value"]["conversation_id"]
            except Exception as e:
                print(str(e))
                pass
    else:
        print(f"Couldn't get {ticket_id} ticket's conversation id")
        return False


def get_conversation(conversation_id):
    uri = f"/v2/apps/{app_id}/conversations/{conversation_id}/messages"
    headers = {'Content-Type': 'application/json'}
    auth = (key_id, secret_key)
    response = requests.get(smooch_url + uri, auth=auth, headers=headers)
    if response.status_code != 200:
        print(f"Couldn't get messages for conversation {conversation_id}", response.text)
    else:
        data = json.loads(response.text)["messages"]
        print("data", data)
        conversation = []
        for event in data:
            try:
                if event["content"]["type"] == "text":
                    conversation.append({"role": "assistant" if event["author"]["displayName"] == "Atlas" else "user",
                                         "content": event["content"]["text"]})
            except Exception:
                pass
        print(conversation)
        return conversation, data[-1].get("content", {}).get("payload", None)


def get_answer(conversation):  # todo change to autogpt trained on 10web's help center
    token = config.get("OPENAI_SECRET_KEY")
    openai.api_key = token

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )

    answer = completion.choices[0].message["content"]
    return answer if answer else False


def message_reply(ticket_id, answer):
    uri = f"/api/v2/tickets/{ticket_id}.json"
    headers = {'Content-Type': 'application/json'}
    auth = (username, password)
    payload = {"ticket": {'comment': {
        "html_body": answer if answer else "this is test response",
        "author_id": config.get("ASSIGNEE_ID"),
        'public': True}
    }}
    response = requests.put(url + uri, json=payload, auth=auth, headers=headers)
    if response.status_code != 200:
        print(f'Error creating comment: {response.status_code} {response.text}')
        return False
    return True


@shared_task()
def reply(ticket_id, via="email"):
    if via == "email":
        messages = get_messages(ticket_id)
        print(messages, ticket_id, via)
        if not messages:
            print(f"Couldn't reply on ticket {ticket_id} because of failed retrieval of the conversation")
            return False
        latest_message = messages[0]["content"]
        latest_responder = messages[0]["role"]
        if len(messages) == 0 or latest_responder == config.get("ASSIGNEE_ID"):
            return None  # so it doesn't reply on its own reply
        elif latest_message == "Assign to Agent":
            assignee = get_least_busy_agent()
            assign.s(ticket_id=ticket_id, assignee=assignee).apply_async()
            return True
        elif latest_message == "Mark as Solved":
            mark_as.s(ticket_id=ticket_id, status="solved").apply_async()
            return True
        else:
            answer = get_answer(messages)

            answer = answer.replace('\n', '<br>')
            answer = "<p>" + answer + "</p>"

            footer = "<p>" + (f"""

Best regards,

Atlas, 10Web's AI Support Companion

You can always connect with Agent

<a href={generate_token(ticket_id, "reassign", "main.agent_assign")} style="background-color: #333; color: white; padding: 10px 20px; border-radius: 20px; text-decoration: none;"><b>Connect me with Support Agent</b></a> 

If your ticket is solved mark it as Solved 

<a href={generate_token(ticket_id, "mark_solved", "main.mark_solved")} style="background-color: #333; color: white; padding: 10px 20px; border-radius: 20px; text-decoration: none;"><b>Mark as Solved</b></a>""").replace(
                '\n', '<br>') + "</p>"

            answer = answer + footer
            print("testing testing testing", answer)

            if message_reply(ticket_id, answer):
                return True
            else:
                print(f"Couldn't reply with message for ticket {ticket_id}")
                return False

    else:
        conversation_id = get_conversation_id(ticket_id)
        if conversation_id:
            conversation, last_payload = get_conversation(conversation_id)
            if conversation[-1]["role"] == "assistant":
                return None
            latest_message = conversation[-1]["content"]
            if latest_message == "Assign to Agent" or last_payload == "/call_support_agent":
                assignee = get_least_busy_agent()
                assign.s(ticket_id=ticket_id, assignee=assignee).apply_async()
                return True
            elif latest_message == "Mark as Solved" or last_payload == "/mark_solved":
                mark_as.s(ticket_id=ticket_id, status="solved").apply_async()
                return True

            answer = get_answer(conversation)

            if not conversation_reply(conversation_id, answer):
                print(f"Error creating comment in conversation: {ticket_id} {conversation_id}")
                return False
            if not message_reply(ticket_id, answer):
                print(f"Error creating comment in ticket: {ticket_id} {conversation_id}")
                return False
            return True


def get_messages(ticket_id):
    uri = f"/api/v2/tickets/{ticket_id}/comments"
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.request(
        "GET",
        url + uri,
        auth=(username, password),
        headers=headers
    )

    if response.status_code == 200:
        print([[conversation['id'], conversation["body"]] for
               conversation in json.loads(response.text)["comments"]])
        conversation = [[conversation['id'], conversation["body"]] for
                        conversation in json.loads(response.text)["comments"]]

        messages = []
        for index in range(len(conversation)):
            messages.append({
                "role": "assistant" if conversation[index][1] == config.get("ASSIGNEE_ID") else "user",
                "content": conversation[index][1]})
        print(messages, "testing the messages")
        return messages
    else:
        print(f"Couldn't get ticket data for ticket {ticket_id}", response.text)
        return False


@shared_task()
def mark_as(ticket_id, status):
    uri = f"/api/v2/tickets/{ticket_id}"

    headers = {'Content-Type': 'application/json'}
    auth = (username, password)

    payload = {"ticket": {
        "status": status}}
    response = requests.put(url + uri, json=payload, auth=auth, headers=headers)

    if response.status_code == requests.codes.ok:
        return True
    else:
        print(f'Error setting status for ticket {ticket_id}: {response.status_code}', response.text)
        return False


def generate_token(ticket_id, salt, route):
    s = URLSafeSerializer(config.get("SECRET_KEY"), salt=salt)
    token = s.dumps(ticket_id)
    email_url = url_for(route, token=token, _external=True)
    print("generated")
    return email_url


class Webhook(Controller):

    def __init__(self, request):
        super().__init__(request)

    def handle_zendesk_webhook(self):
        contents = self.request_json
        try:
            if contents["type"] == "ticket_created":
                print("ticket_created")
                assign.s(ticket_id=contents['ticket_id']).apply_async()
                print("done1")
                reply.s(
                    ticket_id=contents['ticket_id'], via=contents["ticket_via"]) \
                    .apply_async()
                print("done2")
            if contents["type"] == "ticket_updated":
                if contents['assignee_id'] != config.get("ASSIGNEE_ID") or contents["ticket_status"] in ["solved",
                                                                                                         "closed",
                                                                                                         "pending"]:
                    pass  # todo IDK yet maybe delete some logs in mongo
                # todo just forget about this ticket
                else:
                    reply.s(
                        ticket_id=contents['ticket_id'], via=contents["ticket_via"]) \
                        .apply_async()
            return "OK", 200
        except Exception as e:
            print(str(e))
            return "something went wrong, IDK", 500

    def agent_assign(self):
        token = self.request.view_args.get('token')
        s = URLSafeSerializer(config.get("SECRET_KEY"), salt='reassign')

        try:
            ticket_id = s.loads(token)
        except BadData:
            return "Invalid Token", 400

        assignee = get_least_busy_agent()
        assign.s(ticket_id=ticket_id, assignee=assignee).apply_async()
        return "You'll be contacted with our support agent right away", 200

    def mark_solved(self):
        token = self.request.view_args.get('token')
        s = URLSafeSerializer(config.get("SECRET_KEY"), salt='mark_solved')

        try:
            ticket_id = s.loads(token)
        except BadData:
            return "Invalid Token", 400

        mark_as.s(ticket_id=ticket_id, status="solved").apply_async()  # todo closed?
        return "Thank you, your ticket is now marked as Solved", 200

    def track_ticket(self):  # todo to be added, flow-builder
        pass  # todo maybe only reply to chats from here, maybe add to mongo to track, ticket merging possibility
