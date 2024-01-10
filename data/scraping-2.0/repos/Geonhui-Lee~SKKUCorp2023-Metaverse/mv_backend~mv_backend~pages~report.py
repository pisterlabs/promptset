from mv_backend.settings import OPENAI_API_KEY
from django.shortcuts import render
import json
import openai

from mv_backend.lib.database import Database
from mv_backend.lib.common import gpt_model_name

from mv_backend.pages.report_test import ReportTest

class Report:
    npc_conversation_dict = {}
    npc_retrieve_dict = {}
    npc_reflect_dict = {}
    
    npc_normalized_retrieve_dict = {}
    npc_normalized_reflect_dict = {}

def get_parameter(request, name, default_value):
    parameter_value = request.POST.get(name)
    if parameter_value is not None:
        return parameter_value
    else:
        return default_value

def get_llm_content(llm_messages):
    client = openai(
        api_key=OPENAI_API_KEY,
    )
    llm_response = client.chat.completions.create(
        model=gpt_model_name,
        messages=llm_messages
    )
    
    # convert llm_response to json
    print(dict(llm_response))

    #llm_response_demo = {'id': 'chatcmpl-8LmqNeOHpGo7Ujp6EOpaxMMksvIg6', 'choices': [Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content='Reported Evaluation for User0\'s English Conversation Skills\n\n1. Greetings: User0 is making attempts to greet conversation partners. For example, the user said "Halo" when attempting to say "Hello."\n\nImprovement Suggestions: Practice the correct spelling and pronunciation of common greetings such as "Hello."\n\n2. Compliments: User0 is able to express compliments, as shown in the statement, "I think you are cute."\n\nImprovement Suggestions: Continue practicing positive statements and try to expand the vocabulary to express a variety of compliments.\n\n3. Introducing Oneself: User0 can provide their name, as indicated in, "My name is Geonhui."\n\nImprovement Suggestions: Keep practicing self-introduction and learn additional phrases that can be used when meeting someone new.\n\n4. Responding to Introductions: User0 expresses pleasure in meeting others with "Nice to meet you."\n\nImprovement Suggestions: This is well done. Consider also learning other polite phrases connected to introductions, like "How do you do?" or "Pleased to meet you."\n\n5. Asking for Names: User0 makes an attempt to inquire about someone\'s name, using the phrase, "What are you name?"\n\nImprovement Suggestions: Focus on correct grammar structure. The proper question is "What is your name?" Practice making questions in English and pay special attention to the order of words.\n\nOverall, continue practicing these basic conversation elements and pay close attention to spelling and grammar. Regular speaking and listening exercises could greatly benefit the understanding and use of English in conversations.', role='assistant', function_call=None, tool_calls=None))], 'created': 1700203335, 'model': 'gpt-4-1106-preview', 'object': 'chat.completion', 'system_fingerprint': 'fp_a24b4d720c', 'usage': CompletionUsage(completion_tokens=313, prompt_tokens=171, total_tokens=484)}

    llm_content = llm_response.choices[0].message.content
    return llm_content

def gh_render(request):
    PARAMETERS = {
        "username": get_parameter(request, "username", "user0")
    }
    username = PARAMETERS["username"]

    database = Database()

    report_test = ReportTest()
    if (report_test.test_mode == True):
        conversation_documents = report_test.conversation_documents
        cefr_documents = report_test.cefr_documents
        retrieve_documents = report_test.retrieve_documents
        reflect_documents = report_test.reflect_documents
    else:
        conversation_documents = Database.get_all_documents(database, f"{username}", "Conversations")
        cefr_documents = Database.get_all_documents(database, f"{username}", "CEFR")
        retrieve_documents = Database.get_all_documents(database, f"{username}", "Retrieves")
        reflect_documents = Database.get_all_documents(database, f"{username}", "Reflects")


    report = Report()
    
    report.latest_cefr = cefr_documents[-1] if len(list(cefr_documents)) > 0 else "Pre-A1"
    
    for raw_document in conversation_documents:
        document = {
            "speaker": raw_document["name"],
            "listener": raw_document["opponent"],
            "memory": raw_document["memory"],
            "node": raw_document["node"]
        }

        if document["speaker"] == username:
            if document["listener"] not in report.npc_conversation_dict:
                report.npc_conversation_dict[document["listener"]] = []
            report.npc_conversation_dict[document["listener"]].append(document)

    for raw_document in retrieve_documents:
        document = {
            "name": raw_document["name"],
            "retrieve": raw_document["retrieve"]
        }
        if document["name"] not in report.npc_retrieve_dict:
            report.npc_retrieve_dict[document["name"]] = []
        report.npc_retrieve_dict[document["name"]].append(document["retrieve"])

    for raw_document in reflect_documents:
        document = {
            "name": raw_document["name"],
            "reflect": raw_document["reflect"]
        }
        if document["name"] not in report.npc_reflect_dict:
            report.npc_reflect_dict[document["name"]] = []
        report.npc_reflect_dict[document["name"]].append(document["reflect"])
    
    for key in report.npc_retrieve_dict.keys():
        merged_information = '.\n'.join(report.npc_retrieve_dict[key])
        llm_messages = [
            {"role": "system", "content": "The Retrieve information indicates how the NPC (assistant) previously understood the improvements the user should make regarding English conversation skills development. The username is {username}. The previous NPC has interacted with the user as a {key}. All the user messages with the [#] form indicate the user's transcript spoken during the conversation."},
            {"role": "system", "content": "This request is to display the Retrieve information in a reported evaluation format with wordings that fit the reader's comprehension level (CEFR)."},
            {"role": "system", "content": "Focus on what the user has mistaken and what the user should improve. The user's transcript is not necessary to be displayed in the report."},
            {"role": "system", "content": "Username: {username} (The reader prefers the user named as a noun Student, not the actual username.)"},
            {"role": "system", "content": "Reader's CEFR level: {}".format(report.latest_cefr)},
            {"role": "system", "content": "Retrieve logs:\n\n{}".format(merged_information)}
        ]
        for conversation in report.npc_conversation_dict[key]:
            llm_messages.append({
                "role": "system",
                "content": "User's Statement [{}] {}".format(conversation["node"], conversation["memory"])
            })
        report.npc_normalized_retrieve_dict[key] = get_llm_content(llm_messages)
        report.npc_normalized_retrieve_dict[key] = report.npc_normalized_retrieve_dict[key].replace('"', '\\"')

    
    for key in report.npc_reflect_dict.keys():
        merged_information = '.\n'.join(report.npc_reflect_dict[key])
        llm_messages = [
            {"role": "system", "content": f"The Reflect information indicates how the NPC (assistant) previously understood the user's characteristics. The username is {username}. The previous NPC has interacted with the user as a {key}. All the user statements with the [#] form indicate the user's transcript spoken during the conversation."},
            {"role": "system", "content": f"This request is to display the Reflect information in a report format with wordings that fit the reader's comprehension level (CEFR)."},
            {"role": "system", "content": "Username: {username} (The reader prefers the user named as a noun Student, not the actual username.)"},
            {"role": "system", "content": "Reader's CEFR level: {}".format(report.latest_cefr)},
            {"role": "system", "content": "Reflect logs:\n\n{}".format(merged_information)}
        ]
        for conversation in report.npc_conversation_dict[key]:
            llm_messages.append({
                "role": "system",
                "content": "User's Statement [{}] {}".format(conversation["node"], conversation["memory"])
            })
        report.npc_normalized_reflect_dict[key] = get_llm_content(llm_messages)
        report.npc_normalized_reflect_dict[key] = report.npc_normalized_reflect_dict[key].replace('"', '\\"')

    return render(request, 'report.html', {
        "reflect_json": json.dumps(report.npc_normalized_reflect_dict),
        "retrieve_json": json.dumps(report.npc_normalized_retrieve_dict),
    })
    # return render(request, 'report.html', {
    #     "username": "Hello"
    # })