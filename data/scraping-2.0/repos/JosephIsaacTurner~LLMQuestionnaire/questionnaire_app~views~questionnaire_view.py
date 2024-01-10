
from django.shortcuts import render
from questionnaire_app.forms import MedicalQuestionnaireForm
import json
import openai
from django import forms
from django.conf import settings

def create_dynamic_form(fields):
    dynamic_fields = {name: field_type(**attrs) for name, (field_type, attrs) in fields.items()}
    return type('DynamicForm', (forms.Form,), dynamic_fields)

def questionnaire_view(request):
    context = {}
    if request.method == 'POST':
        form = MedicalQuestionnaireForm(request.POST)
        if form.is_valid():
            data_dict = form.cleaned_data
            context['data_dict'] = data_dict
            data_str = json.dumps(data_dict, indent=4)
            # Add to context for debugging
            context['data_str'] = data_str
            access_key = settings.OPENAI_API_KEY
            endpoint_url = "https://api.openai.com/v1/chat/completions"
            openai.api_key = access_key
            background_info = """You are a helpful assistant that is familiar with common medical information, 
                especially diagnosing and treating people's conditions and clarifying their concerns. Not
                only that, but you are a django developer deeply familiar with django form classes and syntax."""
            specific_prompt = f"""Response to the following patient's description of their situation, which is in json format: {data_str} 
                Write a concise and professional message to the patient explaining possible conditions that they may be experiencing. """
            specific_prompt += " Then, think of the top three most relevant follow-up questions given this patient's information, and list each question using the '@@' delimiter."
            # specific_prompt = f"""
            #     Review the following json that is a patient's description
            #     of their situation and request for a medical visit to a physician. 
            #     During your review, consider the most likely medical conditions or diagnoses 
            #     that the patient could be experiencing, and what additional signs or 
            #     questions would be relevant in understanding their situation better. 
            #     Here is the json containing the patient's response to the questionnaire: {data_str}.
            #     Once you are done reviewing the patient's response, design three relevant follow-up questions 
            #     that could help you clarify their concerns and their potential condition, with the goal of 
            #     identifying the cause of their symptoms. Please do not repeat any prior questions that
            #     the patient already responded to or elected not to respond to. If the response json
            #     contains blank or none answers, it is because they were optional fields that the patient 
            #     does not want to answer or are not eligible to answer. If the information is already present in the 
            #     patient's response json, we do not need to repeat the question or ask synonymous questions
            #     that we already know the answers to.
            #     Format these three questions as if they were part of a django form class, including the relevant fields and attributes. 
            #     Then, put these three questions into a json, and exclude any newline characters, like /n.
            #     This is an example skeleton for the json. For the first element in the json, 
            #     I used symptom_progression simply as an example filler to remind you 
            #     of how django form classes work, but you should replace this with the question
            #     you generate, and replace the form fields with the relevant fields for your question. The
            #     questions can be of any type and widget, including CharField, TextInput, Textarea, RadioSelect, ChoiceField, IntField, dropdowns, 
            #     and anything else-- it is all on the table as long as you know how to format it properly for a django form.""" + """
            #     fields = {
            #         "question1": "symptom_progression" = forms.ChoiceField(
            #             choices=[
            #                 ('better', 'Getting Better'),
            #                 ('worse', 'Getting Worse'),
            #                 ('same', 'Staying the Same')
            #             ],
            #             widget=forms.RadioSelect,
            #             label='How have the symptoms progressed?'
            #         )",
            #         "question2": "another django form attribute",
            #         "question3": "another django form attribute,
            #     }
            #     Please return only this json, with no additional explanation or text that is not within the json with these three keys."""
            # # specific_prompt += " Then, think of the top three most relevant follow-up questions given this patient's information, and list each question using the '@@' delimiter."

            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": background_info},
                        {"role": "user", "content": specific_prompt}
                    ]
            )
            message_content = response["choices"][0]["message"]["content"]
            context['message_content'] = message_content
            context['has_message'] = True

    else:
        form = MedicalQuestionnaireForm()
        context['has_message'] = False
    context['form'] = form
    return render(request, 'questionnaire_app/questionnaire_template.html', context)