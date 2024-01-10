# gpt3 completion object

import json

from datetime import datetime, timedelta
from pprint import pprint
import stripe
import os
from os import environ
import uuid
import streamlit as st

import openai

openai_user_id_for_safety_tracking = 6

import pandas as pd

import docx

from app.models.user_models import User, StripeCustomer, Role, UsersRoles, UserPromptForm, UserDocsForm, Presets, Tokens, Transactions, UploadForm, UserDocs, find_or_create_searchdocs, Journals
from app import db

from flask_user import current_user

from sqlalchemy import func, extract
from sqlalchemy.dialects import postgresql

from .s2orc.doc2json.pdf2json.process_pdf import process_pdf_file
 
os.environ.get('OPENAI_KEY') = st.secrets('OPENAI_KEY')

def presets_parser(preset_filename):
    print(preset_filename)
    openfile = "app/presets/" + preset_filename + ".json"
    print('opening file', openfile)


    presetsdf = pd.read_json(openfile, dtype=object)
    #presetsdfprint(presetsdf)

    #search form

    presetsdf['preset_name'] = presetsdf.get('preset_name', "Presets")
    presetsdf['preset_pagetype'] = presetsdf.get('preset_pagetype', "UserPrompt")
    presetsdf['preset_description'] = presetsdf.get('preset_description', "Description of this preset.")
    presetsdf['preset_instructions'] = presetsdf.get('preset_instructions', "Fill in the form.")
    presetsdf['preset_placeholder'] = presetsdf.get('preset_placeholder', "Enter this text:")
    presetsdf['pre_user_input'] = presetsdf.get('pre_user_input', "")
    presetsdf['prompt'] = presetsdf.get('prompt', "")
    presetsdf['post_user_input'] = presetsdf.get('post_user_input',"")
    presetsdf['preset_additional_notes'] = presetsdf.get('preset_additional_notes', "Notes:")

    # request parameters

    presetsdf['engine'] = presetsdf.get('engine', "ada")
    presetsdf['finetune_model'] = presetsdf.get('finetune_model', "")
    presetsdf['temperature'] = presetsdf.get('temperature', 0.7)
    presetsdf['max_tokens'] = presetsdf.get('max_tokens', 100)
    presetsdf['top_p'] = presetsdf.get('top_p', 1.0)
    presetsdf['fp'] = presetsdf.get('fp', 0.5)
    presetsdf['pp'] = presetsdf.get('pp', 0.5)
    presetsdf['stop_sequence'] = presetsdf.get('stop_sequence', ["\n", "<|endoftext|>"])
    presetsdf['echo_on'] = presetsdf.get('echo_on', False)
    presetsdf['search_model'] = presetsdf.get('search_model', "ada")
    presetsdf['model'] = presetsdf.get('model', "curie")
    presetsdf['question'] =presetsdf.get('question',"")
    presetsdf['fileID'] = presetsdf.get('answerhandle',"")
    presetsdf['examples_context'] = presetsdf.get('examples_context', "In 2017, U.S. life expectancy was 78.6 years.")
    presetsdf['examples'] = presetsdf.get('examples', '[["What is human life expectancy in the United States?", "78 years."]]')
    presetsdf['max_rerank'] = presetsdf.get('max_rerank', 10)

    # specify secure db for Journals
    presetsdf['preset_db'] = presetsdf.get('preset_db', 'None')
    
    # metadata

    presetsdf['user'] = presetsdf.get('user', 'testing')
    presetsdf['organization'] = presetsdf.get('organization', 'org-M5QFZNLlE3ZfLaRw2vPc79n2') # NimbleAI


    # print df for convenience
    transposed_df = presetsdf.set_index('preset_name').transpose()
    #print('transposeddf', transposed_df)

    # now read into regular variables

    preset_name = presetsdf['preset_name'][0]
    preset_pagetype = presetsdf['preset_pagetype'][0]
    preset_description  = presetsdf['preset_description'][0]
    preset_instructions = presetsdf['preset_instructions'][0]
    preset_placeholder = presetsdf['preset_placeholder'][0]
    preset_additional_notes =presetsdf['preset_additional_notes'][0]
    pre_user_input = presetsdf['pre_user_input'][0]
    post_user_input = presetsdf['post_user_input'][0]
    prompt = presetsdf['prompt'][0]
    engine = presetsdf['engine'][0]
    finetune_model = presetsdf['finetune_model'][0]
    temperature = presetsdf['temperature'][0]
    max_tokens = presetsdf['max_tokens'][0]
    top_p = presetsdf['top_p'][0]
    fp = int(presetsdf['fp'][0])
    pp = presetsdf['pp'][0]
    stop_sequence = presetsdf['stop_sequence'][0]
    if presetsdf['echo_on'][0] == 'True':
        echo_on = True
    else:
        echo_on = False

    search_model = presetsdf['search_model'][0] 
    model = presetsdf['model'][0]
    question =presetsdf['question'][0]
    fileID = presetsdf['fileID'][0] 
    examples_context = presetsdf['examples_context'][0]
    examples = presetsdf['examples'][0]
    max_rerank = presetsdf['max_rerank'][0]
    preset_db = presetsdf['preset_db'][0]
    user = presetsdf['user'][0]
    organization = presetsdf['organization'][0]

    # then return both df and regular variables

    return presetsdf, preset_name, preset_description, preset_instructions, preset_additional_notes, preset_placeholder, pre_user_input, post_user_input, prompt, engine, finetune_model, temperature, max_tokens, top_p, fp, pp, stop_sequence, echo_on, preset_pagetype, preset_db, user, organization


def gpt3complete(preset_filename, prompt):

    openai_user_id_for_safety_tracking = 6

    if prompt:
        override_prompt = prompt

    presetsdf, preset_name, preset_description, preset_instructions, preset_additional_notes, preset_placeholder, pre_user_input, post_user_input, prompt, engine, finetune_model, temperature, max_tokens, top_p, fp, pp, stop_sequence, echo_on, preset_pagetype, preset_db, user, organization = presets_parser(preset_filename)

    print(presetsdf.transpose())

    print("echo on is", echo_on)

    if override_prompt:
        prompt = override_prompt
    print('override prompt is', prompt)

    promptsubmit = pre_user_input + prompt + post_user_input

    print('promptsubmit is:', promptsubmit)

    if openai_user_id_for_safety_tracking is None:
        openai_user_id_for_safety_tracking = str(current_user.id)

    for item in promptsubmit:
        promptchar= len(promptsubmit)

        response = openai.Completion.create(
            engine=engine,
            prompt=promptsubmit,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=fp,
            presence_penalty=pp,
            logprobs=0,
            echo=False,
            user=openai_user_id_for_safety_tracking
        )
        response_text = "#### " + presetsdf['completion_heading'] + '\n'
        response_text = response_text + response['choices'][0]['text']

        totaltokens = sum(len(i.split()) for i in response['choices'][0]['logprobs']['tokens'])
        print('totaltokens', totaltokens)
        created_on = datetime.utcnow()
        
        """ tokens = Tokens(user_id=current_user.id, totaltokens=totaltokens, created_on=created_on)

        db.session.add(tokens)
        db.session.commit()

        firstdaycurrentmonth = datetime.today().replace(day=1)
        filter_after = firstdaycurrentmonth

        quotastatus = Tokens.query.with_entities(func.sum(Tokens.totaltokens).label('sum')).filter(created_on >= filter_after, Tokens.user_id == current_user.id).all()
        quotastatus = quotastatus[0][0]

        submit = "<|endoftext|>" +  response['choices'][0]['text'] + "\n--\nLabel:"
        safety = openai.Completion.create(engine="content-filter-alpha-c4", prompt=submit, max_tokens=1, temperature=0, top_p=0, user=openai_user_id_for_safety_tracking)

        print(safety)
        
        transactions = Transactions(user_id=current_user.id, totaltokens=totaltokens, pre_user_input=pre_user_input, user_input=prompt, post_user_input=post_user_input, prompt=promptsubmit, response=response_text, safety_rating=safety['choices'][0]['text'], created_on=created_on)

        db.session.add(transactions)
        db.session.commit() """

        return response#, totaltokens, quotastatus, safety 
        # remember text is in item [0] of this tuple response

""" def gpt3answers(answershandle):

    response = openai.Answer.create(
        search_model="ada", 
        model="curie",
        question="which puppy is happy?", 
        documents=["Puppy A is happy.", "Puppy B is sad."],
        examples_context="In 2017, U.S. life expectancy was 78.6 years.", 
        examples=[["What is human life expectancy in the United States?","78 years."]],
        max_tokens=5,
        stop=["\n", "<|endoftext|>"],
)

    pprint(response)
    return response, totaltokens, quotastatus, safety """

def jsonl2openai(filename):
    searchhandle = openai.File.create(file=open(filename), purpose="search")
    answershandle = openai.File.create(file=open(filename), purpose="answers")
    return searchhandle, answershandle

def docx2jsonl(filename):

    doc = docx.Document(filename)
    data = []

    basename = os.path.splitext(os.path.basename(filename))[0]
    print(basename)
    jsonlfilename = 'app/userdocs/' + str(current_user.id) + '/' + basename + '.jsonl'

    for p in doc.paragraphs:
        data.append(p.text)
       
    df = pd.DataFrame(data, columns=["text"])
    df.to_json(jsonlfilename, orient="records", lines=True)
    return jsonlfilename

def pdf2jsonl(filename, tempdir, outdir):
    jsonfile = process_pdf_file(filename, tempdir, outdir)
    filepathname = os.path.splitext(filename)
    basename = os.path.splitext(os.path.basename(filename))[0]
    print(basename)
    # prepare filename to save 
    jsonfilename = outdir + '/' + basename + '.json'
    data = json.load(open(jsonfilename))
    df = pd.DataFrame(data['pdf_parse']['body_text'])
    df1 = df[['text']]
    # prepare filename to save modified df
    jsonlfilename = outdir + basename + '.jsonl'
    df1.to_json(jsonlfilename, orient="records", lines=True)
    return jsonlfilename

def find_or_create_customer(id, user_id, stripeCustomerId, stripeSubscriptionId):
    """ Find existing customer or create new customer """
    customer = Customer.query.filter(Customer.stripeCustomerId == stripeCustomerId).first()
    if not customer:
        customer = Customer(id=id,
        user_id=user_id,
        stripeCustomerId=stripeCustomerId,
        stripeSubscriptionId=stripeSubscriptionId)
        db.session.add(customer)
    return customer

def prepare_quotastatus(user_id):
    created_on = datetime.utcnow()

    firstdaycurrentmonth = datetime.today().replace(day=1)
    filter_after = firstdaycurrentmonth

    print('current_user.id is', current_user.id)
    totaltokens = None # tokens used in current openai call, which is none, until it happens

    quotacheck = Tokens.query.with_entities(func.sum(Tokens.totaltokens).label('sum')).filter(created_on >= filter_after, Tokens.user_id == current_user.id).all()
    #q = query.with_entities(func.sum(Tokens.totaltokens).label('sum')).filter(created_on >= filter_after, Tokens.user_id == current_user.id).all()
    #print(str(quotacheck.statement.compile(dialect=postgresql.dialect())))
    #print(quotacheck)
    quotastatus = quotacheck[0][0]
    if quotastatus == None:
        quotastatus = 0
    print(quotastatus)


    customer = StripeCustomer.query.filter_by(user_id=current_user.id).first()

    if customer:
        subscription = stripe.Subscription.retrieve(
                customer.stripeSubscriptionId)
        product = stripe.Product.retrieve(subscription.plan.product)
        quotalimit = 100 * 1024
        print('user is a customer', 'product is', product['description'])
        product = product['description']
    else:
        quotalimit = 5000
        product = "Free Plan"
    
    if  quotastatus <= quotalimit:
        underquota = True
    else:
        underquota = False

    return totaltokens, underquota, quotastatus, quotalimit, product 

def create_uuid():
    return str(uuid.uuid4())
