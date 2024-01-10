from django.shortcuts import render
from django.http import JsonResponse
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer
import requests
from .models import *
import openai
from . import cmd
from Iris import settings
import openai
from PyPDF2 import PdfReader
import subprocess
from .ask_openai import *
import os
import spacy
from transformers import pipeline

openai.api_key = settings.openai.api_key

import openai
import re



def pdfExtract(file):
    reader = PdfReader(file)
    number_of_pages = len(reader.pages)
    text = ""
    for i in range(number_of_pages):
        page = reader.pages[i]
        text += page.extract_text()
    return text.replace("\n"," ")

def chat_view(request):
    report = Report.objects.filter(user_id=request.user.id).last()
    answers = {}
    response=""
    if request.method == "POST":
        query = request.POST.get('answer')
        if query != " ":
            media_file_path = "/home/cime/Iris"+report.file.url
            subprocess.call(['chmod', '777', media_file_path])
            text = pdfExtract(media_file_path)
        
            # response = ask_questions(text,query,1500,"davinci")
            response = exec(text,query)
            gpt = response
            req = Request.objects.create(query=query, answer=gpt,filename=report.filename)
            req.save()
            answers[query]= gpt
            # answers = AddValueToDict(query, answers, "response"+query, type( "response"))
    requ = Request.objects.filter(filename=report.filename)
    try:
        seq = len(requ) - 1
        r = requ[:seq]
    except:
        r= requ
    context = {
        "report":report,
        "answers":answers,
        "requ":r,
    }
    return render(request, 'window.html',context )

def chat_view_old(request,id):
    report = Report.objects.get(user_id=request.user.id, id=id)
    answers = {}
    response=""
    if request.method == "POST":
        query = request.POST.get('answer')
        if query != " ":
            media_file_path = "/home/cime/Iris"+report.file.url
            subprocess.call(['chmod', '777', media_file_path])
            text = pdfExtract(media_file_path)
        
            response = exec(text,query)
            gpt = response
            req = Request.objects.create(query=query, answer=gpt,filename=report.filename)
            req.save()
            answers[query]= gpt
            # answers = AddValueToDict(query, answers, "response"+query, type( "response"))
    requ = Request.objects.filter(filename=report.filename)
    try:
        seq = len(requ) - 1
        r = requ[:seq]
    except:
        r= requ
    context = {
        "report":report,
        "answers":answers,
        "requ":r,
        
    }
    return render(request, 'window.html',context )





def Audio_chat_view(request):
    report = Report.objects.filter(user_id=request.user.id).last()
    answers = {}
    response=""
    if request.method == "POST":
        query = request.POST.get('answer')
        if query != " ":
            media_file_path = "/home/cime/Iris"+report.file.url
            subprocess.call(['chmod', '777', media_file_path])
            text = pdfExtract(media_file_path)
        
            # response = ask_questions(text,query,1500,"davinci")
            response = exec(text,query)
            if response ==" ":
                response="Désolé Le texte que vous avez fournie ne me permet pas de répondre à cette question"
            
            gpt = response
            req = Request.objects.create(query=query, answer=gpt,filename=report.filename)
            req.save()
            answers[query]= gpt
            # answers = AddValueToDict(query, answers, "response"+query, type( "response"))
    requ = Request.objects.filter(filename=report.filename)
    try:
        seq = len(requ) - 1
        r = requ[:seq]
    except:
        r= requ
    context = {
        "report":report,
        "answers":answers,
        "requ":r,
    }
    return render(request, 'window.html',context )

def Audio_chat_view_old(request,id):
    report = Report.objects.get(user_id=request.user.id, id=id)
    answers = {}
    response=""
    if request.method == "POST":
        query = request.POST.get('answer')
        if query != " ":
            media_file_path = "/home/cime/Iris"+report.file.url
            subprocess.call(['chmod', '777', media_file_path])
            text = pdfExtract(media_file_path)
        
            response = exec(text,query)
            if response ==" ":
                response="Désolé Le texte que vous avez fournie ne me permet pas de répondre à cette question"
            
            gpt = response
            req = Request.objects.create(query=query, answer=gpt,filename=report.filename)
            req.save()
            answers[query]= gpt
            # answers = AddValueToDict(query, answers, "response"+query, type( "response"))
    requ = Request.objects.filter(filename=report.filename)
    try:
        seq = len(requ) - 1
        r = requ[:seq]
    except:
        r= requ
    context = {
        "report":report,
        "answers":answers,
        "requ":r,
        
    }
    return render(request, 'window.html',context )
