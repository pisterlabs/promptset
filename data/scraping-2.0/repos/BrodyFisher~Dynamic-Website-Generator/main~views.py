from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
import os
from django.conf import settings
import cohere
from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))
text = ""
def index(request):
    return render(request, "main/index.html")

def results(request):
    response = co.generate(
    model='command',
    prompt="Your job is to classify text which is going to be used to dynamically generate websites. Your output will look like a typical HTML document, starting with <!DOCTYPE html>. You will read the following text and identify HTML components like headers, body text, forms as well as adding images from the internet:\n\n" + text,
    max_tokens=1000,
    temperature=1.3,
    k=0,
    stop_sequences=[],
    return_likelihoods='NONE')
    print('{}'.format(response.generations[0].text))
    result = response.generations[0].text
    css_response = co.generate(
    model='command',
    prompt="Your job is to create an elegant CSS file which will be used to style the HTML document. You will read the following HTML document and identify HTML components like headers, body text, paragraphs, tables, and forms as well as images. You will style the HTML document with stylish CSS and your output will be CSS code and ONLY CSS code:\n\n" + result,
    max_tokens=1000,
    temperature=1.3,
    k=0,
    stop_sequences=[],
    return_likelihoods='NONE')
    print('{}'.format(css_response.generations[0].text))
    result += "<style>" + css_response.generations[0].text + "</style>"
    pos = result.find('<html>')
    if pos != -1:
        result = "<html contenteditable>" + result[pos + 6:]
        result += "<style>body{background-color: #ebe1db;}</style>"
        result += "<style> table {margin-left:auto; margin-right: auto;background-color: white; padding:5px;border-radius: 7px;border-width:2px;width:screen;white-space: nowrap;}td {border:none;color:gray;}th {text-align: left;border:none;border-radius:5px;color:white;width: max-content;height: max-content;padding:10px;background-color: rgb(40, 16, 54);font-family:Arial, Helvetica, sans-serif;font-size: 1vw;}tr:nth-child(even){background-color: rgb(190, 190, 190);color: #000;;}tr:nth-child(even):hover{background-color: rgb(140, 140, 140);color:black;}tr:hover {background-color: rgb(231, 243, 253);}</style>"

    # Archive the generated html file
    with open('main/static/main/index.html', 'w') as f:
        f.write(result)
    
    return HttpResponse(result)

def search(request):
    # 'Your job is to classify text which is going to be used to dynamically generate websites. You will read the following text and identify HTML components like hearders, body text and etc: Welcome, Jack!\n\nWhat is Cohere?\n\nCohere allows you to implement language AI into your product. Get started and explore Cohere\'s capabilities with the Playground or Quickstart tutorials.'
    global text
    text = request.POST.get('search', "null")
    return HttpResponseRedirect(reverse("results"))

def download_static_files(request, file_path):
    # Construct the full path to the static file
    full_file_path = os.path.join(settings.STATICFILES_DIRS[0], file_path)
    
    # Check if the file exists
    if os.path.exists(full_file_path):
        with open(full_file_path, 'rb') as file:
            response = HttpResponse(file.read(), content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(full_file_path)}"'
            return response
    else:
        # Handle the case where the file does not exist
        return HttpResponse('File not found', status=404)
