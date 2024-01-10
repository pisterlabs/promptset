import os
import openai
import csv

output_path = 'sentences/'
output_file = 'name.csv'

printing_request_prompt = "Suggest a database of 100 sentences a user asks to a chatbot to print some document, always giving the title of the document and the number of pages\n\nSentence,Document Title,Pages\nHello! Can you print my 6000 pages document? It is called report.docx,report.docx,6000\nPrint document.pdf. 1200000 pages,document.pdf,1200000\nIt is possible to print my essay? The document name is essay.pdf and it is 50000 pages long.,essay.pdf,50000\n\nI need to print my contract. The document is called contract.doc and it is 200 pages long.,contract.doc,200\n\nHello, can you print my document called invoice.pdf? It is only 10 pages long.,invoice.pdf,10\n\nCan you print my resume? The document is called resume.doc and it is 2 pages long.,resume.doc,2\n"

openai.api_key = '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'
count = 0
while (count < 3):

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="Generate greeting phrases",
        temperature=0.9,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0.05,
        presence_penalty=0)

    text = response["choices"][0]['text']
    list = text.split(",")
    print(list)
    if len(list) > 3 :
        sentence = ",".join(list[0:len(list)-2])
        list = [sentence,list[len(list)-2],list[len(list)-1]]
    with open('printing_request.csv','a') as csvfile :
        writer = csv.writer(csvfile,delimiter = ',')
        writer.writerow(list)
    count += 1
