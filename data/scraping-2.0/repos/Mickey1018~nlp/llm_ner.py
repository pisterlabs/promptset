import openai
openai.api_type = "azure"
openai.api_base = "https://hkust.azure-api.net"
openai.api_version = "2023-07-01-preview" #2022-12-01, 2023-03-15-preview, 2023-05-15, 2023-06-01-preview, 2023-07-01-preview
openai.api_key = "53f3c40c19b24b40b7f18a67989fcc8a"


prompt = f""" You are an assistant in medical field, the text in triple ticks is getting the output of a medical named entity recognition (NER) model according to the given 
template words in square brackts. 
It is a way to extract structured information from unstructured text. 
this case, the NER model is used to extract information about a medical procedure from a pathology report. 
The output includes information about the length and circumference of the intestine, 
the appearance and size of a tumor, and the status of the margins. It also includes additional findings from the procedure, 
such as the location of lymph nodes and the involvement of the resection margin. 
The output is structured in a way that makes it easy for medical professionals to read and understand the results of the procedure.
Can you try the samething for the case in angle brackets? Please keep the output in json format and include the ner results only.
```{example}```
<{case}>
[{template_words}]
"""

response = openai.Completion.create(
                engine="gpt-35-turbo",
                prompt=prompt,
                temperature=0.1,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=1,
                presence_penalty=1,
                stop=None)

response["choices"][0]["text"].split("\n")