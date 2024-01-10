import os
import openai
import json

openai.api_key = os.getenv("OPENAI_API_KEY")


def ner(docType):
    # inputOCR.txt in the corresponding document directory is the input
    inpFile = open(os.path.join(docType, "inputOCR.txt"), "r")
    OCRtext = inpFile.read()

    headFile = open(os.path.join(docType, "promptHead.txt"), "r")
    promptHead = headFile.read()
    footFile = open(os.path.join(docType, "promptOutput.txt"), "r")
    promptOutput = footFile.read()

    prompt = "\n".join([promptHead, OCRtext, promptOutput])
    # print(prompt)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.2,
        max_tokens=60,
        top_p=1,
        frequency_penalty=0.8,
        presence_penalty=0
    )
    results = (response.choices[0].text).split("\n")
    outputEntities = dict()
    for result in results:
        if (len(result) >= 3):
            keyVal = result.split(":")
            outputEntities[keyVal[0].strip()] = keyVal[1].strip()

    # Extracted entities stored in outputEntities.json in the corresponding document directory
    with open(os.path.join(docType, 'outputEntities.json'), 'w') as json_file:
        json.dump(outputEntities, json_file)
    return outputEntities


# Place Input text in inputOCR.txt in the corresponding Document dir
# Output will be stored in outputEntities.json in the corresponding Document dir
if __name__ == "__main__":
    print(ner("aadhar"))
    # print(ner("pan"))
