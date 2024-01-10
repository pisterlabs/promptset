import openai
import time

# using gpt-3.5 model
openai.api_key = "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"

# Interact with Openai Api GPT-3.5 Model to generate answers to your questions.
def interact_with_image_generator(request, reqNumber, reqSize):
    while True:
        try:
            response = openai.Image.create(
                prompt = request,
                n = reqNumber,
                size = reqSize
            )
            image_url = response['data'][0]['url']
            break
        except Exception as e:
            print("Error:", e)
            print("Retrying in 5 seconds...")
            time.sleep(5)
    return image_url


def interact_with_ChatGPT(question):
    messages = [({
        "role" : "user",
        "content" : question
    })]
    while True:
        try:
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
            reply = response.choices[0].message.content
            break
        except Exception as e:
            print("Error:", e)
            print("Retrying in 5 seconds...")
            time.sleep(5)
    return reply

def generatePrompt(information):
    question = 'Human: {Title : Belts, Topic : Industrie, Effect : photorealistic, Perspective : 4:3}'\
    'AI: "a Industrial, mechanical, metallic belts, outdoor, factory, environment, realistic, dynamic, vivid, 4:3, panoramic, cinematic --ar 2:3 --v 4"'\
    'Human: {Title : Popcorn maker, Topic : Industrie, Effect : painting, Perspective : 4:3}'\
    'AI: "a Industrial, mechanical, metallic Popcorn maker, outdoor, factory, environment, painting, dynamic, vivid, 4:3, panoramic, cinematic --ar 2:3 --v 4"'\
    'Human: {Title : Airplane, Topic : Mining, Effect : vector, Perspective : 16:9}'\
    'AI: "a mining, mechanical, metallic Airplane, outdoor, factory, environment, vector, dynamic, vivid, 16:9, panoramic, cinematic --ar 2:3 --v 4"'\
    f'Human: {{Title : {information["Title"]}, Topic : {information["Topic"]}, Effect : {information["Effect"]}, Perspective : {information["Perspective"]}}}'\
    'AI:'
    prompt = interact_with_ChatGPT(question)
    return prompt[1:-1]


def imageGenerator(imageInformation):
    return interact_with_image_generator(generatePrompt(imageInformation), 1, "256x256")
