import openai
from feautures.custom_voice import speak

openai.api_key = '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'

def cat():
    with open('news.txt', 'r') as f:
        content = f.read()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f" write one mind blowing intrestng fact in hinglish text in english but language hindi  and deatils with only 40 words, dont write  any translataion"}]
    )

    OP = response['choices'][0]['message']['content']
    OP = OP.replace('"', '')
    with open('you.txt', 'a') as f:
        f.write(OP + '\n')
    print(OP)
    speak("hello dosto"  +" " + OP)

    

cat()
