import openai
import get_data

openai.api_key="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"

def run(text):
    question="tomando en cuenta el siguiente texto : \n"
    question+=text
    question+="\n extrae y separa los remates, usar la palabra clave remmatew para separarlos"
    complection=openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        max_tokens=2048)
    
    response=complection.choices[0].text

    print(response)
    
    return response

if __name__=='__main__':
    text=get_data.run("./Ejemplo.txt")
    run(text)