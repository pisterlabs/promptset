import openai
import pinecone

def get_answer(question, index, sources):

    with open('prompts/rules.txt') as r:
        rules = r.read()

    xq = openai.Embedding.create(model='text-embedding-ada-002', input=question)['data'][0]['embedding']
    answers = index.query([xq], top_k = 6, include_metadata=True)
    
    plausible = ''
    i = 0
    source_list = []
    for match in answers['matches']:
        if match['score'] >= 0.75:
            plausible += str(i) + '. ' + match['metadata']['text'] + '\n\n'
            i += 1
            source_list.append(sources[match['metadata']['text']])
            
    header = 'You are given the following question: ' + question + '\n'
    body = 'A crash-course on development gives the following possible answers: ' + plausible + '\n'
    footer = 'Combine the plausible answers to be a coherent and grammatically-correct answer.'
    
    prompt = header + body + footer
    
    chatgpt = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = [
            {'role':'system', 'content': rules},
            {'role':'user', 'content':prompt}
        ]
    )
    
    cleaned = chatgpt['choices'][0]['message']['content'].strip()
    
    return cleaned, source_list