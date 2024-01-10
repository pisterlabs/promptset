import openai
from openai import OpenAI
import json
import os
from langchain.retrievers.you import YouRetriever
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

claim_function = {
        'name': 'claim_extraction',
        'description': 'Extract any claims made in the transcript',
        'parameters': {
            'type': 'object',
            'properties': {
                'claims': {
                    'type': 'array',
                    'description': 'List of claims extracted from transcript',
                    'items': {
                        'type': 'string',
                        'description': 'A claim extracted from transcript'
                    }
                }
            }
        }

    }


EXTRACTION_INSTRUCTIONS = "You are a claim extractor for video transcript. Focus on extracting objective claims about the world, people, or any topic. Clarify the subjects involved. Do not extract subjective claims about the speaker's feelings or opinions."
CLAIM_QUESTION = 'Is this claim TRUE, FALSE, or UNCERTAIN? Please reply with JSON with keys "truth_value" and "explanation". Make the explanation concise and factual.'
client = OpenAI()

# load you.com api key from environment variable
YOU_API_KEY = os.environ['YDC_API_KEY']
yr = YouRetriever()
gptmodel = "gpt-4-1106-preview"
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model=gptmodel), chain_type="stuff", retriever=yr)


async def extract_claims(transcript, video_base_dir):
    '''
    Extracts claims from transcript
    '''
    claims = []
    claims_path = video_base_dir + '/results/claims.json'
    if not os.path.exists(claims_path):
        messages = [{'role': 'system', 'content': EXTRACTION_INSTRUCTIONS},
                    {'role': 'user', 'content': transcript}]
        response = client.chat.completions.create(model='gpt-3.5-turbo-1106',
                                                messages=messages,
                                                tools = [{"type": "function", "function": claim_function}],
                                                )
        print(response.choices[0].message.tool_calls[0].function.arguments)
        claims = json.loads(response.choices[0].message.tool_calls[0].function.arguments)['claims']
        # write claims to file
        os.makedirs(os.path.dirname(claims_path), exist_ok=True)
        with open(claims_path, 'w') as f:
            json.dump(claims, f)
    else:
        with open(claims_path, 'r') as f:
            claims = json.load(f)
    print('claims extracted')
    return claims, claims_path


async def check_one_claim(claim):
    '''
    Checks one claim for its validity
    '''
    prompt = claim + CLAIM_QUESTION
    res = qa.run(prompt)
    res = res.strip('```').strip('json').strip()
    parsed = json.loads(res)
    truth_value = parsed['truth_value']
    explanation = parsed['explanation']
    print(explanation)
    return truth_value, explanation
    
async def check_claims(claims, claims_path):
    '''
    Checks the claims for their validity
    '''
    print('checking claims')
    results = []
    results_path = claims_path.replace('claims', 'results')
    if not os.path.exists(results_path):
        for claim in claims:
            truth_value, explanation = await check_one_claim(claim)
            results.append({'claim': claim, 'truth_value': truth_value, 'explanation': explanation})
        # write results to file
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            print("Writing results to file")
            json.dump(results, f)
    else:
        with open(results_path, 'r') as f:
            print("Reading results from file")
            results = json.load(f)
    return results

if __name__=='__main__': 
    extract_claims('The sky is blue. How are you? The current presiden of Canada is Barack Obama', '/Users/sidb/Development/framecheck/.videos/iRYZjOuUnlU/audio/test')
