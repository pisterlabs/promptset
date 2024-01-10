import sys
sys.path.append('../Gloc')
import asyncio
import time
import openai
from debater_python_api.api.debater_api import DebaterApi
from Credentials.info import *
from Gloc.utils.async_llm import Model
import aiohttp
import json
import re

# set credentials
openai.organization = organization
openai.api_key = openai_api
debater_api = DebaterApi(Debater_api)
claim_boundaries_client = debater_api.get_claim_boundaries_client()

class ClaimDetector():
    def __init__(self):
        pass
    
    async def detect(
            self, sentence: str, 
            boundary_extract:bool=False, 
            llm_classify:bool=False, 
            verbose: bool=False,
            score_threshold: float = 0.5
        ):
        """ 
            Check if there is check-worthy claims in the sentence
            using ClaimBuster API
        """
        # Define the endpoint (url) with the claim formatted as part of it, api-key (api-key is sent as an extra header)
        api_endpoint = f"https://idir.uta.edu/claimbuster/api/v2/score/text/{sentence}"
        request_headers = {"x-api-key": Claim_Buster_api}

        # Send the GET request to the API and store the api response
        async with aiohttp.ClientSession() as session:
            async with session.get(url=api_endpoint, headers=request_headers) as resp:
                api_response = await resp.json()
                if verbose: print("check result: ", api_response)

        # if the score is > .5 --> checkworthy
        if len(api_response['results']) == 0:
            if verbose: print("non-checkworthy: 0")
            return "The claim is not check-worthy", 0
        score = api_response['results'][0]['score']
        if score > score_threshold:
            if llm_classify:
                """
                    create prompt chat to detect if the claim is check-worthy and data related.
                """
                response = await openai.ChatCompletion.acreate(
                    model=Model.GPT3,
                    messages=[
                        {"role": "system", "content": """Label the following claims as 'Y' if they are related to datasets about social issues like Climate Change, Economy, etc.; otherwise 'N' according to the following format.
                        \{
                            "explain": "<TODO: explain why the claim can be verified using data or not. Think step by step.>",
                            "verdict": "<TODO: Y or N>"
                            
                        \}"""},
                        {"role": "user", "content": f"{sentence}"},
                    ]
                )
                res = response["choices"][0]['message']['content']

                verdict = re.search(r'"verdict": "(Y|N)"', res).group(1)
                explain = re.search(r'"explain": "(.*?)"', res, re.DOTALL).group(1)
                if verdict == 'Y':
                    if verbose: 
                        print(f"statistically interesting: {api_response['results'][0]['score']}")
                        print(f"explain: {explain}")
                else:
                    if verbose: print("statistically unrelated")
                    # negative means unrelated to data
                    return explain, -score
            
            # # extract the boundary
            if boundary_extract:
                sentences = [sentence]
                boundaries_dicts = claim_boundaries_client.run(sentences)
                for dic in boundaries_dicts:
                    if dic['span'][0] == dic['span'][1]:
                        dic['claim'] = sentences[0]

                if verbose:
                    print ('In sentence: '+sentences[0])
                    print ('['+str(boundaries_dicts[0]['span'][0])+', '+str(boundaries_dicts[0]['span'][1])+']: '
                        +boundaries_dicts[0]['claim'])
                    print ()

                return boundaries_dicts[0]['claim'], score
            else:
                return sentence, score
            
        else:
            if verbose: print(f"non-checkworthy: {score}")
            return "The claim is not check-worthy", score
    
    async def detect_2(
            self, sentence: str, 
            boundary_extract:bool=False, 
            llm_classify:bool=False, 
            verbose: bool=False,
            score_threshold: float = 0.5
        ):
        """ 
            Check if there is check-worthy claims in the sentence
            using ClaimBuster API
        """
        # Define the endpoint (url) with the claim formatted as part of it, api-key (api-key is sent as an extra header)
        api_endpoint = f"https://idir.uta.edu/claimbuster/api/v2/score/text/{sentence}"
        request_headers = {"x-api-key": Claim_Buster_api}

        # Send the GET request to the API and store the api response
        async with aiohttp.ClientSession() as session:
            claimbuster_task = asyncio.create_task(session.get(url=api_endpoint, headers=request_headers))
            if llm_classify:
                chat_task = asyncio.create_task(openai.ChatCompletion.acreate(
                    model=Model.GPT4,
                    messages=[
                        {"role": "system", "content": """Label the following claims as 'Y' if they are verfiable by datasets about social issues like Climate Change, Economy, etc.; otherwise 'N' according to the following format.
                        \{
                            "explain": "<TODO: explain why the claim can be verified using data or not. Think step by step.>",
                            "verdict": "<TODO: Y or N>"
                            
                        \}"""},
                        {"role": "user", "content": f"{sentence}"},
                    ]
                ))
            else:
                chat_task = None

            # Wait for both tasks to complete
            results = await asyncio.gather(claimbuster_task, chat_task)

            # Process the results
            claimbuster_response = await results[0].json()
            chat_response = results[1]["choices"][0]['message']['content'] if chat_task else None

        # Continue with your code...
        if verbose: print("check result: ", claimbuster_response)
        # if the score is > .5 --> checkworthy
        if len(claimbuster_response['results']) == 0:
            if verbose: print("non-checkworthy: 0")
            return "The claim is not check-worthy", 0
        score = claimbuster_response['results'][0]['score']
        if score > score_threshold:
            if llm_classify:
                """
                    create prompt chat to detect if the claim is check-worthy and data related.
                """

                verdict = re.search(r'"verdict": "(Y|N)"', chat_response).group(1)
                explain = re.search(r'"explain": "(.*?)"', chat_response, re.DOTALL).group(1)
                if verdict == 'Y':
                    if verbose: 
                        print(f"statistically interesting: {claimbuster_response['results'][0]['score']}")
                        print(f"explain: {explain}")
                else:
                    if verbose: print("statistically unrelated")
                    # negative means unrelated to data
                    return explain, -score
            
            # # extract the boundary
            if boundary_extract:
                sentences = [sentence]
                boundaries_dicts = claim_boundaries_client.run(sentences)
                for dic in boundaries_dicts:
                    if dic['span'][0] == dic['span'][1]:
                        dic['claim'] = sentences[0]

                if verbose:
                    print ('In sentence: '+sentences[0])
                    print ('['+str(boundaries_dicts[0]['span'][0])+', '+str(boundaries_dicts[0]['span'][1])+']: '
                        +boundaries_dicts[0]['claim'])
                    print ()

                return boundaries_dicts[0]['claim'], score
            else:
                return sentence, score
            
        else:
            if verbose: print(f"non-checkworthy: {score}")
            return "The claim is not check-worthy", score


async def main():
    detector = ClaimDetector()
    sentence = "More than 80% of Vietnam forest coverage was destroyed by invasion of crickets."
    start = time.perf_counter()
    x,y = await detector.detect_2(sentence, llm_classify=True, verbose=True)
    end = time.perf_counter()
    print(f"Time elapsed: {end - start:0.4f} seconds\n{'@'*100}")

    start = time.perf_counter()
    x,y = await detector.detect(sentence, llm_classify=True, verbose=True)
    end = time.perf_counter()
    print(f"Time elapsed: {end - start:0.4f} seconds")

if __name__ == "__main__":
    asyncio.run(main())