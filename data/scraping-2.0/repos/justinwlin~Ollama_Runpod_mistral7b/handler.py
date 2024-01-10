from langchain.llms import Ollama
import runpod

def handler(event):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = event['input']
    prompt = job_input["prompt"]

    llm = Ollama(model="mistral:7b")
    result = llm.predict(prompt)
    return result

runpod.serverless.start({
    "handler": handler
})
