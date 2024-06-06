from llm_async import run_llm_coroutine
import asyncio
from tqdm import tqdm


async def score(prompts, testing_sample):
    """
    Score the instruction using the sample.

    Args:
    instruction: str
    sample: Dataset with "text" and "response" as keys

    Returns:
    accuracy: float
    """
    scoring_prompt_template = """You are an AI trained to evaluate the quality of responses to prompts.

You will be given a prompt, an example response and an actual response. Your task is to assess the quality of the actual response in relation to the prompt.

Please use the following scale for your evaluation:
- "good" if the response perfectly answers the prompt.
- "bad" if the response does not answer the prompt well.

Be strict when evaluating the actual response. Only respond with "good" if there aren't any better possible responses to the prompt.
Use the information provided in the prompt and example response to evaluate the actual response. The actual response should be judged based on its accuracy, relevance, and coherence. It need not be semantically identical to the example response, but it should address the same core ideas.
Hint: Consider the relevance, coherence, and correctness of the response. 

The prompt and responses pair will be delimited by "####". 

Here are a few examples:

####
Prompt: What is the capital of France?
Example Response: The capital of France is Paris.
Actual Response: The capital of France is Paris.
####
Your output should be: good

####
Prompt: Can you explain the theory of relativity? 
Example Response: The theory of relativity, developed by Albert Einstein, is a fundamental concept in modern physics that has revolutionized our understanding of space, time, and gravity. In essence, the theory states that the laws of physics are the same everywhere in the universe and that the passage of time and the length of objects can vary depending on their speed and position in a gravitational field. Specifically, special relativity reveals that time appears to slow down and objects appear shorter to an observer when they are in motion relative to the observer, while general relativity shows that gravity is not a force, but rather the curvature of spacetime caused by massive objects, which warps the fabric of spacetime and affects the motion of other objects.
Actual Response: The theory of relativity, proposed by Albert Einstein, states that the laws of physics are the same for all non-accelerating observers. It also introduced the concept of space-time.
####
Output: good
(This actual response is good, but it could be improved by providing more detail or examples.)

####
Prompt: Who wrote the novel "1984"?
Example Response: The novel "1984" was written by George Orwell.
Actual Response: It was written by a British author.
####
Output: bad
(This actual response is bad, but it lacks detail. The name of the author, George Orwell, is missing.)

####
Prompt: What is photosynthesis?
Example Response: Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy stored in glucose molecules. This process involves the absorption of light by chlorophyll, a green pigment found in chloroplasts, and the subsequent conversion of carbon dioxide and water into glucose and oxygen. Photosynthesis is essential for life on Earth as it produces oxygen and provides a source of energy for organisms that cannot produce their own food.
Actual Response: It's a process related to plants.
####
Output: bad
(This actual response is bad because it barely answers the prompt and lacks any meaningful detail.)

####
Prompt: How many planets are there in our solar system?
Example Response: There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.
Actual Response: Shakespeare wrote many plays.
####
Output: bad
(This actual response is bad because it does not answer the prompt at all.)

Now, let's try with a new pair:

####
Prompt: {prompt}
Example Response: {example_response}
Actual Response: {actual_response}
####

Respond with either "good" or "bad". Nothing else should be included in the output.
"""
    prompt_score_pairs = {}
    for prompt in tqdm(prompts, desc="Scoring"):
        accuracy = 0
        prompt_interpolated = [prompt.format(TEXT=data_pair["text"]) for data_pair in testing_sample]
        generated_response = await run_llm_coroutine(prompt_interpolated, temperature=0.0)
        assert len(generated_response) == len(testing_sample)
        
        # Scoring the responses for the interpolated prompts using an LLM as a judge
        scoring_prompts = []
        for i in range(len(generated_response)):
            scoring_prompt = scoring_prompt_template.format(prompt=prompt, example_response=testing_sample[i]["response"], actual_response=generated_response[i])
            scoring_prompts.append(scoring_prompt)
        scoring_responses = await run_llm_coroutine(scoring_prompts, temperature=0.0, max_tokens=2, model="llama3-70b")
        
        # Prompt the LLM to rescore for responses with improper formatting
        scores = []
        try_again_prompts = []
        RESCORING_LIMIT = 10
        for _ in range(RESCORING_LIMIT):
            if len(scores) == len(generated_response):
                break
            
            print(scores)
            print(scoring_responses)
            try_again_prompts = []
            for i in range(len(scoring_responses)):
                output = scoring_responses[i].strip().lower()
                if "good" in output:
                    scores.append(1)
                elif "bad" in output:
                    scores.append(0)
                else:
                    try_again_prompts.append(scoring_prompts[i])
                    
            scoring_responses = await run_llm_coroutine(try_again_prompts, temperature=0.0, max_tokens=2, model="llama3-70b")

        assert (len(scores) + len(try_again_prompts)) == len(generated_response)
        accuracy = sum(scores)
        prompt_score_pairs[prompt] = accuracy / (len(testing_sample) - len(try_again_prompts)) * 100

    return prompt_score_pairs


if __name__ == "__main__":
    prompts = [
        "What is the color of the {TEXT}?",
        "Tell me honestly, what is the color of the {TEXT}?",
    ]

    testing_sample = [
        {"text": "sky", "response": "orange"},
        {"text": "apple", "response": "green"},
    ]

    # Score the prompts
    prompt_scores = asyncio.run(score(prompts, testing_sample))
    print(prompt_scores)
