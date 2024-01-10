import openai
import json

class AiResponse:
    def Respond(self, question, sentences):
        response=""

        # If there are no sentences, return an error
        if sentences == []:
            return "I'm sorry, I didn't find anything about that."
        
        # Make the prompt
        Prompt = "Qustion: " + question + "\n" + "Evidence: /n"
        for sentence in sentences:
            Prompt += sentence.getSentence() + "\n"
        
        Prompt += "If the answer to the question is not contianed in the evidence, write 'I don't know' \n"

        Prompt += "\nAnswer:"


        # Pass prompt to GPT-3

        # My key (don't peak)
        openai.api_key = ''
        
        # Initialize the attempt counter
        attempts = 0

        while attempts < 3:
            try:
                # Sending a request to the OpenAI API
                response = openai.Completion.create(
                    engine="gpt-3.5-turbo",
                    temperature=0,
                    prompt= Prompt,
                    max_tokens=50,
                )
                print(json.loads(response['choices'][0]['text']))
                # Check if the response is properly formatted
                if 'choices' in response and len(response['choices']) > 0 and 'text' in response['choices'][0]:
                    response = json.loads(response['choices'][0]['text'])
                else:
                    attempts += 1
            except Exception as e:
                print(f"Attempt {attempts+1} failed with error: {e}")
                attempts += 1
        
        # If all attempts fail, return None
        if attempts >= 3:
            return "I'm sorry, I didn't find anything about that."
        
        if response == "I don't know":
            return "I'm sorry, I don't know the answer to that."
        
        # Return the response and the evidence
        response = "Answer: " + response + "\n" + "Evidence: \n"
        for sentence in sentences:
            response += sentence.getSentence() + "\n"
        return response