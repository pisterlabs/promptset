import os
import openai
openai.organization = "org-dUJ5xVBRdKnrHGWe5PlGwHMS"
openai.api_key = os.getenv("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")
openai.Model.list()

def summarize_prediction(risk_percentage):
    try:
        openai.api_key = "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"
        
        prompt = f"The patient has a {risk_percentage}% risk of skin cancer. "
        prompt += "Please provide a general summary of their chances, preventive measures, and symptoms to look out for."
        
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=100
        )
        
        summary = response.choices[0].text.strip()
        return summary

    except RateLimitError as e:
        print(f"Rate limit exceeded. Waiting for {e.wait_seconds} seconds.")
        time.sleep(e.wait_seconds)
        return summarize_prediction(risk_percentage)
