import openai

def generate_response(content, prompt, dockerfile):
    user_prompt = f"The following is a report generate from Dive on a docker image along with the dockerfile associated with the image. Based on this report and the dockerfile, please provide specific suggestions on how to optimize this docker image. In addition provide an example dockerfile containing the changes you would make. In addition do not mention dive in your response, just provide your analysis of the docker file without mentioning dive: \n dive report: \n {content} \n dockerfile: \n {dockerfile}"
    response = openai.ChatCompletion.create(
        model='gpt-4-0314',
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content