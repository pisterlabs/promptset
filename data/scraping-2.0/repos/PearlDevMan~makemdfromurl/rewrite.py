import openai
def getrewrite(text,filename):
    openai.api_key = 'sk-xxxxxxx'
    # Define the prompt and the original article
 
    path = "result//" + filename + "_rewrited" + ".txt"
    rewritten_article = ""
    # Call the OpenAI API to generate a rewritten version of the article
    with open(path, "w") as f:
        prompt = [
            {"role": "system", "content": "Rewrite the following article, keeping the lines starting with #, ## or ### intact and related article line:"},
            {"role": "user", "content": text}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            #max_tokens=1024,
            #n=1,
            #stop=None,
            temperature=0.3,
        )
        # Extract the rewritten article from the response
        rewritten_article = str(response.choices[0].message["content"]).strip()

        f.write(rewritten_article)
    #print(rewritten_article)
    return rewritten_article

