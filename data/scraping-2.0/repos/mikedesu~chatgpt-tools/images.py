
import os
import openai
import sys
from datetime import datetime

def main():

    t0 = datetime.now()

    openai.organization = 'evildojo'
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    if len(sys.argv)!=3:
        print("usage:")
        print()
        print("python3 images.py <prompt> <n>")
        sys.exit(-1)

    my_prompt = sys.argv[1]
    n = int(sys.argv[2])
    max_token_ct = 2048 - len(my_prompt)
    #my_model = 'text-davinci-003'
    
    #test_obj = openai.Completion.create(
    #  model=my_model,
    #  prompt=my_prompt,
    #  max_tokens=max_token_ct,
    #  temperature=0
    #)
    
    response = openai.Image.create(prompt=my_prompt,
        n=n,
        size="1024x1024"
    )

    #print(response)

    #print(my_prompt)
    #print("-"*20)
    for d in response["data"]:
        print(d["url"],"\n")
    #print("-"*20)

    #text = test_obj["choices"][0]["text"]
    #text = text.strip()

    t1 = datetime.now()

    t_diff = t1-t0

    #print(test_obj)
    #print(text)
    #print(t_diff)


if __name__ == '__main__':
    main()

