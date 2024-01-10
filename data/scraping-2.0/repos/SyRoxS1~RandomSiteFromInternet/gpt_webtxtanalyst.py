from openai import OpenAI
import tiktoken
with open("key.key","r") as f:
    key = f.readline()
client = OpenAI(api_key=key)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def GptWebsiteAnalyser(WebTxt):
    if len(WebTxt) > 3000:
        WebTxt = WebTxt[:3000]
    nbtoken = num_tokens_from_string(WebTxt, "cl100k_base")
    print("TOKENSENDPRICE : " + str(nbtoken) + " in $ : " + str(0.001 * (nbtoken/1000)))
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[    
    {"role": "system", "content": "You are a website analyst and will respond to me briefly what it this website used for or about here is all the text from the website I want the response to be short and clear for exemple something like : this is an apache basic configuration page, or : this is a russian news website, i want it short in really few words"},
    {"role": "user", "content": WebTxt}
  ],
      max_tokens=25  
    )
    completion_message = completion.choices[0].message
    content_field = completion_message.content

    return(content_field)

