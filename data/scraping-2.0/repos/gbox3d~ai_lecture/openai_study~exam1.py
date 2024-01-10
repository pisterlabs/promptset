#%%
import openai

openai.api_key = open("../api_keys/openapi.key.txt").read().strip('\n')

#%%

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role" : "user", "content" : "이재명은 누구입니까?"},
    ])
print(completion)

# %%
#print messages
print(completion.choices[0].message.content)



# %%
