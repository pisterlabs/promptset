import openai
print("WARNING:This feature is highly expermental as the AI Model for coding has not been fleshed out yet.")
input = input("Do you agree y/n")
lmao = input("Your API KEY")

if(input == "n"):
    exit()


openai.api_key = lmao

def generateModel(model,story,tokens):
    completions = openai.Completion.create(
        engine=model,
        prompt=story,
        max_tokens=tokens,
        n=1,
        stop=None,
        temperature=1,
)
    generated = completions.choices[0].text
    return generated
def generateImage(model,story,tokens):
    completions = openai.Image.create(
        engine=model,
        prompt=story,
        max_tokens=tokens,
        n=1,
        stop=None,
        temperature=1,
)
    generated = completions["data"][0]["url"]
    return generated

v = generateModel("davinci","Generate a random story for a video game",8192)
a = generateModel("codex",f"Generate scripts for a video game based on this video game story {v}",8192)
k = generateImage("image-alpha-001",f"Generate art assets for a video game based on a video game story {v}",8192)

class VideoGame():
    def __init__(self,story,assets,scirpts):
        print("Here is your randomly generated video game!")
        print(f"Story:{story}")
        print(f"Assets:{assets}")
        print(f"Scirpts:{scirpts}")

VideoGame(v,k,a)
