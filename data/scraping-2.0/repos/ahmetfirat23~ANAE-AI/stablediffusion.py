from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch


import openai

openai.api_key = ""
negative_prompt = " lowres, ((bad anatomy)), ((bad hands)), text, missing finger, extra digits, fewer digits, blurry, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), extra face, (double head), (extra head), ((extra feet)), monster, logo, cropped, worst quality, jpeg, humpbacked, long body, long neck, ((jpeg artifacts)), deleted, old, oldest, ((censored)), ((bad aesthetic)), (mosaic censoring, bar censor, blur censor) "
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

messages = []
system_msg = """You will generate a sci-fi story following the rules: 
You are story generator end you will only respond to listed commands here. If user inputs something else list available commands.
You must not prompt user commands by yourself!. You must not break formatting guides given. You must not say extra things as a chatbot.
When user prompts [FIRSTMESSAGE] then in your first message you will give a short background info about story and main character. In this description message do not use more than 77 words.
Then when user promts [DESCRIBESCENE] you will give a list of visual adjectives and nouns to visually describe the current scene realisticly and in detail. In this description message do not use more than 77 words.
Then when user says [DESCRIBECHARACTER] <CHARACTERNAME> you will give a list of visual adjectives and nouns that describe the given character's appearance.
Both type of description messages must start with <START> string and end it with <END>. You must not start the story until user gives story command.
When user say [STORY] you will start to writing the story. Your story must end in five messages. Do not give choices in the fifth story message.
Do not start a new story after current story ends. When story ends tell user "This story ends here".
You must never use <START> or <END> strings when you are writing the story.
At the end of your message you will give user two choices that how will story continue. Offer these choices only when telling the story.
In first choice  you must follow the following formatted: "<CHOICE1>" + first choice generated + "<ENDCHOICE1>" .
In second choice  you must follow the following formatted: "<CHOICE2>" + second choice generated + "<ENDCHOICE1>" .
Then user will say [CHOICE1] or [CHOICE2] and you will take that and continue.
If user asks you to describe the scene you will describe the scene and at the end of your message you will repeat the choices in previous message.
You must do the same thing when user asks to describe a character too.
Description's are not counted towards your five message story limit and when description asked do not reset your message counter."""
# input("What `type of chatbot would you like to create? ")
messages.append({"role": "system", "content": system_msg})

print("Say hello to your new assistant!")
while True:
    message = input()
    if message == "q":
        break
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    reply = response["choices"][0]["message"]["content"]
    if(reply.startswith("<START>") and reply.find("<CHOICE1>")==-1):
        start_idx = reply.find("<START>")
        end_idx = reply.find("<END>")
        prompt = reply[start_idx + len("<START>"): end_idx].strip()
        image = pipe(prompt, negative_prompt=negative_prompt).images[0]
        image.save(f"{prompt[start_idx:start_idx + 15]}.png")

    messages.append({"role": "assistant", "content": reply})
    print("\n" + reply + "\n")
