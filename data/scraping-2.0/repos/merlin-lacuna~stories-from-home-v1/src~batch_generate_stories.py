import os
import re
import openai
import datetime
from contextlib import redirect_stdout

oa = openai
oa.api_key = os.getenv("OPENAI_API_KEY")

earth = "davinci:ft-personal-2022-05-08-13-37-54"
water = "davinci:ft-personal:water-2022-03-31-23-56-04"
fire = "davinci:ft-personal:fire-2022-07-06-02-12-31"
air = "davinci:ft-personal:air-2022-07-05-23-19-23"

maxlength = 256
selectedmodel = air

def trim_output(completion):
    try:
        if completion[-1] in ('.', '?', '!'):
            # print("matched end")
            trimmedoutput = completion
        else:
            try:
                # print("matched incomplete")
                re.findall(r'(\.|\?|\!)( [A-Z])', completion)
                indices = [(m.start(0), m.end(0)) for m in re.finditer(r'(\.|\?|\!)( [A-Z])', completion)]
                splittuple = indices[len(indices) - 1]
                trimmedoutput = completion[0:splittuple[0] + 1]
            except:
                trimmedoutput = completion
    except:
        trimmedoutput = completion

    return trimmedoutput


def get_act(myprompt, maxt, element):
    response = openai.Completion.create(
        model=element,
        prompt=myprompt,
        temperature=1,
        max_tokens=maxt,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1,
        stop=["Act "]
    )

    story = response.choices[0].text

    lstory = story.replace("\n", " ")
    lstory = lstory.replace("I'm a forest,", "I am")
    lstory = lstory.replace("I am a forest,", "I am")
    lstory = lstory.replace("I'm just a forest,", "I am")
    lstory = lstory.replace("I am just a forest,", "I am")
    lstory = lstory.replace("Forest: ","")

    return ' '.join(lstory.split())

gentype = "air_island_C02_manuloa"
gencount = 2

intro = "The following play reveals the inner monologue of a rich and fertile island. It is divided into several acts. Throughout these acts, the island describes its inner and outer transformation:\n\nThe first act starts like this:\n\n"

act0descr = "Act 0 description: The c02 levels in air above the island are not not yet known. The island introduces itself and describes its surroundings:\n----\nIsland: They say that I host one the largest volcanoes on Earth. My slopes are gentle and vast. Sometimes I shed tears of liquid lava which flow smoothly down my face in golden rivulets. My crown kisses the sky and I suck in gulps of the rich atmosphere. I have a clear view of the sun and we gaze at each other constantly. I am constantly tasting the unsullied gasses which waft by my mouth and I test them for evidence of poison. I am fearful that the air is slowly becoming poisonous so I always need to taste it."

act1descr = "Act 1: The c02 levels in the air above the island are at dire, extreme levels. The carbon dioxide is causing suffocating greenhouse effects, smothering the island with oppressive heat. The CO2 springs with an implacable crescendo. Its load swells incautiously. It's the burden of the cosmos itself. Everything around stagnates in stasis. A grey, stifling density pervades the entire island's atmosphere, decking everything that exists on its way. An airy wave of faded hopes is ruling.\n----\nIsland: "

act2descr = "Act 2:  And still, the c02 levels in the air above the island are at dire, extreme levels. The carbon dioxide is still causing suffocating greenhouse effects, smothering the island with oppressive heat. The CO2 still springs with an implacable crescendo. Its load swells incautiously. It's the burden of the cosmos itself. Everything around stagnates in stasis. A grey, stifling density pervades the entire island's atmosphere, decking everything that exists on its way. An airy wave of faded hopes is still ruling.\n----\nIsland: "

act3descr = "Act 3:  Things still haven’t changed, the c02 levels in the air above the island are at dire, extreme levels. The carbon dioxide is still causing suffocating greenhouse effects, smothering the island with oppressive heat. The CO2 still springs with an implacable crescendo. Its load swells incautiously. It's the burden of the cosmos itself. Everything around stagnates in stasis. A grey, stifling density pervades the entire island's atmosphere, decking everything that exists on its way. An airy wave of faded hopes is still ruling.\n----\nIsland: "





for x in range(gencount):
    # GET PROMPT FOR ACT1
    prompt = intro + act0descr + act1descr
    print("\n\n<PROMPT>")
    print(prompt)
    print("</PROMPT>\n\n")
    act1raw = get_act(prompt, maxlength, selectedmodel)
    act1 = trim_output(act1raw)
    # print(act1)
    act1static = act1 + '\n\n'

    # GET PROMPT FOR ACT2
    prompt = intro + act0descr + act1descr +  act1static + act2descr
    print("\n\n<PROMPT>")
    print(prompt)
    print("</PROMPT>\n\n")
    act2raw = get_act(prompt, maxlength, selectedmodel)
    # print(act2raw)
    act2 = trim_output(act2raw)
    # print(act2)
    act2static = act2 + '\n\n'

    # GET PROMPT FOR ACT3
    prompt = intro + act0descr + act1descr + act1static + act2descr + act2static + act3descr
    print("\n\n<PROMPT>")
    print(prompt)
    print("</PROMPT>\n\n")
    act3raw = get_act(prompt, maxlength, selectedmodel)
    act3 = trim_output(act3raw)

    story = '\nAct 1: ' + act1static + 'Act 2: ' + act2static + 'Act 3: ' + act3

    # datetime object containing current date and time
    dt_string = datetime.datetime.now().strftime("%d-%m-%Y_%H_%M_%S")

    print('-----------')
    print('\n\n\nSample #' + dt_string + ":")
    print(story)

    finalfile = '../generations/' + dt_string + '_' + gentype + '.txt'

    try:
        with open(finalfile, 'w', encoding="utf-8") as f:
            with redirect_stdout(f):
                print(story)
    except:
        print("File write error")