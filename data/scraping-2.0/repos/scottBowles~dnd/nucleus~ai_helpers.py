from django.conf import settings

OPENAI_API_KEY = settings.OPENAI_API_KEY


def openai_summarize_text(text):
    """
    Summary the given text using openai
    This is first to be used for summarizing long game logs (~13000 character) into a short summary
    """
    import openai

    openai.api_key = OPENAI_API_KEY

    # text = (
    #     "Given the following game log from a game of dungeons and dragons, "
    #     "give it an episode title that is a few words long, "
    #     "a brief description that is a few sentences long, "
    #     "and a synopsis that is a few paragraphs long. "
    #     "Also list any places, characters, races, associations, or items. "
    #     "The response should be in the form of a json object with the following keys: "
    #     '"title", "brief", "synopsis", "places", "characters", "races", "associations",'
    #     ' "items". "places", "characters", "races", "associations", and "items" should be arrays.'
    #     " \n\n The game log is as follows:\n\n" + text + "\n\nResponse:\n\n"
    # )
    # text = (
    #     'Give a synopsis of the following text in a few paragraphs. Be especially sure to retain all events, characters, places, items, and essential details that are mentioned in the log.\n\nText: """\n\n'
    #     + text
    #     + '\n\n""" Response:\n\n'
    # )

    text = (
        '''
        Given the following game log from a role playing game, give it an episode title of a few words and a brief description of a few sentences. Also list all places, characters, races, associations, and items that are mentioned. If you are not sure which something is, include it in both. The response should be in the form of a json object with the following keys: "title", "brief", "places", "characters", "races", "associations", "items". For example:
        '{"title":"My Title","brief":"The Branch, lead by Ego, invents AI using the ReDream. On their way to Hielo, they have to fight off void spiders. They make it to Hielo, and leave the nascent AI to mature.","places":["Hielo"],"characters":["Ego","Void Spiders","AI"],"races":["Void Spiders","AI"],"associations":["The Branch"],"items":["ReDream"]}'
        Text:
        """
    '''
        + text
        + '''
        """
        Response:
        '''
    )

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=text,
        temperature=0,
        max_tokens=357,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=1,
    )
    return response
    # return response["choices"][0]["text"]


def openai_summarize_text_chat(text):
    """
    Summary the given text using an openai chat model
    This is first to be used for summarizing long game logs (~13000 character) into a short summary
    """
    import openai

    openai.api_key = OPENAI_API_KEY

    # text = (
    #     "Given the following game log from a role playing game, "
    #     "give it an episode title that is a few words long, "
    #     "a brief description that is a few sentences long, "
    #     "and a synopsis that is a few paragraphs long. "
    #     "Also list any places, characters, races, associations, or items that are mentioned. "
    #     "The response should be in the form of a json object with the following keys: "
    #     '"title", "brief", "synopsis", "places", "characters", "races", "associations",'
    #     ' "items". "places", "characters", "races", "associations", and "items" should be arrays.'
    #     " \n\n The game log is as follows:\n\n" + text + "\n\nResponse:\n\n"
    # )
    text = (
        '''
        Given the following game log from a role playing game, give it a creative episode title of a few words and a brief description of a few sentences. Also list all places, characters, races, associations, and items that are mentioned. If you are not sure which something is, include it in both. The response should be in the form of a json object with the following keys: "title", "brief", "places", "characters", "races", "associations", "items". For example:
        '{"title":"My Title","brief":"The Branch, lead by Ego, invents AI using the ReDream. On their way to Hielo, they have to fight off void spiders. They make it to Hielo, and leave the nascent AI to mature.","places":["Hielo"],"characters":["Ego","Void Spiders","AI"],"races":["Void Spiders","AI"],"associations":["The Branch"],"items":["ReDream"]}'
        Text:
        """
    '''
        + text
        + '''
        """
        Response:
        '''
    )
    messages = [{"role": "user", "content": text}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        # prompt=text,
        temperature=0.6,
        max_tokens=357,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=1,
    )
    return response
    # return response["choices"][0]["text"]


def openai_titles_from_text_chat(text):
    """
    Summary the given text using an openai chat model
    This is first to be used for summarizing long game logs (~13000 character) into a short summary
    """
    import openai

    openai.api_key = OPENAI_API_KEY

    text = (
        '''
        Given the following game log from a role playing game, provide five possible one-phrase episode titles. One title should be descriptive, one evocative, one pithy, one funny, and one entertaining. The response should be a json object with the key "titles" and the value as an array of five strings. For example:
        '{"titles":["My Title","My Title","My Title","My Title","My Title"]}'
        Text:
        """
    '''
        + text
        + '''
        """
        Response:
        '''
    )
    messages = [{"role": "user", "content": text}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        # prompt=text,
        temperature=0.5,
        max_tokens=357,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=1,
    )
    return response
    # return response["choices"][0]["text"]


# from nucleus.ai_helpers import test_helper
def test_helper():
    from nucleus.gdrive import fetch_airel_file_text
    from nucleus.models import GameLog
    import json

    log = GameLog.objects.first()
    text = fetch_airel_file_text(log.google_id)
    response = openai_titles_from_text_chat(text)
    res_json = response["choices"][0]["message"]["content"]
    try:
        obj = json.loads(res_json)
        return obj
    except:
        print("bad json")
        print(res_json)
        return res_json


def openai_shorten_text(text, percent=70, max_tokens=2000):
    """
    Summary the given text using openai
    This is first to be used for summarizing long game logs (~13000 character) into a short summary
    """
    import openai

    openai.api_key = OPENAI_API_KEY

    text = (
        f"Make the following text {percent}% shorter without losing any content. Especially be sure not to leave out any characters, places, items, etc. Text: "
        + text
        + " Response:"
    )

    response = openai.Completion.create(
        engine="code-davinci-002",
        prompt=text,
        temperature=0.3,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=1,
    )
    return response
    # return response["choices"][0]["text"]


# def summarize_text(text):
#     """
#     Summarizes a text using the google text summarizer
#     """
#     with build("documentai", "v1beta2", developerKey=GOOGLE_API_KEY) as service:
#         body = {
#             "document": {
#                 "content": text,
#                 "mime_type": "text/plain",
#             },
#             "encoding_type": "UTF8",
#         }

#         request = service.documents().analyzeSyntax(body=body)
#         response = request.execute()
#         return response


# def summarize_long_text(text):
#     """
#     Summarize text using huggingface transformers with the LongT5 model
#     """
#     from transformers import pipeline

#     summarizer = pipeline(
#         "summarization",
#         model="Udit191/autotrain-summarization_bart_longformer-54164127153"
#         # "summarization", model="pszemraj/long-t5-tglobal-base-16384-book-summary"
#     )
#     res = summarizer(text)[0]["summary_text"]
#     return res

# model="pszemraj/long-t5-tglobal-base-16384-book-summary"
# [{'summary_text': 'Darnit and General Arrow are on their way to the island of Dr. Whidmoreeau when Tre Arrow shows up with a briefcase full of equipment. He tells them that the bot army is outnumbering the Freelander forces so they can\'t get through without using EMP. They have an island in the center of the map where they want to target. LeFlur comes back with two giant boxes, which she sets down and puts into boxes. She says that there\'s no organic man at all, but that they know it\'s accessible by some griffon-like creatures. The group decides to blow up the island from orbit. It\'s just a simple operation: fly to the Island, set up four boxes on the corners, get away only 60 feet away, and use hysterical detonating devices to disrupt the entire operation. Hrothef asks about the other weapons available for the group, and LeFler says that they don\'t really need anything more than a few bullets. They also think that the islands might be vulnerable to EMP attacks because they seem to possess something called a "hive mind" system. When asked what kind of weaponries the bots carry, Izar replies that they mostly expect Confederation gear. Their attack will probably take place during the sunset, so they plan to shoot at the sun as it rises. As they discuss the bombs, Dorinda becomes nervous about the possibility of being captured by one of the rebels. She wants to keep her distance from the rest of the party since she knows how to dodge any Emps that come after them. In fact, she has been thinking that she can hide behind Sefarinana so that she won\'t be seen by anyone else. Finally, they decide to arm the boxes and put them on the island. They give instructions on how each box should be set up and how to arm parachutes like ropes. After setting up the boxes, Dumbleda flees over the thing. On the island, Dorida sees strange plates on the building. She starts placing her bomb, even though she doesn\'t know exactly what it is. Two robots walk out onto the island carrying twelve bots. One of them knocks off the island while another shoots at him. Both men try to figure out what\'s going on, but both realize that'}]
# model="Udit191/autotrain-summarization_bart_longformer-54164127153"
# [{'summary_text': 'Dorinda, Hrothulf, Izar, and Izar were on the same side of the island as the Bot Line. The Bot Line was composed of a number of bots, some of them armed with plasma rifle fire, some with scuba tanks, and some with other non-lethal weaponry. They were able to move around the island and attack us from all sides, but they were not able to take out any of us. Their armor was strong enough to protect them against EMP blasts, but it was too weak to stop them from hitting us. The bot closest to Ego was alerted to this and tried to charge at her, but was stopped by one of the other bots. The other one tried to follow suit, but failed. The rest of the bot army was able to avoid them, but the one that did succeed was armed with an EMP weapon.'}]

# text = "\ufeff(Between episodes) As we’ve been chilling in our med tent, Darnit has been in an audience with the leader of this whole operation: Tre Arrow. \r\n\r\n\r\nOur girl in green comes in with the briefcase. “Alright. It looks like you’re on.” She opens the briefcase and there are earbuds and go-pros. “They’re sending you in. Plug in. Your friend Darnit should be online in just a moment. He’ll debrief us.”\r\n\r\n\r\n“We figured a way to make this work. Y’all have a part to play, and I and General Arrow will be adjusting from HQ here.”\r\nA small orb in the briefcase opens up and projects a holograph\r\n\r\n\r\n  \r\n\r\n\r\n\r\nDarnit: “The Bot Line currently outnumbers the Freelanders force so they just can’t get through. They’ve had successful aerial moves, but after the first wave the bots use their EMP capacity to eliminate anything aerial. But now we’ve shown up with an aerial assault that is impervious to EMP. You see right in the middle of the map? It’s a floating island that we’re going to target.” \r\n\r\n\r\nLeFleur comes back in with two giant duffel bags. She sets them down and starts pulling out toaster oven-sized gray silver boxes. They have just a couple buttons on them. She says, “We don’t know what’s going on with that island, but we do know it showed up at the same time as the bot army, and we know it’s accessible by the griffons. Your job is to blow it up.”\r\nHrothulf: “There’s no organic guy?”\r\n“Not that we’ve noticed, no.”\r\n“Not to be that guy, but couldn’t we just nuke the whole thing from orbit?”\r\n“There is, what we call in the biz, rules of engagement.”\r\n“We don’t really do that.”\r\n“Guys, you really gotta keep that on the DL ‘cause we’re doing what they can to let us bend the rules and help them. It’s a simple in and out. All you need to do is fly to the island, set up the boxes on the four corners, get at least 60’ away but not more than 100’ away, because the only way we can work this is to use sonic detonators. Within 60’ you get hurt. Outside of 100’ the sonic detonators that aren’t susceptible to EMP won’t work. After you do that you come right back here and the freelanders take o er. We’re hoping that disrupts the bots and we can go in and disrupt the whole operation and hopefully set their operation back 3–5 years minimum. Enough time to give our lawyers to build up a solid case against them. I hesitate to ask, but, any questions?” \r\nHrothulf: “Any secondary objectives?”\r\nLeFleur: “You’ll probably have to deal with assault from the bots. We haven’t seen them fly, but they have proven effective with their nonlethal weaponry. They are quick to self-repair, quick to assist one another in self-repair, and they seem to have some sort of shared hive-mind system, so they’re able to quickly respond to threats.”\r\n“If we see a target of opportunity to we have permission to engage?”\r\n“That’s a hard no. And Whidmore is off the table.”\r\n“What kind of weaponry and weaknesses do the bots have?”\r\n“It’s predominantly what you would expect from Confederation riot gear. A lot of stunning, rubber bullets, things that would dissuade without fatalities. With the EMP they’ve also got decent taser capacity. We’ve seen them shoot their tasers as far as 20’. We’ve attempted to incapacitate and bring one of them back so we can understand them better. However it requires an overwhelming force to knock one out and remove it from the others. But when we did manage with one of them, on the transit back in turned to dust. It seems to have some sort of failsafe.”\r\nEgo: “What do they seem to be made of?”\r\n“Some kind of silicon.”\r\nHrothulf: “What are they powered by?”\r\n“We’re not really sure. The one we did manage to get ahold of just reduced itself to a big old pile of sand. The suns will be in descent in a couple of hours and that’s when you should roll out. When we were doing our drone flyovers we found we could get further at twilight than at any other time, so we want to take advantage of the sunsets.”\r\nHrothulf: “Since we’re going on this mission for you, you gonna hook us up with any weapons? What kind of hardware you got laying around here?”\r\n“We were informed you were quite capable on your own. We are supplying the bombs.”\r\nDarnit: “Yeah I didn’t want to be a burden.”\r\nIzar: “Darnit what is the plan with the bombs?”\r\n“Well there are four bombs and four of you. So it makes sense that each of you would carry a bomb and a detonator. They’re probably 30 lbs of explosives each. We saw how taxing it was for Sefarina in this heat, so we want to keep the load for the griffins light.”\r\n“Do we think it’s going to take all four?”\r\n“To create the diversion we need, yes. Of course, should something go wrong, your safety is more important than the mission, so if one of you needs to pull back we’ll make do. That’s why General Arrow and I will be in your ears and watching through your eyes the whole time.”\r\n“What do we do when Dorinda thinks she knows where Whidmore is and is irresistibly drawn to go after him?”\r\n\r\n\r\nDorinda is a little nervous about the EMP part. A little less about being overtaken by Daryo, but she was thinking, in terms of formation, that she given sufficient distance from the party and made invisible through Sefarina, so she can dodge any EMPs that come at us.\r\n\r\n\r\nHrothulf: “Do you have any suggestions for fighting the bots? What sorts of weaknesses and has anything been successful against them? And follow-up, are they susceptible to their EMP blasts?”\r\n“No to the last one. We’re guessing it’s a frequency thing or something? Weaknesses are hard to identify. Avoiding them is works pretty well, if you can do that. Their armor isn’t exactly durable, but the self-repair thing they have going for them. So we found it was far more effective to concentrate an overwhelming force on one bot. The problem is when one goes down two come to fix it.”\r\nIzar: “Hey what are these giant scuba tanks? (on the map)”\r\n“We’re guessing their either fuel or materials for the terraforming.”\r\nHrothulf: “Hey a question for a friend… are they susceptible to plasma rifle fire?”\r\n“Not that we know of, and not your mission.”\r\nIzar: “Have you noticed any phoenixes?”\r\n“Ugh you’re going back to this? We’ll leave the mythical creatures to the experts—yourselves. Uh, No, is my simple answer.”\r\nHrothulf: “The guard towers on this map. Are they taller than the island or vice versa or what?”\r\n“The island is higher. The guard towers are only about 20’ off the ground, whereas the island is closer to 120’ off the guard.”\r\nHrohtulf: “Does this operation have a cool code name.”\r\nDarnit: “Um, I thought I’d call it blow-up-the-island create-a-diversion, but I suspect you have a more creative name.”\r\nHrothulf: “The Great Blow Up the Island Caper?”\r\nIzar: “The Island of Dr Whidmoreau?”\r\n\r\n\r\nWe’re given instructions on how to arm and set up the boxes and detonate them. They have magnets, so we just slap them onto the building. \r\nWe talk about whether we can drop them. (No.) But what if we have parachutes? Little briefcase-sized parachutes. Like bigger parachutes, but briefcase-sized? Maybe ropes?\r\n“These have lots of explosives. Please just place them where they go, get away, and press the button.”\r\n\r\n\r\nHrothulf: “What are the go-pros for?”\r\nDarnit: “It’s so we can see what you’re doing. So I can micromanage. You know, if needed.”\r\nHrothulf is ready to do something inappropriate.\r\nThey just strap on high on our chests.\r\n\r\n\r\nHrothulf suggests he takes Dorinda’s bomb, leaving Dorinda as an eye high in the sky and able to snipe from the skies.\r\n\r\n\r\nWe have to be aware that our griffins weren’t built to survive on this planet, so their abilities may be limited after some time.\r\n\r\n\r\nWe D-C-B-Ascend! Dorinda flies out, gives us a heads-up over the thing, and does the countdown for us: A-B-C-Descend! We land in the corners of the island, with Dorinda circling above. Dorinda says it looks like no one on the towers seems to be aware of what’s going on.\r\n\r\n\r\nNow on the island, we see this building is emitting a really loud droning sound. It doesn’t seem to be a building you can enter. Maybe just a housing unit for whatever these antennas are zapping away. The entire island has this light vibration as we stand on it.\r\n\r\n\r\nHrothulf has landed at the Northeast corner of the building. He’s covering both East-side corners, while Ego is Southwest and Izar is Northwest. It’ll will take two consecutive uninterrupted rounds to set and arm a bomb. He starts setting the Northeast bomb.\r\n\r\n\r\nEgo perceives out of the corner of her eye that the building of the corner itself has these weird plates on the curvature of the building. They seem modular. It’s really difficult to discern that they exist. It looked initially like a solid piece dropped on top, but Ego has noticed this slightly perceptible difference. She’s not sure what it is and is a little concerned, but she wants to stay on task. She starts setting her bomb, but facing in such a direction that the plates are within view of the plates. She mutters something about the plates, telling Darnit to keep an eye on them.\r\n\r\n\r\nIzar begins placing his bomb as well. Hiare is looking out, keeping Izar attentive through their special bond.\r\n\r\n\r\nDorinda starts to say something about her little droid scanning the building for information, and the plates suddenly open, shifting down. Two bots walk out from each, for twelve bots on the island with us. Dorinda immediately takes one down, knocking it off the island. Over the headset, Darnit says, “Confirmed, they cannot fly.”\r\nThe bots open fire. Little megaman arms shooting rubber bullets at us. Izar hears the shots going off and braces for impact, but the sound of the shot itself is almost immediately hit by the sound of ricocheting as one bot has inadvertently shot another one. She looks and sees a chunk of the silicon armor has been knocked off and fallen behind. Both bots go to pick it up and start reassembling.\r\n\r\n\r\nEgo takes damage from the rubber bullets and it has interrupted her bomb prep.\r\nHrothulf also takes damage, but he is unfazed, focused on the work at hand.\r\n\r\n\r\nDorinda, in our ear, says, “Ok, more importantly, the bots at the towers are paying attention to what’s going on right now. I was able to take out one. I can take another shot and hopefully they won’t be able to triangulate where my shots are coming from . They’re taking out some kind of cannon. I’m guessing it’s some kind of EMP. I want to take out another one but I’m not sure what do you think? \r\nEgo thinks if Dorinda shoots then moves in a random direction, she should have good odds. Darnit says to take out only immediate threats, since we already know the EMP won’t hurt anyone besides her.\r\nShe shoots and takes out another one. The first one she took out was near Hrothulf. This time she took out one close to Ego, hitting it in such a way that it just falls off the edge.\r\n\r\n\r\nHrothulf stays the course still, now arming the bomb.\r\nPoyraz stands between him and the two approaching bots, and he positions himself to completely shield Hrothult. He makes his rod immovable. \r\n\r\n\r\nEgo has a few of these bots coming at her. She disengages from the bomb and fires her railgun at the bot about 8’ away. But she misses, as it whizzes past the bot and into Magmus. \r\nHayete gets between the bots and Ego and casts Shatter so as to hit as many bots as possible. Two bots’ armor shatters like a glass door placed just off and shattering to the ground. The bot closest to Ego is alerted to this.\r\n\r\n\r\nIzar has four bots on her side, but the two closest have one working on healing the other.\r\nIzar attends to arming the bomb.\r\nFei runs down and lunges at the bots closest, grabbing the nearest with her talons and kicking away the other with her lion’s hind, throwing the broken bot over the side of the island.\r\n\r\n\r\nFour orbs are launched simultaneously in a high arc from each of the towers. These golden orbs make their way to us, then there’s a scatter detonation into large spheres of yellow energy that end up surrounding the entire island. Anything electronic on our persons is disabled. But our bombs and detonators are unaffected. The bots also seem unaffected. The entire structure stops humming.\r\n\r\n\r\nEgo notices that the building hum stopped right before the EMPs went off.\r\nThe humming starts again as soon as the EMP dissipates, building back up.\r\n\r\n\r\nA bot charges at Hrothulf and throws a punch with his megaman arm. Hrothulf’s arm recoils downward. He comes back up and licks away the blood on his lip like barbecue sauce and says, “I’ve had roasted quail eggs that were stronger than that”, as he nonchalantly continues armoring the bomb.\r\n\r\n\r\nA bot tries to come at Ego, but Hayete stops it in its tracks with an kick of opportunity.\r\n\r\n\r\nDorinda asks who to shoot, then says, “Hang on. Something’s happening. I’m not quite sure how to say this but the bots are converging. They’re leaving their posts on the perimeter and coming to the center.”\r\nWe now notice the pink wavy lines under the island sucking up about five bots at a time.\r\n“Well, more for me to snipe,” she says."


# {
#   "choices": [
#     {
#       "finish_reason": "length",
#       "index": 0,
#       "logprobs": null,
#       "text": " A group of adventurers is sent on a mission to blow up a floating island that has appeared at the same time as an army of robots. They are given sonic detonators and go-pros to help them complete the mission. The group is briefed on the robots' abilities and weaknesses, as well"
#     }
#   ],
#   "created": 1683848750,
#   "id": "cmpl-7FAGkQkJdQXasiSh2kzd0B7SdLS3P",
#   "model": "text-davinci-003",
#   "object": "text_completion",
#   "usage": {
#     "completion_tokens": 60,
#     "prompt_tokens": 3663,
#     "total_tokens": 3723
#   }
# }
#  A group of adventurers is sent on a mission to blow up a floating island that has appeared at the same time as an army of robots. They are given sonic detonators and go-pros to help them complete the mission. The group is briefed on the robots' abilities and weaknesses, as well
#  A group of adventurers is sent on a mission to blow up a floating island that has appeared at the same time as an army of robots. They are given sonic detonators and go-pros to help them complete the mission. The group is briefed on the robots' abilities and weaknesses, as well


# str = " \n{\n  title: \"The Great Blow Up the Island Caper\",\n  brief: \"The party is sent on a mission to blow up an island and create a diversion for the Freelanders. They must face off against the Bot Line and the mysterious island.\",\n  synopsis: \"The party is sent on a mission to blow up an island and create a diversion for the Freelanders. They must face off against the Bot Line, a force of silicon robots with self-repairing capabilities and a shared hive-mind system. The mysterious island is emitting a loud droning sound and has guard towers that are 20' off the ground. The party must set up four bombs on the four corners of the island and detonate them, while avoiding the EMP blasts of the bots. They must also be aware of the phoenixes and the modular plates on the building. Dorinda snipes from the sky and Hayete casts Shatter to take out the bots. In the end, the bots converge and are sucked up by the pink wavy lines under the island.\",\n  places: \"Med Tent, Floating Island\",\n  characters: \"Darnit, Tre Arrow, LeFleur, Hrothulf, Ego, Izar, Dorinda, Poyraz, Hayete, Fei, Magmus\",\n  associations: \"Confederation, Freelanders, Bot Line, EMP, Sonic Detonators, Phoenixes, Griffins\",\n  items: \"Briefcase, Earbuds, Go-Pros, Toaster Oven-Sized Gray Silver Boxes, Bombs, Detonators, Parachutes, Ropes, Railgun, Rod, Shatter\"\n}"


# '\ufeffThe go-pro has gone all black from Darnit’s side. But Arrow assures it’s glitching temporarily from the EMP and systems are rebooting. Darnit knows the go-pros should be able to survive three of those.\r\n\r\n\r\nThe team is on the floating island. Bots are being sucked up from underneath the island. Hrothulf has set and armed a bomb on the NE corner and Izar set and armed on the NW corner. There are currently three bots near Ego but they are actively repairing themselves. There are four more on the island, in fine condition.\r\n\r\n\r\nDorinda looks for the bot closest to where Hrothulf’s second bomb will be planted and fires her sniper rifle from her invisible perch atop griffin. She blows a huge hole in its chest. It bends to self-heal and instead collapses over itself, falling over the side of the island to the ground below.\r\n\r\n\r\nHrothulf takes a baseball-bat swing with his flaming sword, aiming to smack the bot between him and his goal off the island. He swings for the fence but the bot side-steps and Hrothulf’s inertia takes him to the ground.\r\n\r\n\r\nPoyraz seeks to hold the line against the two soon to be three bots coming.\r\n\r\n\r\nEgo, seeing a clear path, sprints for her bomb site and starts setting the bomb.\r\n\r\n\r\nHayete takes to the air does a loop-de-loop and charges the three bots healing each other. She would charge harder on the ground but this gives more control and aim. She strikes all three and they fly a solid 25’, clawing at the ground to stop their movement and huddling in a robot pile in the center of the island.\r\n\r\n\r\nIzar asks Fei to again throw a bot off the island. A bot nearby is aiming his cannon arm at Poyraz, then there is a *whoosh* and it is no longer there, flying off the island. \r\nMany bots are heading our way, though the bots in the towers are staying in their place. She casts an acid chromatic orb using metamagic to try to reach and hit a bot in one of the towers. From this distance it’s hard to tell exactly what happened, but Izar can see that the bots set down their cannon and began repairs.\r\n\r\n\r\nTwo bots come toward Hrothulf. Poyraz holds them back adeptly, but takes the brunt of some rubber bullet rounds.\r\n\r\n\r\nThe doors on the caterpillar-like island center slide open again. Two more bots emerge from each of the two doors in the middle (north and south). On the western doors of both north and south, one bot emerges. These have megaman arms with an electric bolt jumping from one arm to the other. The formation of bots outside gets closer.\r\n\r\n\r\nDarnit calls Nilchi, planning to come in to provide healing if needed and possibly knock bots off, though he doesn’t want to get in battle. He reiterates that, once the bombs are set, we need to get out of there immediately, and then, potentially provide air support.\r\n“Nope, that won’t be necessary, thank you.” says Arrow. “You have one job. Blow that thing up and come back.”\r\nDarnit gives a mere look of acknowledgement. (NOT a head-nod.)\r\n\r\n\r\nDorinda snipes another as it’s coming out of the door. Ego, arming her bomb, hears a sharp *zip* and looks up to see a bot with a hole in its head, standing still then crumpling to the ground.\r\n\r\n\r\nHrothulf tries to leap up with grace, but really only ably makes it to his feet. Nonetheless: “Alright y’all, let’s try this one more time,” and he makes up for his missed swing with a crushing blow from Watermelon’s Wail. It comes up through the robot fully, so eviscerating the bot that it doesn’t even clang to the ground but instead crumbles to dust, crystalline sand glistening in the setting suns.\r\n\r\n\r\nEgo’s bomb is now set, and she arms it.\r\nHayete is nesting on top of three bots, and two more have just arrived about five feet away from her. She flies at the two that are up, hoping to play a little whack-a-mole. She successfully hits one, sliding east away from Ego. The other is pressed against the caterpillar, with electric charge ready to go.\r\n\r\n\r\nFei goes again for the bot nearest her, but it evades her attack. Hiare is offended on Fei’s behalf and tries to bit the bot, but its armor is too strong so he instead barks at it menacingly.\r\nIzar sees the bot with its electricity charging and says, “You have zappy zap? I’ll show you zappy zaps!” He shocking grasps the bot and it melts the whole nodes of its gun and the whole bot just shuts down. He says, “Hrothulf did you see that!?” and when she looks back the bot has disintegrating to sand. \r\n\r\n\r\nThree blasts emit from the towers. The vibrations from the caterpillar stop again. And Darnit loses audio and visual again.\r\n\r\n\r\nPoyraz takes an attack of opportunity, getting really annoyed by these bots coming at him. He just backhands one off the back of the island and gives an annoyed snarl. Meanwhile he still manages to be in the way of the other two with no trouble.\r\n\r\n\r\nThe one remaining bot on Ego’s side starts running at her, then stops about 7’ away, turns and aims its cannon at Izar, and hits. Izar shrugs off the effects of the taser, maybe due to just using her own electricity, but does feel the pain. “Can’t shock the shocker,” she says.\r\n\r\n\r\nOut of the doors at the southeast corner six bots march out. One of them begins repairing the bot Hayete threw over.\r\n\r\n\r\nUnable to talk to us, Darnit talks with Gen. Arrow and LeFleur, together on the bridge. Arrow suggests that if another EMP goes off the team should just abort. At that point we’d be blind and there’s just too much risk. Darnit is curious about the bots disintegrating and whether that can be wielded as part of the attack. Arrow says the one that disintegrated on their watch was simply being transported, like it was a fail-safe to protect IP or something. Darnit asks whether they use any elemental damage when they attack. Arrow seems a little confused by the question, then says No, we just use blunt force trauma in significant enough amounts to overwhelm their healing.\r\n\r\n\r\nDorinda dive bombs the group of six and throws a sticky bomb (20ft radius) right in the middle of them. She solidly hits all but the one that peeled off to help heal. She can tell that Sefarina is struggling a little to maintain invisibility, but it is still solid for now. The bots have taken damage and are now glued to the ground.\r\n\r\n\r\nHrothulf goes into a Barbaric Rage with a Collosus rage sound and comes with another Watermelon’s Wail. He takes off the bot’s entire arm, and each time it goes to retrieve its arm it cartoonishly kicks it a little away, like it’s glitching a bit. Hrothulf goes to the final bomb site.\r\n\r\n\r\nEgo hops on Hayete and flies around the stuck bots. Hayete flies, leaning to the side so Ego can pick up the one bot’s missing cannon arm. Ego gets it and Hayete lifts another bot with her eagle talons. She takes it into the air and impales it on one of the antenna. The antennae breaks and the bot flails upon it. Ego notices with her passive perception that a whole contingent of the bots—about thirty—coming toward the island suddenly come to a halting stop and just stand there.\r\n\r\n\r\nIzar jumps on Fei and alights. He sees the stuck bots and the one-armed one impaled on the antenna. Izar sees the effect of the antenna and goes to shocking grasp another one. There’s a lot of sizzling. When he shocking grasps he can usually feel the energy. This one the feeling is super intense, like zapping itself. The lights keeping blinking and the antenna, though seared and damaged, is not decommissioned entirely.\r\n\r\n\r\nComms are back on and the caterpillar starts its noisy humming again. The impaled bot gets completely fried. Coming out of its chest are arcs of energy, then it goes to silicon. Then the antenna starts working again and the bots start marching again.\r\n\r\n\r\nMore bots come. Three bots on either side on the west, three in the center. Sticky bots remain stuck.\r\nPoyraz takes his readied action and decides to play with his food. He grabs one with his beak and hulk smashes it a couple times before it yeets it off the edge, in an asymmetric somersault.\r\nThe other bot starts climbing up the caterpillar to try to evade Poyraz.\r\nTwo bots go to put Hrothulf in a restraining hold and Hrothulf lets them. One puts him in full nelson and the other with taser in full charge. Hrothulf takes the one behind in his rage and brings it over his back, blocking the taser and giving it to the bot. He discards it; no longer a threat.'


# {'title': 'Electricity and Fury', 'brief': 'The team is on a floating island, surrounded by bots. They must set and arm bombs on the island while fending off the bots. Dorinda, Hrothulf, Hayete, Fei, and Izar use their skills to fight off the bots while Ego and Darnit set the bombs. In the end, the bots are stopped by an EMP and the team is successful.', 'places': 'Floating Island', 'characters': 'Darnit, Arrow, Hrothulf, Izar, Dorinda, Ego, Hayete, Fei, Poyraz, Nilchi, Gen, LeFleur, Sefarina', 'associations': "EMP, Go-Pro, Bombs, Blunt Force Trauma, Elemental Damage, Electric Bolt, Megaman Arms, Sticky Bomb, Barbaric Rage, Watermelon's Wail, Shocking Grasp, Antenna", 'items': 'Baseball-bat, Flaming Sword, Sniper Rifle, Cannon Arm, Taser, Hulk Smashes'}

# " The team is on a floating island, attempting to set bombs and destroy the bots that are coming from underneath. Dorinda and Ego are sniping and setting bombs, while Hrothulf and Hayete are attacking the bots with physical force. Fei and Izar are using their magical abilities to take out the bots, and Poyraz is using his strength and beak to throw them off the island. Darnit is providing support from the air, and Arrow is giving advice from the bridge. After a few rounds of battle, the team has managed to set the bombs and take out the bots. However, a whole contingent of bots is still coming towards the island, and the team must act quickly to get out of there before the EMP goes off again."


# "{\n\t\"title\": \"The Blind Seer\",\n\t\"brief\": \"The party of adventurers meets the Blind Seer, a mysterious figure who tests them to find the Pantheon of the gods. They must find the scales of the gods and decide where they should reside.\",\n\t\"synopsis\": \"The party of adventurers, seeking Gliten, meets the Blind Seer. He tests them to find the Pantheon of the gods, and they must find the scales of the gods and decide where they should reside. Darnit finds an ebony and ivory scale, Dorinda finds a scale with water and a tree, Ego finds a scale of elegance and beauty, Hrothulf finds an obsidian scale, and Izar and Hiare find two scales, one with a spinning gyroscope and one with a tornado. Dorinda suggests an ordering for five of them, and they find a scale reminiscent of the Lady Saharel. Teresias is playing solitaire with braille cards, and Izar brings forth another scale made of food. In the end, they find the scales of the gods and decide where they should reside.\",\n\t\"places\": [\"Fallucia\", \"Mystra\"],\n\t\"characters\": [\"Teresias\", \"Darnit\", \"Dorinda\", \"Ego\", \"Hrothulf\", \"Izar\", \"Hiare\"],\n\t\"races\": [\"Lizardfolk\"],\n\t\"associations\": [\"Branch of Teresias\", \"Gliten\", \"Dhund Hal-Kah\", \"Griffon Cavalry\", \"Yafel\", \"Magman\", \"Gliton\", \"Gliten\", \"Glitone"
