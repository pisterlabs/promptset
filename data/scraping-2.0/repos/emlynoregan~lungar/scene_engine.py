import setcreds
import openai
import mergedeep
import random
import copy

_diags = 0

def diag(msg, level=5):
    if _diags >= level:
        print(f"***{msg}")

def run_game(scenes, start_scene_id, diagnostics = False, talk_engine="davinci", question_engine="curie",
    npc_talk_max_tokens=64):
    global _diags
    _diags = diagnostics

    scene = {**scenes[start_scene_id]}

    new_scene = True

    history = scene.get('history') or []

    turn = 0

    while not scene.get("gameover"):
        diag(f"turn: {turn}")
        
        # shrink history
        while len(history) > 20:
            remove_ix = random.randint(0, len(history) - 1)
            del history[remove_ix]

        if new_scene:
            diag("new scene")

            turn = scene.get('turn') if not scene.get('turn') is None else turn

            diag(f"turn: {turn}")

            look = scene.get('look')
            if look:
                player_look_lines = get_player_look_lines(scene)

                print("\n".join(player_look_lines))

            new_scene = False
        
        # continue here

        # probably remove this section, npcs don't need actions.
        npcs = scene.get('npcs') or []
        if npcs:
            for _, npc in npcs.items():
                if npc:
                    actions = npc.get('actions') or []
                    if actions:
                        diag(f"npc actions")
                        for action_name, action in actions.items():
                            if action:
                                diag(f"action: {action_name}")
                                min_turn = action.get('min_turn') or 0
                                diag(f"min_turn: {min_turn}")

                                if min_turn <= turn:
                                    if npc_action_fires(npc, action, scene, history, question_engine):
                                        diag(f"action fires!")
                                        to_scene_id = action.get('to_scene')
                                        to_scene = scenes[to_scene_id]
                                        scene = do_merge(scene, to_scene)
                                        new_scene = True

                                        transition_pdesc = action.get('transition_pdesc')
                                        if transition_pdesc:
                                            print(transition_pdesc)

                                        transition_ndesc = action.get('transition_ndesc')
                                        if transition_ndesc:
                                            history.append(transition_ndesc)
                                        break
                                    else:
                                        diag(f"action doesn't fire")
                                else:
                                    diag(f"too early for action")
                        if new_scene:
                            break
                    else:
                        diag(f"no npc actions")
        else:
            diag(f"no npcs")

        actions = scene.get('actions') or []
        if actions:
            diag(f"npc actions")
            for action_name, action in actions.items():
                if action:
                    diag(f"action: {action_name}")
                    min_turn = action.get('min_turn') or 0
                    diag(f"min_turn: {min_turn}")

                    if min_turn <= turn:
                        if npc_action_fires(npc, action, scene, history, question_engine):
                            diag(f"action fires!")
                            to_scene_id = action.get('to_scene')
                            to_scene = scenes[to_scene_id]
                            scene = do_merge(scene, to_scene)
                            new_scene = True

                            transition_pdesc = action.get('transition_pdesc')
                            if transition_pdesc:
                                print(transition_pdesc)

                            transition_ndesc = action.get('transition_ndesc') or action.get('transition_pdesc')
                            if transition_ndesc:
                                history.append(transition_ndesc)
                            break
                        else:
                            diag(f"action doesn't fire")
                    else:
                        diag(f"too early for action")
        else:
            diag(f"no actions")


        if not new_scene:
            #see if npc talks
            npcs = scene.get('npcs') or [] 
            if npcs:
                for npc_name, npc in npcs.items():
                    if npc:
                        shortdesc = npc.get('shortdesc') or npc_name
                        talk_p = npc.get('talk_p') or 1
                        p = random.random()
                        if p <= talk_p:
                            diag(f"npc talks")
                            talk = get_npc_talk(npc, scene, history, talk_engine, npc_talk_max_tokens=npc_talk_max_tokens)
                            full_talk = f"{shortdesc} says:{talk}"
                            history.append(full_talk)
                            print(full_talk)
                            print("")

        if not new_scene:
            # player action
            player_action = None
            player_talk = None
            player = scene.get('player')

            while not (player_action or player_talk):
                player_input = input("You say > ")

                if not player_input:
                    player_action_lines = get_player_action_lines(player)

                    print("\n".join(player_action_lines))
                elif player_input[0] == "/":
                    # this is an action
                    player_command = player_input[1:].lower().strip()
                    if player_command in ["look", "l"]:
                        player_look_lines = get_player_look_lines(scene)

                        print("\n".join(player_look_lines))
                    elif player_command in ["exit", "x"]:
                        exit()
                    else:
                        # check in player actions
                        if not player_command in (player.get('actions') or {}):
                            print(f"The command '{player_command}' is not recognized.")
                        else:
                            player_action = player.get('actions').get(player_command)
                else:
                    # this is talking
                    diag(f"player talks")
                    player_talk = player_input

            print("")

            if player_action:
                diag(f"player action fires!")
                action_ptext = player_action.get('ptext')
                action_ntext = player_action.get('ntext')
                history += [action_ntext] or []
                print(action_ptext)

                to_scene_id = player_action.get('to_scene')
                to_scene = scenes[to_scene_id]
                scene = do_merge(scene, to_scene)
                new_scene = True
            elif player_talk:
                shortdesc = player.get('nshortdesc') or "The player"
                full_talk = f"{shortdesc} says: {player_input}"
                history.append(full_talk)

        turn += 1

def do_merge(scene, to_scene):
    to_scene2 = copy.deepcopy(to_scene)
    scene2 = copy.deepcopy(scene)
    return mergedeep.merge(scene2, to_scene2)

def npc_action_fires(npc, npc_action, scene, history, engine):
    q_and_a_lines = npc_action.get('q_and_a_lines')
    answer = npc_action.get('answer')
    logit_bias = npc_action.get('logit_bias')

    if q_and_a_lines:
        about_lines = npc.get('about_lines') if npc else []
        # talk_prompt = npc.get('talk_prompt') i
        prompt_lines = get_npc_look_lines(scene) + [""] + \
            about_lines + [""] + \
            history + [""] + q_and_a_lines

        prompt = "\n".join(prompt_lines)

        diag (f"npc action prompt: {prompt}", level=4)

        temperature = 0

        completion = openai.Completion.create(
            engine=engine, 
            max_tokens=2, 
            temperature=temperature,
            prompt=prompt,
            frequency_penalty=0,
            logprobs=5,
            logit_bias=logit_bias
        )

        diag(f"completion: {completion}", level=4)

        ai_raw_msg = completion.choices[0].text
        # print(f"ai_raw_msg: {ai_raw_msg}")

        ai_msg_lines = ai_raw_msg.split("\n")

        ai_msg = ai_msg_lines[0]

        npc_answer = ai_msg.lower().strip()

        diag(f"npc answer: {npc_answer}")

        return npc_answer == (answer or "yes")
    else:
        return True

def get_npc_talk(npc, scene, history, engine, npc_talk_max_tokens=128):
    about_lines = npc.get('about_lines')
    talk_lines = npc.get('talk_lines')
    talk_prompt = npc.get('talk_prompt')
    prompt_lines = get_npc_look_lines(scene) + [""] + \
        about_lines + [""] + \
        talk_lines + [""] + \
        history + ["", talk_prompt]

    temperature = 0.8

    prompt = "\n".join(prompt_lines)
    diag(f"npc talk prompt: {prompt}")

    completion = openai.Completion.create(
        engine=engine, 
        max_tokens=npc_talk_max_tokens, 
        temperature=temperature,
        prompt=prompt,
        frequency_penalty=0.1
    )

    ai_raw_msg = completion.choices[0].text
    # print(f"ai_raw_msg: {ai_raw_msg}")

    ai_msg_lines = ai_raw_msg.split("\n")

    ai_msg = ai_msg_lines[0]

    diag(f"ai_msg: {ai_msg}")

    return ai_msg

def get_player_look_lines(scene):
    lines = []
    scene_desc = scene.get('pdesc')
    if scene_desc:
        lines.append(scene_desc)
        # lines.append("")
    
    player = scene.get('player')
    if player:
        # player_desc = player.get('pdesc') or "You are here."
        # if player_desc:
        #     lines.append(player_desc)
        player_items = player.get('items')
        if player_items:
            for _, item in player_items.items():
                if item:
                    item_desc = item.get('desc')
                    if item_desc:
                        lines.append(f"You are holding {item_desc}.")
        # lines.append("")

    lines.extend(get_npc_lines(scene))    

    return lines

def get_npc_look_lines(scene):
    lines = []
    scene_desc = scene.get('ndesc')
    if scene_desc:
        lines.append(scene_desc)
        lines.append("")
    
    player = scene.get('player')
    if player:
        player_desc = player.get('ndesc')
        if player_desc:
            lines.append(player_desc)
            player_items = player.get('items')
            if player_items:
                player_short_desc = player.get('nshortdesc') or "The player"
                for _, item in player_items.items():
                    if item:
                        item_desc = item.get('desc')
                        if item_desc:
                            lines.append(f"{player_short_desc} is holding {item_desc}.")
            lines.append("")

    lines.extend(get_npc_lines(scene))

    return lines

def get_player_action_lines(player):
    lines = [
        "[/L]ook around you",
        "e[/X]it the game"
    ]

    player_actions = player.get('actions')

    if player_actions:
        for action_name, action in player_actions.items():
            if action:
                action_desc = action.get('desc') or action_name
                lines.append(action_desc)

    return lines

def get_npc_lines(scene):
    lines=[]
    npcs = scene.get('npcs') or []
    if npcs:
        for npc_name, npc in npcs.items():
            if npc:
                npc_desc = npc.get('pdesc')
                if npc_desc:
                    lines.append(npc_desc)
                    npc_items = npc.get('items')
                    if npc_items:
                        npc_short_desc = npc.get('shortdesc') or npc_name
                        for _, item in npc_items.items():
                            if item:
                                item_desc = item.get('desc')
                                if item_desc:
                                    lines.append(f"{npc_short_desc} is holding {item_desc}.")
                    lines.append("")
    return lines

    