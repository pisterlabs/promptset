from flask import Blueprint, render_template, current_app as app
from jinja2 import TemplateNotFound, Template
from . import socketio, emitter
from .models import gametime, player, journal, game
import os
import re
from typing import Dict, List
from config import Config
import numpy as np
import asyncio
import textwrap
import math
import threading
import json
import sys
import random
import websockets
from datetime import datetime, timedelta, time
from pydantic import BaseModel
import openai
from collections import defaultdict

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://host.docker.internal:9999/v1"


URI_WS = Config.OOB_URI_WS
state = Blueprint('state', __name__,
                        template_folder='templates',
                        static_folder="static")

default_queue = asyncio.Queue(maxsize=0)
high_queue = asyncio.Queue(maxsize=0)

async def queuing(proposal, next_state):
        print("Queued state: ", next_state)
        game.state = next_state
        if (proposal.get('queue') == 'default'):
            try:
                print("Enqueuing default")
                default_queue.put_nowait(game.state.enqueue(proposal,))
            except asyncio.QueueFull:
                print("Queue is full")
        elif (proposal.get('queue') == 'high'):
            try:
                print("Enqueuing high")
                high_queue.put_nowait(game.state.enqueue(proposal,))
            except asyncio.QueueFull:
                print("Queue is full")

@emitter.on('accepted_proposal')
def handle_accepted_proposal(proposal):
    next_state = compute_next_state(proposal)
    if proposal.get('queue') != None:
        try:
            loop = asyncio.get_event_loop()
            print("Adding task")
            loop.create_task(queuing(proposal, next_state))
        except RuntimeError:
            # If no event loop is running, start one temporarily
            asyncio.run(queuing(proposal, next_state))
        print("Task added")
    else:
        game.state = next_state
        game.state.transition(proposal)

def compute_next_state(accepted_proposal: Dict):
    if (accepted_proposal['action'] == "init_app"):
        return WIZARD_WAITING()
    if (accepted_proposal['action'] == "execute_option"):
        return RESPONSE_COMPUTING()
    if (accepted_proposal['action'] == "lock_time"):
        return TIME_LOCKED()
    if (accepted_proposal['action'] == "load_game"):
        return GAME_LOADING()
    if (accepted_proposal['action'] == "change_stat"):
        return WIZARD_UPDATING()
    if (accepted_proposal['action'] == "select_mission"):
        return MISSION_COMPUTING()
    if (accepted_proposal['action'] == "summarize_journal"):
        return REFLECT_COMPUTING()
    if (accepted_proposal['action'] == "compute_event"):
        return EVENT_COMPUTING()
    if (accepted_proposal['action'] == "compute_concept"):
        return CONCEPT_COMPUTING()
    if (accepted_proposal['action'] == "change_speed"):
        if accepted_proposal['speed'] >= 1.0:
            return GAME_TICKING()
        if accepted_proposal['speed'] < 1.0:
            return GAME_PAUSING()
    else:
        return ("Invalid")

class GameState():
    def can_init_game(self):
        return False
    def can_change_speed(self):
        return True
    def transition(self, game, accepted_proposal=None):
        print("Transitioning to next state")
    async def enqueue(self, game, accepted_proposal):
        print("Enqueuing next state")

class WIZARD_WAITING(GameState):
    def can_init_game(self):
        return True
    def can_change_speed(self):
        return False
    def transition(self, accepted_proposal=None):
        print("Waiting for game to be initialized")
class WIZARD_UPDATING(GameState):
    def can_init_game(self):
        return True
    def can_change_speed(self):
        return False
    def transition(self, accepted_proposal=None):
        calculate_stats(accepted_proposal)
        # test_vllm()

class TIME_LOCKED(GameState):
    def can_init_game(self):
        return False
    def can_change_speed(self):
        return False
    def transition(self, accepted_proposal):
        print("Time locked")
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(stop_ticker())
        except RuntimeError:
            # If no event loop is running, start one temporarily
            asyncio.run(stop_ticker())

class GAME_LOADING(GameState):
    def can_init_game(self):
        return True
    def transition(self, accepted_proposal):
        load_game(accepted_proposal)
        launch_game_ui()

class RESPONSE_COMPUTING(GameState):
    def transition(self, accepted_proposal: Dict):
        complete_event(accepted_proposal)

class GAME_PAUSING(GameState):
    def transition(self, accepted_proposal):
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(stop_ticker())
        except RuntimeError:
            # If no event loop is running, start one temporarily
            asyncio.run(stop_ticker())

class MISSION_COMPUTING(GameState):
    async def enqueue(self, accepted_proposal):
        print("computing mission")
        compute_mission()

class EVENT_COMPUTING(GameState):
    async def enqueue(self, accepted_proposal):
        compute_event()

class CONCEPT_COMPUTING(GameState):
    async def enqueue(self, accepted_proposal):
        compute_concept()

class REFLECT_COMPUTING(GameState):
    async def enqueue(self, accepted_proposal):
        summarize_journal(accepted_proposal['mission_id'])
        return
    
class GAME_TICKING(GameState):
    def transition(self, accepted_proposal):
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(start_ticker(accepted_proposal['speed']))
        except RuntimeError:
            asyncio.run(start_ticker(accepted_proposal['speed']))

# class RepeatingTimer(Timer):
#     def run(self):
#         while not self.finished.wait(self.interval):
#             self.function(*self.args, **self.kwargs)

def test_vllm():
    class AnswerFormat(BaseModel):
        first_name: str
        last_name: str
        year_of_birth: int
        num_seasons_in_nba: int

    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
    """

    def get_prompt(message: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        return f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{message} [/INST]'

    models = openai.Model.list()
    print("Models:", models)

    model = models["data"][0]["id"]

    question = 'Please give me information about Michael Jordan. You MUST answer using the following json schema: '
    question_with_schema = f'{question}{AnswerFormat.schema_json()}'
    schema_json = json.dumps(AnswerFormat.schema_json())

    prompt = get_prompt(question_with_schema)
    stream = True

    print("Prompt:")
    print(prompt)
    print("Answer, With json schema enforcing:")

    result = openai.Completion.create(
        model=model,
        prompt=prompt,
        stream=stream,
        max_tokens=600,
        temperature=0,
        jsonparser=AnswerFormat.schema_json())
    
    print("Completion results:")
    if stream:
        for c in result:
            sys.stdout.write(c.choices[0].text)
            sys.stdout.flush()
    else:
        print(result)

    print("Answer, Without json schema enforcing:")
    result = openai.Completion.create(
        model=model,
        prompt=prompt,
        stream=stream,
        max_tokens=600,
        temperature=0,
        jsonparser=None)

    # Completion API
    # Chat completion API

    print("Completion results:")
    if stream:
        for c in result:
            sys.stdout.write(c.choices[0].text)
            sys.stdout.flush()
    else:
        print(result.choices[0].text)




def calculate_stats(accepted_proposal):
    # Extract people and assets dynamically
    worldview_key = accepted_proposal['values']['worldview']
    socialclass_key = accepted_proposal['values']['socialclass']
    personality_key = accepted_proposal['values']['personality']

    filename = os.path.join(app.root_path, 'templates', 'data', 'stats.json')
    with open(filename) as f:
        data = json.load(f)
    worldview = data['worldview'][worldview_key]
    socialclass = data['socialclass'][socialclass_key]
    personality = data['personality'][personality_key]

    # Calculate stats
    force = round(math.pow(worldview['force'] * socialclass['force'] * personality['force'],0.5))
    diplomacy = round(math.pow(worldview['diplomacy'] * socialclass['diplomacy'] * personality['diplomacy'],0.5))
    insight = round(math.pow(worldview['insight'] * socialclass['insight'] * personality['insight'],0.5))
    commerce = math.pow((worldview['commerce'] * personality['commerce']),0.5)*socialclass['commerce']/2
    debt_score = round(commerce * (socialclass['debt']/socialclass['commerce']))
    commerce = round(commerce)

    player.traits = [
        worldview['pos_trait'],
        socialclass['pos_trait'],
        personality['pos_trait'],
        worldview['neg_trait'],
        socialclass['neg_trait'],
        personality['neg_trait']
    ]

    player.communication = [
        worldview['communication'].split(', ')[0],
        socialclass['communication'].split(', ')[0],
        personality['communication'].split(', ')[0],
        worldview['communication'].split(', ')[1],
        socialclass['communication'].split(', ')[1],
        personality['communication'].split(', ')[1]
    ]    

    player.force = force
    player.diplomacy = diplomacy
    player.insight = insight

    modifier = random.uniform(0.9500, 1.0500)

    player.worldview = worldview
    player.socialclass = socialclass
    player.personality = personality

    player.wealth = round(500*(2^commerce)*modifier-900)
    player.debt = round(500*(2^debt_score)*modifier-900)

    print(f"Player wealth: {player.wealth}")
    print(f"Player debt: {player.debt}")
    

    socketio.emit('update_statmenu', {'force': player.force, 'diplomacy': player.diplomacy, 'insight': player.insight, 'wealth': str(player.wealth)+" fl.", 'debt': str(player.debt)+" fl."})


def compute_concept():
    print("computing concept")
    
    # compute around concept
    # determine significance of concept
    # include interactions

    # compute around character
    # ambition, mission, events

    # compute around events
    # update people disposition
    # update journal
    # update standing
    # update lifestyle


    # compute actions


def launch_game_ui():
    ticker_template = render_template(
    'ticker.html', 
    time = gametime.datetime.strftime('%H:%M'),
    date = gametime.datetime.strftime('%d.%m'),
    year = gametime.datetime.strftime('%Y AD')
    )
    socketio.emit('blank_canvas', {'new_html_content': ticker_template})
    socketio.emit('update_statmenu', {'force': player.force, 'diplomacy': player.diplomacy, 'insight': player.insight, 'wealth': str(player.wealth)+" fl.", 'debt': str(player.debt)+" fl."})
    
    
    if Config.SKIP_GEN_NARRATIVE:
        state_narrative = json.loads(Config.SKIP_VAL_NARRATIVE)
    else:

        schema = get_jsonschema('narrative_state_schema.jinja')
        prompt = generate_state_narrative()

        state_narrative = batch_api(
            prompts=prompt, 
            schemas=schema,
            temperature=0.0,
            frequency_penalty=1.0
            )
        
        print("Result: ", state_narrative)
    
    player.standing = state_narrative['player_standing']
    player.lifestyle = state_narrative['player_lifestyle']

    background = {
        'type': 'background',
        'story': player.background,
        'triggerdate': gametime.datetime - timedelta(days=(int(365.24*player.age*0.2)))
    }

    journal.completed['background']= background

    asyncio.run(game_loop())

async def game_loop():
    asyncio.create_task(action_generator())
    asyncio.create_task(action_consumer())
    while True:
        # You can add any periodic checks or maintenance tasks here
        await asyncio.sleep(10)  # Sleep for a while before continuing the loop


async def action_generator():
    while True:
        if (default_queue.empty() and high_queue.empty()):
            if not journal.active and not any(event['type'] == 'mission_select' for event_id, event in journal.scheduled.items()):
                emitter.emit('generate_action', {'action': 'select_mission', 'queue': 'default'})
            elif journal.active and not any(event['type'] == 'event_challenge' or event['type'] == 'event_confirmation' for event_id, event in journal.scheduled.items()):
                emitter.emit('generate_action', {'action': 'compute_event', 'queue': 'default'})
                #     decide = random.choices(['compute_event','compute_concept'], weights=[odds_events, odds_concepts], k=1)[0]
                #     emitter.emit('generate_action', {'action': decide, 'queue': 'default'})
        await asyncio.sleep(1)

# coroutine to consume work
async def action_consumer():
    print('Consumer: Running')
    # consume work
    while True:
        # get a unit of work
        try:
            item = high_queue.get_nowait()
        except asyncio.QueueEmpty:
            try:
                item = default_queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(1)
                continue
        # report
        await item

# Using partial to encapsulate the argument ()
async def start_ticker(speed):
    try:
        if gametime.ticker:
                gametime.ticker.cancel()
        gametime.ticker = asyncio.create_task(tick(speed))
        await gametime.ticker  # Await the new task
    except (asyncio.exceptions.CancelledError):
        pass

async def stop_ticker():
    if gametime.ticker:
        gametime.ticker.cancel()
        socketio.emit('stop_time')

async def tick(speed):
    tick_speed = 1 / math.pow(speed,0.75)
    while True:
        await asyncio.sleep(tick_speed)
        await update_time(speed)

async def update_time(speed):
    modifier = max(round(speed / 15.13,2),3) - 2
    modifier = math.pow(modifier,3)
    gametime.datetime += timedelta(minutes=modifier)
    
    time = gametime.datetime.strftime('%H:%M')
    date = gametime.datetime.strftime('%d.%m')
    year = gametime.datetime.strftime('%Y AD')

    # for each event in scheduled_events, check if triggerdate is less than datetime
    for event_id, event in journal.scheduled.items():
        if event['triggerdate'] <= gametime.datetime:
            emitter.emit('lock_time')
            if event.get('location', False) != False:
                player.location = event['location']
            socketio.emit('drop_event', {'new_html_content': event['html']})
    socketio.emit('update_time', {'time': time, 'date': date, 'year': year})

# FETCH RESPONSE OOGABOOGA

async def get_ws_result(prompt, json_schema=None, emit_progress_start=0, emit_progress_max=0):
    # while game.socket_occupied:
    #     await asyncio.sleep(0.1)  # wait for 1 second before checking again
    
    # game.socket_occupied = True
    result = await gen_ws_result(prompt, json_schema, emit_progress_start, emit_progress_max)
    # game.socket_occupied = False

    return result
    
async def gen_ws_result(prompt,json_schema,emit_progress_start, emit_progress_max):
    result = ""
    async for response in get_ws_response(prompt,json_schema):
        if(response == "stream_end"):
            print("stream_end")
            return result
        result += response
        if (emit_progress_max!=0):
            result_length = len(result)
            progress = round((result_length+emit_progress_start) / emit_progress_max * 100,0)
            socketio.emit('update_progress', {'progress': progress})
        print(response, end='')
        sys.stdout.flush()  # If we don't flush, we won't see tokens in realtime.

async def get_ws_response(prompt,json_schema):
    request = build_request(prompt,json_schema)
    try:
        async with websockets.connect(URI_WS, ping_interval=None) as websocket:
            await websocket.send(json.dumps(request))

            while True:
                incoming_data = await websocket.recv()
                incoming_data = json.loads(incoming_data)

                match incoming_data['event']:
                    case 'text_stream':
                        yield incoming_data['text']
                    case 'stream_end':
                        yield "stream_end"
                        return
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket connection error: {e}")
        # Handle the error or re-raise it to inform the caller
        # For now, we'll just return an empty list to indicate no data
def build_request(prompt,json_schema):
    request = {
        'prompt': prompt,
        'json_schema': json_schema,
        'max_new_tokens': 8000,
        'auto_max_new_tokens': False,
        'max_tokens_second': 0,
        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',
        'do_sample': True,
        'temperature': 0.2,
        'top_p': 0.37,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1,
        'repetition_penalty_range': 0,
        'top_k': 100,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'guidance_scale': 1,
        'negative_prompt': '',

        'seed': 123,
        'add_bos_token': True,
        'truncation_length': 8192,
        'ban_eos_token': False,
        'custom_token_bans': '',
        'skip_special_tokens': True,
        'stopping_strings': []
    }
    return request


def get_random_value_from_label(difficulty):
    difficulty_ranges = {
        'routine': (1, 1),
        'easy': (2, 2),
        'fair': (3, 4),
        'tricky': (5, 6),
        'challenging': (7, 9),
        'strenuous': (10, 14),
        'hard': (15, 20),
        'very hard': (21, 30),
        'extreme': (31, 41),
        'hardcore': (42, 55),
        'impossible': (56, 70)  # using infinity to capture all higher values
    }

    if difficulty not in difficulty_ranges:
        raise ValueError(f"Unknown difficulty level: {difficulty}")
    low, high = difficulty_ranges[difficulty]
    return random.randint(low, high)

def dice_counts_for_level(level):
    dice_types = [2, 4, 6, 8, 10, 12]
    counts = {2:0, 4:0, 6:0, 8:0, 10:0, 12:0}

    # Add the counts for the d12 die
    counts[12] = (int(level)) // 6

    if level % 6 > 0:
        counts[dice_types[level % 6 - 1]] = 1

    return counts

def roll_dice_for_level(level):    
    counts = dice_counts_for_level(level)
    total = sum(random.randint(1, die) for die, count in counts.items() for _ in range(count))

    return total

def roll_all_dice_for(choice):

    difference = 0
    challenges = choice['challenges']
    challenge_rolls = {}

    if('insight' in challenges):
        challenge_roll = roll_dice_for_level(choice['difficulty'])
        challenge_rolls['Insight'] = challenge_roll
        stat_roll = roll_dice_for_level(player.insight)
        difference += stat_roll - challenge_roll

        print(f"Total for Challenge level {choice['difficulty']}: {challenge_roll}")
        print(f"Total for Player insight level {player.insight}: {stat_roll}")

    if('diplomacy' in challenges):
        challenge_roll = roll_dice_for_level(choice['difficulty'])
        challenge_rolls['Diplomacy'] = challenge_roll
        stat_roll = roll_dice_for_level(player.diplomacy)
        difference += stat_roll - challenge_roll

        print(f"Total for Challenge level {choice['difficulty']}: {challenge_roll}")
        print(f"Total for Player diplomacy level {player.diplomacy}: {stat_roll}")

    if('force' in challenges):
        challenge_roll = roll_dice_for_level(choice['difficulty'])
        challenge_rolls['Force'] = challenge_roll
        stat_roll = roll_dice_for_level(player.force)
        difference += stat_roll - challenge_roll

        print(f"Total for Challenge level {choice['difficulty']}: {challenge_roll}")
        print(f"Total for Player force level {player.force}: {stat_roll}")

    return difference, challenge_rolls

def update_character():
    return

def update_standing():
    return

def update_objects():
    return

def generate_prompt_journal(mission_id):

    instructions = f"""- Analyse the JOURNAL and write a SPR for it. List specific events, actors, objects, locations, decisions, and consequences. Be precise.
    - Render the SPR as a distilled list of succinct statements, assertions, associations, concepts, analogies, and metaphors. The idea is to capture as much, conceptually, as possible but with as few words as possible. Write it in a way that makes sense to you, as the future audience will be another language model, not a human.
    - In prop1_consequences_analysis_of_last_event, analyse the consequences of the last event. What are the consequences for the player? What are the consequences for the other actors? What are the consequences for the objects? What are the consequences for the locations? What are the consequences for the decisions? What are the consequences for the story? What are the consequences for the theme? What are the consequences for the game?
    - In prop2_mission_completed, determine whether the mission is completed based on the last event.
    - In prop3_next_event_narrative, conjecture a new spin to the story and decide on what happens in the next event, while progressing towards mission completion.
    """
    prompt = textwrap.dedent(f"""_sec
    ### System:
    You are a Sparse Priming Representation (SPR) writer for the game events stored in the Player's Journal.
    An SPR is a particular kind of use of language for advanced NLP, NLU, and NLG tasks, particularly useful for the latest generation Large Language Models (LLMs). 
    LLMs are a kind of deep neural network. They have been demonstrated to embed knowledge, abilities, and concepts, ranging from reasoning to planning, and even to theory of mind. These are called latent abilities and latent content, collectively referred to as latent space. 
    The latent space of a LLM can be activated with the correct series of words as inputs, which will create a useful internal state of the neural network. This is not unlike how the right shorthand cues can prime a human mind to think in a certain way. 
    Like human minds, LLMs are associative, meaning you only need to use the correct associations to "prime" another model to think in the same way.
    
    Your mission is to create a SPR for the Journal of the player and output it in JSON. 
    The Journal contains all events and decision the player encountered. The theme of the game is the (Dutch) Golden Age of the Baroque era. 
                                 
    ### Character:
    {return_profile()}
    {return_character()}
    {return_connections()}
    {return_habitus()}

    ### Journal:
    {return_mission(mission_id=mission_id)}
    {return_allevents()}

    ### Instruction: 
    {return_second_person()}
    {instructions}
    """)
    print(prompt)
    return prompt


def summarize_journal(mission_id=None):

    schema = get_jsonschema('spr_schema.jinja')
    prompt = generate_prompt_journal(mission_id)
    result = batch_api(
            prompts=prompt, 
            schemas=schema,
            temperature=0.0,
            frequency_penalty=1.0
            )

    journal.spr = result
    return

def complete_event(accepted_proposal: Dict):
    event_id = accepted_proposal['event_id']
    event = journal.scheduled.get(event_id, False)
    if (event == False):
        print("Event not found")
        socketio.emit('remove_event_modal', {'event_id': event_id})
        return
    event['decision'] = accepted_proposal['option_id']
    if event['type'] == 'mission_select':
        journal.active[event_id] = event
        print("Mission active: ", journal.active[event_id])
    else:
        journal.completed[event_id] = event
        compute_event_response(event_id, event)
    journal.scheduled.pop(event_id)
    socketio.emit('remove_event_modal', {'event_id': event_id})

def compute_event_response(event_id, event):
    choice = event['options'][event['decision']]
    
    if event['type'] == 'event_challenge':
        # perform checks
        outcome, challenge_rolls = roll_all_dice_for(choice)
        # create response event
        response = {}
        response['parent_id'] = event_id
        response['mission_id'] = event['mission_id']
        response['outcome'] = outcome
        if (outcome > 0):
            effects = choice['success_effects']
            response['title'] = "Success!"
        if (outcome <= 0):
            effects = choice['failure_effects']
            response['title'] = "Failure!"
        response['event_body'] = effects['event_body']
        response['options'] = [
            {"player_option": "Continue"}
        ]

        resolved_effects, tooltip = resolve_gameplay_effects(effects['gameplay'], challenge_rolls, choice['difficulty'], outcome)

        response['options'][0]['tooltip_message'] = tooltip
        response['options'][0]['gameplay'] = resolved_effects

        load_event(response, gametime.datetime + timedelta(minutes=random.randint(10, 60)), 'event_confirmation')

    if event['type'] == 'event_confirmation':
        for key, value in choice.items():
            match key:
                case 'wealth_change':
                    player.wealth += value
                case 'notoriety_change':
                    player.notoriety += value
                # case 'character_change':
                #     update_character()
                # case 'standing_change':
                #     update_standing()
                # case 'item_change':
                #     update_objects()
                #     pass
        summarize_journal(event['mission_id'])
        emitter.emit('generate_action', {'action': 'summarize_journal', 'mission_id': event['mission_id'], 'queue': 'high'})

        print(f"Event is confirmed")

    # json_schema = get_jsonschema('event_response_schema.jinja')
    # prompt = generate_prompt_action_analysis(event_id)
    # print("generating event")
    # result_analysis = asyncio.run(get_ws_result(prompt, json_schema))

def resolve_gameplay_effects(gameplay_effects, challenge_rolls, difficulty, outcome):
    #calculate effects
    resolved_effects = {}
    tooltip_message = ""

    if ('direct_wealth_increase' in gameplay_effects):
        resolved_effects['wealth_change'] = round((5*roll_dice_for_level(difficulty) * roll_dice_for_level(player.commerce)))
        tooltip_message += f"Your wealth increases with: {resolved_effects['wealth_change']} fl. \n"

    if ('direct_wealth_decrease' in gameplay_effects):
        resolved_effects['wealth_change'] = round((5*roll_dice_for_level(difficulty) * roll_dice_for_level(player.commerce)))
        tooltip_message += f"Your wealth decreases with: {resolved_effects['wealth_change']} fl. \n"
        resolved_effects['wealth_change'] *= -1

    if('wealth_change' in gameplay_effects) == False:
        resolved_effects['wealth_change'] = 0

    if ('notoriety_increase' in gameplay_effects):
        resolved_effects['notoriety_change']  = roll_dice_for_level(difficulty)
        tooltip_message += f"Your notoriety increases with: {resolved_effects['notoriety_change']} \n"

    if ('notoriety_decrease' in gameplay_effects):
        resolved_effects['notoriety_change']  = roll_dice_for_level(difficulty)
        tooltip_message += f"Your notoriety decreases with: -{resolved_effects['notoriety_change']} \n"
        resolved_effects['notoriety_change'] *= -1

    if('notoriety_change' in gameplay_effects == False):
        resolved_effects['notoriety_change'] = 0

    if ('character_change' in gameplay_effects):
        tooltip_message += f"You are developing some new quirks. \n"
    else:
        resolved_effects['character_change'] = 0

    if ('standing_change' in gameplay_effects):
        tooltip_message += f"People will are starting to view you differently. \n"
    else:
        resolved_effects['standing_change'] = 0

    if 'item_gained' in gameplay_effects:
        resolved_effects['item_change'] = 1
        tooltip_message += f"You received a new item! Let's see what we can do with it. \n"

    if 'item_lost' in gameplay_effects:
        resolved_effects['item_change'] = -1
        tooltip_message += f"You lost an item! Let's investigate the damage. \n"
    
    if ('item_change' in gameplay_effects == False):
        resolved_effects['item_change'] = 0

    for key, value in challenge_rolls.items():
        if (key != 'payment' and outcome > 0):
            resolved_effects[f'{key}_experience'] = value
            tooltip_message += f"{key} experience gain: {value}\n"

    return resolved_effects, tooltip_message.replace("\n", "<br/>")


def generate_prompt_event_response():
    # this should contain the logic for failure and success
    print("generating event response")

def generate_prompt_mission_evaluation():
    print("generating mission evaluation")

def get_difficulty_from(target):

    difficulty_ranges = {
        'routine': (1, 1),
        'easy': (2, 2),
        'fair': (3, 4),
        'tricky': (5, 6),
        'challenging': (7, 9),
        'strenuous': (10, 14),
        'hard': (15, 20),
        'very hard': (21, 30),
        'extreme': (31, 41),
        'hardcore': (42, 55),
        'impossible': (56, float('inf'))  # using infinity to capture all higher values
    }

    for difficulty, (min_val, max_val) in difficulty_ranges.items():
        if min_val <= target <= max_val:
            return difficulty
        
def get_nearby_difficulty_from(target_difficulty):
    difficulty_labels = ['routine', 'easy', 'fair', 'tricky', 'challenging', 'strenuous', 'hard', 'very hard', 'extreme', 'hardcore', 'impossible']

    difficulty_index = difficulty_labels.index(target_difficulty)

    weights = np.ones(len(difficulty_labels))

    # Assign more emphasis on the difficulty index
    for i in range(len(difficulty_labels)):
        weights[i] = 1 / (1 + abs(i - difficulty_index)**2)

    # Normalize the weights to make them sum to 1
    weights /= sum(weights)

    return np.random.choice(difficulty_labels, p=weights)

def compute_event():
    number_of_choices = random.randint(2, 3)

    event_difficulty = get_difficulty_from(player.strength)
    event_difficulty = get_nearby_difficulty_from(event_difficulty)

    # Generate the array of keys
    keys_array = []
    for i in range(number_of_choices):
        key = f"option-{i+1}"
        keys_array.append(key)

    if (Config.SKIP_GEN_EVENT):
        result = Config.SKIP_VAL2
    else:
        schema = get_jsonschema('event_schema.jinja',keys_array)
        mission_ids = list(journal.active.keys())
        print(f"Mission ids: {mission_ids}")
        if len(mission_ids) == 0:
            mission_id = -1
        else:
            count_mission_events = 0
            for mission_id in mission_ids:
                for event_id, event in journal.scheduled.items():
                    if (event['type'] == 'event_challenge' and event.get('mission_id') == mission_id):
                        count_mission_events += 1
                        mission_ids.remove(mission_id)
                        break
            print(f"Mission events: {count_mission_events}")
            if (len(mission_ids)>0):
                mission_id = random.choice(mission_ids)
            else:
                mission_id = -1
        prompt = generate_prompt_event(event_difficulty, mission_id)
        event = batch_api(
            prompts=prompt, 
            schemas=schema,
            temperature=0.0,
            frequency_penalty=0.1
            )

    event['difficulty'] = event_difficulty
    event['options'] = []
    event['mission_id'] = mission_id

    # Iterate over the keys in the object
    for key in list(event.keys()):  # We use list() to create a copy of the keys since we'll be modifying the dictionary
        if "option-" in key:
            event["options"].append(event[key])
            del event[key]

    for option in event['options']:
        multiplier = ((len(event['options'])/2)-event['options'].index(option))*-0.15+1
        option['difficulty'] = round(get_random_value_from_label(event_difficulty)*multiplier)
        option['difficulty_str'] = get_difficulty_from(option['difficulty'])
        if ('payment' in option['challenges']):
            option['amount'] = roll_dice_for_level(option['difficulty']) * 10

    est_time, est_date = generate_schedule(event)
    triggerdate = get_trigger_datetime(est_date, est_time)

    load_event(event, triggerdate, 'event_challenge')


def load_event(event, triggerdate, template):

    # append triggerdate to event
    
    event_id = generate_event_id()

    event['triggerdate'] = triggerdate
    event['type'] = template
    event['html'] = render_template(
            f'{template}.html', 
            name=player.name,
            event=event,
            event_id=event_id
            )
     
    # append event to Game() scheduled_events
    journal.scheduled[event_id] = event


def generate_schedule(event):
    must_time = event['trigger_time_of_day']
    must_date = event['trigger_date']
    event.pop('trigger_date')
    event.pop('trigger_time_of_day')

    return must_time, must_date
    
 
def get_random_time(time_status):
    # Handling time
    if time_status == 'morning':
        hour = random.randint(7, 11)
    elif time_status == 'afternoon':
        hour = random.randint(12, 17)
    elif time_status == 'evening':
        hour = random.randint(18, 23)
    elif time_status == 'night':
        hour = random.randint(0, 6)
    else:
        raise ValueError("Invalid time_status")

    minute = random.randint(0, 59)
    return time(hour, minute)

def get_trigger_datetime(date_status, time_status):

    timesequence = ['morning','afternoon','evening','night']

    if date_status is False:
        date_status = 'tomorrow'

    if time_status is False:
        time_status = 'afternoon'

    # get index for each item in can_time_array from timesequence
    time_status_index = timesequence.index(time_status)
    game_time_status_index = timesequence.index(gametime.status)

     # If in the same part of the day, add a random amount of minutes to triggerdate

    if (date_status == 'now'):
        if (gametime.status == time_status):
            time = (gametime.datetime + timedelta(minutes=random.randint(1, 7))).time()
            date = gametime.datetime.date()
        else:
            date_status = 'today'
    
    if date_status == 'today':
        if (gametime.status == time_status):
            time = (gametime.datetime + timedelta(minutes=random.randint(1, 119))).time()
            date = gametime.datetime.date()
        elif time_status_index > game_time_status_index:
            time = get_random_time(time_status)
            date = gametime.datetime.date()
        else:
            date_status = 'tomorrow'
    
    if date_status == 'tomorrow':
        time = get_random_time(time_status)
        date = (gametime.datetime + timedelta(days=1)).date()

    if date_status == 'this_week':
        time = get_random_time(time_status)
        date = (gametime.datetime + timedelta(days=random.randint(2, 7))).date()
    
    # Combine date and time
    trigger_datetime = datetime.combine(date, time)
    
    return trigger_datetime


def generate_prompt_event(challenge_type,mission_id=-1):

    if (mission_id != -1):
        rule_1 = '- The event should align with the current mission.'
        rule_2 = '- Align the event for the set time of day and location, with the aim of further progressing the current mission.'
    else:
        rule_1 = '- The event should align with the last event and gamestate.'
        rule_2 = '- Write the event for the set time of day and location that is consistent with the current gamestate, with the aim of introducing new narrative elements to the player.'

    match challenge_type:
        case 'routine':
            rule_3 = """- Compose the event so that it requires the player to carry out a simple task, involving minimal thought or effort. For example, making a familiar meal."""
        case 'easy':
            rule_3 = """- Compose the event so that it demands from the player a bit of contemplation and effort. It should be manageable with a few minor obstacles. """
        case 'fair':
            rule_3 = """- Compose the event so that the player is faced with a challenge that needs careful thought to overcome tangible obstacles. For example, navigating through an unfamiliar city without a map."""
        case 'tricky':
            rule_3 = """- Compose the event so that it tests the player's skills and insights intensely. It should demand strategy and a deeper understanding. For example, organizing a complex event with multiple moving parts."""
        case 'challenging':
            rule_3 = """- Compose the event so that it exerts the player significantly both mentally and physically. For example, completing a rigorous obstacle course under a time limit."""
        case 'strenuous':
            rule_3 = """- Compose the event so that it appears formidable for the player, demanding thorough preparation and determination. For example, learning and mastering a new skill in a short period."""
        case 'hard':
            rule_3 = """- Compose the event so that it necessitates the player to go well beyond their usual capabilities, facing what seems nearly overwhelming. For example, leading a team in adverse conditions to achieve a challenging goal."""
        case 'very hard':
            rule_3 = """- Compose the event so that it pushes the player into the unknown, urging them to challenge established conventions and boundaries. For example, developing a groundbreaking solution to a significant problem."""
        case 'extreme':
            rule_3 = """- Compose the event so that its at the boundary of what's conceivable, where the player aims to achieve what's seen as a legendary feat. For example, scaling the highest peaks with minimal equipment."""
        case 'hardcore':
            rule_3 = """- Compose the event so that its almost insurmountable to most. The sheer undertaking itself should be a testament to the player's extreme ambition. For example, revolutionizing global perspectives on a deeply entrenched issue."""
        case 'impossible':
            rule_3 = """- Compose the event so that it stretches the imagination of what's possible, an event that no one has ever thought or dared to attempt. For example, fundamentally altering the course of human history through a singular act."""  
        case 'update':
            rule_3 = """- The goal of this event is to update the player on the evolving state of the game."""

    instructions = f"""
    - Write the event_body directly from on the proceeds and analysis from the last event in the "### Journal".
    - Important: do not repeat the last event. Instead, build on it.
    {rule_1}
    - Write the event_body briefly and succinctly (max 100 words), and keep it natural and easy to read.
    - Write the event based on the input provided by the "### Journal".
    - Take inspiration in writing the event / event_body from the "What may happen next" under "### Journal"

    # Other rules
    - Decide on the event location. Ensure it is not too far away from the current location.
    - Decide on the time of day it should trigger. This can be either morning, afternoon, evening or night.
    {rule_2}
    {rule_3}
    - Write the player_options in the order of the difficulty level, from easiest to hardest. The player must pick only 1 option.
    - For  "challenges" list the skill checks the player should perform, or a payment to perform the action. Do not mention this anywhere else. Use the following info to determine whether or not the player needs to perform a skill check:
    'Insight' revolves around the realm of deep intellectual exploration and esoteric wisdom. Rooted in the synthesis of intuitive perception with structured thought, it captures the essence of understanding phenomena that often transcend conventional boundaries.
    'Force' is the confluence of physical might with the ideals of principled leadership. It emphasizes the exertion of authority driven by both inner strength and a commitment to honorable action.
    'Diplomacy', at its core, is the art of navigating and harmonizing interpersonal relationships. It accentuates the importance of building bridges, fostering communal bonds, and skillfully managing social dynamics.  
    - Do not use the above gameplay jargon in the event_body. Keep it natural and focus on storytelling.
    - For the event_body / narrative effects of each option: write in the present tense and second person (e.g. "You managed to..."). Always directly branch off the options from the event_body. 
    - For "gameplay" list which player's stats would change. Do not mention this anywhere else.
    - For each option/decision write a short line of internal dialogue for the player, that is consistent with the player communication style, character and the consequences of the option. Write in the present tense and first person.
    """
    prompt = textwrap.dedent(f"""
    ### System:
    {return_intro("event writer", "generating a player event for a a mission in JSON format")} 

    ### Character (needed for hooks and ideas):
    {return_profile()}
    {return_character()}
    {return_connections()}
    {return_habitus()}

    ### Journal (very important!):
    {return_mission(mission_id=mission_id)}
    {return_journal_spr()}

    ### Instruction: 
    {return_second_person()}
    {instructions}
    
    ### Response:
    """)

    print(prompt)
    return prompt


def process_json(json_strings,isDict):
    # Remove '\t' characters
    processed_jsons = []
    for json_string in json_strings:
        cleaned_string = re.sub(r"prop\d+_", "", json_string)
        # cleaned_string = json_string.replace("\t", "")
        # incorrect_closing_pos = cleaned_string.rfind('"]}')
        # if incorrect_closing_pos != -1:
        #     # Remove the incorrect closing and correctly close the JSON
        #     last_comma_pos = cleaned_string.rfind(',', 0, incorrect_closing_pos)
        #     if last_comma_pos != -1:
        #         # Remove the last comma and the incorrect closing, then correctly close the JSON
        #         cleaned_string = cleaned_string[:last_comma_pos] + cleaned_string[incorrect_closing_pos:].replace('"]}', '] }')
        #     else:
        #         # If the last comma is not found, just fix the closing
        #         cleaned_string = cleaned_string[:incorrect_closing_pos] + '] }'
        # else:
        #     # If the incorrect closing is not found, return the original string
        #     cleaned_string = cleaned_string
        processed_jsons.append(cleaned_string)
    if isDict:
        dd = defaultdict(dict)
        for x in processed_jsons:
            x = json.loads(x)
            for key, value in x.items():
                dd[key].update(value)
        return dd
    return json.loads('[' + ', '.join(processed_jsons) + ']')



# define a simple coroutine
async def _batch_api(prompts, schemas, temperature, frequency_penalty, emit_progress_max=0):
    tasks = []
    if isinstance(prompts, str):
        tasks = [concept_generator(prompts, schemas, temperature, frequency_penalty, emit_progress_max)]
    if isinstance(prompts, list):
        tasks = [concept_generator(prompt, schema, temperature, frequency_penalty, emit_progress_max) for prompt, schema in zip(prompts, schemas)]
    if isinstance(prompts, dict):
        for category in prompts.keys():
            tasks += [concept_generator(prompt, schema, temperature, frequency_penalty, emit_progress_max, category) for prompt, schema in zip(prompts[category], schemas[category])]
    results = await asyncio.gather(*tasks)
    print(results)
    if isinstance(prompts, str):
        results[0] = re.sub(r"prop\d+_", "", results[0])
        return json.loads(results[0])
    return process_json(results,isinstance(prompts, dict))

async def concept_generator(prompt, schema, temperature, frequency_penalty, emit_progress_max, category=0):
    stream = False
    response = ""
    result = await openai.Completion.acreate(
        model=openai.Model.list()["data"][0]["id"],
        prompt=prompt,
        max_tokens=3000,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        stream = stream,
        jsonparser=schema)
    
    if stream:
        async for c in result:
            # sys.stdout.write(c.choices[0].text)
            # sys.stdout.flush()
            if (emit_progress_max!=0):
                progress = len(c.choices[0].text) / emit_progress_max * 100
                socketio.emit('update_progress', {'progress': progress})
            response += c.choices[0].text
    else:
        if (emit_progress_max!=0):
            progress = len(result.choices[0].text) / emit_progress_max * 100
            socketio.emit('update_progress', {'progress': progress})
        response = result.choices[0].text  

    if category != 0:
        response = '{'+f'"{category}": {response}'+'}'

    print(response)
    return response

def load_game(accepted_proposal: Dict):
    background = f"""Your background: {accepted_proposal['background']}
    """
    relationships = f"""Your relationships: {accepted_proposal['relations']}
    """
    player.starting_data = f"{background if len(accepted_proposal['background']) > 0 else ''}{relationships if len(accepted_proposal['relations']) > 0 else ''}"
    player.name = accepted_proposal['name']
    player.age = int(accepted_proposal['age'])
    player.occupation = accepted_proposal['occupation']
    player.lifestyle = accepted_proposal['lifestyle']
    player.people = accepted_proposal['people']
    player.objects = accepted_proposal['objects']
    player.places = accepted_proposal['places']

    # generate trink items
    print("Is app context:", app.app_context())
    print("Current app:", app._get_current_object())
    game_loader_template = render_template(
        'game_loader.html', 
        name=player.name
        )

    socketio.emit('deploy_loader', {'new_html_content': game_loader_template})

    if (Config.SKIP_GEN_STATE):
        data = Config.SKIP_VAL_STATE
        player.people = Config.SKIP_VAL_PEOPLE
        player.objects = Config.SKIP_VAL_OBJECTS
        player.places = Config.SKIP_VAL_PLACES

    else:

        categories = ['places', 'people', 'objects']
        schemas = []
        prompts = []
        for category in categories:
            wrapper = {'name': category}
            match category:
                case 'places':  
                    wrapper['item_count'] = str(5 - len(player.places))    
                case 'people':
                    wrapper['item_count'] = str(7 - len(player.people))
                case 'objects':
                    wrapper['item_count'] = str(3 - len(player.people))
            schema = get_jsonschema('instancing_schema.jinja',wrapper)
            prompt = generate_item_concepts(category, schema)
            schemas.append(schema)
            prompts.append(prompt)
            print(prompt)

        concept_list = batch_api(
            prompts=prompts, 
            schemas=schemas,
            temperature=0.0,
            frequency_penalty=1.0,
            emit_progress_max=11000
            )
        
        concept_list = {k: v for d in concept_list for k, v in d.items()}

        print("result:", concept_list)

        schemas = {}
        prompts = {}

        for category in categories:
            schemas[category] = []
            prompts[category] = []
            for concept in concept_list.get(category):
                wrapper = {'type': category, 'name': concept}
                schema = get_jsonschema('concept_schema.jinja',wrapper)
                prompt = generate_concept_details(category, schema)
                schemas[category].append(schema)
                prompts[category].append(prompt)
                print(schema)

        items = batch_api(
            prompts=prompts, 
            schemas=schemas,
            temperature=0.0,
            frequency_penalty=1.0,
            emit_progress_max=11000
            )
        
        print("result:", items)

        player.places = {item['name']: {"significance": item['significance'], "owner": item['owner'], "significant": True} for item in items.get('places').values()}
        player.people = {item['name']: {"significance": item['significance'], "location": item['location'], "disposition": item['disposition_towards_player'], "significant": True} for item in items.get('people').values()}
        player.objects = {item['name']: {"significance": item['significance'],"location": item['location'],"owner": item['owner'], "significant": True} for item in items.get('objects').values()}

        data = transform()

        print(data)

        socketio.emit('deploy_start_data', {'data': data})

    
def get_jsonschema(filename, items=None):
    file = os.path.join(app.root_path, 'templates', 'schemas', filename)
    with open(file, 'r') as f:
        template_str = f.read()
    template = Template(template_str)
    if items is None:
        return template.render()
    else:
        return template.render(items=items)
    
def transform():
    result = {
        "name": "My Empire",
        "children": []
    }
    
    dicts_to_iterate = [
        {"dict": player.places, "type": "Location"},
        {"dict": player.objects, "type": "Object"},
        {"dict": player.people, "type": "Entity"}
    ]

    for d in dicts_to_iterate:
        for key, value in d["dict"].items():
            child = {
                "name": key,
                "type": d["type"],
                "significance": value["significance"],
            }
            result["children"].append(child)
    return result

def generate_state_narrative(brainstorm_results = None):
    instructions = """
    - Understand the gamestate fully and constellate items from your background, relationships, occupation and lifestyle into interconnected narratives and themes.
    - Distill the narrative into three interconnected sections: player's background, player's lifestyle and player's standing. 
    - For the player's background, focus on player's important locations, their origin and hooks to the present and future.
    - For the player's lifestyle, focus on job & occupation and daily activities and expenses.
    - For the player's standing, focus on describing the player's reputation among communities, groups and organizations.
    - Be sure to write critically, holistically and realistically.
    - Do not use anything from the example JSON above. Be be creative and think outside the box. Go for a different take, structure and approach.
    - Be sure to write in second person, always considering everything from the perspective of the player.        
    """
    prompt = textwrap.dedent(f"""
    {return_intro("expert game state conceptualizer / narrative writer", "generating a narrative summary in JSON format")}
    {return_profile()}
    {return_character()}
    {player.starting_data}
    {return_connections()}
    {return_habitus()}
    ### Instruction: 
    {return_second_person()}
    {instructions}
    """)
    
    print(prompt)
    return prompt
    
    
def generate_item_concepts(type,json_state_schema):

    places = """{
            "places": [
                "Your REDACTED in REDACTED",
                "Your REDACTED room",
                "REDACTED",
                "REDACTED Port",
                "REDACTED Castle",
                ...
            ]
        }"""
    
    people = """{
            "people": [
                "Your [REDACTED] [REDACTED]",
                "[REDACTED] [REDACTED]",
                "[REDACTED]",
                "[REDACTED] working on your [REDACTED]",
                ...
            ]
        }"""

    objects = """{
            "objects": [
                "Your [REDACTED] in [REDACTED]",
                "[REDACTED]",
                "Your [REDACTED]",
                "Your [REDACTED] of [REDACTED]",
                "[REDACTED] the [REDACTED]",
                "Your [REDACTED] 'The [REDACTED] of [REDACTED]'",
                ...
            ]
        }"""
    
    match type:
        case 'places':
            definition = "specific places, whether natural or constructed that are of interest to the player. Examples: Cities, your house, natural landscapes, man-made structures, mystical realms."
            samples = places
            rule = '- Be sure to include: 1) one or more of your resting places OR your home AND 2) the place(s) where you earn money and spend your wealth.'
            location_format= ""

        case 'people':
            definition = "humans (EXCLUDING YOURSELF), animals, or sentient beings that are of interest to the player. Examples: Family members, friends, rivals, historical figures, pets, mythical creatures."
            samples = people
            rule = f"""- Be sure to include at least one nemesis OR individual OR faction OR group challenging the player. 
    - Do NOT list the player / EXCLUDE yourself  (= {player.name}) from the list."""
            location_format= f""", at a town, city or points of interest,"""

        case 'objects':
            definition = "tangible, inanimate items that are of interest to the player. These are typically interactable items that have a physical presence. Examples: Tools, weapons, vehicles, consumables, documents."
            samples = objects
            location_format= f""", at a town, city or points of interest,"""
            rule = '- Be sure to list at least some of the more important gear, outfit and possessions of the player. Do not list locations or places.'

    instructions = f"""- Aim for {type} with flavor that touch upon the player's characteristics, relationships, worldview, personality, social class, occupation and background.
    - Ensure that the {type} are significant to the player in the sense that the player can engage or interact with them.
    - Exclude {type} that do not align with the player's social class or wealth disposition.
    - Assign specific relatable English or Dutch names to the {type}. Never use jargon, sample or generic names (e.g. "entity-x", "[REDACTED]").
    - For all {type} in the provided array, ensure each can exist in the baroque era{location_format} and are of significance to the player.
    - Instances of '{type}' can exclusively be {definition}
    - Per item, be brief and succinct - just name the item, then proceed with the next entry.
    {rule}
    """

    prompt = textwrap.dedent(f"""
    ### System:
    {return_intro("expert game state generator/brainstormer", "generating an array of game items in JSON format")}  

    You MUST answer using the following json schema: {json_state_schema}

    ### Character (needed for hooks and ideas):
    {return_profile()}
    {return_character()}
    {player.starting_data}
    {return_connections()}
    {return_habitus()}

    ### Instruction:
    {return_second_person()}
    {instructions}
    
    ### Response:
    """)
    
    print(prompt)
    return prompt


def generate_concept_details(type,json_state_schema):
    parsed_json = json.loads(json_state_schema)
    compact_json_schema = json.dumps(parsed_json)

    match type:
        case 'places':
            definition = "specific places, whether natural or constructed that are of interest to the player. Examples: Cities, your house, natural landscapes, man-made structures, mystical realms."
            location_format= ""
            attributes="'significance' and 'owner' (not necesarily the player)"
            rule= f"- Determine who is the most likely owner. In most cases it is not the player / {player.name}. Instead hypothesize a specific 'name' who is the most likely owner of place."

        case 'people':
            definition = "humans (EXCLUDING YOURSELF), animals, or sentient beings that are of interest to the player. Examples: Family members, friends, rivals, historical figures, pets, mythical creatures."
            location_format= f""", at a town, city or points of interest,"""
            attributes="'significance', 'location' and 'disposition_towards_player'"
            rule="- Ensure that the player has a relationship with this person."

        case 'objects':
            definition = "tangible, inanimate items that are of interest to the player. These are typically interactable items that have a physical presence. Examples: Tools, weapons, vehicles, consumables, documents."
            location_format= f""", at a town, city or points of interest,"""
            attributes="'significance', 'location' and 'owner'"
            rule="- Determine who is the most likely owner."

    instructions = f"""
    - For all {type} in the provided array, expand on each required attribute according to the json schema. These are {attributes}.
    - Align the {type} with the {player.name}'s worldview, personality, social class, occupation and background. The more personal, signigicant and specific to {player.name} the better.
    {rule}
    - Be sure to write in second person, always considering the {type} from the perspective of the player ({player.name}). E.g. "Your father.." rather than "{player.name}'s father...".
    - Never mention the player's name ({player.name}) in the {type} attributes, instead always refer to the player as 'you' or 'your'.
    - For all {type} in the provided array, ensure each can exist in the baroque era{location_format} and are of significance to the player.
    - Ensure the descriptions should be consistent with historical facts and the cultural context of the Dutch Golden Age of the Baroque era.
    - Instances of '{type}' can exclusively be {definition}
    - Use the following JSON schema for your response: {compact_json_schema}
    """

    prompt = textwrap.dedent(f"""
    ### System:
    {return_intro("expert game state generator/brainstormer", "generating an array of game items in JSON format")}  

    ### Character (needed for hooks and ideas):
    {return_profile()}
    {return_character()}
    {player.starting_data}
    {return_connections()}
    {return_habitus()}

    ### Instruction: 
    {return_second_person()}
    {instructions}
    
    ### Response:
    """)
    
    print(prompt)
    return prompt
    
def run_async_in_thread(coroutine):
    """
    Run an async coroutine in a new thread, and return the result.
    """
    result = None
    exception = None

    def thread_target(loop, coroutine):
        nonlocal result, exception
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(coroutine)
        except Exception as e:
            exception = e
        finally:
            loop.close()

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=thread_target, args=(loop, coroutine))
    thread.start()
    thread.join()  # Wait for the thread to complete

    if exception:
        raise exception  # Re-raise any exception that occurred in the thread

    return result


def batch_api(prompts, schemas, temperature, frequency_penalty, emit_progress_max=0):
    api_args = {
        "prompts": prompts, 
        "schemas": schemas,
        "temperature": temperature,
        "frequency_penalty": frequency_penalty,
        "emit_progress_max": emit_progress_max
    }
    reponse = run_async_in_thread(_batch_api(**api_args))
    return reponse
    
def compute_mission():

    if (Config.SKIP_GEN_MISSION):
        result = Config.SKIP_VAL_MISSION
    else:
        schema = get_jsonschema('mission_schema.jinja')
        prompt = generate_prompt_mission()
        print(prompt)
        result = batch_api(
            prompts=prompt, 
            schemas=schema,
            temperature=0.0,
            frequency_penalty=1.0
            )
    event = result

    event['missions'] = [value for key, value in event['missions'].items()]

    # Create a new JSON object with the original array-based structure

    print(event['missions'])

    load_event(event, gametime.datetime + timedelta(hours=3), 'mission_select')

def generate_prompt_mission():
    instructions = f"""- Generate THREE missions that are warranted by the player state and game state.
    - Ensure that the generated missions BRANCH from the current gamestate and are NOT SEQUENTIAL TO EACH OTHER.
    - Ensure the missions are concrete, clear, interesting and easy to understand. Write in the present tense and second person.
    - Keep the plot of each mission mysterious and open-ended. Do not reveal the outcome of the missions.
    - Ensure the missions are consistent with the player's character, relationships & reputation, occupation and worldview.
    - Write missions that are significant to the player and resolve the player's current challenges.
    """
    prompt = textwrap.dedent(f"""
    {return_intro("expert mission writer", "generating a player mission in JSON format")}
    {return_profile()}
    {return_character()}
    {return_connections()}
    {return_habitus()}
    ### Instruction: 
    {return_second_person()}
    {instructions}
    """)
    return prompt
    
def generate_event_id():
    event_id = journal.counter
    journal.counter += 1
    return int(event_id)

def return_profile(event = None):
    profile = f"""
    Your name: {player.name} (this is you - don't refer to yourself from the third perspective!)
    Your age: {player.age}
    Your occupation: {player.occupation}
    Your current location:{player.location}"""
    if event is None:
        return profile
    else:
        time = f"""The time of day: {event['trigger_time_of_day']}
        """
        profile += time
    return profile

def return_character():
    character = f"""Your worldview: {player.worldview_str}
    Your social class: {player.socialclass_str}
    Your personality: {player.personality_str}
    Your traits: {player.traits_str}
    Your communication style: {player.communication_str}"""
    return character

def return_connections():
    people = ""
    standing = ""
    notoriety = f"Your notoriety: {player.notoriety_str}\n"
    if len(player.standing) > 1:
        standing= f"Your standing: {player.standing}\n\n"
    if len(player.sig_people_str) > 1:
        people = f"Your significant people:\n{player.sig_people_str}\n"
    return notoriety + standing + people

def return_habitus():
    lifestyle = ""
    objects = ""
    places = ""
    finance = f"Your financial health: {player.financial_health_str}\n"
    if len(player.lifestyle) > 1:
        lifestyle = f"Your lifestyle: {player.lifestyle}\n\n"
    if len(player.sig_places_str) > 1:
        places = f"Your significant places:\n{player.sig_places_str}\n"
    if len(player.sig_objects_str) > 1:
        objects = f"Your significant objects:\n{player.sig_objects_str}\n"
    return places + objects + finance + lifestyle

def return_mission(mission_id = -1, event_id = -1):
    if (event_id != -1):
        print(f"Event id: {event_id}")
        mission_id = int(journal.scheduled.get(event_id).get('mission_id',-1))
        print(f"Mission id: {mission_id}")
    if (mission_id == -1):
        mission = None
        return ""
    else:
        mission = journal.active.get(mission_id)
        mission = mission['missions'][mission['decision']]
        mission = mission['title'] + ":  " + mission['narrative']
        mission = f"""The current active mission (important):\n{mission}"""
        return mission
    
def return_allevents():
    counter = 0
    progress = "\n\n# Your mission log (descending from the earliest / oldest event to the most recent / latest event):\n"
    for event in journal.readme:
        counter += 1
        if counter == len(journal.readme):
            # perform some code if this is the last event in the list
            progress += "\nThis is the latest event in your mission log:\n"
        progress += event.get('story') + "\n"
    if counter == 0:
        progress = ""
    return progress
    
def return_journal_spr():
        progress = ""
        if hasattr(journal, 'spr'):
            progress = f"""Analysis of last event: {journal.spr['consequences_analysis_of_last_event']}\n
            What may happen next: {journal.spr['next_event_narrative']}
            """
        return progress

def return_event(event = None, executed_option = None):
    if not journal.scheduled or event is None:
        event = None
    else:
        event = f"""# The event that was triggered (important):\n {event['title']}": "{event['event_body']}\n\n
        # The option the player picked (important):\n {executed_option}"""
    return event

def return_intro(role, task):
    intro = f"""You are a {role} for grand strategy games. 
    The theme of the game is the (Dutch) Golden Age of the Baroque era.              
    You are tasked with {task} that is consistent with the current gamestate.
    """
    return intro

def return_second_person():
    instructions = f"""- Always write in English in the SECOND perspective - as if written for the player. E.g. "You have.." or "You are.." or "You need to.." or "Your x.." or "You are facing..". So do not say "{player.name}'s father", instead say "your father"."""
    return instructions