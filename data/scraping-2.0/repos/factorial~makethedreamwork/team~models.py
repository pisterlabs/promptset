from django.conf import settings
from django.db import models
from django.utils import timezone

from team.ai import openai_call, openai_image, lookahead_filter
from team.utils import approximate_word_count, persist_image
from team import prompts
import urllib.parse
import requests
import os
import openai
import json
import time
import uuid
import threading
import random


class Team(models.Model):
    created = models.DateTimeField(default=timezone.now)
    guid = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    objective = models.CharField(null=False, blank=False, max_length=511)
    description = models.TextField(null=True, blank=True)
    generation_progress_percent = models.PositiveSmallIntegerField(default=0)
    private = models.BooleanField(default=False, null=False)
    tokens_used = models.PositiveIntegerField(default=0)
    moderator_image_url = models.URLField(max_length=1023, null=True, blank=True)

    def __str__(self):
        return f"{self.objective} ({self.guid}) private={self.private}"

    @classmethod
    def create_and_generate_team(cls, OBJECTIVE):
        prompt = prompts.CHECK_OFFENSE.format(OBJECTIVE=OBJECTIVE)
        print(prompt)
        result = next(openai_call(prompt, max_tokens=100))
        print("Offensive? "+ result)
        if not result or 'yes' in result.lower():
            return None

        # make new team, get guid
        new_team = cls(objective=OBJECTIVE)
        new_team.save()
        guid = new_team.guid
        print(f"Launching thread to generate_team {new_team}...")

        t = threading.Thread(target=Team.generate_team,args=[new_team], daemon=True)
        t.start()

        print(f"Thread for generate_team {new_team} launched.")
        return guid

    def generate_team(self):
        print(f"Generating team for {self.objective} guid {self.guid}")
        start_time = time.time()

        progress_points_total = 0

        context = ""
        print(f"Prompt building for team {self}")
        prompt = prompts.DEFINE_TEAM.format(objective=self.objective)
        print(f"Calling openai for team {self}")
        result = next(openai_call(prompt, max_tokens=3000))
        print(result)
        if not result:
            print(f"Something went wrong getting team roles for {self}.")
            return

        self.description=result
        self.tokens_used = 0
        self.generation_progress_percent = 10
        self.save()

        context = prompt + result

        role_tasks = {}
        try:
            role_tasks = json.loads(f"{result}")
            print(f"Parsed {self} into json {role_tasks}")
        except:
            print("Couldn't parse {result}. No role_tasks for {self}")

        valid_json = False
        while not valid_json:
            prompt = prompts.LIST_ROLES.format(context=context)
            result = next(openai_call(prompt))
            result = "["+result
            print(result)
            if not result:
                print(f"Something went wrong getting openai JS array roles out of team description for {self}.")
                return
            self.tokens_used = 0
            self.generation_progress_percent = 20 
            self.save()

            context = prompt + result

            try:
                roles = json.loads(f"{result}")
                valid_json = True
            except:
                print(f"Something went wrong PARSING openai's JS array roles out of team description for {self} Asking again forever.")
                prompt = prompts.JAVASCRIPT_ERROR.format(context=context)

        progress_points = 0
        steps_per_role = 3
        total_progress_points = len(roles)*steps_per_role
        base_progress_percent = 20
        
        # generate moderator image (not a real role)
        #self.moderator_image_url = persist_image(openai_image(prompt=prompts.MODERATOR_AVATAR))
        #self.save()

        for role in roles:
            print(f"Generating role {role} for {self}...")
            loop_context = context
            prompt = prompts.NEW_MEMBER_QUESTIONS.format(context=loop_context, role=role, objective=self.objective)
            result = next(openai_call(prompt, max_tokens=3000))
            print(result)
            if not result:
                print("Something went wrong getting new member questions for {role}.")
                return
            progress_points = progress_points + 1
            self.tokens_used = 0
            self.generation_progress_percent = round(base_progress_percent + ((100-base_progress_percent)*(progress_points/total_progress_points)))
            self.save()

            new_role = self.role_set.create(
                    name=role,
                    questions_text=result,
            )

            loop_context = prompt + result

            prompt = prompts.GENERATE_HANDBOOK.format(context=loop_context, role=role)

            result = next(openai_call(prompt, max_tokens=3000))
            print(result)
            if not result:
                print("Something went wrong getting expert answers for {role}.")
                return


            new_role.guide_text =result
            if role in role_tasks:
                print(f"Yes, {role} is in {role_tasks}")
                role_tasks_to_save=role_tasks[role]
            elif role in role_tasks[list(role_tasks.keys())[0]]:
                print(f"Yes, {role} is in first key of {role_tasks}")
                role_tasks_to_save=role_tasks[list(role_tasks.keys)[0]][role]
            else:
                print(f"No, {role} is not in {role_tasks}, so this role gets no tasks.")
                role_tasks_to_save=None

            if role_tasks_to_save:
                if isinstance(role_tasks_to_save, list):
                    print("Saving tasks as a list from list")
                    new_role.tasks_list_js_array=json.dumps(role_tasks_to_save)
                elif isinstance(role_tasks[role], dict):
                    print("Saving tasks as a list from dict")
                    new_role.tasks_list_js_array=json.dumps([role_tasks_to_save[key] for key in role_tasks[role]])
                else:
                    print("Saving tasks as a string")
                    new_role.tasks_list_text=json.dumps(role_tasks_to_save)
            else:
                print(f"So yeah. Saving.")

            new_role.save()

            progress_points = progress_points + 1
            self.tokens_used = 0
            self.generation_progress_percent = round(base_progress_percent + ((100-base_progress_percent)*(progress_points/total_progress_points)))
            self.save()

            new_role.generate_image()

            progress_points = progress_points + 1
            self.generation_progress_percent = round(base_progress_percent + ((100-base_progress_percent)*(progress_points/total_progress_points)))
            self.save()

        end_time = time.time()
        print(end_time)
        total_time_mins = (end_time - start_time)/60
        print(f"Generating {self} took {total_time_mins} minutes. Now downloading images from openai...")

        for role in self.role_set.all():
            role.persist_image()
        
        end_time = time.time()
        total_time_mins = (end_time - start_time)/60
        

        # calculate_cost()
        cost_per_token = (0.002/1000)
        cost_per_image = (0.018)
        #cost = (self.tokens_used * cost_per_token) + (cost_per_image * self.role_set.count())
        #print(f"After downloading {self.role_set.count()} images, {self} took {total_time_mins} minutes & {self.tokens_used} tokens (${cost}).")


    def generate_chat(self, human_role_guids: list = None):
        new_chat = Chat.objects.create(team_id=self.guid)
        new_chat.save()
        new_chat.next_speaker_name = self.role_set.first().name
        new_chat.save()
        guid = new_chat.guid
        if not human_role_guids:
            human_role_guids = []

        for role in self.role_set.all():
            if str(role.guid) in human_role_guids:
                print(f"Not summarizing for human role {role.name}")
                new_chat.human_roles.add(role)
                continue
            else:
                print(f"{role.guid} not in human guids {human_role_guids}")

            print(f"Making AI system prompt for role {role.name}")
            prompt = prompts.AI_ROLE_PROMPT.format(role=role.name, objective=self.objective, responsibilities=f"{role.tasks_list_js_array}{role.tasks_list_text}")
            role.ai_prompt = prompt
            role.save()

        print(f"Made AI agents for {self}. Next speaker is {new_chat.next_speaker_name}. Returning chat object.")
        return new_chat


    def get_template_context(self):
        retval = {
                "generation_progress_percent": self.generation_progress_percent,
                "team": {
                    "team": self,
                    "objective": self.objective,
                    "description": self.description,
                    "roles": {
                    }
                }
        }

        for role in self.role_set.all():
            retval["team"]["roles"][role.name] = {
                "role": role,
                "questions": role.questions_text,
                "guide": role.guide_text,
                "tasks_string":  role.tasks_list_text or "",
                "image_url": role.image_url,
                "linkedin_url": "https://www.linkedin.com/search/results/all/?keywords="+ urllib.parse.quote_plus(role.name, safe=''),
            }
            try:
                retval["team"]["roles"][role.name]["tasks"] = json.loads(role.tasks_list_js_array)
            except:
                if role.tasks_list_js_array:
                    print(f"Couldn't parse {role.tasks_list_js_array} on {role.guid} which is a {role.name}")


        return retval

class Role(models.Model):
    guid = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.TextField(null=False, blank=False)
    questions_text = models.TextField(null=True, blank=True)
    guide_text = models.TextField(null=True, blank=True)
    tasks_list_js_array = models.TextField(null=True, blank=True)
    tasks_list_text = models.TextField(null=True, blank=True)
    image_url = models.URLField(max_length=1023, null=True, blank=True)
    ai_prompt = models.TextField(null=True, blank=True)


    team = models.ForeignKey(Team, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.name} ({self.guid}) on {self.team.objective} team"

    def generate_image(self):
        print(f"Generating image for {self}")
        
        mascfem = random.choice(["male ", "female "])
        city = "Atlanta"
        prompt = prompts.AVATAR_FROM_CITY.format(mascfem=mascfem, role=self.name, city=city, add='')
        new_image_url = openai_image(prompt=prompt)
        if not new_image_url:
            # try generating a stranger instead
            prompt = prompts.AVATAR.format(mascfem=mascfem)
            new_image_url = openai_image(prompt=prompt)

        print(f"Response generating image for {self}: {new_image_url}")
        if new_image_url:
            self.image_url=new_image_url
            self.save()
        else:
            print(f"That's a major error generating image for {self}")

    def persist_image(self):
        self.image_url = persist_image(self.image_url)
        self.save();
        print(f"Saved image. It's at {self.image_url} now")

class Chat(models.Model):
    guid = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    log = models.TextField(null=True, blank=True)
    log_historical = models.TextField(null=True, blank=True)
    next_speaker_name = models.TextField(null=True, blank=True)

    team = models.ForeignKey(Team, on_delete=models.CASCADE)
    human_roles =  models.ManyToManyField(Role)

    def __str__(self):
        return f"Chat ({self.guid}) for {self.team.objective} team"

    def summarize_and_save(self):
        print("*************** summarizing and saving chat *****")
        end_of_session_message = "## END OF MEETING SUMMARY\n\n"
        yield_dict = { 'data': end_of_session_message }
        yield f"data: {json.dumps(yield_dict)}\n\n"
        new_log_item = f"""{end_of_session_message}"""

        chatlog = self.log
        summary_system_prompt = prompts.SUMMARIZER
        print(f"Summarizing Chat {self.guid} so far with {summary_system_prompt}")
        summary_messages = [{"role": "system", "content": summary_system_prompt }]
        # summary must not fail. a token is about 3/4 of a word.
        token_count = approximate_word_count(f"{summary_system_prompt}\n{chatlog}") * (4/3)
        max_token_count = 3000

        if token_count > max_token_count:
            scale_factor = max_token_count/token_count
            substrlen = int(len(chatlog)*scale_factor)
            chatlog = chatlog[-substrlen:]
            print(f"Summarizing last {approximate_word_count(chatlog)} of chatlog")
            print(f"{chatlog}")

        prompt = f"{chatlog}\n{end_of_session_message}"

        result = openai_call(prompt,role="user", max_tokens=1200, previous_messages=summary_messages, stream=True)
        # Stream the response line by line to the client
        summary = ""
        for message in result:
            try:
                summary += message['choices'][0]['delta']['content']
                yield_dict = { 'data': message['choices'][0]['delta']['content'] }
                print(f"yielding {yield_dict}")
                yield f"data: {json.dumps(yield_dict)}\n\n"
            except Exception as e:
                print('bare except {e}')
                pass

        # save the log so the chat can be rendered as old log + summary + current log
        self.log_historical = f"{self.log_historical or ''}\n{self.log}\n"
        # but start over with chat.log = just the summary as chat.log
        startover_message = f"""
# CHAT LOG - TEAM OBJECTIVE = {self.team.objective}

## Moderator
Welcome back, team. Continue work on your objective.
"""
        yield_dict = { 'data': startover_message }
        yield f"data: {json.dumps(yield_dict)}\n\n"

        new_log_item += f"""
{summary}

{startover_message}
"""
        self.log = new_log_item
        self.next_speaker_name = self.team.role_set.first().name
        self.save()
        return new_log_item


    def do_one_chat_round(self):
        moderator_prompt = prompts.TASK_FINDER
        if self.next_speaker_name is None:
            self.next_speaker_name = self.team.role_set.first().name
        waiting_for_human_input = self.next_speaker_name in self.human_roles.all().values_list('name', flat=True)
        print(f"waiting for human? {waiting_for_human_input}, next spker {self.next_speaker_name}, not in {self.human_roles.all().values_list('name', flat=True)}")
        while not waiting_for_human_input:
            template_context={}
            role_list = []
            for role in self.team.role_set.all():
                role_list.append(role)

            print(f"Rotating through all roles in team {self.team}: {role_list}")
            for idx, role in enumerate(role_list):
                # find the new next speaker after this one
                if role and role.name != self.next_speaker_name:
                    print(f"NOT chatting with {role.name}, isnt next speaker {self.next_speaker_name}")
                    continue

                try:
                    next_role_name = role_list[idx+1].name
                except:
                    next_role_name = role_list[0].name
                break

            print(f"Chatting with role: {role}, next up will be {next_role_name}.")
            print(f"first a moderator prompt.")

            # Moderator
            system_prompt = moderator_prompt.format(role=role.name)
            previous_messages = [ {"role": "system", "content": system_prompt} ]
            openai_role = "user"
            prompt = self.log
            print(f"Moderating with {system_prompt}")
            yield_dict = { 'data': "\n## Moderator\n" }
            yield f"data: {json.dumps(yield_dict)}\n\n"
            result = openai_call(prompt,role=openai_role, max_tokens=500, previous_messages=previous_messages, stream=True)
            # Stream the response line by line to the client
            chat_log_update = ""
            for message in result:
                if message is False:
                    # error. probably too long.
                    # Summarize chat so far.
                    print("Summarizinggg.")
                    for item in self.summarize_and_save():
                        yield item
                    self.save()
                    continue

                try:
                    next_thing = message['choices'][0]['delta']['content']
                    if next_thing.strip().lower() == 'as':
                        #print(f"next thing is {next_thing}")
                        # lookahead 5 to filter out "as an ai language model" or the like
                        next_thing = lookahead_filter(result, next_thing, ['as an ai language model, ', 'as the ai language model, ', 'as a language model, '])
                    chat_log_update += next_thing
                    yield_dict = { 'data': next_thing }
                    yield f"data: {json.dumps(yield_dict)}\n\n"
                except Exception as e:
                    print(f"Lazy except. {e}")
                    pass

            print(f"yielding data:\\n twice")
            yield_dict = { 'data': "\n\n" }
            yield f"data: {json.dumps(yield_dict)}\n\n"

            new_log_line = f"\n## Moderator\n{chat_log_update}\n\n"
            print(f"Saving new line: {new_log_line}")
            self.log += new_log_line
            self.save()

            # Now talk to the speaker.
            print(f"Now time to talk to {role}") 
            if role in self.human_roles.all():
                # Done with chat, pausing for human response
                print(f"Waiting on human: {role}")
                waiting_for_human_input=True
                self.next_speaker_name = role.name
                self.save()
                break

            #chatting with an AI role

            # TODO HERE: remodel chat as list of messages so we can make previous_messages mark "assistant"
            # all the previous messages from this role.
            # Then, refactor each role into two system prompts: info sharing, then task suggestion. each task is
            # either info gathering or other (thus, necessary for a person to do and eternally undone unless human says so)
            # then maybe make moderator a permanent fixture of this structure who summarizes each group of messages into tasks & facts?
            system_prompt = f"""{role.ai_prompt}"""
            user_prompt = f"""{self.log or ""}\n\n"""
            previous_messages = [ {"role": "system", "content": system_prompt} ]
            openai_role = "user"
            prompt = user_prompt

            yield_dict = { 'data': f"## {role.name}\n" }
            yield f"data: {json.dumps(yield_dict)}\n\n"
            result = openai_call(prompt,role=openai_role, max_tokens=1000, previous_messages=previous_messages, stream=True)

            # Stream the response line by line to the client
            chat_log_update = ""
            for message in result:
                if message is False:
                    print("Summarizinggg.")
                    for item in self.summarize_and_save():
                        yield item
                    self.save()
                
                try:
                   
                    next_thing = message['choices'][0]['delta']['content']
                    if next_thing.strip().lower() == 'as':
                        #print(f"next thing is {next_thing}")
                        # lookahead 5 to filter out "as an ai language model" or the like
                        next_thing = lookahead_filter(result, next_thing, ['as an ai language model, ', 'as the ai language model, ', 'as a language model, '])
                    chat_log_update += next_thing
                    yield_dict = { 'data': next_thing }
                    yield f"data: {json.dumps(yield_dict)}\n\n"
                except Exception as e:
                    print(f"Laziness error {e}")
                    pass

            yield_dict = { 'data': "\n\n" }
            yield f"data: {json.dumps(yield_dict)}\n\n"
            self.log += f"## {role.name}\n{chat_log_update}\n\n"
            self.next_speaker_name = next_role_name
            self.save()
           
            # Done with the role ring 
            if self.human_roles.count() == 0:
                print(f"no human role names possible. THIS IS NOW A HUGE PROBLEM because i stopped caring if summaries are made.")
        return


