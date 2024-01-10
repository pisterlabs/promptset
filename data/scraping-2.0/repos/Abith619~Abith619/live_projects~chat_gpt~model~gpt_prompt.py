from odoo import fields, models, api
from datetime import date, datetime,timedelta
from odoo.exceptions import ValidationError, UserError
import openai

class GptPrompt(models.Model):
    _name = "gpt.prompt"
    _rec_name = 'question'
    
    token = fields.Char("Token", default='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')
    question = fields.Char('Question')
    prompt = fields.Text(string='Prompt ', default=f'I want You to Be Act as A Rasa Bot Superviser .Create the below files for this question: for Rasa bot training.You  Must Maintain The nlu File Structure.Use Postgresql for Db Queries And Get Partner_id from Tracker.File Structure : [*nlu.yml[Create This file structue  : nlu_startintent :[intent] example : |[15 examples]]nlu_end*stories.yml[Create This file structue  : stories_startstory: [story Title]steps:-[steps]stories_end],*Actions. py[write custom action  : Action_startClassClassname:[queris in postgresql]Action_end].].')
    
    @api.constrains('question')
    def generate_project(self):
        openai.api_key = "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"

        completions = openai.Completion.create(
            model="text-davinci-003",
            # prompt=f"I want You to Be Act as A Rasa Bot Superviser .Create the below files for this question: {self.question} for Rasa bot training.You  Must Maintain The nlu File Structure.Use Postgresql for Db Queries And Get Partner_id from Tracker.File Structure : [*nlu.yml[Create This file structue  : nlu_startintent :[intent] example : |[15 examples]]nlu_end*stories.yml[Create This file structue  : stories_startstory: [story Title]steps:-[steps]stories_end],*Actions. py[write custom action  : Action_startClassClassname:[queris in postgresql]Action_end].].",
            prompt=self.prompt+self.question,
            temperature=0.7,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
                )

        paragraph = completions.choices[0].text.strip()
        # raise ValidationError(paragraph)
        try:
            nlu = ((paragraph.split("nlu_startintent:"))[1].split("nlu_end"))[0]
            story = ((paragraph.split("stories_start"))[1].split("stories_end"))[0]
            api_data = ((paragraph.split("Action_start"))[1].split("Action_end"))[0]
            intent = ((nlu.split("intent:")[1]).split("examples:"))[0]
            story_title = ((paragraph.split("story:"))[1].split("steps:"))[0]
            action = (story.split("action:"))[-1]
            data = {
                'intent': intent,
                'example_set' :  nlu,
                'story_title': story_title,
                'story_set' :  story,
                'action_api' :  api_data,
                'intent_domain' : intent,
                'action_domain' : action,
                    }
            create_intent = self.env['rasa.intent'].create(data)
        except:
            pass
