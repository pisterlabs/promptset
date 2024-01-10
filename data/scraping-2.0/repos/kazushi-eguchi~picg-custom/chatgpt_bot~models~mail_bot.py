from odoo import api, fields, models, _
from odoo.exceptions import UserError

import openai
import json
import aiohttp
import asyncio
import requests
from bs4 import BeautifulSoup as BS


class ChatGptBot(models.AbstractModel):
    _inherit = 'mail.bot'
    _description = 'ChatGPT OdooBot'

   


#################################################################################################
#                                       ORM FUNCTION                                            #
#################################################################################################

    def create_content(self):
        if self.text_chatgpt:
        # raise UserError(_("This feature is not available yet."))
            openai.api_key = self.env['ir.config_parameter'].sudo().get_param('chatgpt_api_key')
            
            ex_json = {
                "title" : "titolo",
                "description" : "descrizione",
                "keywords" : "keywords",
            }
            # print(json.dumps(ex_json))
            # return str(ex_json)
            try:
                response = openai.Completion.create(
                model="text-davinci-003",
                prompt=self.text_chatgpt+". Scrivi la risposta in html tra un <div> e un /div>, con titoli, paragrafi, link. al termine crea un json con doppi apici come delimitatore del contenuto ma che rispetti questa matrice e che description sia massimo di 160 caratteri, "+json.dumps(ex_json) + "per poter scrivere i campi necessari al indicizzazione tramite SEO",
                temperature=0.7,
                max_tokens=3000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                user = self.name,
                )
                res = response['choices'][0]['text']

                
                text = res.split('</div>',1)[0]
                seo = res.split('</div>', 1)[1]
                
                if seo:
                    try:
                        res = json.loads(seo)
                        self.website_meta_title = res['title']
                        self.website_meta_description = res['description']
                        self.website_meta_keywords = res['keywords']
                    except:
                        pass
                        raise UserError(_("The response is not a valid json."))
                self.content = text
                if self.is_publish_now:
                    self.is_published = True
                else:
                    self.is_published = False
                self.is_elaborate = True
            except Exception as e:
                raise UserError(_(e))
            # self.create_seo()
            """
            # for resp in openai.Completion.create(
            #     model='text-davinci-003',
            #     prompt=self.text_chatgpt,
            #     max_tokens=512, stream=True):
            #         txt += resp.choices[0].text
            #         print(txt)
            """
 
        
#################################################################################################
#                                    ONCHANGE && COMPUTE                                        #
#################################################################################################




#################################################################################################
#                                      CUSTOM FUNCTION                                          #
#################################################################################################
    
    
    

    def _get_answer(self, record, body, values, command=False):
        odoobot_state = self.env.user.odoobot_state

        res = super(ChatGptBot, self)._get_answer(record, body, values, command=False)

        
        get_last_message = self.env['mail.channel'].search([('id', '=', record.id)]).message_ids.ids
        messages = self.env['mail.message'].search([('id', 'in', get_last_message)], order='id desc', limit=2).mapped('body')
        old_conv = ""
        for msg in messages:
            if msg:
                old_conv += BS(msg, 'html.parser').get_text() + "\n"
        # body = old_conv
       
        if body =="#enable":
            self.env.user.odoobot_state = 'chatgpt'
            return "ChatGpt enabled"
        if body =="#disable":
            self.env.user.odoobot_state = 'disabled'
            return "ChatGpt disabled"
        
        list = {'o_mail_notification', 'o_mail_redirect', 'o_channel_redirect'}
        msg_sys = [ele for ele in list if(ele in body)]
        
        
        
        
        if odoobot_state == 'chatgpt' and not msg_sys:
            print(odoobot_state)
            self.with_delay().risposta(record, body)
            print("res::::::::::", res)
            return 
        else:
            return res

    
    
    
    

    
    def risposta(self, record, body): 
        
        openai.api_key = self.env['ir.config_parameter'].sudo().get_param('chatgpt_api_key')
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=body,
            temperature=0.7,
            max_tokens=3000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=[" Human:", " AI:"],
            echo = False,
            )
        gpt = "<strong>OpenAI: </strong>"+(response['choices'][0]['text'])
        odoobot_id = self.env['ir.model.data']._xmlid_to_res_id("base.partner_root")
        mod_response = self.env['mail.channel'].with_context(chatgpt=True).browse(record.id).message_post(body=gpt, message_type='comment', subtype_xmlid='mail.mt_comment', author_id=odoobot_id)
        return mod_response