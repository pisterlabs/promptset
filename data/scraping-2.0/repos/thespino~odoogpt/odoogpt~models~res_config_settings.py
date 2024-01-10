# -*- coding: utf-8 -*-

from odoo import models, fields, api, _


class ResConfigSettings(models.TransientModel):
    _inherit = ['res.config.settings', 'odoogpt.settings.mixin']
    _name = 'res.config.settings'


    odoogpt_openai_api_key = fields.Char(related='company_id.odoogpt_openai_api_key', readonly=False)
    odoogpt_openai_model = fields.Char(related='company_id.odoogpt_openai_model', readonly=False)
    odoogpt_openai_max_tokens = fields.Integer(related='company_id.odoogpt_openai_max_tokens', readonly=False)
    odoogpt_openai_temperature = fields.Float(related='company_id.odoogpt_openai_temperature', readonly=False)


    # CHAT CUSTOMIZATION ===========================================================================
    odoogpt_chat_method = fields.Selection(related='company_id.odoogpt_chat_method', readonly=False)
    odoogpt_chat_system_message = fields.Text(related='company_id.odoogpt_chat_system_message', readonly=False)
    odoogpt_openai_prompt_prefix = fields.Char(related='company_id.odoogpt_openai_prompt_prefix', readonly=False)
    odoogpt_openai_prompt_suffix = fields.Char(related='company_id.odoogpt_openai_prompt_suffix', readonly=False)


    def set_values(self):
        res = super().set_values()
        self.env.company.odoogpt_openai_api_key = self.odoogpt_openai_api_key
        self.env.company.odoogpt_openai_model = self.odoogpt_openai_model
        self.env.company.odoogpt_openai_max_tokens = self.odoogpt_openai_max_tokens
        self.env.company.odoogpt_openai_temperature = self.odoogpt_openai_temperature
        self.env.company.odoogpt_chat_method = self.odoogpt_chat_method
        self.env.company.odoogpt_chat_system_message = self.odoogpt_chat_system_message
        self.env.company.odoogpt_openai_prompt_prefix = self.odoogpt_openai_prompt_prefix
        self.env.company.odoogpt_openai_prompt_suffix = self.odoogpt_openai_prompt_suffix
        return res

    @api.model
    def get_values(self):
        res = super().get_values()
        res.update(
            odoogpt_openai_api_key = self.env.company.odoogpt_openai_api_key,
            odoogpt_openai_model = self.env.company.odoogpt_openai_model,
            odoogpt_openai_max_tokens = self.env.company.odoogpt_openai_max_tokens,
            odoogpt_openai_temperature = self.env.company.odoogpt_openai_temperature,
            odoogpt_chat_method = self.env.company.odoogpt_chat_method,
            odoogpt_chat_system_message = self.env.company.odoogpt_chat_system_message,
            odoogpt_openai_prompt_prefix = self.env.company.odoogpt_openai_prompt_prefix,
            odoogpt_openai_prompt_suffix = self.env.company.odoogpt_openai_prompt_suffix
        )
        return res

    def odoogpt_openai_model_select_from_db(self):
        """Get Models from OpenAI api and show selector wizard"""
        self.env['odoogpt.openai.model'].sudo().refresh_from_api(format='model')
        return self.env.ref('odoogpt.odoogpt_openai_model_select_wizard_act_window').read()[0]