# -*- coding: utf-8 -*-

from odoo import models, fields, api, _


class ResUsers(models.Model):
    _inherit = ['res.users', 'odoogpt.settings.mixin']
    _name = 'res.users'


    @property
    def SELF_READABLE_FIELDS(self):
        return super().SELF_READABLE_FIELDS + [
            'odoogpt_openai_api_key_customize',
            'odoogpt_openai_api_key',
            'odoogpt_openai_model_customize',
            'odoogpt_openai_model',
            'odoogpt_openai_max_tokens_customize',
            'odoogpt_openai_max_tokens',
            'odoogpt_openai_temperature_customize',
            'odoogpt_openai_temperature',
            'odoogpt_chat_method_customize',
            'odoogpt_chat_method',
            'odoogpt_chat_system_message_customize',
            'odoogpt_chat_system_message',
            'odoogpt_openai_prompt_prefix_customize',
            'odoogpt_openai_prompt_prefix',
            'odoogpt_openai_prompt_suffix_customize',
            'odoogpt_openai_prompt_suffix',
        ]

    @property
    def SELF_WRITEABLE_FIELDS(self):
        return super().SELF_WRITEABLE_FIELDS + [
            'odoogpt_openai_api_key_customize',
            'odoogpt_openai_api_key',
            'odoogpt_openai_model_customize',
            'odoogpt_openai_model',
            'odoogpt_openai_max_tokens_customize',
            'odoogpt_openai_max_tokens',
            'odoogpt_openai_temperature_customize',
            'odoogpt_openai_temperature',
            'odoogpt_chat_method_customize',
            'odoogpt_chat_method',
            'odoogpt_chat_system_message_customize',
            'odoogpt_chat_system_message',
            'odoogpt_openai_prompt_prefix_customize',
            'odoogpt_openai_prompt_prefix',
            'odoogpt_openai_prompt_suffix_customize',
            'odoogpt_openai_prompt_suffix',
        ]


    odoogpt_openai_api_key_customize = fields.Boolean(string='Customize OpenAI Api token', default=False)
    odoogpt_openai_model_customize = fields.Boolean(string='Customize OpenAI Model', default=False)
    odoogpt_openai_max_tokens_customize = fields.Boolean(string='Customize OpenAI Max tokens', default=False)
    odoogpt_openai_temperature_customize = fields.Boolean(string='Customize OpenAI Temperature', default=False)
    odoogpt_chat_method_customize = fields.Boolean(string='Customize Chat method', default=False)
    odoogpt_chat_system_message_customize = fields.Boolean(string='Customize System message', default=False)
    odoogpt_openai_prompt_prefix_customize = fields.Boolean(string='Customize OpenAI Prompt prefix', default=False)
    odoogpt_openai_prompt_suffix_customize = fields.Boolean(string='Customize OpenAI Prompt suffix', default=False)


    # Compute OdooGPT settings to be able to always read the user value and so get the real value. 
    # If setting is customized on user, get the user setting, otherwise get the company setting.
    # TODO: In future, generalize this, will be hard to maintain

    odoogpt_openai_api_key = fields.Char(compute='_compute_odoogpt_openai_api_key', store=True, required=False)
    @api.depends('company_id.odoogpt_openai_api_key', 'odoogpt_openai_api_key_customize')
    def _compute_odoogpt_openai_api_key(self):
        for rec in self:
            if rec.odoogpt_openai_api_key_customize:
                try:
                    rec.odoogpt_openai_api_key = rec.odoogpt_openai_api_key
                except:
                    rec.odoogpt_openai_api_key = False
            else:
                rec.odoogpt_openai_api_key = rec.company_id.odoogpt_openai_api_key

    odoogpt_openai_model = fields.Char(compute='_compute_odoogpt_openai_model', store=True, required=False)
    @api.depends('company_id.odoogpt_openai_model', 'odoogpt_openai_model_customize')
    def _compute_odoogpt_openai_model(self):
        for rec in self:
            if rec.odoogpt_openai_model_customize:
                try:
                    rec.odoogpt_openai_model = rec.odoogpt_openai_model
                except:
                    rec.odoogpt_openai_model = False
            else:
                rec.odoogpt_openai_model = rec.company_id.odoogpt_openai_model

    odoogpt_openai_max_tokens = fields.Integer(compute='_compute_odoogpt_openai_max_tokens', store=True, required=False)
    @api.depends('company_id.odoogpt_openai_max_tokens', 'odoogpt_openai_max_tokens_customize')
    def _compute_odoogpt_openai_max_tokens(self):
        for rec in self:
            if rec.odoogpt_openai_max_tokens_customize:
                try:
                    rec.odoogpt_openai_max_tokens = rec.odoogpt_openai_max_tokens
                except:
                    rec.odoogpt_openai_max_tokens = False
            else:
                rec.odoogpt_openai_max_tokens = rec.company_id.odoogpt_openai_max_tokens

    odoogpt_openai_temperature = fields.Float(compute='_compute_odoogpt_openai_temperature', store=True, required=False)
    @api.depends('company_id.odoogpt_openai_temperature', 'odoogpt_openai_temperature_customize')
    def _compute_odoogpt_openai_temperature(self):
        for rec in self:
            if rec.odoogpt_openai_temperature_customize:
                try:
                    rec.odoogpt_openai_temperature = rec.odoogpt_openai_temperature
                except:
                    rec.odoogpt_openai_temperature = False
            else:
                rec.odoogpt_openai_temperature = rec.company_id.odoogpt_openai_temperature

    odoogpt_chat_method = fields.Selection(compute='_compute_odoogpt_chat_method', store=True, required=False)
    @api.depends('company_id.odoogpt_chat_method', 'odoogpt_chat_method_customize')
    def _compute_odoogpt_chat_method(self):
        for rec in self:
            if rec.odoogpt_chat_method_customize:
                try:
                    rec.odoogpt_chat_method = rec.odoogpt_chat_method
                except:
                    rec.odoogpt_chat_method = False
            else:
                rec.odoogpt_chat_method = rec.company_id.odoogpt_chat_method

    odoogpt_chat_system_message = fields.Text(compute='_compute_odoogpt_chat_system_message', store=True, required=False)
    @api.depends('company_id.odoogpt_chat_system_message', 'odoogpt_chat_system_message_customize')
    def _compute_odoogpt_chat_system_message(self):
        for rec in self:
            if rec.odoogpt_chat_system_message_customize:
                try:
                    rec.odoogpt_chat_system_message = rec.odoogpt_chat_system_message
                except:
                    rec.odoogpt_chat_system_message = False
            else:
                rec.odoogpt_chat_system_message = rec.company_id.odoogpt_chat_system_message

    odoogpt_openai_prompt_prefix = fields.Char(compute='_compute_odoogpt_openai_prompt_prefix', store=True, required=False)
    @api.depends('company_id.odoogpt_openai_prompt_prefix', 'odoogpt_openai_prompt_prefix_customize')
    def _compute_odoogpt_openai_prompt_prefix(self):
        for rec in self:
            if rec.odoogpt_openai_prompt_prefix_customize:
                try:
                    rec.odoogpt_openai_prompt_prefix = rec.odoogpt_openai_prompt_prefix
                except:
                    rec.odoogpt_openai_prompt_prefix = False
            else:
                rec.odoogpt_openai_prompt_prefix = rec.company_id.odoogpt_openai_prompt_prefix

    odoogpt_openai_prompt_suffix = fields.Char(compute='_compute_odoogpt_openai_prompt_suffix', store=True, required=False)
    @api.depends('company_id.odoogpt_openai_prompt_suffix', 'odoogpt_openai_prompt_suffix_customize')
    def _compute_odoogpt_openai_prompt_suffix(self):
        for rec in self:
            if rec.odoogpt_openai_prompt_suffix_customize:
                try:
                    rec.odoogpt_openai_prompt_suffix = rec.odoogpt_openai_prompt_suffix
                except:
                    rec.odoogpt_openai_prompt_suffix = False
            else:
                rec.odoogpt_openai_prompt_suffix = rec.company_id.odoogpt_openai_prompt_suffix


    def odoogpt_openai_model_select_from_db(self):
        """Get Models from OpenAI api and show selector wizard"""
        self.env['odoogpt.openai.model'].sudo().refresh_from_api(format='model')
        action = self.env.ref('odoogpt.odoogpt_openai_model_select_wizard_act_window').sudo().read()[0]

        action['context'] = {
            'default_res_id': self.id,
            'default_res_model': self._name,
        }

        return action