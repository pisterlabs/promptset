# Copyright 2023 Akretion France (http://www.akretion.com/)
# @author: Alexis de Lattre <alexis.delattre@akretion.com>
# License AGPL-3.0 or later (https://www.gnu.org/licenses/agpl).

from odoo import api, fields, models, tools, _
from odoo.exceptions import UserError
from collections import defaultdict
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    logger.debug('Cannot import openai')

class ImportShowLogs(models.TransientModel):
    _name = "import.show.logs"
    _description = "Pop-up to show warnings after import"

    logs = fields.Html(readonly=True)

    @api.model
    def _import_speedy(self, chatgpt=False):
        logger.debug('Start to prepare import speedy')
        speedy = {
            'chatgpt': chatgpt,
            'field2label': {},
            'logs': [],
        # 'logs' should contain a list of dict :
        # {'msg': 'Checksum IBAN wrong',
        #  'value': 'FR9879834739',
        #  'vals': vals,  # used to get the line
        #                   (and display_name if partner has been created)
        #  'field': 'res.partner,email',
        #  'reset': True,  # True if the data is NOT imported in Odoo
        # }
        }
        if chatgpt:
            openai_api_key = tools.config.get('openai_api_key', False)
            if not openai_api_key:
                raise UserError(_(
                    "Missing entry openai_api_key in the Odoo server configuration file."))
            openai.api_key = openai_api_key
            speedy['openai_tokens'] = 0
        return speedy

    def _field_label(self, field, speedy):
        if field not in speedy['field2label']:
            field_split = field.split(',')
            ofield = self.env['ir.model.fields'].search([
                ('model', '=', field_split[0]),
                ('name', '=', field_split[1]),
                ], limit=1)
            if ofield:
                speedy['field2label'][field] = ofield.field_description
            else:
                speedy['field2label'][field] = '%s (%s)' % (
                    field_split[1], field_split[0])
        return speedy['field2label'][field]

    def _import_logs2html(self, speedy):
        line2logs = defaultdict(list)
        field2logs = defaultdict(list)
        for log in speedy['logs']:
            if log['vals'].get('line'):
                line2logs[log['vals']['line']].append(log)
            if log.get('field'):
                field2logs[log['field']].append(log)
        html = '<p><small>For the logs in <span style="color: red">red</span>, the data was <b>not imported</b> in Odoo</small><br/>'
        if speedy.get('chatgpt'):
            html += '<small><b>%d</b> OpenAI tokens where used</small></p>' % speedy['openai_tokens']
        html += '<h1>Logs per line</h1>'
        for line, logs in line2logs.items():
            log_labels = []
            for log in logs:
                log_labels.append(
                    '<li style="color: %s"><b>%s</b>: <b>%s</b> - %s</li>' % (
                        log.get('reset') and 'red' or 'black',
                        self._field_label(log['field'], speedy),
                        log['value'],
                        log['msg'],
                        ))
            h3 = 'Line %s' % line
            if log['vals'].get('id'):
                h3 += ': %s (ID %d)' % (log['vals']['display_name'], log['vals']['id'])
            html += '<h3>%s</h3>\n<p><ul>%s</ul></p>' % (h3, '\n'.join(log_labels))
        html += '<h1>Logs per field</h1>'
        for field, logs in field2logs.items():
            log_labels = []
            for log in logs:
                line_label = 'Line %s' % log['vals'].get('line', 'unknown')
                if log['vals'].get('id'):
                    line_label += ' (%s ID %d)' % (log['vals']['display_name'], log['vals']['id'])
                log_labels.append(
                    '<li style="color: %s"><b>%s</b>: <b>%s</b> - %s</li>' % (
                        log.get('reset') and 'red' or 'black',
                        line_label,
                        log['value'],
                        log['msg'],
                        ))
            html += '<h3>%s</h3>\n<p><ul>%s</ul></p>' % (
                self._field_label(field, speedy), '\n'.join(log_labels))
        return html

    def _import_result_action(self, speedy):
        action = {
            'name': 'Result',
            'type': 'ir.actions.act_window',
            'res_model': 'import.show.logs',
            'view_mode': 'form',
            'target': 'new',
            'context': dict(self._context, default_logs=self._import_logs2html(speedy)),
            }
        return action

    def _prepare_create_date(self, vals, speedy):
        create_date = vals.get('create_date')
        create_date_dt = False
        if isinstance(create_date, str) and len(create_date) == 10:
            try:
                create_date_dt = datetime.strptime(create_date, '%Y-%m-%d')
            except Exception as e:
                speedy['logs'].append({
                    'msg': "Failed to convert '%s' to datetime: %s" % (create_date, e),
                    'value': vals['create_date'],
                    'vals': vals,
                    'field': 'product.product,create_date',
                    'reset': True,
                    })
        elif isinstance(create_date, datetime):
            create_date_dt = create_date
        if create_date_dt and create_date_dt.date() > fields.Date.context_today(self):
            speedy['logs'].append({
                'msg': 'create_date %s cannot be in the future' % create_date_dt,
                'value': create_date,
                'vals': vals,
                'field': 'product.product,create_date',
                'reset': True,
                })
        return create_date_dt
