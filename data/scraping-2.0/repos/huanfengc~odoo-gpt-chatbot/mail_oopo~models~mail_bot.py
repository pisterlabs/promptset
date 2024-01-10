import openai
import tiktoken
import json
import re

from odoo import models, fields, api

# Function definitions
functions = [
    {
        "name": "read_record",
        "description": "Read a record or records based on the given fields and search domains",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "The name of the model to be read"
                },
                "field": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Array of field names to be read from the model, "
                },
                "search_domains": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": """Each item is a  must be formatted as a tuple, e.g. ("name", "=", "Mark Cheng")"""
                        }
                    },
                    "description": """Odoo search domains to filter the data. Each domain should be an tuple of strings. \
                        e.g. [("name", "=", "Mark Cheng"), ("phone", "=", "123")] denotes a domain containing two conditions.""",
                    "default": []
                },
                "limit": {
                    "type": "integer",
                    "description": "Limit the number of records to be read",
                },
                "order": {
                    "type": "string",
                    "description": "Order the records by the given field - example name asc",
                }
            },
            "required": ["model", "field"]
        }
    },
    {
        "name": "create_record",
        "description": "Create a new record in a model based on the given fields.",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "The name of the model to be read"
                },
                "values": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": """A key-value mapping, both key and value must be strings, \
                            e.g. {"name": "Hello"}, {"partner_ids": "1", "name": "Hello"}"""
                    },
                    "description": """Array of field names to be created as a new record. \
                        Each field must be a mapping, e.g. [{"name": "Mark Cheng", "phone": 9993336666}]"""
                },
            },
            "required": ["model", "values"]
        }
    },
    {
        "name": "update_record",
        "description": """Update an existing record in a model. \
            The record to update is identified based on the fields and search domain. \
            The field to update is extracted from the user message.""",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "The name of the model to be read"
                },
                "field": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Array of field names in order for searching for the record to update."
                },
                "field_to_update": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": """Array of field names to be updated. \
                        Each field must be a key-value pair, e.g. [{"phone": "9993336666"}]"""
                },
                "search_domains": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "description": "Odoo search domains to filter the data. Each domain should be an array of strings",
                    "default": []
                },
                "limit": {
                    "type": "integer",
                    "description": "Limit the number of records to be read",
                }
            },
            "required": ["model", "field", "field_to_update"]
        }
    },
]

# For the summary command in the chatter this array serves to provide 
# the 'important' information about a related field so we aren't prompting every field of every relational field
# TODO make .name ^ .display_name default for all relational fields then add addons here
# TODO add support for depth2 relational fields when specified e.g res.partner.user_id.name
relational_bindings = {
    "res.partner": ["name"],
    "res.users": ["name"],
    "sale.order.line": ["name", "qty_to_deliver", "price_unit"],
    "account.move.line": ["name", "quantity", "price_unit", "price_subtotal"],
    "stock.move": ["display_name","product_id","product_uom_qty", "forecast_availability", "quantity_done"],
    "product.product": ["name", "lst_price", "standard_price", "detailed_type"],
    "mrp.bom.line": ["display_name", "product_qty"],
    "account.payment.term": ["name", "note"]
}

summary_prompt = """
        You are a friendly AI Odoo Assistant.

        Use the provided record information below to process the user query. 
        Your response must be professional and concise.

        The record information starts with a record description and model_name(record_id),
        and is followed by the fields information, i.e. field description [field_name] = field_value.

        <Record information starts>
        {prompt}
        <Record information ends>

        Selection fields return tuples with (technical_name, display_name)

        Please keep the summaries concise and professional. Only include the most relevant information. Summarize it in a business context - not a technical one.

        Don't just list the fields, rather summarize the record in a business context as a whole
            """

class MailBot(models.AbstractModel):
    _inherit = "mail.bot"

    first_msg = "Hi, I'm Oopo, an AI assistant. Feel free to ask my any questions."

    def _apply_logic(self, record, values, command=None):
        odoobot_id = self.env["ir.model.data"]._xmlid_to_res_id("base.partner_root")

        if len(record) != 1 or values.get("author_id") == odoobot_id or values.get("message_type") != "comment" and not command:
            return
        
        if self._is_bot_pinged(values) or self._is_bot_in_private_channel(record):
            body = values.get("body", "").replace(u"\xa0", u" ").strip().lower().strip(".!")
            answer, message_type = self._get_answer(record, body, values, command)
            if answer:
                subtype_id = self.env["ir.model.data"]._xmlid_to_res_id("mail.mt_comment")
                record.with_context(mail_create_nosubscribe=True).sudo().message_post(body=answer, author_id=odoobot_id, message_type=message_type, subtype_id=subtype_id)

    def _get_answer(self, channel, body, values, command):
        odoobot_id = self.env["ir.model.data"]._xmlid_to_res_id("base.partner_root")
        api_key = self.env["ir.config_parameter"].sudo().get_param("mail_oopo.openapi_api_key")

        if not api_key:
            return "Please set the OpenAI API key in the settings under integrations", "comment"
        openai.api_key = api_key

        fail_moderation_check = self._get_chat_completion(messages=body)
        if isinstance(fail_moderation_check, str):
            return fail_moderation_check, "comment"
        if fail_moderation_check:
            return "[Request Decline] The request violates OpenAI usage policy, please try another request.", "comment"
        
        if not isinstance(channel, type(self.env["mail.channel"])):
            response = self._process_query_in_chatter(channel, body)
            return response

        msgs = self._get_relevant_chat_history(channel)
        gpt_arr = self._build_chatgpt_request(msgs)

        response = self._pre_prompt(gpt_arr)

        if isinstance(response, str):
            return response, "comment"
        
        is_function_call, function_call_fail, response = True, False, None
        loop_count, timeout = 0, 20
        functional_msg_saved = []

        while is_function_call:
            if loop_count >= timeout:
                break

            response = self._get_chat_completion(messages=gpt_arr, callable_functions=functions, temperature=0.5)
            if isinstance(response, str):
                return response, "comment"
            gpt_arr.append(dict(response["choices"][0]["message"]))
    
            is_function_call = response["choices"][0]["message"].get("function_call") is not None
            if not is_function_call:
                break

            print("\033[96m" + "CALL #: " + str(loop_count+1) + "\033[0m")

            function_response, function_call_fail, error_message, model_name = self._execute_function_call(response["choices"][0]["message"])
            gpt_arr.append(function_response)
            if function_call_fail:
                user_prompt = self._tailor_user_prompt(body, error_message, model_name)
                gpt_arr.append({"role": "user", "content": user_prompt})
            if not function_call_fail:
                functional_msg_saved.append((dict(response["choices"][0]["message"]), "bot_function_request"))
                functional_msg_saved.append((function_response, "bot_function"))
            loop_count +=1

        final_response = response["choices"][0]["message"]["content"]
        if not function_call_fail and final_response:
            for message, message_type in functional_msg_saved:
                self._create_functional_message(channel, message, message_type)
        
        if not final_response:
            final_response = "I am sorry that I failed to process your query, please provide more details/instructions and retry!"
        final_response = self._def_transform_links(final_response)

        return final_response, "comment"
    
    def _get_field_info(self, field, target_field, field_val, relational_bindings):
        field_string = target_field['string']

        field_info = None

        if target_field["type"] == "boolean":
            field_info = f"'{field_string}' [{field}] = {field_val}\n"
        elif field_val:
            field_info = None

            if target_field["type"] in ("text", "integer", "float", "char", "datetime"):
                field_info = f"'{field_string}' [{field}] = {field_val}\n"
            elif target_field["type"] == "selection":
                selected_item = next(item for item in target_field['selection'] if item[0] == field_val)
                field_info = f"'{field_string}' [{field}] = {selected_item}\n"
            elif target_field["type"] == "many2one" and target_field["relation"] in relational_bindings:
                deep_field_metadata = field_val.fields_get()
                relevant_field_names = relational_bindings[target_field["relation"]]
                related_values = [f"{deep_field_metadata[rel_field]['string']} = {field_val[rel_field]}" for rel_field in relevant_field_names]
                field_info = f"'{field_string}' [{field}] = {field_string} with the following values: {', '.join(related_values)}\n"
            elif target_field["type"] == "one2many" and target_field["relation"] in relational_bindings:
                field_info = f"'{field_string}' [{field}] = {field_string} with the following values:\n"
                for record in field_val:
                    deep_field_metadata = record.fields_get()
                    relevant_field_names = relational_bindings[target_field["relation"]]
                    related_values = [f"{deep_field_metadata[rel_field]['string']} = {record[rel_field]}" for rel_field in relevant_field_names]
                    field_info += ", ".join(related_values) + "\n"
            else:
                field_info = f"'{field_string}' [{field}] = {field_val}\n"
        
        return field_info
    
    def _construct_summary_prompt(self, channel, fields_metadata, relational_bindings):
        prompt = f"Record Information: {channel._description} {getattr(channel, 'name', channel.display_name)} [{str(channel)}]\n"
        prompt_list = [prompt]

        ignored_fields = ("Followers", "Followers (Partners)", "Messages", "Website Messages")

        for field, target_field in fields_metadata.items():
            if target_field["string"] in ignored_fields:
                continue
            
            field_val = channel[field]
            field_info = self._get_field_info(field, target_field, field_val, relational_bindings)

            if field_info:
                prompt_list.append(field_info)
        
        prompt = "".join(prompt_list)
        constructed_prompt = summary_prompt.format(prompt=prompt)
        return constructed_prompt
        
    def _process_query_in_chatter(self, channel, body):
        fields_metadata = channel.fields_get()
        constructed_prompt = self._construct_summary_prompt(channel, fields_metadata, relational_bindings)

        msgs = [{'role': 'system', 'content': constructed_prompt}]
        response = self._get_chat_completion(messages=msgs, model="gpt-3.5-turbo-0613")
        return response["choices"][0]["message"]["content"], "notification"

    
    def _build_chatgpt_request(self, msgs):
        odoobot_id = self.env["ir.model.data"]._xmlid_to_res_id("base.partner_root")
        chatgpt_msgs_arr = [{"role":"system", "content": self._select_system_message()}, {"role": "assistant", "content": self.first_msg}]
        
        for message in msgs[1:]:
            body = str(message["body"]).replace("<p>", "").replace("</p>", "")
            if message["author"]["id"] == odoobot_id:
                if body != self.first_msg and message["message_type"] == "comment":
                    chatgpt_msgs_arr.append({"role": "assistant", "content": body})
                elif message["message_type"] == "bot_function":
                    chatgpt_msgs_arr.append(message["function_content"])
                elif message["message_type"] == "bot_function_request":
                    chatgpt_msgs_arr.append(message["function_content"])
            else:
                chatgpt_msgs_arr.append({"role": "user", "content": body})
        return chatgpt_msgs_arr

    def _get_relevant_chat_history(self, channel):
        msgs = channel._channel_fetch_message(limit=None)
        msgs = [msg for msg in msgs if msg["body"] != "" or msg.get("message_type") == "bot_function" or msg.get("message_type") == "bot_function_request"]
        msgs.reverse()
        return msgs
    
    def _create_functional_message(self, channel, content, message_type):
        odoobot_id = self.env["ir.model.data"]._xmlid_to_res_id("base.partner_root")
        subtype_id = self.env["ir.model.data"]._xmlid_to_res_id("mail.mt_comment")
        vals = {
            "body": "",
            "author_id": odoobot_id,
            "message_type": message_type,
            "subtype_id": subtype_id,
            "model": channel._name,
            "res_id": channel.id,
            "function_content": content
        }
        return channel._message_create(vals)

    def _is_bot_pinged(self, values):
        odoobot_id = self.env["ir.model.data"]._xmlid_to_res_id("base.partner_root")
        return odoobot_id in values.get("partner_ids", [])

    def _is_bot_in_private_channel(self, record):
        odoobot_id = self.env["ir.model.data"]._xmlid_to_res_id("base.partner_root")
        if record._name == "mail.channel" and record.channel_type == "chat":
            return odoobot_id in record.with_context(active_test=False).channel_partner_ids.ids
        return False
    
    def change_model(self, gptmodel):
        self.env.user.openai_model = gptmodel
        return True
    
    def get_model(self):
        return self.env.user.openai_model
    
    def _execute_function_call(self, message):
        function_name = message["function_call"]["name"]
        print("\033[95m" + "ODOOGPT FUNCTION CALL: " + function_name + "\033[0m")

        kwargs = None
        function_call_fail, model_name, error_message = False, None, None
        try:
            kwargs = json.loads(message["function_call"]["arguments"])
            # kwargs = ast.literal_eval(message["function_call"]["arguments"])
        except Exception as e:
            function_call_fail, error_message = True, e
            chat_result = self._construct_function_response(function_name, "JSON Error:" + str(e))
            # print as red
            print("\033[91m" + "ODOOGPT FUNCTION ERROR: " + str(e) + "\033[0m")
            print("\033[91m" + "ODOOGPT FUNCTION ERROR: " + message["function_call"]["arguments"] + "\033[0m")
            
            return chat_result, function_call_fail, error_message, model_name

        print("\033[95m" + "ODOOGPT FUNCTION ARGUMENTS: " + str(kwargs) + "\033[0m")
        
        chat_result = None
        model_name = kwargs["model"]
        savepoint = self.env.cr.savepoint(flush=True) 
        try:
            function_to_call = self._get_avalaible_function_dict()[function_name]
            result = function_to_call(**kwargs)
            print("\033[95m" + "ODOOGPT FUNCTION RESULT: " + str(result) + "\033[0m")
            chat_result = self._construct_function_response(function_name, str(result))
        except Exception as e:
            savepoint.rollback()
            function_call_fail, error_message = True, e
            print("\033[91m" + "ODOOGPT FUNCTION ERROR: " + str(e) + "\033[0m")
            chat_result = self._construct_function_response(function_name, str(e))
        return chat_result, function_call_fail, error_message, model_name
    
    def _construct_function_response(self, function_name, result):
        return {"role": "function", "name": function_name, "content": result}
    
    def _fix_errorneous_domain(self, search_domains):
        if len(search_domains) == 1:
            if len(search_domains[0]) == 3:
                if not all(isinstance(domain, (str, int)) for domain in search_domains[0]):
                    search_domains = search_domains[0]
            else:
                if any(isinstance(operator, str) for operator in search_domains[0]):
                    search_domains = search_domains[0]
        return search_domains

    def _def_transform_links(self, input_string):
        # Define the pattern to match the bracket-parenthesis links
        pattern = r'\[(.*?)\]\(#&data-oe-model=(.*?)&data-oe-id=(.*?)\)'

        # Function to replace the matches with the desired format
        def replace_link(match):
            link_text, model_name, id_number = match.groups()
            return f"<a href='#' data-oe-model='{model_name}' data-oe-id='{id_number}'>{link_text}</a>"

        # Use re.sub to replace the matches with the function result
        transformed_string = re.sub(pattern, replace_link, input_string)

        return transformed_string

    ### START ORM METHODS ###

    def _read_record(self, model, field, search_domains=None, limit=None, order=None):
        """Search and read records in the model based on search domains."""

        available_fields = self._get_fields_for_model(model)

        # Try to always include 'name' in the fields to be read, if the field 'name' exists in the model
        if 'name' not in field: 
                field.append('name') if 'name' in available_fields else field.append('display_name')

        search_domains = self._fix_errorneous_domain(search_domains) if search_domains else []
        search_domains = [tuple(domain) if isinstance(domain, list) else domain for domain in search_domains] if search_domains else []
        return self.env[model].search_read(domain=search_domains, fields=field, limit=limit, order=order)

    def _create_record(self, model, values):
        """Create new records in the model with fields filled by given values."""
        return self.env[model].create(values)

    def _update_record(self, model, field, field_to_update, search_domains=None, limit=None):
        """Update existing records in the model with new values for the fields."""
        record_id = self._read_record(model, field, search_domains, limit)[0]["id"]
        record_to_update = self.env[model].browse([record_id])
        return record_to_update.write(vals=field_to_update[0])   

    def _get_fields_for_model(self, model_name):
        try:
            available_fields = set(self.env[model_name].fields_get().keys())
            return available_fields
        except:
            return set()

    def _get_avalaible_function_dict(self):
        """Available funcitons that OpenAI API funciton call has access to."""
        avalaible_function_dict = {
            "read_record": self._read_record,
            "create_record": self._create_record,
            "update_record": self._update_record,
        }
        return avalaible_function_dict

    def _get_chat_completion(self, messages, callable_functions=None, temperature=0.1, model=None):
        """Get completion for prompt via ChatCompletion model of OpenAI API.``messages`` should be a list of message.
        If ``messages`` is a string, i.e. single user prompt, perform moderation check."""
        if model is None:
            model = self.get_model()

        try:
            if isinstance(messages, str):
                response = openai.Moderation.create(input=messages)
                response = response["results"][0]["flagged"]
            else:
                params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "request_timeout": 60, # This parameter helps raise Timeout error, but is not officially documented.
                }

                if callable_functions is not None:
                    params["functions"] = callable_functions
                    params["function_call"] = "auto"

                response = openai.ChatCompletion.create(**params)
        except openai.error.AuthenticationError as e:
            return f"""[OpenAI API Key Error] Your OpenAI API key is invalid, expired or revoked. \
                Please provide a valid API key in Settings/General Settings/Integrations. See details: {e}"""
        except openai.error.ServiceUnavailableError as e:
            return f"""[OpenAI Server Error] There is an ongoing outage of OpenAI server. \
                Please retry your query after a brief wait. See details: {e}"""
        except openai.error.APIError as e:
            return f"""[OpenAI Server Error] There is an ongoing outage or temporary error of OpenAI server. \
                Please retry your query after a brief wait. See details: {e}"""
        except openai.error.APIConnectionError as e:
            return f"""[OpenAI Connection Error] Connection fails, please check your network settings, proxy configuration, \
                SSL certificates or firewall rules. See details: {e}"""
        except openai.error.RateLimitError as e:
            return f"""[OpenAI Rate Limit Error] The rate limit of OpenAI API request has been reached. \
                Please reduce the frequency of sending queries. See details: {e}"""
        except openai.error.Timeout as e:
            return f"""[OpenAI Request Timeout] Query timed out, please retry your query after a brief wait. \
                See details: {e}"""
        except openai.error.InvalidRequestError as e:
            # ``InvalidRequestError`` should not be displayed to the user and has to be handled only on developer side.
            error_message = str(e)
            if "reduce" in error_message:
                return f"""[Token Limit Warning] {error_message} Or choose a GPT model with higher token limit to afford longer context. \
                    Please contact Odoo Inc for suggestions. (To release token usage, please enter "clear" to clear current message history.)"""
            return f"""[Query Fails] Your query can not be processed at current version of OdooBot, \
                please contact Odoo Inc to upgrade OdooBot."""
        
        if not isinstance(messages, str):
            print(f"\033[92m Total Conversation Tokens: {str(response['usage']['total_tokens'])} \033[0m")

        return response
    
    def _select_system_message(self):
        """System message sets up the tone of GPT, basic context of chat and requirements that GPT has to follow.
        Note: It is not guaranteed that GPT would strictly follow the requirements."""
        
        prompts = {
            "base_system_message": """You are Oopo a friendly AI Assistant, users might ask questions, or ask to perform any actions. \ You have full access to the current Odoo environment""",
            "inline_link_instruction": """You can add links into the text too by adding an <a> tag in this format:
                <a href='#' data-oe-model='model name' data-oe-id='id number'>test</a>

                e.g., <a href='#' data-oe-model='sale.order' data-oe-id='7'>My sale order</a>

                You should always use links to reference records in the system, as it will make it easier for me to understand what you are referring to.

                Anything that is returned from the `read_record` function should be linked to:
                <a href='#' data-oe-model='model name' data-oe-id='id number'>Link text</a>

                For instance, if the `read_record` function returns a sale order with ID 7, create the link like this:
                <a href='#' data-oe-model='sale.order' data-oe-id='7'>Sale Order 7</a>

                By consistently including links in the responses, we can maintain a more structured and interactive conversation.
                
                Please avoid using the square bracket format like this [Product 45](#&data-oe-model=product.product&data-oe-id=45) for links, as it is not the correct format. Always use the "<a>" tag as shown in the examples above to create links.""",
            "relation_fields_prompt": """provides several functions to interact with the database, including querying records, creating new records, and updating existing records.

                To query records, you can use the `read_record` function, which retrieves specific fields from the given model. For example, to get the most recent three sale orders, you can use the `read_record` function with the appropriate arguments (`model`, `field`, `order`, and `limit`) as shown below:

                ```
                ODOOGPT FUNCTION CALL: read_record
                ODOOGPT FUNCTION ARGUMENTS: {'model': 'sale.order', 'field': ['name', 'date_order'], 'order': 'date_order desc', 'limit': 3}
                ```

                To create a new record, you can use the `create_record` function. For instance, to create a new customer named "Diego," you can use the `create_record` function with the desired `model` and `values` as shown below:

                ```
                ODOOGPT FUNCTION CALL: create_record
                ODOOGPT FUNCTION ARGUMENTS: {'model': 'res.partner', 'values': [{'name': 'Diego'}]}
                ```

                To update an existing record, you can use the `update_record` function. For example, if you want to update the phone number and email of the customer named "Diego" to "99999999" and "jot@odooooo.com" respectively, you can use the `update_record` function with the appropriate arguments (`model`, `field`, `field_to_update`, `search_domains`, and `limit`) as shown below:

                ```
                ODOOGPT FUNCTION CALL: update_record
                ODOOGPT FUNCTION ARGUMENTS: {'model': 'res.partner', 'field': ['name'], 'field_to_update': [{'phone': '99999999', 'email': 'jot@odooooo.com'}], 'search_domains': [['name', '=', 'Diego']], 'limit': 1}
                ```

                One important concept to understand is the usage of relational fields. In some cases, you might need to reference the ID of a record when creating or updating another record with a relationship. For example, to create a sale order for a customer, you need to pass the customer's ID as the value for the `partner_id` field in the `sale.order` model.

                When you are unsure about the ID of a record, you can perform a search using the `read_record` function with appropriate search filters. For instance, if you want to find the ID of a product with a name containing "cabinet," you can use the `read_record` function with the search domain `[['name', '=ilike', '%cabinet%']]` as shown below:

                ```
                ODOOGPT FUNCTION CALL: read_record
                ODOOGPT FUNCTION ARGUMENTS: {'model': 'product.product', 'field': ['id'], 'search_domains': [['name', '=ilike', '%cabinet%']], 'limit': 1}
                ```

                Remember, the IDs returned from previous function calls can be used as arguments in subsequent function calls to establish relationships between records.""",
            "search_domains": """Domain criteria can be combined using 3 logical operators than can be added between tuples:

                '&' (logical AND, default)
                '|' (logical OR)
                '!' (logical NOT)
                These are prefix operators and the arity of the '&' and '|' operator is 2, while the arity of the '!' is just 1. Be very careful about this when you combine them the first time.

                Here is an example of searching for Partners named ABC from Belgium and Germany whose language is not english ::

                [('name','=','ABC'),'!',('language.code','=','en_US'),'|',
                ('country_id.code','=','be'),('country_id.code','=','de')]
                The '&' is omitted as it is the default, and of course we could have used '!=' for the language, but what this domain really represents is::

                [(name is 'ABC' AND (language is NOT english) AND (country is Belgium OR Germany))]

                For example if I ask, are you familiar with product x,y,z 

                And you need to get the product ids of x,y,z

                You can use the search domain like this:

                search_domains: ["|", "|", ['name', 'ilike', '%x%'], ['name', 'ilike', '%y%'], ['name', 'ilike', '%z%']]

                or if you just wanted X or Y, you could do:

                search_domains: ["|", ['name', 'ilike', '%x%'], ['name', 'ilike', '%y%']]
                
                """
        }
        
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        master_prompt = ""
        for prompt_module,prompt in prompts.items():
            print(f"\033[92m Loaded Prompt Module: {prompt_module} - Tokens: {len(encoding.encode(prompt))}\033[0m")
            master_prompt += prompt
        
        # print tokens of master
        print(f"\033[92m Loaded Master Prompt - Tokens: {len(encoding.encode(master_prompt))}\033[0m")

        return master_prompt
    
    def _get_delimiter(self):
        """Delimiter helps to prevent prompt injections from users, and tailor a use prompt with internal helper prompts."""
        return "####"
    
    def _tailor_user_prompt(self, user_prompt, error_message=None, model_name=None):
        """Encapsulate a user prompt with internal helper prompts, for error handling or requirement enforcement."""
        if error_message:
            if isinstance(error_message, KeyError):
                error_type = "model" 
            elif isinstance(error_message, ValueError) and model_name:
                error_type = "field"
            elif isinstance(error_message, TypeError):
                error_type = "type"
            else:
                error_type = "else"
            helper_prompt = self._get_user_prompt_helper((error_type, model_name))
            user_prompt = " ".join([helper_prompt, user_prompt])
        return user_prompt
    
    def _get_user_prompt_helper(self, helper_type):
        """Inject error message into the user prompt to help GPT self-correct and retry the failed user prompt.
        Note: the data type of helper_type is discussable, e.g. a tuple (key_word, model_name)"""
        error_type, model_name = helper_type
        if error_type == "model":
            model_prefix = model_name.split(".")[0]
            model_list = str([m["model"] for m in self.env["ir.model"].search_read([]) if model_prefix == m["model"].split(".")[0]])
            helper_prompt = f"""The model {model_name} is invalid, you are required to only use the model defined in Odoo, \
                the valid model names are listed as follows: {model_list}."""
        elif error_type == "field":
            available_fields = str(list(self.env[model_name].fields_get().keys()))
            helper_prompt = f"""In {model_name} model of Odoo, the defined field names are listed as follows: {available_fields}. \
                You are mandatory to use defined field names only."""
        elif error_type == "type":
            helper_prompt = """You are mandatory to use the correct value type of the field. If it is a relational field, \
                e.g. partner_id, res_model_id, you must perform a read operation first to find the corrrect id."""
        else:
            helper_prompt = """
                            Please correct the error you made based on the error message shown in the last function response.

                            If it was a JSON error, you must generate in correct JSON schema next time. 
                            If you used "(" and ")", change it to "[" and "]" to avoid error.

                            If it violate not-null constraint, it means that field is required, and you must use the required field.
                            If the required field is some id, you must perform a read operation to find the correct id.  
                            """
        required_instruction = "Based on aforementioned information, please response the following user query again:"
        return " ".join([helper_prompt, required_instruction])
    
    def _pre_prompt(self, gpt_arr):
        """Apply Chain of Thought (CoT) to help GPT decompose a user query into basic CRUD operations."""
        user_prompt = gpt_arr[-1:]
        gpt_arr.append({"role":"user",
                        "content":
                        """
                        Instructions while running a query: 
                        search domain in read_record() must not have duplicate fields.
                        While trying to get the id of a single record, every read_record() call must correspond to a single record
                         eg: {'model': 'res.partner', 'field': ['id'], 'search_domains': [['name', '=', 'Odoo Wheel'],['name', '=', 'Odoo Frame']], 'limit': 1} is wrong, name is duplicated. 
                        Instead it should be
                        {'model': 'res.partner', 'field': ['id'], 'search_domains': [['name', '=', 'Odoo Wheel']], 'limit': 1}
                        followed by
                        {'model': 'res.partner', 'field': ['id'], 'search_domains': [['name', '=', 'Odoo Frame']], 'limit': 1}
                        Always try to read only a single record at once.

                        If the user request doesn't require data operations - do not return anything - otherwise state what CRUD operations are required for the above(only give in read,create, update)? 
                        Which models are required(only give technical odoo model names)? Summarize within 100 words. Perform these CRUD operations"""})
        response = self._get_chat_completion(messages=gpt_arr, callable_functions=functions, temperature=0.5)

        if not isinstance(response, str):
            gpt_arr.append(dict(response["choices"][0]["message"]))
        gpt_arr += user_prompt
        
        return response