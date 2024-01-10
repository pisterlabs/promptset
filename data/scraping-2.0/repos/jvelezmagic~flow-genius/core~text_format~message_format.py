from langchain.schema import messages_to_dict

from core.models.intents import Intent, ActionParameter


def format_conversation_with_incoming(messages, message: str) -> str:
    messages = (
        f"{msg.get('type')}: {msg.get('data').get('content')}"
        for msg in messages_to_dict(messages)
    )

    conversation = "\n".join(messages)
    return f"{conversation} \n human: {message}"


def parse_intents(business_intents: list[Intent], intents_text: str) -> list[Intent]:
    intents = [
        line.strip().lstrip("- ").split(": ")
        for line in intents_text.strip().split("\n")
    ]

    if not intents:
        return []

    active_intents = []

    for intent, is_active in intents:
        if is_active == "True":
            active_intents.append(intent)

    if not active_intents:
        return []

    return [intent for intent in business_intents if intent.name in active_intents]


def parse_customer_information(
    action_params: list[ActionParameter], customer_info: str
) -> dict:
    parsed_info = {}

    for param in action_params:
        # Extract the field name and format from the ActionParameter object
        field_name = param.field
        field_format = param.format

        # Search for the corresponding field in the customer information
        search_string = f"{field_name}: "
        start_index = customer_info.find(search_string)
        if start_index == -1:
            # If the field is required but not found in the customer information, raise an exception
            if param.required:
                raise ValueError(
                    f"Required field '{field_name}' not found in customer information"
                )
            # If the field is not required and not found in the customer information, skip it
            else:
                continue

        # Extract the value of the field from the customer information
        end_index = customer_info.find("\n", start_index)
        if end_index == -1:
            value = customer_info[start_index + len(search_string) :].strip()
        else:
            value = customer_info[start_index + len(search_string) : end_index].strip()

        # Remove any trailing non-numeric characters from the value if it's an integer
        if field_format == "integer":
            value = value.rstrip("'")
            value = int(value)
        # Validate the format of the value if specified in the ActionParameter object
        elif field_format == "float":
            value = value.rstrip("'")
            value = float(value)
        # Add the parsed value to the result dictionary

        if value == "":
            raise ValueError(
                f"Required field '{field_name}' not found in customer information"
            )
        parsed_info[field_name] = value
    return parsed_info


def parse_data_parameters(
    action_params: list[ActionParameter], customer_info: str
) -> dict:
    return parse_customer_information(action_params, customer_info)
    pairs = customer_info.split("\n")

    pattern = r"[^a-zA-Z0-9]+"

    data: dict = {}
    for pair in pairs:
        key, value = pair.split(": ")
        key = key.strip("'")
        value = value.strip("'")

        f = 0
        b = False
        while f < len(action_params) and b is False:
            action = action_params[f]
            if action.field == key:
                b = True
                if action.format == "integer":
                    value = int(value)
                elif action.format == "float":
                    value = float(value)
                elif action.format == "string":
                    value = str(value)
            f += 1

        data[key] = value

    return data


def parse_action_confirmation(action_confirmation_text: str) -> bool:
    return action_confirmation_text.count("True") == 2
