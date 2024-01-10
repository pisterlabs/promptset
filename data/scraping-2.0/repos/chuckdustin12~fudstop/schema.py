from openai import OpenAI
import os
import asyncio
from webull_options.webull_options import WebullOptions
import json
from dotenv import load_dotenv
load_dotenv()
import datetime
client = OpenAI(api_key=os.environ.get('YOUR_OPENAI_KEY'))
sdk = WebullOptions(access_token=os.environ.get('ACCESS_TOKEN'), osv=os.environ.get('OSV'), did=os.environ.get('DID'))
def serialize_record(record):
    """Convert asyncpg.Record to a dictionary, handling date serialization."""
    return {key: value.isoformat() if isinstance(value, datetime.date) else value 
            for key, value in dict(record).items()}

tools = [
    {
        "type": "function",
        "function": {
            "name": "filter_options",
            "description": "Filter options based on several different keyword arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_id": {
                        "type": "integer",
                        "description": "Identifier for the ticker.",
                    },
                    "belong_ticker_id": {
                        "type": "integer",
                        "description": "Identifier for the belonging ticker.",
                    },
                    "open_min": {
                        "type": "number",
                        "description": "Minimum value for the opening price.",
                    },
                    "open_max": {
                        "type": "number",
                        "description": "Maximum value for the opening price.",
                    },
                    "high_min": {
                        "type": "number",
                        "description": "Minimum value for the highest price.",
                    },
                    "high_max": {
                        "type": "number",
                        "description": "Maximum value for the highest price.",
                    },
                    "low_min": {
                        "type": "number",
                        "description": "Minimum value for the lowest price.",
                    },
                    "low_max": {
                        "type": "number",
                        "description": "Maximum value for the lowest price.",
                    },
                    "strike_price_min": {
                        "type": "integer",
                        "description": "Minimum strike price.",
                    },
                    "strike_price_max": {
                        "type": "integer",
                        "description": "Maximum strike price.",
                    },
                    "pre_close_min": {
                        "type": "number",
                        "description": "Minimum pre-close price.",
                    },
                    "pre_close_max": {
                        "type": "number",
                        "description": "Maximum pre-close price.",
                    },
                    "open_interest_min": {
                        "type": "number",
                        "description": "Minimum open interest.",
                    },
                    "open_interest_max": {
                        "type": "number",
                        "description": "Maximum open interest.",
                    },
                    "volume_min": {
                        "type": "number",
                        "description": "Minimum volume.",
                    },
                    "volume_max": {
                        "type": "number",
                        "description": "Maximum volume.",
                    },
                    "latest_price_vol_min": {
                        "type": "number",
                        "description": "Minimum latest price volume.",
                    },
                    "latest_price_vol_max": {
                        "type": "number",
                        "description": "Maximum latest price volume.",
                    },
                    "delta_min": {
                        "type": "number",
                        "description": "Minimum delta value.",
                    },
                    "delta_max": {
                        "type": "number",
                        "description": "Maximum delta value.",
                    },
                    "vega_min": {
                        "type": "number",
                        "description": "Minimum vega value.",
                    },
                    "vega_max": {
                        "type": "number",
                        "description": "Maximum vega value.",
                    },
                    "imp_vol_min": {
                        "type": "number",
                        "description": "Minimum implied volatility.",
                    },
                    "imp_vol_max": {
                        "type": "number",
                        "description": "Maximum implied volatility.",
                    },
                    "gamma_min": {
                        "type": "number",
                        "description": "Minimum gamma value.",
                    },
                    "gamma_max": {
                        "type": "number",
                        "description": "Maximum gamma value.",
                    },
                    "theta_min": {
                        "type": "number",
                        "description": "Minimum theta value.",
                    },
                    "theta_max": {
                        "type": "number",
                        "description": "Maximum theta value.",
                    },
                    "rho_min": {
                        "type": "number",
                        "description": "Minimum rho value.",
                    },
                    "rho_max": {
                        "type": "number",
                        "description": "Maximum rho value.",
                    },
                    "close_min": {
                        "type": "number",
                        "description": "Minimum closing price.",
                    },
                    "close_max": {
                        "type": "number",
                        "description": "Maximum closing price.",
                    },
                    "change_min": {
                        "type": "number",
                        "description": "Minimum change value.",
                    },
                    "change_max": {
                        "type": "number",
                        "description": "Maximum change value.",
                    },
                    "change_ratio_min": {
                        "type": "number",
                        "description": "Minimum change ratio.",
                    },
                    "change_ratio_max": {
                        "type": "number",
                        "description": "Maximum change ratio.",
                    },
                    "expire_date": {
                        "type": "string",
                        "description": "Expiration date of the option (YYYY-MM-DD).",
                    },
                    "open_int_change_min": {
                        "type": "number",
                        "description": "Minimum open interest change.",
                    },
                    "open_int_change_max": {
                        "type": "number",
                        "description": "Maximum open interest change.",
                    },
                    "active_level_min": {
                        "type": "number",
                        "description": "Minimum active level.",
                    },
                    "active_level_max": {
                        "type": "number",
                        "description": "Maximum active level.",
                    },
                    "cycle_min": {
                        "type": "number",
                        "description": "Minimum cycle value.",
                    },
                    "cycle_max": {
                        "type": "number",
                        "description": "Maximum cycle value.",
                    },
                    "call_put": {
                        "type": "string",
                        "description": "Type of option to filter ('call' or 'put').",
                    },
                    "option_symbol": {
                        "type": "string",
                        "description": "Symbol of the option.",
                    },
                    "underlying_symbol": {
                        "type": "string",
                        "description": "Underlying symbol for the option.",
                    },
                    "underlying_price_min": {
                        "type": "number",
                        "description": "Minimum underlying price.",
                    },
                    "underlying_price_max": {
                        "type": "number",
                        "description": "Maximum underlying price.",
                    },
                },
                "required": [],  # Add required fields here if any
            },
        }
    }
]
async def filter_options(**kwargs):
    """
    Filters the options table based on provided keyword arguments.
    Usage example:
        await filter_options(strike_price_min=100, strike_price_max=200, call_put='call',
                                expire_date='2023-01-01', delta_min=0.1, delta_max=0.5)
    """
    # Start with the base query
    query = f"SELECT underlying_symbol, strike_price, call_put, expire_date, theta FROM public.options WHERE "
    params = []
    param_index = 1

    # Mapping kwargs to database columns and expected types, including range filters
    column_types = {
        'ticker_id': ('ticker_id', 'int'),
        'belong_ticker_id': ('belong_ticker_id', 'int'),
        'open_min': ('open', 'float'),
        'open_max': ('open', 'float'),
        'high_min': ('high', 'float'),
        'high_max': ('high', 'float'),
        'low_min': ('low', 'float'),
        'low_max': ('low', 'float'),
        'strike_price_min': ('strike_price', 'int'),
        'strike_price_max': ('strike_price', 'int'),
        'pre_close_min': ('pre_close', 'float'),
        'pre_close_max': ('pre_close', 'float'),
        'open_interest_min': ('open_interest', 'float'),
        'open_interest_max': ('open_interest', 'float'),
        'volume_min': ('volume', 'float'),
        'volume_max': ('volume', 'float'),
        'latest_price_vol_min': ('latest_price_vol', 'float'),
        'latest_price_vol_max': ('latest_price_vol', 'float'),
        'delta_min': ('delta', 'float'),
        'delta_max': ('delta', 'float'),
        'vega_min': ('vega', 'float'),
        'vega_max': ('vega', 'float'),
        'imp_vol_min': ('imp_vol', 'float'),
        'imp_vol_max': ('imp_vol', 'float'),
        'gamma_min': ('gamma', 'float'),
        'gamma_max': ('gamma', 'float'),
        'theta_min': ('theta', 'float'),
        'theta_max': ('theta', 'float'),
        'rho_min': ('rho', 'float'),
        'rho_max': ('rho', 'float'),
        'close_min': ('close', 'float'),
        'close_max': ('close', 'float'),
        'change_min': ('change', 'float'),
        'change_max': ('change', 'float'),
        'change_ratio_min': ('change_ratio', 'float'),
        'change_ratio_max': ('change_ratio', 'float'),
        'expire_date': ('expire_date', 'date'),
        'open_int_change_min': ('open_int_change', 'float'),
        'open_int_change_max': ('open_int_change', 'float'),
        'active_level_min': ('active_level', 'float'),
        'active_level_max': ('active_level', 'float'),
        'cycle_min': ('cycle', 'float'),
        'cycle_max': ('cycle', 'float'),
        'call_put': ('call_put', 'str'),
        'option_symbol': ('option_symbol', 'str'),
        'underlying_symbol': ('underlying_symbol', 'str'),
        'underlying_price_min': ('underlying_price', 'float'),
        'underlying_price_max': ('underlying_price', 'float'),
    }

    # Dynamically build query based on kwargs
    query = "SELECT underlying_symbol, strike_price, call_put, expire_date FROM public.options WHERE open_interest > 0"

    # Dynamically build query based on kwargs
    for key, value in kwargs.items():
        if key in column_types and value is not None:
            column, col_type = column_types[key]

            # Sanitize and format value for SQL query
            sanitized_value = sdk.sanitize_value(value, col_type)

            if 'min' in key:
                query += f" AND {column} >= {sanitized_value}"
            elif 'max' in key:
                query += f" AND {column} <= {sanitized_value}"
            else:
                query += f" AND {column} = {sanitized_value}"
            print(query)
    query += " LIMIT 25"
    conn = await sdk.db_manager.get_connection()

    try:
        # Execute the query
        return await conn.fetch(query)
    except Exception as e:
        print(f"Error during query: {e}")
        return []
    finally:
        await conn.close()
async def run_conversation(query):
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": f"{query}?? Please response in tabulated format for ease of readability and give a summary underneath it. Use 'fancy' Ensure to only return options that expire past today's date."}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "filter_options": filter_options,  # Assuming filter_options is an async function
        }

        messages.append(response_message)  # extend conversation with assistant's reply

        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)

            records = await function_to_call(**function_args)

            # Process each record for serialization
            processed_records = [serialize_record(record) for record in records]

            # Serialize the list of processed records
            serialized_response = json.dumps(processed_records)

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": serialized_response,
            })

        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response


