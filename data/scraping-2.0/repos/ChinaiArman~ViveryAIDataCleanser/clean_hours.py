"""
Clean Hours Script

@author Arman Chinai
@version 2.1.2

This script uses Azure OpenAI to clean and format pantry hours within the Bulk Upload File template. 
The input and output of this program is a Bulk Upload File (CSV).
The hours are placed into a single column within the Bulk Upload File.
The hours are then flattened into plaintext, and parsed line-by-line to a fine-tuned, Azure OpenAI model.
The resulting hours are then tested for AI errors, with failing tests resulting in a flagged hour for manual review.
Finally, the hours are unflattened and reformatted back into the Bulk Upload File.

---> OPERATIONAL INSTRUCTIONS <---

Package Imports:
    * OpenAI            * Pandas            * Datetime
    * Argparse          * Regex

API Keys (stored in keys.py):
    * Azure OpenAI - North Central US: Contact Arman for API Key.

Instructions:
    1) Package Imports:
        a) Create a new terminal.
        b) Run `pip install -r requirements.txt`.
    2) API Keys:
        a) Create a new file `keys.py` within the directory at the same level as `clean_hours.py`.
        b) Contact Arman (arman@vivery.org) for the API Key.
        c) Create a new python variable `North_CENTRAL_API_KEY` with the received API Key.
    3) Prepare the Bulk Upload File
        a) Within the Bulk Upload File, add a new column `Hours Uncleaned`.
        b) Paste all unformatted/uncleaned hours into this column. Ensure these unformatted hours are in the row with the associating Pantry/Location. Save these changes.
        c) Add the Bulk Upload File to the working directory at the same level as `clean_hours.py`.
    4) Run the following command within the terminal: `python clean_hours.py "{path to Bulk Upload File from working directory}"`.

Desired Output:
    * A new CSV file will be present within the working directory, with the name ending in "_HOURS_CLEANED".
    * The file will contain the hours for each pantry cleaned and formatted into their respective rows.
    * Any hours that failed the testing round will remain in the `Hours Uncleaned` column for manual review.

Still have questions? Send an email to `arman@vivery.org` with the subject line `Clean Hours - {question}`.
"""


# PACKAGE IMPORTS
import openai
import argparse, os, shutil
import pandas as pd
import re
from datetime import datetime
import time

# LOCAL FILE IMPORTS


# AI CONSTANTS
from keys import NORTH_CENTRAL_API_KEY as OAI_API

# MISC CONSTANTS
INT_TO_DAY_OF_MONTH = {"1": ["1st", "First"], "2": ["2nd", "Second"], "3": ["3rd", "Third"], "4": ["4th", "Fourth"], "5": ["5th", "Fifth"], "": ""}
DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HOUR_TYPES = ["Weekly", "Every Other Week", "Day of Month", "Week of Month"]
UNCLEANED_HOURS_COLUMN = "Hours Uncleaned"
INVALID_CHARACTERS = "/"




# HELPERS
def create_id_hours_dict(df: pd.DataFrame) -> dict:
    """
    Create a dictionary mapping `Program External IDs` to `Hours Uncleaned` from a DataFrame.

    Args:
        - `df` (pd.DataFrame): A Pandas DataFrame containing data.

    Preconditions:
        - The DataFrame `df` must have columns `Program External ID` and `Hours Uncleaned`.
        - `Program External ID` column should contain unique identifiers.
        - `Hours Uncleaned` column should contain the hours data.

    Returns:
        - dict: A dictionary mapping `Program External IDs` (str) to `Hours Uncleaned` (str).

    Raises:
        - None

    Example:
        >>> import pandas as pd
        >>> data = {
        ...     'Program External ID': ['ID1', 'ID2', 'ID3'],
        ...     'Hours Uncleaned': ['x', 'y', 'z']
        ... }
        >>> df = pd.DataFrame(data)
        >>> result = create_id_hours_dict(df)
        >>> print(result)
        {'ID1': 'x', 'ID2': 'y', 'ID3': 'z'}
    """
    id_hours_dict = {}

    for _, row in df.iterrows():
        id_hours_dict[row["Program External ID"]] = str(row[UNCLEANED_HOURS_COLUMN]).strip()

    return id_hours_dict


def call_oai(prompt: str) -> str:
    """
    Calls the `Vivery Clean Hours Training Model` to format uncleaned hours into "bulk-upload-ready" hour entries. 

    Args:
        - `prompt` (str): An hour entry to be cleaned using the `Vivery Clean Hours Training Model`.

    Preconditions:
        - The OpenAI API key and other configuration details should be correctly set up in a separate `keys.py` file and imported with the constants at the top of the file.
            
            >>> API_KEY = {
                "key": "...",
                "base": "...",
                "engine": "..."
            }

        - The `prompt` should be a string.
        - The `Vivery Clean Hours Training Model` and `Microsoft Azure OAI` services must be online and operational to be called upon.

    Returns:
        - str: The hours, cleaned and formatted for the bulk upload file template. 

    Raises:
        - None

    Example:
        >>> response = call_oai("Every Monday, from 3pm-5pm")
        >>> print(response)
        'Monday,15:00,17:00,,,,,,,,Weekly,,,'
    """
    openai.api_type = "azure"
    openai.api_base = OAI_API["base"]
    openai.api_version = "2023-09-15-preview"
    openai.api_key = OAI_API["key"]
    response = openai.Completion.create(
        engine=OAI_API["engine"],
        prompt=f"{prompt}",
        temperature=0.2,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
        stop=["%%"]
    )
    time.sleep(0.05)
    print("\tOAI API Response: " + response["choices"][0]["text"])
    return response["choices"][0]["text"]


def format_hours_iteratively(id_hours_dict: dict) -> dict:
    """
    Creates a dictionary of `Program External IDs` and their formatted-hour counterparts. 

    Args:
        - `id_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the original unformatted hour values as values.

    Preconditions:
        - The `id_hours_dict` should be a dictionary with `Program External IDs` as keys and string representations of unformatted hours as values.
        - All `call_oai` preconditions must be satisfied.

    Returns:
        - dict: A dictionary containing `Program External IDs` as keys and formatted hour values as values.

    Raises:
        - None

    Example:
        >>> id_hours = {
        ...     "ID1": "Every Monday, from 3pm-5pm",
        ...     "ID2": "3rd Tuesday and Wednesday, from 9am-10am"
        ... }
        >>> formatted_hours = format_hours_iteratively(id_hours)
        >>> print(formatted_hours)
        {
            "ID1": "Monday,15:00,17:00,,,,,,,,Weekly,,,", 
            "ID2": "Tuesday,9:00,10:00,,,,,,,3,Day of Month,,,;Wednesday,9:00,10:00,,,,,,,2,Day of Month,,,"
        }
    """
    cleaned_hours_dict = {}

    for key, value in id_hours_dict.items():
        new_value = call_oai(value)
        new_value = new_value
        cleaned_hours_dict[key] = new_value
    
    return cleaned_hours_dict


def filter_invalid_values(id_hours_dict: dict, cleaned_hours_dict: dict, is_valid_hours_dict: dict) -> dict:
    """
    Removes cleaned hour entries that failed one or more tests during the testing phase. This process flags them for human review, returning the hours to their original, unformatted state.

    Args:
        - `id_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the original unformatted hour values as values.
        - `cleaned_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the cleaned/formatted hour values as values.
        - `is_valid_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Preconditions:
        - `id_hours_dict` should be a dictionary with the `Program External IDs` as keys and string representations of hours as values.
        - `cleaned_hours_dict` should be a dictionary with the `Program External IDs` as keys and cleaned/formatted hour values as values.
        - `is_valid_hours_dict` should be a dictionary with the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Returns:
        - dict: A dictionary containing the `Program External IDs` as keys and the hours in their final state (invalid hours returned to original state, valid hours in a formatted state).

    Raises:
        - None

    Example:
        >>> id_hours = {
        ...     "ID1": "Every Monday, from 3pm-5pm",
        ...     "ID2": "3rd Tuesday and Wednesday, from 9am-10am"
        ... }
        >>> cleaned_hours = {
        ...     "ID1": "Monday,15:00,17:00,,,,,,,,Weekly,,,",
        ...     "ID2": "Tuesday,9:00,10:00,,,,,,,3,Day of Month,,,;Wednesday,9:00,10:00,,,,,,,2,Day of Month,,,"
        ... }
        >>> is_valid = {
        ...     "ID1": True,
        ...     "ID2": False
        ... }
        >>> valid_hours = filter_invalid_values(id_hours, cleaned_hours, is_valid)
        >>> print(valid_hours)
        {
            "ID1": "Monday,15:00,17:00,,,,,,,,Weekly,,,",
            "ID2": "3rd Tuesday and Wednesday, from 9am-10am"
        }
    """
    valid_hours_dict = {}

    for key, _ in cleaned_hours_dict.items():
        if is_valid_hours_dict[key]:
            valid_hours_dict[key] = cleaned_hours_dict[key]
        else:
            valid_hours_dict[key] = id_hours_dict[key]
            
    return valid_hours_dict


def convert_id_hours_dict_to_df(cleaned_hours_dict: dict, is_valid_hours_dict: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the cleaned hours dictionary into a DataFrame, filling in formatted hour entries when valid and preserving original data for invalid entries 

    Args:
        - `cleaned_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the cleaned/formatted hour values as values.
        - `is_valid_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.
        - `df` (pd.DataFrame): The original DataFrame containing program data.

    Preconditions:
        - 'cleaned_hours_dict' should be a dictionary with `Program External IDs` as keys and cleaned/formatted hour values as values.
        - 'is_valid_hours_dict' should be a dictionary with `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.
        - 'df' should be a pandas DataFrame containing program data with a "Program External ID" column.

    Returns:
        - pd.DataFrame: A new DataFrame with cleaned/formatted hour values for valid entries and original data for invalid entries. This DataFrame's format matches that of the `Bulk Upload` file

    Raises:
        - None

    Example:
        >>> program_data = pd.DataFrame({
        ...     "Program External ID": ["ID1", "ID2"],
        ...     # Other program data columns...
        ... })
        >>> cleaned_hours = {
        ...     "ID1": "Monday,15:00,17:00,,,,,,,,Weekly,,,",
        ...     "ID2": "3rd Tuesday and Wednesday, from 9am-10am"
        ... }
        >>> is_valid = {
        ...     "ID1": True,
        ...     "ID2": False
        ... }
        >>> cleaned_hours_df = convert_id_hours_dict_to_df(cleaned_hours, is_valid, program_data)
        >>> print(cleaned_hours_df)
            "Program External ID"   # Unrelated Columns   # Formatted Hours Related Columns (CSV)
        0              "ID1"          ...     ...           "Monday","15:00","17:00",,,,,,,,"Weekly",,,,      
        1              "ID2"          ...     ...           ,,,,,,,,,,,,,"3rd Tuesday and Wednesday, from 9am-10am",  
    """
    # Create new DF
    cleaned_hours_df = pd.DataFrame(columns=df.columns)

    # Iterate over Program IDs
    for id in df["Program External ID"].to_list():
        new_entries = []
        row = df.loc[df['Program External ID'] == id].values.tolist()[0]    # if this line produces an error, some programs are missing program IDs within the bulk upload file.

        # Create new row if valid cleaning
        if is_valid_hours_dict[id]:
            row = row[0:len(df.columns) - 15]
            list_of_entries = cleaned_hours_dict[id].split(";")
            for entry in list_of_entries:
                entry = entry.split(',')
                entry = row + entry + [""]
                new_entries.append(entry)

        # Add row to new DF
        if is_valid_hours_dict[id]:
            for entry in new_entries:
                try:
                    cleaned_hours_df.loc[len(cleaned_hours_df)] = entry
                except ValueError:
                    cleaned_hours_df.loc[len(cleaned_hours_df)] = row + [""] * 15
        else:
            cleaned_hours_df.loc[len(cleaned_hours_df)] = row
    
    # Return DF
    return cleaned_hours_df




# TESTS
def test_valid_day_of_week(_: any, cleaned_hours_dict: dict, is_valid_dict: dict) -> dict:
    """
    Test the validity of the day of the week entries in the cleaned hours dictionary.

    Args:
        - `_` (dict): [UNUSED] `id_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the original unformatted hour values as values.
        - `cleaned_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the cleaned/formatted hour values as values.
        - `is_valid_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Preconditions:
        - `cleaned_hours_dict` should be a dictionary with the `Program External IDs` as keys and cleaned/formatted hour values as values.
        - `is_valid_hours_dict` should be a dictionary with the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Returns:
        - dict: An updated `is_valid_dict` with the validity of day of the week entries for each program.

    Raises:
        - None

    Example:
        >>> cleaned_hours = {
        ...     "ID1": "Monday,15:00,17:00,,,,,,,,Weekly,,,",
        ...     "ID2": "Mursday,12:00,13:00,,,,,,3,,Week of Month,,,",
        ...     "ID3": "Tuesday,9:00,10:00,,,,,,,3,Day of Month,,,;Wednesday,9:00,10:00,,,,,,,2,Day of Month,,,"
        ... }
        >>> is_valid = {
        ...     "ID1": True,
        ...     "ID2": True,
        ...     "ID3": False
        ... }
        >>> updated_validity = test_valid_day_of_week({}, cleaned_hours, is_valid)
        >>> print(updated_validity)
        {
            "ID1": True,
            "ID2": False,
            "ID3": False
        }
    """
    for key, value in cleaned_hours_dict.items():
        is_valid = True
        list_of_entries = value.split(";")

        for value in list_of_entries:
            value = value.split(",")
            try:
                is_valid = value[0] in DAYS_OF_WEEK and is_valid
            except:
                is_valid

        is_valid_dict[key] = is_valid_dict[key] and is_valid

    return is_valid_dict


def test_valid_entry_format(_: dict, cleaned_hours_dict: dict, is_valid_dict: dict) -> dict:
    """
    Test the validity of entry format in the cleaned hours dictionary.

    Args:
        - `_` (dict): [UNUSED] `id_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the original unformatted hour values as values.
        - `cleaned_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the cleaned/formatted hour values as values.
        - `is_valid_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Preconditions:
        - `cleaned_hours_dict` should be a dictionary with the `Program External IDs` as keys and cleaned/formatted hour values as values.
        - `is_valid_hours_dict` should be a dictionary with the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Returns:
        - dict: An updated `is_valid_dict` with the validity of entry format for each program.

    Raises:
        - None

    Example:
        >>> cleaned_hours = {
        ...     "ID1": "Monday,15:00,17:00,,,,,,,,Weekly,,,",
        ...     "ID2": "Monday,12:00,13:00,3,Week of Month;",
        ...     "ID3": "Tuesday,9:00,10:00,,,,,,,3,Day of Month,,,;Wednesday,9:00,10:00,,,,,,,2,Day of Month,,,"
        ... }
        >>> is_valid = {
        ...     "ID1": True,
        ...     "ID2": True,
        ...     "ID3": False
        ... }
        >>> updated_validity = test_valid_entry_format({}, cleaned_hours, is_valid)
        >>> print(updated_validity)
        {
            "ID1": True,
            "ID2": False,
            "ID3": False
        }
    """
    for key, value in cleaned_hours_dict.items():
        count_semicolons = value.count(";")
        count_commas = value.count(",")
        is_valid = 13 + count_semicolons * 13 == count_commas
        is_valid = (count_semicolons < 1 and count_commas == 13) or (count_semicolons >= 1 and count_commas > 13) and is_valid
        is_valid_dict[key] = is_valid_dict[key] and is_valid

    return is_valid_dict


def test_valid_open_closed_hours(_: dict, cleaned_hours_dict: dict, is_valid_dict: dict) -> dict:
    """
    Test the validity of open and closed hours format in the cleaned hours dictionary.

    Args:
        - `_` (dict): [UNUSED] `id_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the original unformatted hour values as values.
        - `cleaned_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the cleaned/formatted hour values as values.
        - `is_valid_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Preconditions:
        - `cleaned_hours_dict` should be a dictionary with the `Program External IDs` as keys and cleaned/formatted hour values as values.
        - `is_valid_hours_dict` should be a dictionary with the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Returns:
        - dict: An updated `is_valid_dict` with the validity of open and closed hours format for each program.

    Raises:
        - None

    Example:
        >>> cleaned_hours = {
        ...     "ID1": "Monday,15:00,17:00,,,,,,,,Weekly,,,",
        ...     "ID2": "Monday,12pm,1pm,,,,,,3,,Week of Month,,,",
        ...     "ID3": "Tuesday,9:00,10:00,,,,,,,3,Day of Month,,,;Wednesday,9:00,10:00,,,,,,,2,Day of Month,,,"
        ... }
        >>> is_valid = {
        ...     "ID1": True,
        ...     "ID2": True,
        ...     "ID3": False
        ... }
        >>> updated_validity = test_valid_open_closed_hours({}, cleaned_hours, is_valid)
        >>> print(updated_validity)
        {
            "ID1": True,
            "ID2": False,
            "ID3": False
        }
    """
    time_regex = re.compile("^([01]?[0-9]|2[0-3]):[0-5][0-9]$")

    for key, value in cleaned_hours_dict.items():
        is_valid = True
        list_of_entries = value.split(";")

        for value in list_of_entries:
            value = value.split(",")
            try:
                is_valid = value[1] != "" and value[2] != "" and is_valid
                is_open_hour_valid = re.search(time_regex, value[1])
                is_closed_hour_valid = re.search(time_regex, value[2])
                is_valid = is_open_hour_valid != None and is_closed_hour_valid != None and is_valid
            except:
                is_valid = False

        is_valid_dict[key] = is_valid_dict[key] and is_valid
    
    return is_valid_dict
            

def test_close_hour_greater_than_open_hour(_: dict, cleaned_hours_dict: dict, is_valid_dict: dict) -> dict:
    """
    Test if the closing hour is greater than the opening hour for each program in the cleaned hours dictionary.

    Args:
        - `_` (dict): [UNUSED] `id_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the original unformatted hour values as values.
        - `cleaned_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the cleaned/formatted hour values as values.
        - `is_valid_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Preconditions:
        - `cleaned_hours_dict` should be a dictionary with the `Program External IDs` as keys and cleaned/formatted hour values as values.
        - `is_valid_hours_dict` should be a dictionary with the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Returns:
        - dict: An updated `is_valid_dict` with the validity of closing hours being greater than opening hours for each program.

    Raises:
        - None

    Example:
        >>> cleaned_hours = {
        ...     "ID1": "Monday,15:00,17:00,,,,,,,,Weekly,,,",
        ...     "ID2": "Monday,14:00,13:00,,,,,,3,,Week of Month,,,",
        ...     "ID3": "Tuesday,9:00,10:00,,,,,,,3,Day of Month,,,;Wednesday,9:00,10:00,,,,,,,2,Day of Month,,,"
        ... }
        >>> is_valid = {
        ...     "ID1": True,
        ...     "ID2": True,
        ...     "ID3": False
        ... }
        >>> updated_validity = test_close_hour_greater_than_open_hour({}, cleaned_hours, is_valid)
        >>> print(updated_validity)
        {
            "ID1": True,
            "ID2": False,
            "ID3": False
        }
    """
    for key, value in cleaned_hours_dict.items():
        is_valid = True
        list_of_entries = value.split(";")

        for value in list_of_entries:
            value = value.split(",")
            try:
                is_valid = datetime.strptime(value[2], "%H:%M") > datetime.strptime(value[1], "%H:%M") and is_valid
            except:
                is_valid = False

        is_valid_dict[key] = is_valid_dict[key] and is_valid

    return is_valid_dict


def test_day_of_month_formatting(id_hours_dict: dict, cleaned_hours_dict: dict, is_valid_dict: dict) -> dict:
    """
    Test the formatting and validity of `Day of Month` entries in cleaned hours for each program.

    Args:
        - `id_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the original unformatted hour values as values.
        - `cleaned_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the cleaned/formatted hour values as values.
        - `is_valid_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Preconditions:
        - `cleaned_hours_dict` should be a dictionary with the `Program External IDs` as keys and cleaned/formatted hour values as values.
        - `is_valid_hours_dict` should be a dictionary with the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Returns:
        - dict: An updated `is_valid_dict` with the validity of `Day of Month` entries in cleaned hours for each program.

    Raises:
        - None

    Example:
        >>> id_hours = {
        ...     "ID1": "Every Monday, from 3pm-5pm",
        ...     "ID2": "3rd Tuesday and Wednesday, from 9am-10am"
        ...     "ID3": "Friday 3pm to 5pm Every Year"
        ... }
        >>> cleaned_hours = {
        ...     "ID1": "Monday,15:00,17:00,,,,,,,,Weekly,,,",
        ...     "ID2": "Thursday,13:00,14:00,,,,,,,7,Day of Month,,,",
        ...     "ID3": "Friday,15:00,17:00,,,,,,,,Year of Week,,,"
        ... }
        >>> is_valid = {
        ...     "ID1": True,
        ...     "ID2": True,
        ...     "ID3": False
        ... }
        >>> updated_validity = test_day_of_month_formatting(id_hours, cleaned_hours, is_valid)
        >>> print(updated_validity)
        {
            "ID1": True,
            "ID2": False,
            "ID3": False
        }
    """
    for key, value in cleaned_hours_dict.items():
        is_valid = True
        list_of_entries = value.split(";")

        for value in list_of_entries:
            value = value.split(",")
            try:
                if value[10] == "Day of Month":
                    is_valid = value[9].isdigit() and value[8] == "" and is_valid
                    is_valid = (any(day_of_month_value.lower() in id_hours_dict[key].lower() for day_of_month_value in INT_TO_DAY_OF_MONTH[value[9]]) or value[9] == "") and is_valid
            except:
                is_valid = False

        is_valid_dict[key] = is_valid_dict[key] and is_valid

    return is_valid_dict


def test_week_of_month_formatting(id_hours_dict: dict, cleaned_hours_dict: dict, is_valid_dict: dict) -> dict:
    """
    Test the formatting and validity of `Week of Month` entries in cleaned hours for each program.

    Args:
        - `id_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the original unformatted hour values as values.
        - `cleaned_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the cleaned/formatted hour values as values.
        - `is_valid_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Preconditions:
        - `cleaned_hours_dict` should be a dictionary with the `Program External IDs` as keys and cleaned/formatted hour values as values.
        - `is_valid_hours_dict` should be a dictionary with the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Returns:
        - dict: An updated `is_valid_dict` with the validity of `Week of Month` entries in cleaned hours for each program.

    Raises:
        - None

    Example:
        >>> id_hours = {
        ...     "ID1": "Every Monday, from 3pm-5pm",
        ...     "ID2": "3rd Week Tuesday and Wednesday, from 9am-10am"
        ...     "ID3": "Friday 3pm to 5pm Every Year"
        ... }
        >>> cleaned_hours = {
        ...     "ID1": "Monday,15:00,17:00,,,,,,,,Weekly,,,",
        ...     "ID2": "Thursday,14:00,13:00,,,,,,,7,Week of Month,,,",
        ...     "ID3": "Friday,15:00,17:00,,,,,,,,Year of Week,,,"
        ... }
        >>> is_valid = {
        ...     "ID1": True,
        ...     "ID2": True,
        ...     "ID3": False
        ... }
        >>> updated_validity = test_week_of_month_formatting(id_hours, cleaned_hours, is_valid)
        >>> print(updated_validity)
        {
            "ID1": True,
            "ID2": False,
            "ID3": False
        }
    """
    for key, value in cleaned_hours_dict.items():
        is_valid = True
        list_of_entries = value.split(";")

        for value in list_of_entries:
            value = value.split(",")
            try:
                if value[10] == "Week of Month":
                    is_valid = value[8].isdigit() and value[9] == "" and is_valid
                    is_valid = (any(day_of_week_value.lower() in id_hours_dict[key].lower() for day_of_week_value in INT_TO_DAY_OF_MONTH[value[8]]) or value[8] == "") and is_valid
            except:
                is_valid = False

        is_valid_dict[key] = is_valid_dict[key] and is_valid

    return is_valid_dict


def test_weekly_formatting(_: dict, cleaned_hours_dict: dict, is_valid_dict: dict) -> dict:
    """
    Test the formatting and validity of `Weekly` and `Every Other Week` entries in cleaned hours for each program.

    Args:
        - `_` (dict): [UNUSED] `id_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the original unformatted hour values as values.
        - `cleaned_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the cleaned/formatted hour values as values.
        - `is_valid_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Preconditions:
        - `cleaned_hours_dict` should be a dictionary with the `Program External IDs` as keys and cleaned/formatted hour values as values.
        - `is_valid_hours_dict` should be a dictionary with the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Returns:
        - dict: An updated `is_valid_dict` with the validity of `Weekly` and `Every Other Week` entries in cleaned hours for each program.

    Raises:
        - None

    Example:
        >>> cleaned_hours = {
        ...     "ID1": "Monday,15:00,17:00,,,,,,,,Weekly,,,",
        ...     "ID2": "Friday,14:00,13:00,,,,,,3,,Weekly,,,",
        ...     "ID3": "Tuesday,9:00,10:00,,,,,,,3,Day of Month,,,;Wednesday,9:00,10:00,,,,,,,2,Day of Month,,,"
        ... }
        >>> is_valid = {
        ...     "ID1": True,
        ...     "ID2": True,
        ...     "ID3": False
        ... }
        >>> updated_validity = test_weekly_formatting({}, cleaned_hours, is_valid)
        >>> print(updated_validity)
        {
            "ID1": True,
            "ID2": False,
            "ID3": False
        }
    """
    for key, value in cleaned_hours_dict.items():
        is_valid = True
        list_of_entries = value.split(";")

        for value in list_of_entries:
            value = value.split(",")
            try:
                if value[10] == "Weekly" or value[10] == "Every Other Week":
                    is_valid = value[8] == "" and value[9] == "" and is_valid
            except:
                is_valid = False

        is_valid_dict[key] = is_valid_dict[key] and is_valid

    return is_valid_dict


def test_all_null_values_empty_string(_: dict, cleaned_hours_dict: dict, is_valid_dict: dict) -> dict:
    """
    Test whether all columns that are always suppose to be null in cleaned hours are represented as null for each program.

    Args:
        - `_` (dict): [UNUSED] `id_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the original unformatted hour values as values.
        - `cleaned_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the cleaned/formatted hour values as values.
        - `is_valid_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Preconditions:
        - `cleaned_hours_dict` should be a dictionary with the `Program External IDs` as keys and cleaned/formatted hour values as values.
        - `is_valid_hours_dict` should be a dictionary with the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Returns:
        - dict: An updated `is_valid_dict` with the validity of all null values in cleaned hours for each program.

    Raises:
        - None

    Example:
        >>> cleaned_hours = {
        ...     "ID1": "Monday,15:00,17:00,,,,,,,,Weekly,,,",
        ...     "ID2": "Friday,14:00,13:00,3,3,3,3,3,3,3,Weekly,3,3,",
        ...     "ID3": "Tuesday,9:00,10:00,,,,,,,3,Day of Month,,,;Wednesday,9:00,10:00,,,,,,,2,Day of Month,,,"
        ... }
        >>> is_valid = {
        ...     "ID1": True,
        ...     "ID2": True,
        ...     "ID3": False
        ... }
        >>> updated_validity = test_all_null_values_empty_string({}, cleaned_hours, is_valid)
        >>> print(updated_validity)
        {
            "ID1": True,
            "ID2": False,
            "ID3": False
        }
    """
    for key, value in cleaned_hours_dict.items():
        is_valid = True
        list_of_entries = value.split(";")

        for value in list_of_entries:
            value = value.split(",")
            try:
                is_valid = value[3] == "" and value[4] == "" and value[5] == "" and value[6] == "" and value[11] == "" and value[12] == "" and value[13] == "" and is_valid
            except:
                is_valid = False

        is_valid_dict[key] = is_valid_dict[key] and is_valid

    return is_valid_dict


def test_valid_hour_types(_: dict, cleaned_hours_dict: dict, is_valid_dict: dict) -> dict:
    """
    Test whether the hour types in cleaned hours are valid for each program.

    Args:
        - `_` (dict): [UNUSED] `id_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the original unformatted hour values as values.
        - `cleaned_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the cleaned/formatted hour values as values.
        - `is_valid_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Preconditions:
        - `cleaned_hours_dict` should be a dictionary with the `Program External IDs` as keys and cleaned/formatted hour values as values.
        - `is_valid_hours_dict` should be a dictionary with the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Returns:
        - dict: An updated `is_valid_dict` with the validity of hour types in cleaned hours for each program.

    Raises:
        - None

    Example:
        >>> cleaned_hours = {
        ...     "ID1": "Monday,15:00,17:00,,,,,,,,Weekly,,,",
        ...     "ID2": "ID3": "Friday,15:00,17:00,,,,,,,,Year of Week,,,",
        ...     "ID3": "Tuesday,9:00,10:00,,,,,,,3,Day of Month,,,;Wednesday,9:00,10:00,,,,,,,2,Day of Month,,,"
        ... }
        >>> is_valid = {
        ...     "ID1": True,
        ...     "ID2": True,
        ...     "ID3": False
        ... }
        >>> updated_validity = test_valid_hour_types({}, cleaned_hours, is_valid)
        >>> print(updated_validity)
        {
            "ID1": True,
            "ID2": False,
            "ID3": False
        }
    """
    for key, value in cleaned_hours_dict.items():
        is_valid = True
        list_of_entries = value.split(";")

        for value in list_of_entries:
            value = value.split(",")
            try:
                is_valid = value[10] in HOUR_TYPES and is_valid
            except:
                is_valid = False

        is_valid_dict[key] = is_valid_dict[key] and is_valid

    return is_valid_dict


def test_valid_case_length(id_hours_dict: dict, _: dict, is_valid_dict: dict) -> dict:
    """
    Test if the length of the case in 'id_hours_dict' is less than 100 characters for each key.

    Args:
        - `id_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the original unformatted hour values as values.
        - `_` (dict): [UNUSED] A dictionary containing the `Program External IDs` as keys and the cleaned/formatted hour values as values.
        - `is_valid_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Preconditions:
        - `id_hours_dict` should be a dictionary with program IDs as keys and case descriptions as values.
        - `is_valid_dict` should be a dictionary indicating the initial validity state for each case.

    Returns:
        dict: An updated dictionary ('is_valid_dict') with the validity of each case based on the length criterion.

    Raises:
        None

    Example:
        >>> id_hours_dict = {
        ...     "ID1": "Monday-Friday 3-5pm",
        ...     "ID2": "This is a very long case that exceeds 100 characters",
        ...     "ID3": "Thursday 11am-2pm every other week"
        ... }
        >>> is_valid_dict = {
        ...     "ID1": True,
        ...     "ID2": True,
        ...     "ID3": False
        ... }
        >>> result = test_valid_case_length(id_hours_dict, {}, is_valid_dict)
        >>> print(result)
        {
            "ID1": True,
            "ID2": False,
            "ID3": False
        }
    """
    for key, value in id_hours_dict.items():
        is_valid_dict[key] = len(value) < 100 and is_valid_dict[key]
    return is_valid_dict


def test_valid_case_characters(id_hours_dict: dict, _: dict, is_valid_dict: dict) -> dict:
    """
    Test if case descriptions in 'id_hours_dict' contain any invalid characters.

    Args:
        - `id_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and the original unformatted hour values as values.
        - `_` (dict): [UNUSED] A dictionary containing the `Program External IDs` as keys and the cleaned/formatted hour values as values.
        - `is_valid_hours_dict` (dict): A dictionary containing the `Program External IDs` as keys and Boolean values indicating whether the hour value is valid.

    Preconditions:
        - `id_hours_dict` should be a dictionary with program IDs as keys and case descriptions as values.
        - `is_valid_dict` should be a dictionary indicating the initial validity state for each case.

    Returns:
        dict: An updated dictionary ('is_valid_dict') with the validity of each case based on the length criterion.

    Raises:
        None

    Example:
        >>> INVALID_CHARACTERS = "/"
        >>> id_hours_dict = {
        ...     "ID1": "Monday-Friday 3-5pm",
        ...     "ID2": "Tuesday/Thursday 6-9pm",
        ...     "ID3": "Thursday 11am-2pm every other week"
        ... }
        >>> is_valid_dict = {
        ...     "ID1": True,
        ...     "ID2": True,
        ...     "ID3": False
        ... }
        >>> result = test_valid_case_length(id_hours_dict, {}, is_valid_dict)
        >>> print(result)
        {
            "ID1": True,
            "ID2": False,
            "ID3": False
        }
    """
    for key, value in id_hours_dict.items():
        is_valid = True
        for character in INVALID_CHARACTERS:
            is_valid = character not in value and is_valid
        is_valid_dict[key] = is_valid and is_valid_dict[key]
    return is_valid_dict




# MAIN
if __name__ == "__main__":
    # Define console parser
    parser = argparse.ArgumentParser(description="Clean a bulk upload files hours")
    # Add file argument
    parser.add_argument("file", action="store", help="A bulk upload file")
    # Console arguments
    args = parser.parse_args()

    # Create DataFrame
    df = pd.read_csv(args.file)
    # Move CSV
    shutil.move(args.file, "csvs/" + args.file.replace("csvs\\", ""))
    # Create id_hours Dictionary
    id_hours_dict = create_id_hours_dict(df)
    # Create is_valid_hours Dictionary
    is_valid_hours_dict = {key: True for key, _ in id_hours_dict.items()}

    # Parse Hours through OAI
    print("Calling OpenAI Fine-Tuned Model...")
    cleaned_hours_dict = format_hours_iteratively(id_hours_dict)

    # Test OAI Hours 
    print("\nTesting OpenAI Fine-Tuned Model responses...")
    validation_tests = [
        test_day_of_month_formatting,
        test_week_of_month_formatting,
        test_weekly_formatting,
        test_valid_hour_types,
        test_valid_day_of_week,
        test_valid_open_closed_hours,
        test_close_hour_greater_than_open_hour,
        test_all_null_values_empty_string,
        test_valid_entry_format,
        test_valid_case_length,
        test_valid_case_characters
    ]
    [test(id_hours_dict, cleaned_hours_dict, is_valid_hours_dict) for test in validation_tests]

    # PRINT TESTING RESULTS (CAN BE REMOVED LATER)
    for key, value in is_valid_hours_dict.items():
        print("\tProgram ID: " + str(key) + "\t\tResult: " + str(value))

    # Check Values Still Valid
    valid_id_hours_dict = filter_invalid_values(id_hours_dict, cleaned_hours_dict, is_valid_hours_dict)

    # Convert Back to DF
    cleaned_hours_df = convert_id_hours_dict_to_df(cleaned_hours_dict, is_valid_hours_dict, df)
    if not os.path.isdir('csvs'):
        os.mkdir('csvs')
    cleaned_hours_df.to_csv("csvs/" + args.file.replace(".csv", "").replace("csvs\\", "") + "_HOURS_CLEANED.csv")
    # cleaned_hours_df.to_csv(args.file.replace(".csv", "") + "_HOURS_CLEANED.csv")
