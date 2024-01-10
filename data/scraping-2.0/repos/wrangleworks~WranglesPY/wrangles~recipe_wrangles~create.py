"""
Functions to create new columns
"""
import uuid as _uuid
from typing import Union as _Union
import math as _math
import pandas as _pd
import numpy as _np
from jinja2 import (
    Environment as _Environment,
    FileSystemLoader as _FileSystemLoader,
    BaseLoader as _BaseLoader
)
import requests as _requests
from ..connectors.test import _generate_cell_values
from .. import openai as _openai

def bins(df: _pd.DataFrame, input: _Union[str, list], output: _Union[str, list], bins: _Union[int, list], labels: _Union[str, list] = None, **kwargs) -> _pd.DataFrame:
    """
    type: object
    description: Create a column that groups data into bins
    additionalProperties: false
    required:
      - input
      - output
      - bins
    properties:
      input:
        type:
          - array
        description: Name of input column
      output:
        type:
          - array
        description: Name of new column
      bins:
        type:
          - integer
          - array
        description: Defines the number of equal-width bins in the range
      labels:
        type:
          - string
          - array
        description: Labels for the returned bins
    """
    if output is None: output = input
    
    # Ensure input and outputs are lists
    if not isinstance(input, list): input = [input]
    if not isinstance(output, list): output = [output]
    
    # Ensure input and output are equal lengths
    if len(input) != len(output):
        raise ValueError('The lists for input and output must be the same length.')

    for in_col, out_col in zip(input, output):
      # Dealing with positive infinity. At end of bins list
      if isinstance(bins, list):
          if bins[-1] == '+':
              bins[-1] = _math.inf
          
          # Dealing with negative infinity. At start of bins list
          if bins[0] == '-':
              bins[0] = -_math.inf
      
      # Set to string in order to be able to fill NaN values when using where
      df[out_col] = _pd.cut(
          x=df[in_col],
          bins=bins,
          labels=labels,
          **kwargs
      ).astype(str)
    
    return df


def column(df: _pd.DataFrame, output: _Union[str, list], value = None) -> _pd.DataFrame:
    """
    type: object
    description: Create column(s) with a user defined value. Defaults to None (empty).
    additionalProperties: false
    required:
      - output
    properties:
      output:
        type:
          - string
          - array
        description: "Name or list of names of new columns or column_name: value pairs."
      value:
        type:
          - string
        description: (Optional) Value(s) to add in the new column(s). If using a dictionary in output, value can only be a string.
    """    
    
    # if value is a list then raise and error
    if isinstance(value, list):
        raise ValueError('Value must be a string and not a list')
    
    # Get list of existing columns
    existing_column = df.columns
    
    # Get number of rows in df
    rows = len(df)
    # Get number of columns created
    cols_created = len(output)
    # If a string provided, convert to list
    if isinstance(output, str):
      if output in existing_column:
        raise ValueError(f'"{output}" column already exists in dataFrame.')
      output = [output]
    
    # gather the columns and values in a dictionary, if not a dict then use value as the value of dictionary
    output_dict = {}
    for out in output:
        if isinstance(out, dict):
            # get the first key and value only and append dictionary to output_dict
            temp_key, temp_value = list(out.items())[0]
            output_dict.update({temp_key: temp_value})
        else:
            output_dict.update({out: value})
                
    # Check if the list of outputs exist in dataFrame
    check_list = [x for x in (output_dict.keys()) if x in existing_column]
    if len(check_list) > 0:
      raise ValueError(f'{check_list} column(s) already exists in the dataFrame') 
    
    for output_column, values_list in zip(output_dict.keys(), output_dict.values()):
        # Data to generate
        data = _pd.DataFrame(_generate_cell_values(values_list, rows), columns=[output_column])
        data.set_index(df.index, inplace=True)  # use the same index as original to match rows
        # Merging existing dataframe with values created
        df = _pd.concat([df, data], axis=1)

    return df


def embeddings(
    df: _pd.DataFrame,
    input: str,
    api_key: str,
    output: str = None,
    batch_size: int = 100,
    threads: int = 10,
    output_type: str = "python list",
    model: str = "text-embedding-ada-002",
    retries: int = 0
) -> _pd.DataFrame:
    """
    type: object
    description: Create an embedding based on text input.
    additionalProperties: false
    required:
      - input
      - api_key
    properties:
      input:
        type:
          - string
          - array
        description: The column of text to create the embeddings for.
      output:
        type:
          - string
          - array
        description: The output column the embeddings will be saved as.
      api_key:
        type: string
        description: The OpenAI API key.
      batch_size:
        type: integer
        description: The number of rows to submit per individual request.
      threads:
        type: integer
        description: >-
          The number of requests to submit in parallel.
          Each request contains the number of rows set as batch_size.
      output_type:
        type: string
        description: >-
          Output the embeddings as a numpy array or a python list
          Default - python list.
        enum:
          - numpy array
          - python list
      retries:
        type: integer
        description: >-
          The number of times to retry if the request fails.
          This will apply exponential backoff to help with rate limiting.
    """
    if output is None: output = input

    if not isinstance(input, list): input = [input]
    if not isinstance(output, list): output = [output]

    if len(input) != len(output):
        raise ValueError('The lists for input and output must be the same length.')

    if output_type not in ["python list", "numpy array"]:
        raise ValueError('Output_type must be of value "numpy array" or "python list"')

    for input_col, output_col in zip(input, output):
        df[output_col] = _openai.embeddings(
            df[input_col].to_list(),
            api_key,
            model,
            batch_size,
            threads,
            retries
        )

        if output_type == 'python list':
            df[output_col] = [
                list(row)
                for row in df[output_col].values
            ]

    return df


def guid(df: _pd.DataFrame, output: _Union[str, list]) -> _pd.DataFrame:
    """
    type: object
    description: Create column(s) with a GUID.
    additionalProperties: false
    required:
      - output
    properties:
      output:
        type:
          - string
          - array
        description: Name or list of names of new columns
    """
    return uuid(df, output)


def index(df: _pd.DataFrame, output: _Union[str, list], start: int = 1, step: int = 1) -> _pd.DataFrame:
    """
    type: object
    description: Create column(s) with an incremental index.
    additionalProperties: false
    required:
      - output
    properties:
      output:
        type:
          - string
          - array
        description: Name or list of names of new columns
      start:
        type: integer
        description: (Optional; default 1) Starting number for the index
      step:
        type: integer
        description: (Optional; default 1) Step between successive rows
    """
    # If a string provided, convert to list
    if isinstance(output, str): output = [output]

    # Loop through and create incremental index
    for output_column in output:
        df[output_column] = _np.arange(start, len(df) * step + start, step=step)

    return df


def jinja(df: _pd.DataFrame, template: dict, output: list, input: str = None) -> _pd.DataFrame:
    """
    type: object
    description: Output text using a jinja template
    additionalProperties: false
    required:
      - output
      - template
    properties:
      input:
        type: string
        description: |
          Specify a name of column containing a dictionary of elements to be used in jinja template.
          Otherwise, the column headers will be used as keys.
      output:
        type: string
        description: Name of the column to be output to.
      template:
        type: object
        description: |
          A dictionary which defines the template/location as well as the form which the template is input.
          If any keys use a space, they must be replaced with an underscore.  Note: spaces within column names
          are replaced by underscores (_).
        additionalProperties: false
        properties:
          file:
            type: string
            description: A .jinja file containing the template
          column:
            type: string
            description: A column containing the jinja template - this will apply to the corresponding row.
          string:
            type: string
            description: A string which is used as the jinja template
    """
    if isinstance(output, list):
        output = output[0]
    
    if input:
        input = input[0]
        input_list = df[input]
    else:
        input_list = df.to_dict(orient='records')

    # Replace any spaces in keys with underscores
    input_list = [
        {
            key.replace(' ', '_'): val
            for key, val in row.items()
        }
        for row in input_list
    ]

    if len(template) > 1:
        raise Exception('Template must have only one key specified')

    # Template input as a file
    if 'file' in template:
        environment = _Environment(loader=_FileSystemLoader(''),trim_blocks=True, lstrip_blocks=True)
        desc_template = environment.get_template(template['file'])
        df[output] = [desc_template.render(row) for row in input_list]

    # Template input as a column of the dataframe
    elif 'column' in list(template.keys()):
        df[output] = [
            _Environment(loader=_BaseLoader).from_string(template).render(row)
            for (template, row) in zip(df[template['column']], input_list)
        ]
        
    # Template input as a string
    elif 'string' in list(template.keys()):
        desc_template = _Environment(loader=_BaseLoader).from_string(template['string'])
        df[output] = [desc_template.render(row) for row in input_list]
  
    else:
        raise Exception("'file', 'column' or 'string' not found")

    return df


def uuid(df: _pd.DataFrame, output: _Union[str, list]) -> _pd.DataFrame:
    """
    type: object
    description: Create column(s) with a UUID.
    additionalProperties: false
    required:
      - output
    properties:
      output:
        type:
          - string
          - array
        description: Name or list of names of new columns
    """
    # If a string provided, convert to list
    if isinstance(output, str): output = [output]

    # Loop through and create uuid for all requested columns
    for output_column in output:
        df[output_column] = [_uuid.uuid4() for _ in range(len(df.index))]

    return df
