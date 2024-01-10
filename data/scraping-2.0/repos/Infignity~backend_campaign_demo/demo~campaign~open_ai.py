'''importing program modules'''
import os
import re
import json
from langchain.llms import OpenAI
from langchain.chains import llm
from .templates import analyst_template, campaign_template, email_prompt


class LangChainAI:
    '''Company Data analysis'''

    def __init__(self):
        self.open_ai_key = os.environ.get("OPENAI_API_KEY")
        self.llm_ai = OpenAI(temperature=0.9, openai_api_key=self.open_ai_key)

    def analysis_extractor(self, text):
        '''extract the company analyzed data'''
        # Split the text into lines
        lines = f"{text}".strip().split('\n')
        # Initialize variables
        data = {}
        current_main_header = ""
        current_sub_header = ""
        current_body = []

        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            if not re.match(r"^\d+", line) and line.endswith(":"):
                # Identify main headers
                current_main_header = line[:-1]
                data[current_main_header] = {}
            # elif re.match(r"^\d+\.\w+\.", line):
            elif re.match(r"^\d+.*:$", line):
                # Identify sub-headers
                current_sub_header = line
                data[current_main_header][current_sub_header] = []
            elif line.startswith("- "):
                # Add content to the current sub-header's body
                current_body.append(line[2:])
            else:
                pass
            # Update the data structure
            if current_main_header and current_sub_header:
                data[current_main_header][current_sub_header] = current_body
        output_data = {}
        for main_header, sub_headers in data.items():
            flattened_body = []
            for _, body in sub_headers.items():
                flattened_body.extend(body)
            output_data[main_header] = flattened_body
        return json.dumps(output_data, indent=4)
    
    def analysis_extractor_main(self, text):
        '''extract the company analyzed data'''
        # Define regular expressions to extract data
        main_header_pattern = r'^\d+\.\s+(.*?):$'
        sub_header_pattern = r'^\s*([a-z]\.)\s(.*?):$'
        content_pattern = r'^\s*-\s(.+)$'
        # Initialize variables to store extracted data
        data = {}
        current_main_header = ""
        current_sub_header = ""

        # Split the sample_text into lines
        lines = f"{text}".strip().split('\n')

        # Iterate through the lines and extract data
        for line in lines:
            main_header_match = re.match(main_header_pattern, line)
            sub_header_match = re.match(sub_header_pattern, line)
            content_match = re.match(content_pattern, line)
            
            if main_header_match:
                current_main_header = main_header_match.group(1)
                data[current_main_header] = {}
            elif sub_header_match:
                current_sub_header = sub_header_match.group(2)
                if current_sub_header and current_main_header:
                    data[current_main_header][current_sub_header] = []
            elif content_match and (current_main_header in data
                                    and current_sub_header in
                                    data[current_main_header]):
                data[current_main_header][current_sub_header].append(
                    content_match.group(1))
        return data

    def job_analysis_extractor(self, text):
        '''job list extractors'''
        # a regular exp pattern to match numbered list items along with headers
        pattern = r'(?P<number>\d+)\.\s(?P<header>[^:]+):\s(?P<items>.+)'
        # Use re.finditer to find all matching sections
        matches = re.finditer(pattern, text)
        data = {}
        # Iterate through the matched sections and extract data
        for match in matches:
            header = match.group('header').strip()  # Extract the header
            items = match.group('items').strip().split(', ')
            data[header] = items
        return data

    def extract_and_format_data(self, text):
        """extract dataset"""
        # Initialize a dictionary to store extracted data
        extracted_data = {}
        # Split the text into lines
        lines = text.split('\n')

        # Initialize variables to track the current section and data
        current_section = None
        current_data = []
        # Iterate through each line and extract data
        for line in lines:
            # Check if the line matches a header pattern
            header_match = re.match(r'^([A-Z][a-zA-Z\s&]+):$', line.strip())
            if header_match:
                # If a new header is found, store the previous data (if any)
                if current_section and current_data:
                    extracted_data[current_section] = '\n'.join(current_data)
                    current_data = []

                # Extract the header text (without the colon)
                current_section = header_match.group(1)
            else:
                # Store the line as data under the current section
                if current_section:
                    current_data.append(line.strip())

        # Store the last section's data (if any)
        if current_section and current_data:
            extracted_data[current_section] = '\n'.join(current_data)
        return extracted_data

    def reduce_token_length(self, text, max_tokens=2000):
        """
        Truncate the input text to the specified maximum number of tokens.
        """
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
        else:
            return ' '.join(tokens[:max_tokens])

    def get_ai_data(self, company_data, json_data):
        ''' a llm function to get data base on some content'''
        chained_llm = llm.LLMChain(
            llm=self.llm_ai,
            prompt=analyst_template
        )
        company_text = ' '.join(company_data)
        # reduce the number of token
        truncated_data = self.reduce_token_length(company_text, 500)
        input_data = {
            'company_data': truncated_data,
            'company_json': json_data,
        }
        # force result generation for a number of loop
        max_iterations = 10
        i = 0
        analysis_result = None
        while i < max_iterations and analysis_result is None:
            analysis_result = chained_llm.run(input_data)
            i += 1
        analysis_result = self.analysis_extractor_main(analysis_result)
        # print(analysis_result)
        # print("="*10, 'analysis data', "="*10,)
        # analyze the job description
        campaign_llm = llm.LLMChain(
            llm=self.llm_ai,
            prompt=campaign_template
        )
        campaign_data = None
        x = 0
        while x < max_iterations and campaign_data is None:
            campaign_data = campaign_llm.run(input_data)
            x += 1
        campaign_result = self.job_analysis_extractor(campaign_data)
        # print(campaign_result)
        # print("="*10, 'campaign data', "="*10,)
        return analysis_result, campaign_result

    def email_generator(
            self,
            person_json_data,
            company_data):
        ''' an email generator language model'''
        chained_llm = llm.LLMChain(
            llm=self.llm_ai,
            prompt=email_prompt
        )
        input_data = {
            'company_data': company_data,
            'person_json_data': person_json_data
        }
        i = 0
        data = None
        # try to generate data for some number of time if data is empty
        max_iterations = 10
        while i < max_iterations and data is None:
            data = chained_llm.run(input_data)
            i += 1
        return data
