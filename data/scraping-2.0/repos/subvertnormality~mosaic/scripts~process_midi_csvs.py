# This is designed to create tables of midi devices for use in mosaic. The scipt takes midi definition CSVs 
# from https://github.com/usercamp/midi/ and outputs a table file. It uses chatgpt to create two four letter tokens 
# designed to be used as identifiers for controls. The output needs to be edited manually as it's not perfect.


import os
import csv
import json
import openai
import os
import csv
import json
import time

# Set your open AI org to enviroment variable OPENAI_API_ORG
openai.organization = os.getenv("OPENAI_API_ORG")
# Set your open AI API key to enviroment variable OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_short_descriptor(full_string, retries=10, delay=5):
    for i in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                temperature=0.5,
                messages=[
                    {"role": "system", "content": "You are an assistant designed to abbreviate a given input that represent concepts in modern synthesizers into two four letter uppercase tokens seperated by a '-'. Only ever respond in the format 'XXXX-XXXX'. Always respond with 9 characters in total. Fill in blanks with spaces ' '. Never return anything else. Please optimise for readability and understanding of a human musician using a synthesizer where these abbreviations are the only information they have to understand a parameter does. Use your understanding of synthesisers to appraise what is the most important information in the input. Here are some examples. 'FM parameters syn 1 Ratio A' would be 'SYN1-RATA'. 'FM parameters syn 2 A level' would be 'SYN2-ALVL'. 'Filter parameters filter type' would be 'FLTR-TYPE'. 'Amp parameters chorus send' would be 'CHOR-SEND'. 'Apm parameters amp reverb send' would be 'REVB-SEND'. 'LFO parameters LFO 2 trig mode' would be 'LFO2-TRGM'. 'FX parameters reverb highpass filter' would be 'RVRB-HPFL'. 'Amp parameters Amp delay send' would be 'DLAY-SEND'. 'Amp parameters Amp reverb send' would be 'REVB-SEND'. "},
                    {"role": "user", "content": full_string}
                ]
            )
            result = response['choices'][0]['message']['content'].strip()
            print(f"Result: {result}")
            return result[:9]
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            time.sleep(delay)
    print("Failed to get result after multiple attempts.")
    return None

def to_lua_table(input_dict):
    lua_table = "{\n"
    for key, value in input_dict.items():
        if isinstance(value, str):
            if value == "nil":
                lua_table += f'  ["{key}"] = nil,\n'
            else:
                lua_table += f'  ["{key}"] = "{value}",\n'
        else:
            lua_table += f'  ["{key}"] = {value},\n'
    lua_table += "}"
    return lua_table

def create_lua_table(device, data):
    lua_list = "\n"
    for item in data:
        lua_list += f'{to_lua_table(item)},\n'
    lua_list += ""
    lua_table = "[\"" + f'{device}\"] = {{{lua_list}\n'
    return lua_table

def process_csv_file(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        processed_data = []
        for row in reader:
            id = "{}_{}".format(row["section"].lower().replace(" ", "_"), row["parameter_name"].lower().replace(" ", "_"))
            name = row["parameter_name"]
            # Call the ChatGPT API to generate the short descriptors
            short_descriptors = generate_short_descriptor(row["section"] + " " + row["parameter_name"])
            short_descriptor_1 = short_descriptors[:4]
            short_descriptor_2 = short_descriptors[-4:]
            processed_data.append({
                "id": id,
                "name": name,
                "cc_msb": int(row["cc_msb"]) if row["cc_msb"] else "nil",
                "cc_lsb": int(row["cc_lsb"]) if row["cc_lsb"] else "nil",
                "cc_min_value": int(row["cc_min_value"]) if row["cc_min_value"] else "nil",
                "cc_max_value": int(row["cc_max_value"]) if row["cc_max_value"] else "nil",
                "nrpn_msb": int(row["nrpn_msb"]) if row["nrpn_msb"] else "nil",
                "nrpn_lsb": int(row["nrpn_lsb"]) if row["nrpn_lsb"] else "nil",
                "nrpn_min_value": int(row["nrpn_min_value"]) if row["nrpn_min_value"] else "nil",
                "nrpn_max_value": int(row["nrpn_max_value"]) if row["nrpn_max_value"] else "nil",
                "short_descriptor_1": short_descriptor_1,
                "short_descriptor_2": short_descriptor_2,
            })
        return processed_data

def main():
    with open('output.lua', 'w') as output_file:
        output_file.write("local midi_devices = {")
        for file in os.listdir():
            if file.endswith(".csv"):
                device = os.path.splitext(file)[0]
                processed_data = process_csv_file(file)
                lua_table = create_lua_table(device, processed_data) + "}"
                output_file.write(lua_table.rstrip('\n').rstrip(',') + ',\n')
        output_file.write("}")

if __name__ == "__main__":
    main()
