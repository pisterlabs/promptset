import hashlib
import os
import subprocess
import sys
import pefile 
import magic
import requests
import oletools.oleid
from oletools.olevba import VBA_Parser, TYPE_OLE, TYPE_OpenXML, TYPE_Word2003_XML, TYPE_MHTML
import openai
from termcolor import colored  # You'll need to install termcolor: pip install termcolor
import math
import hashlib




SUSPICIOUS_PE_IMPORTS = [
    "VirtualAlloc", "VirtualProtect", "WriteProcessMemory", "ReadProcessMemory",
    "VirtualFree", "LoadLibrary", "GetProcAddress", "LdrLoadDll", "CreateFile",
    "WriteFile", "ReadFile", "DeleteFile", "CreateProcess", "OpenProcess",
    "TerminateProcess", "InjectThread", "WSASocket", "connect", "send", "recv",
    "InternetOpen", "InternetOpenUrl", "RegOpenKey", "RegSetValue", "RegCreateKey",
    "RegDeleteKey", "SetWindowsHookEx", "GetKeyState", "IsDebuggerPresent",
    "CheckRemoteDebuggerPresent", "OpenService", "StartService", "CreateService", "DeleteService"
]

# prompt_list = [
#         {"role": "user", "content": "Today you are going to be a malware anaylst that is going to investigate some information that was obtained from some malicious files."},
#         {"role": "user", "content": f"Please provide details about the av results? {av_result}"},
#         {"role": "user", "content": f"Please provide details about the strings in the following list that was pulled from the malicious file? {file_strings}"},
#         {"role": "user", "content": "Are there any strings that stick out or are suspicious? If so, what are they?"},
#         {"role": "user", "content": f"Pleaes analaze the following function, symbol or system calls that were pulled from the file? {suspicious_api_system_calls}"},
#         {"role": "user", "content": "Are there any function, symbol or system call that stick out or are suspicious? If so, what are they?"},
#         {"role": "user","content": "What else would be helpful to understand from this analyse from this file?"},
#         {"role": "user","content": "Can you generate a Yara configuration that would help identify and classify this malware?"}

#     ]

# ChatGPT API using OpenAI Python SDK
def get_details_from_chatgpt(prompt_list, file_strings="None", suspicious_api_system_calls="None", av_result="None"):
    
    openai.api_key = os.getenv("OPENAI_API_KEY")

    message_prompts = []

    for p in prompt_list:
       
        message_prompts.append(p)
        
        completion = openai.ChatCompletion.create(

            model="gpt-3.5-turbo-16k",
            messages=message_prompts,
            temperature = 0
            
        )

        #print(completion.choices[0].message)
        message_prompts.append(completion.choices[0].message)

    return message_prompts

# Submit the file to VirusTotal
def get_virus_total_report(file_path):
    # Submit the file to VirusTotal
        vt_api_key = os.getenv('VT_API_KEY')
        vt_report = None

        if vt_api_key:
            url = "https://www.virustotal.com/api/v3/files"
            headers = {
                "x-apikey": vt_api_key,
                "accept": "application/json",
            }
            files = {'file': (file_path, open(file_path, 'rb'))}
            try:
                response = requests.post(url, headers=headers, files=files)
                if response.status_code == 200:
                    data = response.json()
                    resource_id = data['data']['id']
                    vt_report = f"VirusTotal Report: https://www.virustotal.com/gui/file/{resource_id}"
            except Exception as e:
                print(f"Error submitting file to VirusTotal: {e}")
        return vt_report

# Identify if the file is an office file
def is_office_file(file_path):
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)
    print(file_type)
    return "ms-office" in file_type

# Improved VirusTotal reporting with colored outputs
def get_virus_total_information(hashes):

    i = 0
    av_result = []

    api_key = os.getenv('VT_API_KEY')

    for key, value in hashes.items():
        if key == "SHA256":
            url = f'https://www.virustotal.com/api/v3/files/{value}'
            headers = {'Accept': 'application/json', 'x-apikey': api_key}
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                json_response = response.json()
            
                # Display results with color for better clarity
                av_result.append(f"Magic - " + str(json_response['data']['attributes']['magic']))
                malicious = json_response['data']['attributes']['last_analysis_stats']['malicious']
                if malicious > 0:
                    av_result.append(f"Malicious Results: {malicious}")
                else:
                    av_result.append(f"Malicious Results: {malicious}")

                last_analysis_results = json_response['data']['attributes']['last_analysis_results']
            

               # Get the top 5 AV results
               #print(colored("Top 5 AV Results:", 'red'))

                for key, value in last_analysis_results.items():
                    if i <= 5:
                        if value['category'] == 'malicious' or value['category'] == 'suspicious':
                           av_result.append(f"{key}: {value['result']}")
                        #av_result.append(value['result'])
                        i += 1
            
            else:
                print("Error: " + str(response.status_code))

    return av_result
        
# identify file type
def identify_file_type(file_path):
    try:
        with open(file_path, 'rb') as file:
            # Read the first few bytes (commonly 8-16 bytes)
            file_signature = file.read(16)

        # Check for specific magic bytes or patterns
        if file_signature.startswith(b'\x4D\x5A'):  # MZ header for Windows PE executable
            return "Windows PE Executable"
        elif file_signature.startswith(b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1'):  # OLE file
            return "OLE Document (e.g., MS Office)"
        elif file_signature.startswith(b'\x50\x4B\x03\x04'):  # Zip archive (common in MS Office documents)
            return "MS Office Document (ZIP format)"
        elif file_signature.startswith(b'\x3C\x3F\x78\x6D\x6C'):  # XML or HTML file (common in VBS)
            return "XML/HTML File (e.g., VBS Script)"
        else:
            try:
                file_extension = os.path.splitext(file_path)[1].lower()

                file_type_mapping = {
                    ".vbs": "VBScript",
                }

                return file_type_mapping[file_extension]
            except Exception as e:
                print(f"Error identifying file type: {e}")
                return "Unknown File Type"

    except Exception as e:
        print(f"Error identifying file type: {e}")

# Analyze VBA Macros
def analyze_vba(file_path):
    vba_results = []
    vbaparser = VBA_Parser(file_path)
    vba_macro_code = []
    analysis = []
    prompt = []
    segment = 0
    i =0

    if vbaparser.detect_vba_macros():
        vba_results.append("VBA Macros Detected")

        for vba_code in vbaparser.extract_macros():

            # Get the VBA code analysis
            if i == 0:
                analysis = vbaparser.analyze_macros()
                vba_results.append(analysis)
                i += 1

            vba_macro_code.append(vba_code)

    else:

        vba_results.append("No VBA Macros Detected")

    #Gets the VBA code analysis from ChatGPT
    for code_segments in vba_macro_code:
        if segment == 0:

            segment = 1
            prompt.append({"role": "user", "content": f"VBA Code Segment {segment}: {code_segments} VBA Code Analysis:"})
            segment += 1
        
        else:
            
            prompt.append({"role": "user", "content": f"VBA Code Segment {segment}: {code_segments} VBA Code Analysis:"})
    
    prompt.append({"role": "user", "content": f"The following is the analysis of the VBA code from OleTools VBA_Parser.AnalyzeMacro.: {analysis}"})
    prompt.append({"role": "user", "content": "If the analysis of the VBA code from the OleTool discovered Base64 encoded strings, please decode them and provide the decoded code."})    
    prompt.append({"role": "user", "content": "Please format the VBA code to be human-readable and then add your comments inline."})
    prompt.append({"role": "user", "content": "What is your final analysis of this code?"})
    prompt.append({"role": "user", "content": "What other information would be helpful for you to analyze this file further?"})
            
    vba_results.append(get_details_from_chatgpt(prompt))

    return vba_results

# Calculate the entropy of a file
def calculate_entropy(data):
    entropy = 0
    
    if data:
        for x in range(256):
            p_x = float(data.count(chr(x))) / len(data)
            if p_x > 0:
                entropy += - p_x * math.log(p_x, 2)
    
    return entropy

# PE Analysis
def pe_analysis(file_path):
    try:
        pe = pefile.PE(file_path)
        pe_info = {
            "ImageBase": hex(pe.OPTIONAL_HEADER.ImageBase),
            "EntryPoint": hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint),
            "NumberOfSections": pe.FILE_HEADER.NumberOfSections,
            "TimeDateStamp": pe.FILE_HEADER.TimeDateStamp,
            "NumberOfSymbols": pe.FILE_HEADER.NumberOfSymbols,
            "Machine": hex(pe.FILE_HEADER.Machine),
            "SizeOfOptionalHeader": pe.FILE_HEADER.SizeOfOptionalHeader,
            "Characteristics": hex(pe.FILE_HEADER.Characteristics),
            "DLL": pe.OPTIONAL_HEADER.DllCharacteristics,
            "Entropy": round(pe.sections[0].get_entropy(),),
        }


        # Check for suspicious imports
        suspicious_imports = []
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                if imp.name:
                    if imp.name.decode('utf-8') in SUSPICIOUS_PE_IMPORTS:
                        suspicious_imports.append(imp.name.decode('utf-8'))

        if suspicious_imports: 
            pe_info["Suspicious Imports"] = suspicious_imports
        


        # iterate through the PE sections and print the section information
        section_info = []
        for section in pe.sections:
            section_name = section.Name.decode('utf-8').rstrip('\x00')
            virtual_size = section.Misc_VirtualSize
            virtual_address = section.VirtualAddress
            raw_size = section.SizeOfRawData
            characteristics = hex(section.Characteristics)
            entropy = round(section.get_entropy(),3)
    
            section_info.append({
                "Name": section_name,
                "VirtualSize": virtual_size,
                "VirtualAddress": virtual_address,
                "RawSize": raw_size,
                "Characteristics": characteristics,
                "Entropy": entropy,
            })

        pe_info["Sections"] = section_info

        # Check for suspicious sections
        suspicious_sections = []
        for section in pe.sections:
            if section.IMAGE_SCN_MEM_WRITE and section.IMAGE_SCN_MEM_EXECUTE:
                suspicious_sections.append(section.Name.decode('utf-8'))
            elif section.IMAGE_SCN_MEM_WRITE and section.IMAGE_SCN_MEM_READ:
                suspicious_sections.append(section.Name.decode('utf-8'))
            elif section.IMAGE_SCN_MEM_EXECUTE and section.IMAGE_SCN_MEM_READ:
                suspicious_sections.append(section.Name.decode('utf-8'))
            elif section.IMAGE_SCN_MEM_WRITE and section.IMAGE_SCN_MEM_READ and section.IMAGE_SCN_MEM_EXECUTE:
                suspicious_sections.append(section.Name.decode('utf-8'))

        if suspicious_sections:
            pe_info["Suspicious Sections"] = suspicious_sections

        return pe_info
    
    except Exception as e:
        print(f"Error reading PE: {e}")
        return None



# Calculate md5, sha1, sha256 hashes of a file
# Hashing
def get_file_hash(filepath):
    
    BLOCK_SIZE = 65536
    
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1()
    md5 = hashlib.md5()

    
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(BLOCK_SIZE)
            if not data:
                break
            md5.update(data)
            sha1.update(data)
            sha256.update(data)

    return {
        'MD': md5.hexdigest(),
        'SHA1': sha1.hexdigest(),
        'SHA256': sha256.hexdigest()
    }


# Packer Detection
def detect_packer(file_path):
    
    try:
         pe = pefile.PE(file_path)

    except Exception as e:
        print(f"Error Reading PE: {e}")
        return 0

    suspicious_packer = ['ASPack', 'ASProtect', 'PECompact', 'PELock', 'PESpin', 'UPX', 'VMProtect', 'WinRAR', 'WinZip']

    for section in pe.sections:
        for s in suspicious_packer:
            if s in section.Name.decode('utf-8'):
                return s
    return None


def static_analysis(file_path, min_length=4):
    try:

        static_info = {}

        # Identify the file type
        file_type = identify_file_type(file_path)


        if file_type == "Windows PE Executable":

            static_info = {
                "File Type": file_type,
                "PE Analysis": pe_analysis(file_path),
            }
            
            # Calculate hashes
            hashes = get_file_hash(file_path)

            static_info["File Hashes"] = hashes

            # Use the 'file' command to identify the file type
            magic_instance = magic.Magic()
            file_type = magic_instance.from_file(file_path)

       
            # Extract static strings from the binary
            str_result = []
            with open(file_path, 'rb') as file:
                current_string = b""
                while True:
                    byte = file.read(1)
                    if byte == b"":
                        break
                    if 32 <= ord(byte) < 126:
                        current_string += byte
                    else:
                        if len(current_string) > min_length:
                            str_result.append(current_string.decode(errors='replace'))
                        current_string = b""
            
                static_info["Strings"] = str_result

            # Detect packer
            packer = detect_packer(file_path)

            if packer:
                static_info["Packer"] = packer
            else:
                static_info["Packer"] = "None"

            # Checking virus total for posture of the file
            av_result = get_virus_total_information(hashes)

            static_info["AV Results"] = av_result

        elif file_type == "OLE Document (e.g., MS Office)":
            vba_results = analyze_vba(file_path)

            # Calculate hashes
            hashes = get_file_hash(file_path)

            # Checking virus total for posture of the file
            av_result = get_virus_total_information(hashes)

            # Checking virus total for posture of the file
            av_report = get_virus_total_report(file_path)
             
            
            static_info = {
                "File Type": file_type,
                "File Hashes": hashes,
                "AV Results": av_result,
                "VBA Analysis": vba_results,
            }

        elif file_type == "MS Office Document (ZIP format)":
            vba_results = analyze_vba(file_path)

            # Calculate hashes
            hashes = get_file_hash(file_path)
           
            # Checking virus total for posture of the file
            av_result = get_virus_total_information(hashes)

            # Checking virus total for posture of the file
            av_report = get_virus_total_report(file_path)

            static_info = {
               "File Type": file_type,
               "File Hashes": hashes,
               "AV Results": av_result,
               "Total Virus Report": av_report,
               "VBA Analysis": vba_results,
            }

        elif file_type == "VBScript":
            static_info = {
                "File Type": file_type,
            }
        else:
            static_info = {
                "File Type": "Unknown File Type",
            } 

        return static_info

    except Exception as e:
        print(f"Static analysis error: {str(e)}")
        return None



def main():

    hash = {}
    suspicious_found = []
    strings_found = []

    if len(sys.argv) != 2:
        print("Usage: python3 inspector.py <filename>")
        sys.exit(1)

    file_path = sys.argv[1]

    print(colored(f"Analyzing {file_path}...","blue"))

    print(f"Static Analysis Results:\n")
    for key, value in static_analysis(file_path).items():
        if key == "PE Analysis":
            print("PE Analysis: ")
            for k, v in value.items():
                print(f"     {k}: {v}")
        elif key == "File Hashes":
            print("File Hashes: ")
            for k, v in value.items():
                print(f"     {k}: {v}")
                hash[k] = v
        elif key == "AV Results":
            print(colored("AV Results: ", "red"))
            print(colored(f"{value[0]}", "green"))
            print(colored(f"{value[1]}", "red"))
            for i in range(2, len(value)):
                print(colored(f"     {value[i]}", "white"))
        elif key == "Total Virus Report":
            print(colored("Total Virus Report: ", "green"))

        elif key == "VBA Analysis":
            print(colored("VBA Analysis:","blue"))
            if value[0] == "VBA Macros Detected":
                print(colored(f"Macro was detected in the file: {file_path}", "red"))
            print("VBA Code Analysis from OleTools VBA_Parser.AnalyzeMacro: ")
            for messages in value[1]:
                print(messages)
            
            print()
            print(colored("Chat GPT Analysis of VBA Code: ", "blue"))
           
            for message in value[2]:
                role = message['role']
                content = message['content']
                #Process the extracted information as needed
                print(colored(f"Role: {role}", "green"))
                print(colored(f"Content: {content}","white"))
                print()
            
                
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()