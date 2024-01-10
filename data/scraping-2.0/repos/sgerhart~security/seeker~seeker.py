import hashlib
import os
import sys
import requests
import pefile
import elftools
import openai
import pprint

from elftools.elf.elffile import ELFFile
from datetime import datetime
from pwd import getpwuid
from termcolor import colored  # You'll need to install termcolor: pip install termcolor


SUSPICIOUS_PE_IMPORTS = [
    "VirtualAlloc", "VirtualProtect", "WriteProcessMemory", "ReadProcessMemory",
    "VirtualFree", "LoadLibrary", "GetProcAddress", "LdrLoadDll", "CreateFile",
    "WriteFile", "ReadFile", "DeleteFile", "CreateProcess", "OpenProcess",
    "TerminateProcess", "InjectThread", "WSASocket", "connect", "send", "recv",
    "InternetOpen", "InternetOpenUrl", "RegOpenKey", "RegSetValue", "RegCreateKey",
    "RegDeleteKey", "SetWindowsHookEx", "GetKeyState", "IsDebuggerPresent",
    "CheckRemoteDebuggerPresent", "OpenService", "StartService", "CreateService", "DeleteService"
]

SUSPICIOUS_ELF_IMPORTS = [
    "ptrace", "fork", "execve", "open", "read", "write", "kill", 
    "dlopen", "mmap", "mprotect", "socket", "connect", "send", "recv", 
    "system", "popen", "chmod", "chown", "unlink"
]

OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'


# File Metadata extraction
def get_file_info(filepath, min_length=5):
    type = ""
    info = os.stat(filepath)
    
    # Extract embedded strings
    with open(filepath, 'rb') as file:
        str_result = []
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

    # Extract header info
    with open(filepath, 'rb') as file:
        header = file.read(4)

        if header == b'\x7fELF':
            type = "ELF"
        elif header[:2] == b'MZ':
            type = "PE"
        else:
            return "Unknown file type"

    return {
        'file size': str(info.st_size),
        'file owner': getpwuid(info.st_uid).pw_name,
        'file type': type,
        'creation-time': str(datetime.fromtimestamp(info.st_ctime))[:19],
        'modified': str(datetime.fromtimestamp(info.st_mtime))[:19],
        'strings': str_result
    }


# PE File Analysis
def analyze_pe(file_path):
    suspicious_found = []
    pe = pefile.PE(file_path)

    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        for imp in entry.imports:
            if imp.name.decode('utf-8') in SUSPICIOUS_PE_IMPORTS:
                suspicious_found.append(imp.name.decode('utf-8'))
    return suspicious_found

# ELF File Analysis
def analyze_elf(file_path):
    suspicious_found = []
    with open(file_path, 'rb') as f:
        elf_file = ELFFile(f)
        for section in elf_file.iter_sections():
            if isinstance(section, elftools.elf.sections.SymbolTableSection):
                for symbol in section.iter_symbols():
                    if symbol.name in SUSPICIOUS_ELF_IMPORTS:
                        suspicious_found.append(symbol.name)
    return suspicious_found


# Improved VirusTotal reporting with colored outputs
def get_virustotal_report(api_key, hashes):

    i = 0
    av_result = []

    for key, value in hashes.items():
        url = f'https://www.virustotal.com/api/v3/files/{value}'
        headers = {'Accept': 'application/json', 'x-apikey': api_key}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            json_response = response.json()
            
            # Display results with color for better clarity
            print(colored(f"Magic - " + str(json_response['data']['attributes']['magic']), 'white'))
            malicious = json_response['data']['attributes']['last_analysis_stats']['malicious']
            if malicious > 0:
                print(colored(f"Malicious Results: {malicious}", 'red'))
            else:
                print(colored(f"Malicious Results: {malicious}", 'green'))

            last_analysis_results = json_response['data']['attributes']['last_analysis_results']
            print()

            # Get the top 5 AV results
            print(colored("Top 5 AV Results:", 'red'))

            for key, value in last_analysis_results.items():
                if i <= 5:
                    if value['category'] == 'malicious' or value['category'] == 'suspicious':
                        print(f"{key}: {value['result']}")
                        av_result.append(value['result'])
                    i += 1
            
        else:
            print("Error: " + str(response.status_code))

    return av_result
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
def detect_packer(filepath):
    
    try:
         pe = pefile.PE(filepath)

    except Exception as e:
        print(f"Error Reading PE: {e}")
        return 0

    suspicious_packer = ['ASPack', 'ASProtect', 'PECompact', 'PELock', 'PESpin', 'UPX', 'VMProtect', 'WinRAR', 'WinZip']

    for section in pe.sections:
        for s in suspicious_packer:
            if s in section.Name.decode('utf-8'):
                return s
    return None


# ChatpGPT API using requests library (Transitioned to OpenAI API)
""" def get_detail_from_chatgpt(api_key, strings, suspicious_api_system_calls):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    data = {
        'model': 'gpt-3.5-turbo',
        'messages':[{
        'role': 'user',
        'content': f'Provide details about the function or symbol: {string}'
        }],
        'max_tokens': 150
        }

    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        print(response.json()['choices'][0]['message']['content'])
        #return response.json()["choices"][0]["text"].strip()
    else:
        return f"Error: {response.text}" """

# ChatGPT API using OpenAI Python SDK
def get_detail_from_chatgpt(api_key, file_strings, suspicious_api_system_calls, av_result):

    prompt_list = [
        {"role": "user", "content": "Today you are going to be a malware anaylst that is going to investigate some information that was obtained from some malicious files."},
        {"role": "user", "content": f"Please provide details about the av results? {av_result}"},
        {"role": "user", "content": f"Please provide details about the strings in the following list that was pulled from the malicious file? {file_strings}"},
        {"role": "user", "content": "Are there any strings that stick out or are suspicious? If so, what are they?"},
        {"role": "user", "content": f"Pleaes analaze the following function, symbol or system calls that were pulled from the file? {suspicious_api_system_calls}"},
        {"role": "user", "content": "Are there any function, symbol or system call that stick out or are suspicious? If so, what are they?"},
        {"role": "user","content": "What else would be helpful to understand from this analyse from this file?"},
        {"role": "user","content": "Can you generate a Yara configuration that would help identify and classify this malware?"}

    ]

    openai.api_key = api_key

    message_prompts = []

    for p in prompt_list:
        message_prompts.append(p)
        #print(message_prompts)
        completion = openai.ChatCompletion.create(

            model="gpt-3.5-turbo-16k",
            messages=message_prompts,
            temperature = 0
            
        )

        #print(completion.choices[0].message)
        message_prompts.append(completion.choices[0].message)

    return message_prompts


def main():

    hash = {}
    suspicious_found = []
    strings_found = []
    

    # Get API Keys from environment variables (export API_KEY=xxxxx)
    total_virus_key = os.environ.get('VT_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')


    if len(sys.argv) != 2:
        print("Usage: python3 test.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    # Gather file info and display
    print(colored("File Information: ", 'blue') )
    file_info = get_file_info(filename)
    for key, value in file_info.items():
        if key != 'strings':
            print(f"{key}: {value}")
        else:
             print(f"The following Strings were found:")
             for s in value:
                 strings_found.append(s)
             print(strings_found)
    print()
   
    # PE and ELF Analysis
    if file_info['file type'] == "PE":
        suspicious_found = analyze_pe(filename)
        print(colored("Anaylzing PE File", 'blue'))
        if suspicious_found:
            print(f"The following API calls where found: {suspicious_found}")
            print()
    elif file_info['file type'] == "ELF" :
        suspicious_found = analyze_elf(filename)
        print(colored("Anaylzing ELF File", 'blue'))
        if suspicious_found:
            print(f"The following System Calls and Functions where found: {suspicious_found}")
            print()

    

    
    # Gather file hashes and display
    print(colored("File Hashes: ", 'blue') )
    for key, value in get_file_hash(filename).items():
        print(key + ": " + value)
        if key == 'SHA256':
            hash[key] = value
    print()


    # Check if file is packed
    print(colored("Packer Detection: ", 'blue') )
    if file_info['file type'] != 'ELF':
        packer = detect_packer(filename)
        if packer != 0:
            print(colored(f"File is packed with {packer}", 'red'))
        else:
            print(colored("File is not packed", 'green'))
    else:
        print(colored("File is not packed", 'green'))
    print()


    # VirusTotal report
    print(colored("VirusTotal Report: ", 'blue') )
    av_result = get_virustotal_report(total_virus_key, hash)
    print()
    
    # ChatGPT API
    print(colored("ChatGPT API: ", 'blue'))

    results = get_detail_from_chatgpt(openai_key, strings_found, suspicious_found, av_result)
    print(len(results))

    for message in results:
        role = message['role']
        content = message['content']

        if role == 'user':
            print(colored(f"User: {content}", 'green'))
            print()
        if role == 'assistant':
            print(colored(f"Assitant: {content}", 'white'))
            print()


if __name__ == "__main__":
    main()