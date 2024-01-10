import json
import configparser
import openai
import os
BYPASS = False
def askForKeys():
    gpt_key = input("Please enter your GPT key, you can get it at https://platform.openai.com/account/api-keys \n\
    Gpt key: ")
    elevenlab_key = input("Please enter your ElevenLab key, you can get it at https://beta.elevenlabs.io/subscription (if you don't have one, just press Enter)\n\
    Elevenlab key: ")
    if not elevenlab_key:
        elevenlab_key = 'null'

    # Now that we have the keys, write them into the file
    with open(file_path, 'w') as file:
        file.write('[API_KEYS]\n')  # add this line
        file.write('GPT_KEY=' + gpt_key + '\n')
        file.write('ELEVENLAB_KEY=' + elevenlab_key + '\n')

def get_keys(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    
    gpt_key = config.get('API_KEYS', 'GPT_KEY', fallback=None)
    elevenlab_key = config.get('API_KEYS', 'ELEVENLAB_KEY', fallback=None)
    
    return gpt_key, elevenlab_key
def checkGPTKeyValidity(gpt_key):
    openai.api_key = gpt_key
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt="This is a test.",
            max_tokens=5
        )
    except:
        return False
    else:
        return True
def checkForAvailableASRModels():
    availablemodel = []

    return ["whisper-online","whisper","lee-ueno"]
def checkForAvailableTTSModels():
    availablemodel = []
    return ["elevenlabs", "gtts","pytts",]
def checkForAvailableLLMModels():
    availablemodel = []
    return ["gpt-3.5-turbo-0613"]

def select_model(model_list, model_type):
    print(f"Available {model_type} models:")
    for i, model in enumerate(model_list, 1):
        print(f"{i}. {model}")
    while True:
        try:
            selected = int(input(f"Select a {model_type} model by entering its number: ")) - 1
            if 0 <= selected < len(model_list):
                return selected
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
def CreateNewAIConf():
    ASRmodelList = checkForAvailableASRModels()
    TTSmodelList = checkForAvailableTTSModels()
    LLMmodelList = checkForAvailableLLMModels()

    ASRmodelSelected = select_model(ASRmodelList, "ASR")
    TTSmodelSelected = select_model(TTSmodelList, "TTS")
    LLMmodelSelected = select_model(LLMmodelList, "LLM")

    config = {
    "ASR_model": ASRmodelList[ASRmodelSelected],
    "TTS_model": TTSmodelList[TTSmodelSelected],
    "LLM_model": LLMmodelList[LLMmodelSelected],
    }
    
    # Create a path to the new config file
    config_file_path = os.path.join(local_dir, "config.json")

    # Write the config to the file
    with open(config_file_path, 'w') as config_file:
        json.dump(config, config_file)
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
    return config
    
def load_config(local_dir, bypass = False):
    # Create a path to the config file
    config_file_path = os.path.join(local_dir, "config.json")

    # Check if the config file exists
    if os.path.exists(config_file_path):
        # Load the config from the file
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)

        # Print the current configuration
        print("Current configuration:")
        for key, value in config.items():
            print(f"{key}: {value}")

        # Ask the user if they want to use the current configuration
        if not bypass:
            choice = input("Do you want to use the current configuration? (y/n) ")
        else: 
            choice = "y"
        if choice.lower() == 'y':
            return config
        elif choice.lower() == 'n':
            CreateNewAIConf()
            return load_config(local_dir)
        else:
            print("Invalid choice. Please enter y or n.")
            return load_config(local_dir)

    else:
        print(f"No config file found at {config_file_path}")
        CreateNewAIConf()
        return load_config(local_dir)

if __name__ == "__main__":
    local_dir = "localdir"
    file_name = "api_keys.txt"
    # Check if directory exists
    if not os.path.isdir(local_dir):
        # If not, create the directory
        os.makedirs(local_dir)
    file_path = os.path.join(local_dir, file_name)
    if not os.path.isfile(file_path):
        askForKeys()
    valid_key = False
    while valid_key == False:
        gpt_key, elevenlab_key = get_keys(file_path)
        if not checkGPTKeyValidity(gpt_key):
            print("Invalid GPT key, please make sure the key is valid (There is not check on elevenlabs key yet, be carefull too")
            askForKeys()
        else:
            valid_key = True

    config = load_config(local_dir, BYPASS)

    # Extract the models from the configuration
    ASRmodelList = ASRmodelList = checkForAvailableASRModels()
    TTSmodelList = TTSmodelList = checkForAvailableTTSModels()
    LLMmodelList = LLMmodelList = checkForAvailableLLMModels()
    try:
        # Extract the selected model indices from the configuration
        ASRmodelSelected = ASRmodelList.index(config.get('ASR_model'))
        TTSmodelSelected = TTSmodelList.index(config.get('TTS_model'))
        LLMmodelSelected = LLMmodelList.index(config.get('LLM_model'))
    except IndexError:
        CreateNewAIConf()

    from system import System
    system = System(ASRmodelList, TTSmodelList, LLMmodelList, ASRmodelSelected , TTSmodelSelected , LLMmodelSelected, gpt_key = gpt_key, elevenlab_key = elevenlab_key)
    
    from IHM import TalkingIndicator
    indicator = TalkingIndicator(system)

    import threading
    system_thread = threading.Thread(target=system.run)
    system_thread.start()
    indicator.run()