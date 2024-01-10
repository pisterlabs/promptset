import openai

HAS_PALM = True
# if the palm package is not installed, ignore it
try:
    import google.generativeai as palm
except ImportError:
    HAS_PALM = False
    pass


def init_openai_key():
    # read openAI key from txt file, if file is not found, prompt for key
    try:
        with open('openai.key', 'r') as f:
            openai_key = f.read()
    except FileNotFoundError:
        openai_key = input("Please enter your openAI key: ")
        with open('openai.key', 'w') as f:
            f.write(openai_key)

    # set openAI key
    openai.api_key = openai_key


def init_palm_key():
    # read palm key from txt file, if file is not found, prompt for key
    try:
        with open('palm.key', 'r') as f:
            palm_key = f.read()
    except FileNotFoundError:
        palm_key = input("Please enter your palm key: ")
        with open('palm.key', 'w') as f:
            f.write(palm_key)

    # set palm key
    palm.configure(api_key=palm_key)

    # for model in palm.list_models():
    #     print(model)


def init_keys():
    init_openai_key()
    if HAS_PALM:
        init_palm_key()
