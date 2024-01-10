import os
from models.AutoModel import AutoModel
from models.OpenAIChatModel import OpenAIChatModel
from models.OpenAIModel import OpenAIModel
from models.PaLMModel import PaLMModel
from models.TextGenerationModel import TextGenerationModel

def load_openai_model(args):
    # If there exists a ".openai_api_key" file, use that as the API key.
    if os.path.exists(".openai_api_key"):
        with open(".openai_api_key", "r") as f:
            openai_api_key = f.read().strip()
    else:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    assert len(openai_api_key) > 0, (
        "OpenAI API key not found. "
        + "Either create a '.openai_api_key' file or "
        + "set the OPENAI_API_KEY environment variable."
    )
    if args.model_type == "openai":
        model = OpenAIModel(model=args.model_name, api_key=openai_api_key)
    elif args.model_type == "openai-chat":
        model = OpenAIChatModel(model=args.model_name, api_key=openai_api_key)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    return model

def load_palm_model(args):
    # If there exists a ".palm_api_key" file, use that as the API key.
    if os.path.exists(".palm_api_key"):
        with open(".palm_api_key", "r") as f:
            palm_api_key = f.read().strip()
            print("palm_api_key", palm_api_key)
    else:
        palm_api_key = os.getenv("PALM_API_KEY")
    assert len(palm_api_key) > 0, (
        "PaLM API key not found. "
        + "Either create a '.palm_api_key' file or "
        + "set the PALM_API_KEY environment variable."
    )
    model = PaLMModel(model=args.model_name, api_key=palm_api_key)
    return model

def load_automodel(args):
    model = AutoModel(batch_size=1, path=args.model_name)
    return model

def load_text_generation_model(args):
    model = TextGenerationModel(args.tgi_server_url, args.max_workers)
    return model

def load_model(args):
    if "openai" in args.model_type:
        model = load_openai_model(args)
    elif args.model_type == "palm":
        model = load_palm_model(args)
    elif args.model_type == "automodel":
        model = load_automodel(args)
    elif args.model_type == "hf-textgen":
        model = load_text_generation_model(args)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    return model
