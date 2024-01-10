import os
from models import DeviceEnum, Wish
from channel.langchain.LangChainWrapper import LangChainWrapper
from channel.llamaindex.LlamaIndexWrapper import LlamaIndexWrapper
# from channel.vllm.VllmWrapper import VllmWrapper
# from channel.vmwarevllmapi.VmwareVllmApiWrapper import VmwareVllmApiWrapper
import time
import argparse

from common.config import setup_logging_from_args, setup_logging_from_file
from common.logging_decorator import log_entry_exit, SensitiveData

@log_entry_exit()
def process_wish(wish: Wish):
    if wish.integrator == "Langchain":
        langChainWrapper = LangChainWrapper(wish)
        grant = langChainWrapper.run()
        return grant
    elif wish.integrator == "Llamaindex":
        llamaIndexWrapper = LlamaIndexWrapper(wish)
        grant = llamaIndexWrapper.run()
        return grant
    # elif wish.integrator == "VmwareVllmApi":
        # vmwareVllmApiWrapper = VmwareVllmApiWrapper(wish)
        # grant = vmwareVllmApiWrapper.run()
        # return grant
    else:
        raise Exception("Channel not supported")

@log_entry_exit()
def create_directory_structure():
    # Get the directory of the current script
    root_path = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path for the new directory structure
    # new_dir_path = os.path.join(script_dir, "vector-store/langchain/faiss")

    # Create the directory structure if it doesn't exist
    # os.makedirs(new_dir_path, exist_ok=True)

    return root_path

def main(args):
    """The main function."""
    if args.config_file:
        setup_logging_from_file(args.config_file)
    else:
        setup_logging_from_args(args.log_level)

    # Start time
    start_time = time.time()
    wish = Wish(rootPath=create_directory_structure(), device=args.device, modelLocation=args.modelLocation, documentName=args.documentName, modelName=args.modelName, integrator=args.integrator, inferenceServer=args.inferenceServer, vectorDatabase=args.vectorDatabase ,whisper=args.whisper)
    grant = process_wish(wish)
    print(grant)
    # End time
    end_time = time.time()

    # Total time taken
    totalTimeTaken = (end_time - start_time) * 1000
    print()
    print(f"Runtime of the program is {totalTimeTaken}")
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log_level", type=str, default="INFO", 
        help="Set the logging level (e.g., DEBUG, INFO, WARNING)"
    )

    parser.add_argument(
        "--config_file", type=str, default=None,
        help="Path to an alternate configuration file"
    )

    parser.add_argument(
        "--device", type=str, default="cpu", 
        help="Device to run the model on."
    )
    
    parser.add_argument(
        "--modelLocation", type=str, default="local", 
        help="Location of the model."
    )

    parser.add_argument(
        "--documentName", type=str, default="Business Conduct.pdf", 
        help="Name of the document."
    )

    parser.add_argument(
        "--modelName", type=str, default="Llama", 
        help="Name of the model."
    )

    parser.add_argument(
        "--integrator", type=str, default="Langchain", 
        help="Name of the integrator."
    )

    parser.add_argument(
        "--inferenceServer", type=str, default="LocalFile", 
        help="Name of the inference server."
    )

    parser.add_argument(
        "--vectorDatabase", type=str, default="FAISS", 
        help="Name of the vector database."
    )

    parser.add_argument(
        "--whisper", type=str, default="what is Legal Holds?", 
        help="The question."
    )

    args = parser.parse_args()
    main(args)    
