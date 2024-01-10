import sys
from application import *
import openai
import os
from util import *

if len(sys.argv) > 1:
    if "OPENAI_API_KEY" not in os.environ:
        raise Exception("OPENAI_API_KEY environment variable not set")
    if "AWS_ACCESS_KEY" not in os.environ:
        raise Exception("AWS_ACCESS_KEY environment variable not set")
    if "AWS_SECRET_KEY" not in os.environ:
        raise Exception("AWS_SECRET_KEY environment variable not set")
    
    if sys.argv[1] == "--delete":
        delete_persistent_variable("OPENAI_API_KEY")
        delete_persistent_variable("AWS_ACCESS_KEY")
        delete_persistent_variable("AWS_SECRET_KEY")
        print("All keys removed from system environment variables")
        sys.exit(0)
        
    
    openai.api_key = os.environ["OPENAI_API_KEY"]

    prompt = " ".join(sys.argv[1:]) 
    run(prompt) #TODO: this is too much code change it later

else:
    set_variable_if_not_exists("OPENAI_API_KEY")
    set_variable_if_not_exists("AWS_ACCESS_KEY")
    set_variable_if_not_exists("AWS_SECRET_KEY") 

    print("All keys ready. Use ./ChatSynthesis.py <prompt> to generate text for speech synthesis.")

