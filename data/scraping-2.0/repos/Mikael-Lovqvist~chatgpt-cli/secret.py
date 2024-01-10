#Builtins
from pathlib import Path

#Third party
import openai

#Note - to access your OpenAI API key, go to https://platform.openai.com/account/api-keys

#This is not rocket surgery level secret storage, any application under your user will have access and if you don't set permissions, so will other users.
#I am not an authority on good practices here, use your own judgement. I did look a bit at https://www.freedesktop.org/wiki/Specifications/secret-storage-spec/
#but then I didn't want to complicate things and decided on using the plain text file.
openai.api_key = (Path.home() / '.config/OpenAI/primary-account.api_key').read_text().strip()


#You could also specify it via the environment which with proper containerization probably is fine, but then again, I don't know
#This is how OpenAI's docs suggested:

#	import os
#	openai.api_key = os.environ['OPENAI_API_KEY']

#Or you could just hardcode it into this module, just make sure you don't check it into your git repository if you do.
#The main reason I went with the external file is to decrease the probability of me or anyone using this repository from doing it by mistake.

#	openai.api_key = ...

