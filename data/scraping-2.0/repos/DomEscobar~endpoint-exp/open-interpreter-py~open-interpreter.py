import interpreter
import openai

interpreter.model = 'gpt-3.5-turbo'
interpreter.auto_run = True
openai.api_key="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"
# interpreter.chat('Use google maps to get all asian restaurants at Heilbronn')
interpreter.chat("Extract all emails from the website mmmake.com") # Executes a single command