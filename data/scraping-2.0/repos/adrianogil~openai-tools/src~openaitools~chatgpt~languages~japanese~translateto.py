# openaitools.chatgpt.languages.japanese.translateto
import sys
import os


def get_translation(str_to_translate):
	from openaitools.chatgpt.chatcli import get_chatgpt_output

	prompt = \
f"""Translate the following text to Japanese:

{str_to_translate}
"""
	translated_str = get_chatgpt_output(prompt)
	return translated_str


if __name__ == '__main__':
	if len(sys.argv) <= 1:
		str_to_translate = input('> ')
		print("")
	else:
		input_str = sys.argv[1]
		if os.path.isfile(input_str):
			file_path = input_str
			with open(file_path, 'r') as file_handler:
			    str_to_translate = json.load(file_handler)
		else:
			str_to_translate = input_str
	output = get_translation(str_to_translate)
	print(output)
