import os
import openai
import json
from dotenv import load_dotenv
import argparse
from utils import (
  format_prompt_for_openai_completion, get_completion, create_openai_file, list_openai_files, list_finetunes, 
  create_finetune, get_finetune, calculate_fine_tuning_cost, list_models, get_completions_from_input, fix_escaped_unicode)
import readline
from utils_highlight import highlight_substrings, print_colored_text
import asyncio
from constants import model_name_to_id

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

async def main():
  assert len(os.getenv("OPENAI_API_KEY")) > 0, "Please set your OPENAI_API_KEY in your .env file"

  parser = argparse.ArgumentParser()
  parser.add_argument("step", type=str, help="phase to run in (test, file, tune, list, cost)")
  parser.add_argument("-m", "--model", type=str, help="model to use (underline, highlight, emphasis, or custom)", default="underline")
  parser.add_argument("--debug", action="store_false", help="hide debug info")
  parser.add_argument("-f", "--file", type=str, help="path to file to upload or file ID to use for fine tuning")
  parser.add_argument("-l", "--list", type=str, help="in list mode, what to list (files, finetunes)")
  parser.add_argument("-oai", "--open_ai_model", type=str, help="OpenAI model to use for fine tuning (default: babbage)", default="babbage")
  parser.add_argument("--finetune_id", type=str, default=None)

  args = parser.parse_args()

  assert args.step in ["test", "file", "tune", "list", "cost"], "Please specify a valid step (test, file, tune, list, cost)"

  if args.step == "test":
    assert args.model in model_name_to_id, "Please specify a valid model (underline, highlight, emphasis)"
    model = model_name_to_id[args.model]

    while True:
      tag = input("Tag: ")
      bodyText = input("Body Text: ")
      
      if args.model != "underline":
        underlines = input("JSON formatted underlines: ")
      else:  
        underlines = None

      if underlines:
        underlines = fix_escaped_unicode(underlines)
      bodyText = fix_escaped_unicode(bodyText)
      
      output = await get_completions_from_input(tag, bodyText, model, underlines=underlines, debug=args.debug)
      if output is None:
        print("No output")
        continue

      output_str, loc = output
      print_colored_text(output_str)
      if args.debug:
        print(loc)
  elif args.step == "file":
    if not args.list:
      assert args.file is not None and os.path.isfile(args.file), "Please specify a file to upload"
      model_name = os.path.basename(args.file).split(".")[0]
      create_openai_file(model_name, args.file)
    else:
      list_openai_files()
  elif args.step == "tune":
    assert args.file is not None, "Please specify a valid OpenAI file ID to use for fine tuning"
    create_finetune(args.file, args.open_ai_model, args.model)
  elif args.step == "list":
    assert args.list in ["files", "finetunes", "models"], "Please specify a valid list type (files, finetunes, models)"
    if args.list == "files":
      list_openai_files()
    elif args.list == "models":
      list_models()
    else:
      if args.finetune_id is None:
        list_finetunes()
      else:
        get_finetune(args.finetune_id)
  elif args.step == "cost":
    assert args.file is not None and os.path.isfile(args.file), "Please specify a valid file path"
    calculate_fine_tuning_cost(args.file)


if __name__=="__main__":
  loop = asyncio.new_event_loop()
  loop.run_until_complete(main())