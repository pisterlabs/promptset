import click
from openai import OpenAI
from icecream import ic
from support import functions as sf
# from support.functions import generate_image, save_picture, get_dalle_prompt_based_on_input, execution_time_decorator, save_text_to_file
import urllib3
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
import support.functions as sf
import concurrent.futures


openAIClient = OpenAI()
model_to_chat = "gpt-3.5-turbo-1106"
# model_to_chat = "gpt-4-1106-preview"
model_to_image = "dall-e-3"
# 1792x1024
PIC_SIZE="1024x1024"
# PIC_SIZE="1792x1024"
PIC_QUALITY="hd"



@click.group()
def cli():
    """CLI Application for Various Tasks"""
    pass


@cli.command()
@click.option('-p', '--prompt', default=None, help='Prompt text for generating an idea.')
@click.option('-i', '--inputfile', default=None, type=click.Path(exists=True), help='Input file with text for generating the idea.')
@click.option('-0', '--outputfile', required=True, type=click.Path(), help='Output file to save the generated idea.')
def idea(prompt, outputfile, inputfile):
    """
    Generate an Idea
    This method generates a prompt for generating pictures on a given prompt ideas. The generated idea is saved to an output file.
    Args:
        prompt (str): The prompt text for generating the idea.
        outputfile (str): The path of the output file to save the generated idea.
    Returns:
        None
    Example Usage:
        idea("--prompt 'give me a picture of a beautiful woman with handsome man.' --outputfile 'my_idea.txt'")
    """
    if prompt is None and inputfile is None:
        raise click.UsageError("You must provide either a prompt or an input file.")
    if prompt is not None and inputfile is not None:
        raise click.UsageError("You can't provide both a prompt and an input file.")

    if prompt is not None:
        text_prompt = prompt
    if inputfile is not None:
        with open(inputfile, 'r') as file:
            text_prompt = file.read().strip()

    print("\tGenerating prompt based on idea ...")
    sf.generate_and_save_idea(text_prompt, outputfile, openAIClient, model_to_chat)
    click.echo(f'Idea generated and saved to {outputfile}.')




@cli.command()
@click.option('-i', '--input_file', type=click.File('r'),  help='Input file with prompt text.')
@click.option('-s', '--style', type=str, help='List of styles to apply to the picture.[comma separated]')
@click.option('-r', '--random_num', type=int, default=0, help='Generate number of different random styles')
@click.option('-o', '--output_file', type=str, help='Where to save picture')
@click.option('-w', '--workers_num', type=int, default=3, help='Number of workers to use for parallel execution.')
def multistyle(input_file, style, output_file, workers_num, random_num):
    # Implement logic to generate pictures with specified styles
    if random_num != 0 and style is not None:
        raise Exception("You can't specify both random_num and list_of_styles. Please specify only one of them.")

    if random_num != 0:
        print(f"Generating num of random styles: {random_num}")
        list_of_styles = sf.get_random_styles_from_file(random_num)
    else:
        list_of_styles = style.split(",")

    print(f"List of styles: {list_of_styles}")

    # Define a function to perform task in parallel
    initial_idea_prompt = input_file.read()
    def task_gen_adopted_prompt(initial_idea_prompt, style,output_file):
        print(f"Processing style: {style}")
        additional_user_prompt = ""
        adopted_prompt = sf.generate_adopted_prompt(additional_user_prompt, initial_idea_prompt, style, openAIClient, model_to_chat)
        output_adopted_prompt_file = "temp/multi/03_adopted_prompt.txt"
        output_file_path = sf.replace_last_path_part_with_datetime(output_adopted_prompt_file, style)
        sf.save_text_to_file(adopted_prompt, output_file_path)
        image = sf.generate_image(adopted_prompt, openAIClient, size=PIC_SIZE, quality=PIC_QUALITY)
        output_file = sf.replace_last_path_part_with_datetime(output_file, style)
        sf.save_picture(output_file, image)

        return adopted_prompt

    # Use ThreadPoolExecutor to run the tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers_num) as executor:
        executor.map(task_gen_adopted_prompt, [initial_idea_prompt]*len(list_of_styles), list_of_styles, [output_file]*len(list_of_styles))

    pass


@cli.command()
@click.option('-i', '--input_file', type=click.File('r'),  help='Input file with prompt text.')
@click.option('-p', '--prompt', type=str, help='Additional Prompt text for generating a picture.')
@click.option('-s', '--style', type=str, help='Style to apply to the picture.')
@click.option('-o', '--output_file', type=str, help='Where to save picture')
def picByStyle(input_file, prompt, style, output_file):
    """
    This method generates a picture based on a given style and prompt text.

    Parameters:
    - input_file: The input file that contains the prompt text. Should be opened in 'r' mode.
    - prompt: Additional prompt text to be used for generating the picture.
    - style: The style to apply to the picture.
    - output_file: The path where the generated picture will be saved. Should be a string.

    Returns:
    None

    Example usage:
    picByStyle(open('prompt.txt', 'r'), "Generate a beautiful landscape", "landscape_style", "output.png")
    """
    initial_idea_prompt = input_file.read()
    additional_user_prompt = prompt
    adopted_prompt = sf.generate_adopted_prompt(additional_user_prompt, initial_idea_prompt, style, openAIClient, model_to_chat)

    image = sf.generate_image(adopted_prompt, openAIClient, size=PIC_SIZE, quality=PIC_QUALITY)
    output_file = sf.replace_last_path_part_with_datetime(output_file, style)
    sf.save_picture(output_file, image)
    click.echo(f'Picture generated with style "{style}" based on the input prompt and saved:\n---\n{output_file}')
    ic(f"Picture saved to {output_file}")


@cli.command()
@click.option('-i', '--input_file', type=click.File('r'),  help='Input file with prompt text.')
@click.option('-o', '--output_file', type=str, help='Where to save picture')
def picFromPromptFile(input_file, output_file):
    initial_idea_prompt = input_file.read()
    image = sf.generate_image(initial_idea_prompt, openAIClient, size=PIC_SIZE, quality=PIC_QUALITY)
    output_file = sf.replace_last_path_part_with_datetime(output_file, "")
    sf.save_picture(output_file, image)
    click.echo(f'Picture generated from file "{input_file}" based on the input prompt and saved:\n---\n{output_file}')
    ic(f"Picture saved to {output_file}")

if __name__ == '__main__':
    cli()