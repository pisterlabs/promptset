import ast_comments as ast
import asyncio
import openai
import click
import glob
import os
import rich
import time
# import tqdm

from rich.progress import Progress

# import commentator
from . import commentator
from . import strip_comments
from . import strip_types

async def commentate_one_file(index, file, check, language, progress):
    code = file.read()

    if language == "Python":
        try:
            ast.parse(code)
        except SyntaxError:
            # Failed to parse.
            return

    function_count = 0
    for func_name in commentator.enumerate_functions(code):
        the_code = commentator.extract_function_source(code, func_name)
        #if not (commentator.has_docstring(the_code) and commentator.has_types(the_code)):
        function_count += 1
        
    if function_count == 0:
        return

    # from tqdm import tqdm
    pbar = progress.add_task(f"{file.name}", total=function_count)
    # pbar = tqdm(total=function_count, desc=file.name, leave=False, unit='function') # position=index,
    
    (result, successes) = await commentator.commentate(file.name, check, code, pbar, progress, language)
    if result and not check:
        save_path = os.path.join(os.getcwd(), "backup")
        os.makedirs(os.path.dirname(os.path.join(save_path, file.name)), exist_ok=True)
        with open(os.path.join(save_path, file.name), 'w') as f:
            # Generate import statement first.
            import_stmt = commentator.generate_import(ast.parse(code))
            if import_stmt:
                code = import_stmt + "\n" + code
            f.write(code)
        with open(file.name, 'w') as f:
            f.write(result)
        #if successes > 1:
        #    print(f"Successfully commentated {successes} functions.")
        #elif successes == 1:
        #    print(f"Successfully commentated {successes} function.")
        #else:
        #    print("Unable to commentate any functions. See 'commentator.log'.")
    else:
        pass
        #print(f"Failed to process {file.name}.")
    
async def do_it(api_key, check, language, progress, *files):
    openai.api_key = api_key
    file_list = list(*files)
    tasks = [commentate_one_file(index, file, check, language, progress) for (index, file) in enumerate(file_list)]
    await asyncio.gather(*tasks)

def print_version(ctx, param, value):
    import importlib.metadata
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"commentator version {importlib.metadata.metadata('python-commentator')['Version']}")
    ctx.exit(0)

async def func_one_file(index, file, progress, func):
    with open(file.name, 'r') as f:
        code = f.read()
    try:
        the_ast = ast.parse(code)
    except SyntaxError:
        # Failed to parse.
        return
        
    function_count = len(commentator.enumerate_functions(code))
    if function_count == 0:
        return

    # from tqdm import tqdm
    pbar = progress.add_task(f"{file.name}", total=function_count)
    # pbar = tqdm(total=function_count, desc=file.name, leave=False, unit='function') # position=index,

    result = func(the_ast)
    if result:
        save_path = os.path.join(os.getcwd(), "backup")
        os.makedirs(os.path.dirname(os.path.join(save_path, file.name)), exist_ok=True)
        with open(os.path.join(save_path, file.name), 'w') as f:
            f.write(code)
        with open(file.name, 'w') as f:
            f.write(result)
    else:
        pass

async def strip_types_one_file(index, file, progress):
    await func_one_file(index, file, progress, strip_types.strip_types)

async def strip_comments_one_file(index, file, progress):
    await func_one_file(index, file, progress, strip_comments.strip_comments)
    
async def strip_types_helper(progress, *files):
    file_list = list(*files)
    tasks = [strip_types_one_file(index, file, progress) for (index, file) in enumerate(file_list)]
    await asyncio.gather(*tasks)

def do_strip_types(progress, files):
    asyncio.run(strip_types_helper(progress, files))

async def strip_comments_helper(progress, *files):
    file_list = list(*files)
    tasks = [strip_comments_one_file(index, file, progress) for (index, file) in enumerate(file_list)]
    await asyncio.gather(*tasks)

def do_strip_comments(progress, files):
    asyncio.run(strip_comments_helper(progress, files))
    
@click.command()
@click.argument('file', nargs=-1, type=click.File('r'))
@click.option('--api-key', help="OpenAI key.", default=commentator.api_key(), required=False)
@click.option('--check', is_flag=True, help="Instead of adding comments, check for inconsistencies between the code and comments.", required=False, default=False)
@click.option('--language', help="Write all comments in the (human) language of your choice (default=English).", required=False, default=None)
@click.option('--version', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True, help="Print the current version number and exit.")
@click.option('--strip-types/--no-strip-types', default=False, help="Just strip existing types and exit.")
@click.option('--strip-comments/--no-strip-comments', default=False, help="Just strip existing comments and exit.")
def main(file, api_key, check, language, strip_types, strip_comments):
    """Automatically adds comments to your code.

    See https://github.com/plasma-umass/commentator for more information.
    """
    with Progress() as progress:
        if strip_types:
            do_strip_types(progress, file)
        if strip_comments:
            do_strip_comments(progress, file)
        if strip_types or strip_comments:
            return
        asyncio.run(do_it(api_key, check, language, progress, file))

main()
