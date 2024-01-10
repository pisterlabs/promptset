import polars as pl
import json
import time
import openai
import os
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

from config import IMG_DIR


def download_image(url: str, category: str, city: str, number: str, option: str) -> None:
    category, city, option = map(lambda x: x.replace(' ', '_').replace('-', '_') , [category, city, option])
    save_dir = Path(f'{IMG_DIR}/{category}/{city}')
    save_dir.mkdir(parents=True, exist_ok=True)
    image_name = Path(f'{number}_{option}.jpg')
    save_path = save_dir/image_name
    try:
        response = requests.get(url)
        # Open the downloaded image using PIL
        with Image.open(BytesIO(response.content)) as image:
            # Define the desired size and save
            new_size = (1024, 1024)
            resized_image = image.resize(new_size)
            resized_image.save(save_path, format='JPEG')
        return image_name   
    except IOError as err:
        print(f'An error {err} was occured while downloading image of {number}.{option} for {city}')
    except Exception as err:
        print(f'Unexpected error was occured while downloading image {url}')
        

def correct_image_names():
    rep = {"'":"", ".":"", "__":"_"}
    for file in Path(f'{IMG_DIR}/city_attractions').rglob('*.jpg'):
        new_stem = file.stem
        for key, value in rep.items():
            if key in file.stem:
                new_stem = new_stem.replace(key, value)
        file.rename(file.with_stem(new_stem))


def resize_images(folder_path: Path | str, to_size: tuple=(1024, 1024)) -> None:
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)
    try:
        for file_path in folder_path.rglob('*.jpeg'):
            print(file_path)
            try:
                img = Image.open(file_path)
                if img.size == to_size:
                    print(f"Skipping {file_path.name} (already {to_size}.")
                else:
                    img = img.resize(to_size, Image.ANTIALIAS)
                    img.save(file_path)
                    print(f"Resized {file_path.name} successfully.")
            except Exception as e:
                print(f"Error resizing {file_path.name}: {str(e)}")
    except StopIteration as err:
        print(err)
    except Exception as err:
        print(f'\n It was something wrong with {file_path}: {err}')
        

def resize_image(file_path: Path | str, to_size: tuple=(1024, 1024)) -> None:
    try:
        img = Image.open(file_path)
        if img.size == to_size:
            print(f"Skipping {file_path.name} (already {to_size}.")
        else:
            img = img.resize(to_size, Image.ANTIALIAS)
            img.save(file_path)
            print(f"Resized {file_path.name} to {to_size} successfully.")
    except Exception as e:
        print(f"Error resizing {file_path.name}: {str(e)}")


def is_valid_link(url: str, timeout: int=10) -> bool:
    try:
        requests.head(url, timeout=timeout, allow_redirects=False).raise_for_status()
        return True
    except requests.exceptions.RequestException as err:
        print("An error occurred during the request:", err)
        return False


def get_prompts_GPT(prompt_json_path: Path | str) -> dict:
    with open(prompt_json_path, 'r') as f:
        return json.load(f)
    

def limit_calls_per_minute(max_calls):
    """
    Decorator that limits a function to being called `max_calls` times per minute,
    with a delay between subsequent calls calculated based on the time since the
    previous call.
    """
    calls = []
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Remove any calls from the call history that are older than 1 minute
            calls[:] = [call for call in calls if call > time.time() - 60]
            if len(calls) >= max_calls:
                # Too many calls in the last minute, calculate delay before allowing additional calls
                time_since_previous_call = time.time() - calls[-1]
                delay_seconds = 60 / max_calls - time_since_previous_call
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
            # Call the function and add the current time to the call history
            result = func(*args, **kwargs)
            calls.append(time.time())
            return result
        return wrapper
    return decorator
    
  
@limit_calls_per_minute(3)    
def get_response_GPT(prompt: str, api_key: str='OPENAI_API_KEY_CT_2') -> str:
    openai.organization = os.getenv('OPENAI_ID_CT')
    openai.api_key = os.getenv(api_key)
    # print(f'prompt = ')
    try:
        response = openai.ChatCompletion.create(model='gpt-3.5-turbo',
                                                messages=[
                                                            #{"role": "system", "content": f"Act as an {role}"},
                                                            {'role': 'user', 'content': prompt}
                                                        ],
                                                temperature=0)  
        print(f"\n{response['choices'][0]['message']['content']}") 
        return response['choices'][0]['message']['content']
    except openai.OpenAIError as err:
        print("An error occurred during the OpenAI request:", err)
        return None 
     

@limit_calls_per_minute(3)    
def get_images_DALLE(prompt: str, n: int=1, size: str='512x512', api_key: str='OPENAI_API_KEY_CT_2') -> list:
    openai.organization = os.getenv('OPENAI_ID_CT')
    openai.api_key = os.getenv(api_key)
    try:
        response = openai.Image.create(prompt=prompt, n=n, size=size)
        return [item['url'] for item in response['data']]
    except openai.OpenAIError as err:
        print("An error occurred during the OpenAI request:", err)
        return None
    

def elapsed_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        elapsed_minutes = int(elapsed // 60)
        elapsed_seconds = int(elapsed % 60)
        elapsed_hours = int(elapsed_minutes // 60)
        elapsed_minutes %= 60
        print(f'Elapsed time for {func.__name__}: {elapsed_hours} hours, '
                f'{elapsed_minutes} minutes, {elapsed_seconds} seconds')
        return result
    return wrapper

    
def load_json(path: str) -> dict:
    """
    This function loads a JSON file from a given path. The purpose of this function is to provide a convenient way to load JSON data from a file, while also handling potential errors that may occur during the loading process.

    Args:
        path (str): given path

    Returns:
        _type_: JSON data from a file
    """
    try:
        with open(Path(path), 'r') as f:
            return json.load(f)
        
    except (FileNotFoundError, PermissionError, IsADirectoryError, TypeError) as e:
        print(f"Error loading JSON file '{path}': {str(e)}")
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file '{path}': {str(e)}")
        
    except Exception as e:
        print(f"An unexpected error occurred while loading JSON file '{path}': {str(e)}")
    
    
    
if __name__ == '__main__':
    # print(get_cities())
    pass