



''' A library designed to initialize required modules for MLOps Tools by Nullzero

Options include:
- WandB
- Langchain
- MLFlow (ToDo)

'''
import os
import subprocess
import sys
from DependMLMoi.utils.logs.log_tool import _logger
from datetime import datetime, timedelta

from pathlib import Path
from arg_parser import parse_args
from datetime import datetime, timedelta
from dotenv import load_dotenv, find_dotenv, __version__ as dotenv_version
from langchain.callbacks import WandbCallbackHandler, StdOutCallbackHandler

# Custom imports
from DependMLMoi.constants import (
    REQUIRED_DOTENV_VERSION, DEBUG, LOGGING, QUIET,
    REQUIREMENTS_PATH, LOGS_DIR, AUTO_INSTALL, NAME, CUSTOM)

_logger = _logger
args = parse_args()

if args.quiet:
    _logger.basicConfig(level=logging.CRITICAL)
elif args.debug == True:
    _logger.basicConfig(level=logging.DEBUG)
elif args.logging == LEVEL:
    _logger.basicConfig(level=logging.INFO)
else:
    _logger.basicConfig(level=logging.WARNING)

# Constants
LOGS_DIR = Path('./utils/logs')
CACHE_FILE = LOGS_DIR / 'setup_cache.log'
CACHE_DURATION = timedelta(hours=12)
REQUIRED_DOTENV_VERSION = '1.0.0'  # Replace with actual version


''' Libraries to check:

'''
LIBRARIES = ["WandB", "Langchain", "MLFlow"]

if CUSTOM is not None:
    LIBRARIES.append(CUSTOM)
    _logger.info(LIBRARIES)


def update_cache_timestamp():
    CACHE_FILE.write_text(datetime.now().isoformat())

def install_dotenv():
    print(f"Installing python-dotenv=={REQUIRED_DOTENV_VERSION}...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'python-dotenv=={REQUIRED_DOTENV_VERSION}'])
    global load_dotenv, find_dotenv
    from dotenv import load_dotenv, find_dotenv
   

def check_dotenv(REQUIRED_DOTENV_VERSION, auto_install):
    if dotenv_version == REQUIRED_DOTENV_VERSION and auto_install:
        install_dotenv()
    else:
        _logger.error("Dotenv version is not up to date. Please update it to the latest version.")
    return True

def check_library_installed(library_name, auto_install=False):
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'show', library_name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        if auto_install or input(f"Do you wish to install {library_name}? (y/n): ").lower() == 'y':
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', library_name])
        else:
            return f"{library_name} is not installed. Please install it before continuing."

def prompt_openai_api_key():
    if not os.getenv('OPENAI_API_KEY'):
        openai_api_key = input("Enter your OPENAI_API_KEY or 'no' to skip: ").strip()
        if openai_api_key.lower() != 'no':
            Path('.env').write_text(f"OPENAI_API_KEY={openai_api_key}\n", mode='a')

def check_wandb_login():
    if not any(key.startswith('WAND') for key in os.environ):
        try:
            print("Enter your WANDB API KEY")
            Path('.env').write_text(f"'WANDB_API_KEY'={wandb.login()}\n", mode='a')
        except Exception as e:
            _logger.error(e)
        import wandb
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        if NAME == None:
            project = f"default-{datetime.date(datetime.now())}"
            wandb.config(project=project)
            _logger.info("Default project name: {}".format(project))
        else:
            wandb.config(project=NAME)
            _logger.info("Project name: {}".format(NAME))
    
    if TYPE == "LLM" | "GPT" | "GPTApp":
        session_group = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        wandb_callback = WandbCallbackHandler(
        job_type="inference",
        project=project,
        group=f"minimal_{session_group}",
        username = (os.getenv('USER') | os.getenv('USERNAME')),
        tags=["test"])
        
        callbacks = [StdOutCallbackHandler(), wandb_callback]

        return callbacks

def update_gitignore():
    gitignore_path = Path('.gitignore')
    if '.env' not in gitignore_path.read_text():
        gitignore_path.write_text('\n# dotenv environment variables file\n.env\n', mode='a')

def is_setup_required():
    LOGS_DIR.mkdir(exist_ok=True)
    if not CACHE_FILE.exists():
        return True
    timestamp = datetime.fromisoformat(CACHE_FILE.read_text().strip())
    return datetime.now() > timestamp + CACHE_DURATION


# Main function to run all setup steps
def debug_setup(AUTO_INSTALL=False, DEBUG=False):
    if DEBUG:
        print(f"00x ---- checking dotenv installed ---- x00")
    try: 
        check_dotenv()
        load_dotenv(find_dotenv())
    except Exception as e:
        print(e)


# This function is now the main entry point to the library. It should be called explicitly.
def run_setup(auto_install=False):
    # Check if dotenv is installed and meets the required version
    if not check_dotenv(REQUIRED_DOTENV_VERSION, auto_install):
        sys.exit("The required dotenv version is not installed.")

    # Example usage
    if is_setup_required():
        update_cache_timestamp()
        check_dotenv()
        for lib in libraries:
            try:
                check_library_installed(lib, auto_install=True)
            except Exception as e:
                _logger.error({}+" error thrown", e)   
        try:
            prompt_openai_api_key()
            check_wandb_login()
            update_gitignore()
            logging.info("Setup complete.")
        except Exception as e:
            _logger.error({}+" error thrown", e)    
        except CACHE_DURATION:
            _logger.info("Setup already completed within the last 12 hours:\n {}", format(cache))



# The functions below are part of the library's API
def setup(auto_install=False):
    """The public API function to setup dependencies."""
    run_setup(auto_install=auto_install)

# The code below is not executed on import, which is a good practice for libraries
if __name__ == "__main__":
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        logging.warning("No .env file found.")
    # Example: Run setup with AUTO_INSTALL if the corresponding environment variable is set
    setup(auto_install=os.getenv('AUTO_INSTALL', 'False').lower() in ('true', '1', 't'))






