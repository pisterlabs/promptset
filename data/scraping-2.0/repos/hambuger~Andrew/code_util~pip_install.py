import subprocess

from config.global_logger import logger
from openai_util.function_call.openaifunc_decorator import openai_func
import importlib

install_package = []


@openai_func
def install_modules(modules_list: list):
    """
    install the modules list  needed by the code
    :param modules_list: the module name list should be installed
    """
    # Use regex to match imported modules
    # import_pattern = re.compile(r'^(?:from\s+)?(\w+)(?:\s+import\b|$)', re.MULTILINE)
    # modules_to_install = set(match.group(1) for match in import_pattern.finditer(code))
    global install_package
    # Check and install missing modules
    for module in modules_list:
        if module in install_package:
            continue
        try:
            importlib.import_module(module)
            logger.debug(f"{module} is already installed.")
            install_package.append(module)
        except ImportError:
            logger.debug(f"{module} is not installed. Installing...")
            try:
                # Use a subprocess to execute the pip install command and capture the error output to a variable
                subprocess.run(['pip', 'install', module], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True,
                               check=True, encoding='utf-8')
                logger.debug(f"Successfully installed {module}.")
                install_package.append(module)
            except subprocess.CalledProcessError as e:
                # output error output of command execution
                logger.debug(f"Failed to install {module}. Error: {e.stderr}")
                return f"Failed to install {module}. Error: {e.stderr}"
    return "success"
