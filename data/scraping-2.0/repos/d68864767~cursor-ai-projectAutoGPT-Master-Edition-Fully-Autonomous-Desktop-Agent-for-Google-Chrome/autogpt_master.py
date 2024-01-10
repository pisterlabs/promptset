```python
# autogpt_master.py

# Importing required libraries
import asyncio
import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from config import OPENAI_API_KEY, CHROME_PATH, TASK_CYCLE

# Importing custom chrome scripts
from chrome_scripts import custom_script

# Setting up Chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Setting up Chrome driver
webdriver_service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)

# Function to execute tasks
async def execute_tasks():
    for task in TASK_CYCLE:
        if task == 'custom_script':
            custom_script(driver, OPENAI_API_KEY)
        else:
            print(f"Task {task} not recognized.")
        await asyncio.sleep(1)  # Sleep for a second between tasks

# Main function
def main(continuous):
    loop = asyncio.get_event_loop()
    while True:
        loop.run_until_complete(execute_tasks())
        if not continuous:
            break

# Command line arguments handling
if __name__ == "__main__":
    continuous = '--continuous' in sys.argv
    main(continuous)
```
