```python
# chrome_scripts.py

# Importing required libraries
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from openai import OpenAI, GPT3Completion

# Initialize OpenAI with API key
openai = OpenAI(api_key=OPENAI_API_KEY)

# Function to execute custom script
def custom_script(driver, api_key):
    # Navigate to a specific URL (replace 'https://www.google.com' with your desired URL)
    driver.get('https://www.google.com')

    # Wait until a specific element is loaded (replace 'element_id' with the id of the element you're waiting for)
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, 'element_id'))
    )

    # Interact with the element (this is just an example, replace with your desired interaction)
    element.click()

    # Use GPT-3.5 to generate some text (replace 'prompt' with your desired prompt)
    gpt = GPT3Completion(api_key)
    response = gpt.create(prompt='prompt', max_tokens=100)

    # Print the generated text
    print(response.choices[0].text.strip())
```
