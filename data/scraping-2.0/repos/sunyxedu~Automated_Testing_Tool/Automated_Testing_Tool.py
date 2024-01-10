import unittest
from appium import webdriver
from appium.webdriver.common.appiumby import AppiumBy
import openai
import time
import os
import dotenv

def set_openai_api_env():
    dotenv.load_dotenv()
    if os.getenv("OPENAI_API_BASE") is not None:
        openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_key = os.getenv("OPENAI_API_KEY")

set_openai_api_env()

capabilities = dict(
    platformName='Android',
    automationName='uiautomator2',
    deviceName='Android',
    appPackage='com.quizlet.quizletandroid',
    appActivity='.ui.RootActivity',
    language='en',
    locale='US',
    noReset=True,
)

appium_server_url = 'http://localhost:4723'

def prompting(goal, elements, driver, prev_choices=[], ban_list = []):
    choices = [elem for elem in elements if elem.is_enabled() and elem.text]
    choice_texts = [elem.text for elem in choices]
    print(ban_list)
    print("prev", prev_choices)
    for ban in ban_list:
        if ban in choice_texts:
            choice_texts.remove(ban)
    print(choice_texts)
    prompt = f"""Your Goal: {goal} 
                Previous actions: {prev_choices}
                Your Choices: {choice_texts}:
                Think step by step toward your ultimate choice, and write it out loud.
                Ultimately, in an additional, seperate line, output one, and only one, of the choices, 
                in the format of "CHOICE: <CHOICE>".
                If you feel stuck and want to go back, output BAK instead in the last line.
            """
    messages = [
        {"role": "system", "content": prompt}
    ]
    while True:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        content = completion.choices[0].message.content
        print(f"result: {content}")
        if content == "DONE":
            return True, True, content
        
        ctn = content.split('\n')[-1]
        if "BAK" in ctn:
            driver.back()
            messages.append({"role": "assistant", "content": "go back a page"})
            messages.append({"role": "system", "content": "Successfully executed, please continue"})
            continue
        for choice in choices:
            if choice.text in ctn:
                choice.click()
                print("----- Clicked", choice.text)
                return False, True, choice.text
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "system", "content": "The choice you made is not valid. Please try again."})


class TestAppium(unittest.TestCase):
    def setUp(self) -> None:
        self.driver = webdriver.Remote(appium_server_url, capabilities)

    def tearDown(self) -> None:
        if self.driver:
            self.driver.quit()

    def test_find_clickable_elements(self):
        flag = False
        GOAL = "I want to turn on Dark Mode in the settings."
        # bad_choices = []
        used_choices = []
        while not flag:
            cnt = 0
            while cnt < 5:
                try:
                    time.sleep(3)
                    all = self.driver.find_elements(AppiumBy.XPATH, ".//*")
                    ended, success, choice = prompting(GOAL, all, self.driver, used_choices, [])
                    used_choices = used_choices + [choice]
                    flag = flag or ended
                    if ended:
                        user_ok = input("Did you do it? (y/n)")
                        if user_ok != "y":
                            success = False
                    if not success:
                        self.driver.back()
                        flag = False
                except Exception as e:
                    print(e)
                    cnt += 1
                    continue
            if cnt == 5:
                break
                
if __name__ == '__main__':
    unittest.main()