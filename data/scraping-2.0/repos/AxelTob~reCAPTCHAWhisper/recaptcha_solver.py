# Import the necessary libraries
import openai
import requests
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set your OpenAI API key
openai.api_key = 'your-api-key'

# depends on your browser language settings. e.g. swedish: 'reCAPTCHA-utmaningen löper ut om två minuter'
# Title of the reCAPTCHA challenge iframe
RECAPTCHA_CHALLENGE_TITLE = 'recaptcha challenge expires in two minutes'


def download_audio_file(audio_url):
    """
    Downloads the audio file from the given URL.
    """
    response = requests.get(audio_url)
    audio_file_path = 'captcha_audio.mp3'
    with open(audio_file_path, 'wb') as f:
        f.write(response.content)
    return audio_file_path


def transcribe_audio_file(audio_file_path):
    """
    Transcribes the audio file using OpenAI's Whisper model.
    """
    with open(audio_file_path, 'rb') as audio_file:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file
        )
    return transcript['text'].strip()


def click_recaptcha_checkbox(driver):
    """
    Clicks the reCAPTCHA checkbox.
    """
    driver.switch_to.default_content()
    driver.switch_to.frame(driver.find_element(
        By.XPATH, ".//iframe[@title='reCAPTCHA']"))
    driver.find_element(By.ID, "recaptcha-anchor-label").click()
    driver.switch_to.default_content()


def request_audio_captcha(driver):
    """
    Requests the audio version of the captcha.
    """
    driver.switch_to.default_content()
    driver.switch_to.frame(driver.find_element(
        By.XPATH, f".//iframe[@title='{RECAPTCHA_CHALLENGE_TITLE}']"))
    driver.find_element(By.ID, "recaptcha-audio-button").click()


def solve_audio_captcha(driver):
    """
    Solves the audio captcha.
    """
    audio_url = driver.find_element(By.ID, "audio-source").get_attribute('src')
    audio_file_path = download_audio_file(audio_url)
    transcript = transcribe_audio_file(audio_file_path)
    driver.find_element(By.ID, "audio-response").send_keys(transcript)
    driver.find_element(By.ID, "recaptcha-verify-button").click()


def main():
    """
    Main function to execute the captcha solver.
    """
    driver = webdriver.Chrome(executable_path=ChromeDriverManager().install())
    driver.get("https://www.google.com/recaptcha/api2/demo")
    click_recaptcha_checkbox(driver)
    time.sleep(1)
    request_audio_captcha(driver)
    time.sleep(1)
    solve_audio_captcha(driver)
    time.sleep(15)


if __name__ == "__main__":
    main()
