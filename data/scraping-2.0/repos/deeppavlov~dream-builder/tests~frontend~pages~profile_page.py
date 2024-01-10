from .base_page import BasePage
from tests.frontend.locators.locators import ProfilePageLocators
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tests.frontend.config import openai_token


class ProfilePage(BasePage):
    def switch_to_personal_tokens_tab(self):
        textarea = self.browser.find_element(*ProfilePageLocators.PERSONAL_TOKENS)
        textarea.click()

    def switch_to_personal_account_tab(self):
        textarea = self.browser.find_element(*ProfilePageLocators.ACCOUNT_TAB)
        textarea.click()

    def click_change_language(self):
        textarea = self.browser.find_element(*ProfilePageLocators.CHANGE_LANGUAGE)
        textarea.click()

    def select_english_language(self):
        textarea = self.browser.find_element(*ProfilePageLocators.RADIOBUTTON_ENGLISH)
        textarea.click()

    def select_russian_language(self):
        textarea = self.browser.find_element(*ProfilePageLocators.RADIOBUTTON_RUSSIAN)
        textarea.click()

    def click_save_button_language_mw(self):
        textarea = self.browser.find_element(*ProfilePageLocators.SAVE_BUTTON)
        textarea.click()

    def click_cancel_button_language_mw(self):
        textarea = self.browser.find_element(*ProfilePageLocators.CANCEL_BUTTON)
        textarea.click()

    def enter_token(self):
        textarea = self.browser.find_element(*ProfilePageLocators.TOKEN_TEXTAREA)
        textarea.click()
        textarea.send_keys(openai_token)

    def open_choose_service_dropdown(self):
        button = self.browser.find_element(*ProfilePageLocators.CHOOSE_TOKEN_SERVICES_DROPDOWN)
        button.click()

    def choose_service(self):
        button = self.browser.find_element(*ProfilePageLocators.CHOOSE_TOKEN_SERVICE_OPENAI)
        button.click()

    def click_enter_token_button(self):
        edit_button = (
            WebDriverWait(self.browser, 3)
            .until(EC.element_to_be_clickable(ProfilePageLocators.ENTER_TOKEN_BUTTON))
            .click()
        )

    def click_remove_button(self):
        button = self.browser.find_element(*ProfilePageLocators.REMOVE_TOKEN)
        button.click()

    def check_successfully_created_toast(self):
        success_toast = WebDriverWait(self.browser, 10).until(
            EC.presence_of_element_located(ProfilePageLocators.SUCCESS_TOAST)
        )

    def click_close_button(self):
        success_toast = (
            WebDriverWait(self.browser, 3)
            .until(EC.presence_of_element_located(ProfilePageLocators.CLOSE_BUTTON))
            .click()
        )
