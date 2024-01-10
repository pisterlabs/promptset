from openai import OpenAI
import logging
import traceback
import random
import sqlite3
from decouple import config

logging.basicConfig(filename='price_cleanup.log', level=logging.INFO)


class PriceCleanup:
    """
    This class is designed to clean up price data in a SQLite database.
    Uses OPENAI
    """

    def __init__(self, database_path):
        """
        Initializes the PriceCleanup object with the path to the SQLite database.
        """
        self.database_path = database_path
        self.openai_client = self.initialize_openai_client()
        self.openai_counter = 0

    def initialize_openai_client(self):
        """
        Initializes the OpenAI client using API keys from the environment.
        """
        key_preface = "openai_"
        keys = [config(key_preface + str(i)) for i in range(1, 10)]
        return OpenAI(api_key=random.choice(keys))

    def analyze_price_with_openai(self, text):
        """
        Analyzes the given text to extract price information using the OpenAI API.
        """
        sys_prompt = (
            "You are a system who tries to figure out the price and currency from given text"
            "from the user input if you get more than one price, you reply the lowest and the highest among them"
            "you do not add any additional information. "
            "You add the currency after the price like 1 USD or 2 BTC"
            "lowest and highest price will be separated by ,"
            "if no price found, or you didnt understand the command, return Unknown"
            "respons should be in the following format in str - X USD or whatever currency!"
            "respons for lowest and highest should be in X USD, Y USD format"
            "if the text is without any currency, then response should in X aa format"
            "for lowest and highest, response should be in X aa, Y aa format"
        )
        try:
            response = self.openai_client.chat.completions.create(
                # model="gpt-3.5-turbo",
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": text}],
            )
            message_content = response.choices[0].message.content
            self.openai_counter += 1
            print("Open AI Requests ", self.openai_counter)
            return self.parse(message_content)
        except Exception as e:
            logging.error(f"Error while querying OpenAI: {
                          traceback.format_exc()}")
            return None

    @staticmethod
    def parse(json_string):
        """
        Parses the JSON string to extract the relevant information.
        """
        return json_string.strip("[]")

    def price_cleanup(self):
        """
        Cleans up Price data in the SQLite database.
        """
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT id, price FROM new_data")
                rows = cursor.fetchall()
                cached = dict()

                for row in rows:
                    original_id, original_price = row
                    logging.info(f"Processing price for ID: {original_id}")

                    if original_price is None:
                        original_price = "Unknown"

                    updated_price = cached.get(original_price.lower())
                    if updated_price is None:
                        updated_price = self.analyze_price_with_openai(
                            original_price)
                        if updated_price is not None:
                            cached[original_price.lower()] = updated_price
                        else:
                            updated_price = "Unknown"

                    cursor.execute("UPDATE new_data SET price = ? WHERE id = ?",
                                   (updated_price, original_id))
                    conn.commit()
                    logging.info(f"Updated price for ID: {original_id} from '{
                        original_price}' to '{updated_price}'")
                    print(f"Updated price for ID: {original_id} from '{
                        original_price}' to '{updated_price}'")
            except Exception as e:
                logging.error(f"Error in Price Cleanup: {e}")
                print(f"Error in Price Cleanup: {e}")


if __name__ == "__main__":
    logging.info("Program started")
    print("Clean up 2: Fix the Prices!.")

    try:
        cleanup = PriceCleanup('compromised_assets.db')
        cleanup.price_cleanup()
    except Exception as e:
        logging.error(f"Exception in main: {e}")
        print(f"Exception in main: {e}")
    finally:
        logging.info("Program ended")
