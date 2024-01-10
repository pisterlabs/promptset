from openai import OpenAI
import logging
import traceback
import random
import sqlite3
from decouple import config
import pycountry

logging.basicConfig(filename='location_cleanup.log', level=logging.INFO)


class LocationCleanup:
    """
    This class is designed to clean up location data in a SQLite database.
    It uses pycountry to standardize country names and falls back to OpenAI for complex location parsing.
    """

    def __init__(self, database_path):
        """
        Initializes the LocationCleanup object with the path to the SQLite database.
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

    def analyze_location_with_openai(self, text):
        """
        Analyzes the given text to extract location information using the OpenAI API.
        """
        sys_prompt = (
            "You are a system who tries to figure out the overall location by country "
            "from the user input if you get more than one location, you reply in list "
            "you do not add any additional information. "
            "response structure ['list_of_country' or 'a single country' in alpha-2] "
            "if multiple location/cities, you return the country name in alpha-2"
            "if no location found, or you didnt understand, return Unknown"
        )
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
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

    @staticmethod
    def find_country(location):
        """
        Tries to find the country using pycountry. First attempts to find a direct country match.
        If unsuccessful, tries to find a subdivision and then the corresponding country.
        Returns the alpha-2 code of the country if found, otherwise returns None.
        """
        try:
            # Attempt to find a direct country match
            country = pycountry.countries.search_fuzzy(location)[0]
            return country.alpha_2
        except Exception:
            try:
                # Attempt to find a subdivision and then get the corresponding country
                subdivision = pycountry.subdivisions.search_fuzzy(location)[0]
                country = pycountry.countries.get(
                    alpha_2=subdivision.country_code)
                if country:
                    return country.alpha_2
            except Exception:
                # If both attempts fail, return None
                return None

    def location_cleanup(self):
        """
        Cleans up location data in the SQLite database.
        """
        index = 0
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT id, location FROM new_data")
                rows = cursor.fetchall()
                cached = {
                    "Unknown": "Unknown"
                }

                for row in rows:
                    original_id, original_location = row
                    logging.info(f"Processing location for ID: {original_id}")

                    if original_location is None:
                        original_location = "Unknown"

                    locations = original_location.split(',')
                    new_locations = set()

                    for loc in locations:
                        loc = loc.strip()
                        country = self.find_country(
                            loc) or cached.get(loc.lower())
                        if not country:
                            country = self.analyze_location_with_openai(loc)
                            if country:
                                cached[loc.lower()] = country
                        if country:
                            new_locations.add(country)

                    if new_locations:
                        new_locations = list(new_locations)
                        updated_location = ', '.join(new_locations)
                        cursor.execute("UPDATE new_data SET location = ? WHERE id = ?",
                                       (updated_location, original_id))
                        conn.commit()  # lol i forgot!
                        logging.info(f"Updated location for ID: {original_id} from '{
                                     original_location}' to '{updated_location}'")
                        print(f"Updated location for ID: {original_id} from '{
                              original_location}' to '{updated_location}'")
            except Exception as e:
                logging.error(f"Error in location_cleanup: {e}")
                print(f"Error in location_cleanup: {e}")


if __name__ == "__main__":
    logging.info("Program started")
    print("Clean up 2: Fix the locations.")

    try:
        cleanup = LocationCleanup('compromised_assets.db')
        cleanup.location_cleanup()
    except Exception as e:
        logging.error(f"Exception in main: {e}")
        print(f"Exception in main: {e}")
    finally:
        logging.info("Program ended")
