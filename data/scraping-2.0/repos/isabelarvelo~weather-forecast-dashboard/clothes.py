import openai


class clothesRecommender:

    def __init__(self, api_key):
        self.api_key = api_key

    def get_gpt4_response(self, temp, feels_like, rain, humid, uv, wind, description):
        """
        Get a response from the GPT-4 API based on the given weather conditions using the chat completions API.

        Parameters:
        - temp (float): Current temperature in Fahrenheit .
        - feels_like(float): Current apparent temperature in Fahrenheit .
        - rain (float): Current Probability of Rain
        - humid (float): Current humidity as a percentage
        - uv (float): Current UV Index
        - description (str): Current weather condition (e.g., 'Rain', 'Sunny', etc.)
        - wind(float): Current wind speed in km/h.
        - api_key (str): Your OpenAI API key.

        Returns:
        - str: GPT-4's response.
        """

        # Initialize the OpenAI API
        openai.api_key = self.api_key

        # Generate the message for the API based on the weather conditions
        message_content = f"""You are a polite, helpful, and personable personal assistant that offers 
                                gender-neutral clothing advice 
                                appropriate for the current weather conditions. Don't make any assumptions about the time of day.
                                Considering it's {temp}°F out,
                                feels like {feels_like}°f, has a {rain}% chance of rain 
                                with  {humid}% humidity, a UV index of {uv}, wind speeds of {wind} mi/h, 
                                and the general condition is {description}, what would you recommend I wear 
                                in 1-3 sentences? Please do not restate the weather conditions. Complete the sentence
                                Wear ...
                                """

        # Create a list of messages for the chat API
        messages = [{"role": "user", "content": message_content}]

        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Extract the assistant's response from the API response
        recommendation = response['choices'][0]['message']['content']

        return recommendation.strip()
