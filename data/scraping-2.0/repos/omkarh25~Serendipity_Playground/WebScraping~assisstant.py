# import requests

# def query_openai_assistant(prompt, model="text-davinci-003", api_key="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"):
#     """
#     Send a query to the OpenAI Assistant API.

#     :param prompt: The prompt or question to ask the AI.
#     :param model: The model to use. Defaults to 'text-davinci-003'.
#     :param api_key: Your API key for OpenAI.
#     :return: The response from the AI.
#     """
#     url = "https://api.openai.com/v1/assistants/YOUR_ASSISTANT_ID/messages"
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json"
#     }
#     data = {
#         "model": model,
#         "messages": [{"role": "system", "content": "This is a test"}]
#     }

#     response = requests.post(url, headers=headers, json=data)
#     return response.json()

# # Example usage
# response = query_openai_assistant("What is the capital of France?")
# print(response)
# asst_uhQlHg2Zelg1Fsxl4cFBLiUv



# import openai

# def generate_text(prompt, model="text-davinci-003", api_key="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"):
#     openai.api_key = api_key

#     try:
#         response = openai.Completion.create(
#             model=model,
#             prompt=prompt,
#             max_tokens=150
#         )
#         return response.choices[0].text.strip()
#     except Exception as e:
#         return str(e)

# # Example usage
# response = generate_text("Give the geocode of Rakesh Fantasy Garden", api_key="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")
# print(response)




# import openai
# import pandas as pd

# def generate_text(prompt, model="text-davinci-003", api_key="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"):
#     openai.api_key = api_key

#     try:
#         response = openai.Completion.create(
#             model=model,
#             prompt=prompt,
#             max_tokens=150
#         )
#         return response.choices[0].text.strip()
#     except Exception as e:
#         return str(e)

# def get_geocodes_from_excel(file_path, api_key):
#     # Read the Excel file
#     df = pd.read_excel(file_path)

#     # Print the column names for troubleshooting
#     print("Columns in the file:", df.columns.tolist())

#     # Check if 'location' column exists (note the lowercase 'l')
#     if 'location' not in df.columns:
#         return "The Excel file does not have a 'location' column."

#     # Create a new column for geocodes
#     df['Geocode'] = ''

#     # Iterate over each location and get the geocode
#     for index, row in df.iterrows():
#         location = row['location']  # Use 'location' with lowercase 'l'
#         prompt = f"Give the geocode of {location}"
#         geocode = generate_text(prompt, api_key=api_key)
#         df.at[index, 'Geocode'] = geocode

#     # Save the results to a new Excel file
#     output_file = 'geocodes_output1.xlsx'
#     df.to_excel(output_file, index=False)

#     return f"Geocodes saved to {output_file}"

# # Example usage
# file_path = r'C:\Users\91861\OneDrive\Desktop\bhoodevi\WebScraping\coo.xlsx'  # Use the correct file path
# api_key = "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"  # Replace with your actual OpenAI API key
# result = get_geocodes_from_excel(file_path, api_key)
# print(result)
# import openai
# import pandas as pd

# def generate_text(prompt, api_key, model="text-davinci-003"):
#     openai.api_key = api_key

#     try:
#         response = openai.Completion.create(
#             model=model,
#             prompt=prompt,
#             max_tokens=150
#         )
#         return response.choices[0].text.strip()
#     except Exception as e:
#         return f"Error: {str(e)}"

# def get_geocodes_from_excel(file_path, api_key):
#     try:
#         # Read the Excel file
#         df = pd.read_excel(file_path)
#     except Exception as e:
#         return f"Error reading the Excel file: {str(e)}"

#     # Print the column names for troubleshooting
#     print("Columns in the file:", df.columns.tolist())

#     # Check if 'location' column exists
#     if 'location' not in df.columns:
#         return "The Excel file does not have a 'location' column."

#     # Create a new column for geocodes
#     df['Geocode'] = ''

#     # Iterate over each location and get the geocode
#     for index, row in df.iterrows():
#         location = row['location']
#         prompt = f"Give the geocode of {location}"
#         geocode = generate_text(prompt, api_key=api_key, model="text-davinci-003")
#         df.at[index, 'Geocode'] = geocode
#         print(f"Processed location: {location} - Geocode: {geocode}")  # Print each geocode for verification

#     # Save the results to a new Excel file
#     try:
#         output_file = r'C:\Users\91861\OneDrive\Desktop\bhoodevi\WebScraping\geocodes_output1.xlsx'
#         df.to_excel(output_file, index=False)
#         return f"Geocodes saved to {output_file}"
#     except Exception as e:
#         return f"Error saving the Excel file: {str(e)}"

# # Example usage
# file_path = r'C:\Users\91861\OneDrive\Desktop\bhoodevi\WebScraping\coo.xlsx'
# api_key = "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"
# result = get_geocodes_from_excel(file_path, api_key)
# print(result)

# import openai
# import pandas as pd

# def generate_text(prompt, api_key, model="text-davinci-003"):
#     openai.api_key = api_key

#     try:
#         response = openai.Completion.create(
#             model=model,
#             prompt=prompt,
#             max_tokens=150
#         )
#         return response.choices[0].text.strip()
#     except Exception as e:
#         return f"Error: {str(e)}"

# def parse_geocode(geocode):
#     # Assuming the geocode format is "latitude, longitude"
#     try:
#         latitude, longitude = geocode.split(", ")
#         return float(latitude), float(longitude)
#     except Exception as e:
#         return None, None

# def get_geocodes_from_excel(file_path, api_key):
#     try:
#         # Read the Excel file
#         df = pd.read_excel(file_path)
#     except Exception as e:
#         return f"Error reading the Excel file: {str(e)}"

#     # Check if 'location' column exists
#     if 'location' not in df.columns:
#         return "The Excel file does not have a 'location' column."

#     # Create new columns for latitude and longitude
#     df['Latitude'] = ''
#     df['Longitude'] = ''

#     # Iterate over each location and get the geocode
#     for index, row in df.iterrows():
#         location = row['location']
#         prompt = f"Give the geocode of {location}"
#         geocode = generate_text(prompt, api_key=api_key, model="text-davinci-003")
#         latitude, longitude = parse_geocode(geocode)
#         df.at[index, 'Latitude'] = latitude
#         df.at[index, 'Longitude'] = longitude

#     # Save the results to a new Excel file
#     try:
#         output_file = r'C:\Users\91861\OneDrive\Desktop\bhoodevi\WebScraping\geocodes_output1.xlsx'
#         df.to_excel(output_file, index=False)
#         return f"Geocodes saved to {output_file}"
#     except Exception as e:
#         return f"Error saving the Excel file: {str(e)}"

# # Example usage
# file_path = r'C:\Users\91861\OneDrive\Desktop\bhoodevi\WebScraping\coo.xlsx'
# api_key = "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"
# result = get_geocodes_from_excel(file_path, api_key)
# print(result)

# import openai
# import pandas as pd
# import os
# from dotenv import load_dotenv

# load_dotenv(r"C:\Users\91861\OneDrive\Desktop\bhoodevi\WebScraping\.env")

# def generate_text(prompt, api_key, model="text-davinci-003"):
#     openai.api_key = api_key

#     try:
#         response = openai.Completion.create(
#             model=model,
#             prompt=prompt,
#             max_tokens=150
#         )
#         return response.choices[0].text.strip()
#     except Exception as e:
#         print(f"Error in generate_text: {e}")
#         return None

# def parse_geocode(geocode):
#     try:
#         latitude, longitude = geocode.split(", ")
#         return float(latitude), float(longitude)
#     except Exception as e:
#         print(f"Error in parse_geocode: {e}")
#         return 'N/A', 'N/A'

# def get_geocodes_from_excel(file_path, api_key):
#     try:
#         df = pd.read_excel(file_path)

#         if 'Location' not in df.columns:
#             print("The Excel file does not have a 'Location' column.")
#             return

#         df['Latitude'] = 'N/A'
#         df['Longitude'] = 'N/A'

#         for index, row in df.iterrows():
#             location = row['Location']
#             prompt = f"Give the geocode of {location}"
#             geocode = generate_text(prompt, api_key=api_key)
#             if geocode:
#                 latitude, longitude = parse_geocode(geocode)
#             else:
#                 latitude, longitude = 'N/A', 'N/A'
            
#             df.at[index, 'Latitude'] = latitude
#             df.at[index, 'Longitude'] = longitude

#         output_file = 'geocodes_output_new.xlsx'
#         df.to_excel(output_file, index=False)
#         print(f"Geocodes saved to {output_file}")

#     except Exception as e:
#         print(f"Error in get_geocodes_from_excel: {str(e)}")


# # Replace the file path and API key with your actual file path and new API key
# file_path = r'coo.xlsx'  # Update the file path as needed
# api_key = os.getenv("assisstant_api")# Use your new API key
# result = get_geocodes_from_excel(file_path, api_key)
# print(result)


# import openai
# import pandas as pd
# import os
# import time
# from dotenv import load_dotenv

# load_dotenv(r"C:\Users\91861\OneDrive\Desktop\bhoodevi\WebScraping\.env")

# def generate_text(prompt, api_key, model="text-davinci-003"):
#     openai.api_key = api_key

#     try:
#         response = openai.Completion.create(
#             model=model,
#             prompt=prompt,
#             max_tokens=150
#         )
#         return response.choices[0].text.strip()
#     except Exception as e:
#         print(f"Error in generate_text: {e}")
#         return None

# def parse_geocode(geocode):
#     try:
#         latitude, longitude = geocode.split(", ")
#         return float(latitude), float(longitude)
#     except Exception as e:
#         print(f"Error in parse_geocode: {e}")
#         return 'N/A', 'N/A'

# def get_geocodes_from_excel(file_path, api_key):
#     try:
#         df = pd.read_excel(file_path)

#         if 'Location' not in df.columns:
#             print("The Excel file does not have a 'Location' column.")
#             return

#         df['Latitude'] = 'N/A'
#         df['Longitude'] = 'N/A'

#         for index, row in df.iterrows():
#             location = row['Location']
#             prompt = f"Give the geocode of {location}"
#             geocode = generate_text(prompt, api_key=api_key)
#             if geocode:
#                 latitude, longitude = parse_geocode(geocode)
#             else:
#                 latitude, longitude = 'N/A', 'N/A'
            
#             df.at[index, 'Latitude'] = latitude
#             df.at[index, 'Longitude'] = longitude

#             # Add a delay between API calls
#             time.sleep(1)

#         output_file = 'geocodes_output_new.xlsx'
#         df.to_excel(output_file, index=False)
#         print(f"Geocodes saved to {output_file}")

#     except Exception as e:
#         print(f"Error in get_geocodes_from_excel: {str(e)}")

# # Replace the file path and API key with your actual file path and new API key
# file_path = r'coo.xlsx'  # Update the file path as needed
# api_key = os.getenv("assisstant_api")  # Use your new API key
# get_geocodes_from_excel(file_path, api_key)

# import pandas as pd
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter
# import time

# def geocode_locations(file_path):
#     # Load the Excel file
#     df = pd.read_excel(file_path)

#     # Check if 'Location' column exists
#     if 'Location' not in df.columns:
#         print("The Excel file does not have a 'Location' column.")
#         return

#     # Initialize the geocoder with rate limiter
#     geolocator = Nominatim(user_agent="geoapiExercises")
#     geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

#     # Geocode each location
#     for index, row in df.iterrows():
#         try:
#             location = geocode(row['Location'])
#             if location:
#                 df.at[index, 'Latitude'] = location.latitude
#                 df.at[index, 'Longitude'] = location.longitude
#             else:
#                 df.at[index, 'Latitude'] = 'N/A'
#                 df.at[index, 'Longitude'] = 'N/A'
#         except Exception as e:
#             print(f"Error processing {row['Location']}: {e}")
#             df.at[index, 'Latitude'] = 'N/A'
#             df.at[index, 'Longitude'] = 'N/A'

#         # Print progress
#         print(f"Processed {index + 1}/{len(df)} locations")

#     # Save the results to a new Excel file
#     output_file = 'geocoded_locations1.xlsx'
#     df.to_excel(output_file, index=False)
#     print(f"Geocoded data saved to {output_file}")

# # Replace with your file path
# file_path = r'C:\Users\91861\OneDrive\Desktop\bhoodevi\WebScraping\coo.xlsx'
# geocode_locations(file_path)

# import pandas as pd
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter
# import traceback

# def geocode_locations(file_path):
#     try:
#         # Load the Excel file
#         df = pd.read_excel(file_path)

#         # Check if 'Location' column exists
#         if 'Location' not in df.columns:
#             print("The Excel file does not have a 'Location' column.")
#             return

#         # Initialize the geocoder with rate limiter
#         geolocator = Nominatim(user_agent="geoapiExercises")
#         geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

#         # Geocode each location
#         for index, row in df.iterrows():
#             try:
#                 location = geocode(row['Location'])
#                 if location:
#                     df.at[index, 'Latitude'] = location.latitude
#                     df.at[index, 'Longitude'] = location.longitude
#                 else:
#                     df.at[index, 'Latitude'] = 'N/A'
#                     df.at[index, 'Longitude'] = 'N/A'
#             except Exception as e:
#                 print(f"Error processing {row['Location']}: {e}")
#                 traceback.print_exc()
#                 df.at[index, 'Latitude'] = 'N/A'
#                 df.at[index, 'Longitude'] = 'N/A'

#             # Print progress
#             print(f"Processed {index + 1}/{len(df)} locations")

#         # Save the results to a new Excel file
#         output_file = 'geocoded_locations.xlsx'
#         df.to_excel(output_file, index=False)
#         print(f"Geocoded data saved to {output_file}")

#     except Exception as e:
#         print(f"General Error: {e}")
#         traceback.print_exc()

# # Replace with your file path
# file_path = r'C:\Users\91861\OneDrive\Desktop\bhoodevi\WebScraping\coo.xlsx'
# geocode_locations(file_path)

# import pandas as pd
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter
# import traceback

# def geocode_locations(file_path):
#     try:
#         df = pd.read_excel(file_path)

#         if 'Location' not in df.columns:
#             print("The Excel file does not have a 'Location' column.")
#             return

#         geolocator = Nominatim(user_agent="geoapiExercises")
#         geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)  # Reduced delay

#         for index, row in df.iterrows():
#             location_name = row['Location']
#             try:
#                 location = geocode(location_name)
#                 if location:
#                     df.at[index, 'Latitude'] = location.latitude
#                     df.at[index, 'Longitude'] = location.longitude
#                     print(f"Processed: {location_name} -> Lat: {location.latitude}, Long: {location.longitude}")
#                 else:
#                     df.at[index, 'Latitude'] = 'N/A'
#                     df.at[index, 'Longitude'] = 'N/A'
#                     print(f"Location not found: {location_name}")
#             except Exception as e:
#                 print(f"Error processing {location_name}: {e}")
#                 df.at[index, 'Latitude'] = 'N/A'
#                 df.at[index, 'Longitude'] = 'N/A'

#         output_file = 'geocoded_locations123.xlsx'
#         df.to_excel(output_file, index=False)
#         print(f"Geocoded data saved to {output_file}")

#     except Exception as e:
#         print(f"General Error: {e}")
#         traceback.print_exc()

# file_path = r'C:\Users\91861\OneDrive\Desktop\bhoodevi\WebScraping\coo.xlsx'
# geocode_locations(file_path)


import pandas as pd
import googlemaps
import os
import time
from dotenv import load_dotenv

load_dotenv(r"C:\Users\91861\OneDrive\Desktop\bhoodevi\WebScraping\.env")

def geocode_with_google(file_path, api_key):
    df = pd.read_excel(file_path)
    gmaps = googlemaps.Client(key=api_key)

    for index, row in df.iterrows():
        try:
            geocode_result = gmaps.geocode(row['Location'])
            if geocode_result:
                location = geocode_result[0]['geometry']['location']
                df.at[index, 'Latitude'] = location['lat']
                df.at[index, 'Longitude'] = location['lng']
            else:
                df.at[index, 'Latitude'] = 'N/A'
                df.at[index, 'Longitude'] = 'N/A'
            print(f"Processed: {row['Location']}")
        except Exception as e:
            print(f"Error: {e}")
            df.at[index, 'Latitude'] = 'N/A'
            df.at[index, 'Longitude'] = 'N/A'
        time.sleep(1)  # To prevent exceeding query limit

    output_file = 'geocoded_with_google2.xlsx'
    df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")

# Replace with your file path and API key
file_path = r'C:\Users\91861\OneDrive\Desktop\bhoodevi\WebScraping\coo.xlsx'
api_key = os.getenv("Google1_api_key")
geocode_with_google(file_path, api_key)      
