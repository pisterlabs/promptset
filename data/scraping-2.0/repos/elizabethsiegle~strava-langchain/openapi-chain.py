from langchain.llms import OpenAI
from langchain.requests import Requests
from langchain.chains import OpenAPIEndpointChain
from langchain.tools import OpenAPISpec, APIOperation
from langchain.llms import OpenAI

from dotenv import dotenv_values
# import os
# from langchain.utilities.twilio import TwilioAPIWrapper

config = dotenv_values(".env")

spec = OpenAPISpec.from_file("data_files/swagger.yml")
operation = APIOperation.from_openapi_spec(spec, '/athlete/activities', "get")

strava_token = config.get('STRAVA_TOKEN')
headers = {
    "Authorization": "Bearer {strava_token}".format(strava_token = strava_token),
    "Content-Type": "application/json"
}

llm = OpenAI(openai_api_key=config.get('OPENAI_API_KEY'), temperature=0)
api_chain = OpenAPIEndpointChain.from_api_operation(
    operation, 
    llm, 
    requests=Requests(headers=headers), 
    verbose=False,
    return_intermediate_steps=True
)
last_run_far = api_chain("how far was Lizzie's last run? how many minutes did it take her?")
print(last_run_far['output']) # Lizzie's last run was 1752.5 meters and it took her 1322 minutes.
most_recent_run = api_chain("When was her most recent run?")
print(most_recent_run['output']) # The most recent run was on July 6th, 2023 at 9:59am local time and was 1752.5 meters long.
last_ride = api_chain("when was Lizzie's last activity of sport_type ride?")
print(last_ride['output'])
num_runs = api_chain("how many runs has Lizzie done")
print(num_runs['output']) # The API response does not contain any information about the number of workouts of type ride that the athlete has done. Please provide more information about the athlete and the type of workout you are looking for
num_ride_workouts = api_chain("how many workouts of type ride has she done?")
print(num_ride_workouts['output']) # The API response does not contain any information about the number of workouts of type ride that the athlete has done. Please provide more information about the athlete and the type of workout you are looking for
num_swims = api_chain("How many swims has she done?") 
print(num_swims['output']) # The API response does not contain any information about swims. It contains information about a walk that was taken on July 6th, 2023.
fastest_half_run = api_chain("What was Lizzie's fastest half marathon run?")
print(fastest_half_run['output']) # The API response does not contain any information about Lizzie's fastest half marathon run. Please provide more information about the run you are looking for.
fastest_run_km = api_chain("What was her fastest run of at least 21.0975km?")
print(fastest_run_km['output']) # The API response does not contain any information about the fastest run of at least 21.0975km. Please provide more information about the run you are looking for.
walks_elev_gain_over100 = api_chain("How many walks with total elevation gain over 100 has she taken?")
print(walks_elev_gain_over100['output']) 

# twilio = TwilioAPIWrapper(
#     account_sid=os.environ['TWILIO_ACCOUNT_SID'],
#     auth_token=os.environ['TWILIO_AUTH_TOKEN'],
#     from_number=os.environ['MY_TWILIO_NUMBER']
# )
# twilio.run(output2['output'], os.environ['MY_PHONE_NUMBER'])

