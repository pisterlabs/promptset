import spotifyRecommendScript
import pickle
import tensorflow as tf
import json
import sys
sys.path.append("../backend/ml_models/config.py")
# from config import OPENAI_API_KEY, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI
import os
from dotenv import load_dotenv


load_dotenv()

# CLIENT_ID = "85828d1937e346c8a174c74766c1bb89"
# CLIENT_SECRET = "de99228ea35a4287bd2f1e25d35dec36"
# REDIRECT_URI = "http://localhost:8081/callback"

os.environ["SPOTIPY_CLIENT_ID"] = os.getenv("SPOTIFY_CLIENT_ID")
os.environ["SPOTIPY_CLIENT_SECRET"] = os.getenv("SPOTIFY_CLIENT_SECRET")
os.environ["SPOTIPY_REDIRECT_URI"] = os.getenv("SPOTIFY_REDIRECT_URI")

# CLIENT_ID = SPOTIFY_CLIENT_ID
# CLIENT_SECRET = SPOTIFY_CLIENT_SECRET
# REDIRECT_URI = SPOTIFY_REDIRECT_URI

def script_run(input_mood):
    try:

        # file_path = '/usr/src/app/ml_models/spotify_recommendation/tokenizer.pkl'
        # file_path = '../backend/ml_models/spotify_recommendation/tokenizer.pkl'
        # if os.path.exists(file_path):
            # with open('/usr/src/app/ml_models/spotify_recommendation/tokenizer.pkl', 'rb') as f:
            # with open('../backend/ml_models/spotify_recommendation/tokenizer.pkl', 'rb') as f:
                # scaler = pickle.load(f)

        # model = tf.keras.models.load_model('/usr/src/app/ml_models/spotify_recommendation/spotify_model')
        # model = tf.keras.models.load_model('../backend/ml_models/spotify_recommendation/spotify_model')

        # result = spotifyRecommendScript.getRecommendation(input_mood, model, scaler)

        # print({input_mood})
        # result = '["1HNkqx9Ahdgi1Ixy2xkKkL", "1ei3hzQmrgealgRKFxIcWn", "7eJMfftS33KTjuF7lTsMCx"]'
        # result = '["1bzsi0xA8MxBZICdtB23pO", "6XSCLKQcavtn1PaLAhmo1x", "0sfCRhw6gS3b1WepmTVPZE", "6u8KnNnGthVY4PaWiH2mgt", "4FIGOj3O3cyIf6Dg2TDoVv", "0FSk1OsGWoTCS8lA74x5lF", "55eUtE9v3uatR0y0NijM6N", "1bzsi0xA8MxBZICdtB23pO", "6XSCLKQcavtn1PaLAhmo1x", "0sfCRhw6gS3b1WepmTVPZE"]'
        result = '["0wjno8lHmDep5uhLVowoSA", "42maX10SkL9u6KM2dCdgPm", "0ABeQP5ffDRiZTe3aorRGn", "3QPFWCPWmL7fed8jCQ8jty", "59s47nDkFz1UNXujnpzEPt", "4c2JHXheN3t6Wh13lqj91s", "6eABugchKfRPyrzUpv7UzJ", "42maX10SkL9u6KM2dCdgPm", "6PJrnbdGmbiXjFceEFlnaR", "4cFm2ZZcb4mQMXwKCeJgzs"]'
        # result = json.dumps(result)
        # response = str(result)
        output = {"result": result}

        output_json = json.dumps(output)
        # sys.stdout.flush()
        # print(result)
        print(output_json)

    except Exception as e:
        error_message = str(e)
        output = {"error2011": error_message}

        output_json = json.dumps(output)
        print(output_json)
        sys.stdout.flush()

        # return(output)
        # print(output)
    

if __name__ == "__main__":
    # spotifyRecommendScript.initialize()

    # Print current working directory
    # print("Current working directory:", os.getcwd())

    # file_path = '../spotify_recommendation/tokenizer.pkl'
    # print("File exists:", os.path.exists(file_path))

    # #######################################################################################################################################
    # '''Load saved model and scaler'''

    # if os.path.exists(file_path):
    #     with open('ml_models/spotify_recommendation/tokenizer.pkl', 'rb') as f:
    #         scaler = pickle.load(f)

    # model = tf.keras.models.load_model('ml_models/spotify_recommendation/spotify_model')

    # #######################################################################################################################################

    # print(json.dumps({"Results": spotifyRecommendScript.getRecommendation(sys.argv[1], model, scaler)}))

    # if len(sys.argv) > 1:
    #     # The first command-line argument (sys.argv[1]) will be the input_mood
    input_mood = sys.argv[1]
    #     # input_mood = 'joy'

    # Call the function with the received input_mood
    script_run(input_mood)
    # print({input_mood})

    #     # script_run("happiness")
        
    # else:
    #     # If no input_mood is provided, handle the case accordingly
    #     print(json.dumps({"error": "No mood provided"}))
    #     sys.stdout.flush()
    # script_run('joy')

    # file_path = 'ml_models/spotify_recommendation/tokenizer.pkl'
    # if os.path.exists(file_path):
    #     with open('ml_models/spotify_recommendation/tokenizer.pkl', 'rb') as f:
    #         scaler = pickle.load(f)

    # model = tf.keras.models.load_model('ml_models/spotify_recommendation/spotify_model')
    # print(spotifyRecommendScript.getRecommendation('joy', model, scaler))



# import spotifyRecommendScript
# import pickle
# import tensorflow as tf
# import json
# import sys

# if __name__ == "__main__":
#     # spotifyRecommendScript.initialize()

#     #######################################################################################################################################
#     '''Load saved model and scaler'''
#     print('Loading saved model and scaler')
#     with open('ml_models/spotify_recommendation/tokenizer.pkl', 'rb') as f:
#         scaler = pickle.load(f)

#     model = tf.keras.models.load_model('ml_models/spotify_recommendation/spotify_model')
#     print('Model loaded successfully')
#     #######################################################################################################################################
#     print(spotifyRecommendScript.getRecommendation("joy", model, scaler))

#     print('Done')
#     # print(json.dumps({"Results": spotifyRecommendScript.getRecommendation(sys.argv[1], model, scaler)}))