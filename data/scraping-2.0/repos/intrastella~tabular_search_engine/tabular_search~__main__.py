import argparse
import warnings

from request_engine import PreviewModel

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-o_key", "--openai_api_key", help="API key from OpenAI.", required=True, nargs='+')
parser.add_argument("-b_key", "--bing_api_key", help="API key from Bing.", required=True, nargs='+')
parser.add_argument("-n", "--o_name", help="File name of excel.", dest="name", required=True)
parser.add_argument("-r", "--request", help="Your request.", dest="rows", required=True, nargs='+')
parser.add_argument("-add", "--add_info", help="Required additional info to your request.", dest="columns",
                    required=False, nargs='+')

args = parser.parse_args()

pm = PreviewModel(**vars(args))
pm.run()
