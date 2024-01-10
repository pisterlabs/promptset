import logging
import os
from pathlib import Path

import boto3
import environ
import openai

BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env()
env_file = os.path.join(BASE_DIR, '.env')
environ.Env.read_env(env_file)

# Load the .env file

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BUCKET_NAME = env("BUCKET_NAME")
OPENAI_API_KEY = env("OPENAI_API_KEY")

# print(BUCKET_NAME)
# print(OPENAI_API_KEY)

AWS_ACCESS_KEY_ID = env("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = env("AWS_SECRET_ACCESS_KEY")
AWS_S3_REGION_NAME = 'ap-northeast-2'

AWS_STORAGE_BUCKET_NAME = BUCKET_NAME
AWS_BUCKET_ACL = "public-read"
AWS_S3_CUSTOM_DOMAIN = '%s.s3.%s.amazonaws.com' % (
    AWS_STORAGE_BUCKET_NAME, AWS_S3_REGION_NAME)
AWS_S3_OBJECT_PARAMETERS = {
    'CacheControl': 'max-age=86400',
}
DEFAULT_FILE_STORAGE = "config.storages.MediaStorage"


# AWS CLIENT INITIALIZE
s3 = boto3.client('s3',
                  region_name="ap-northeast-2",
                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

polly = boto3.client('polly',
                     region_name="ap-northeast-2",
                     aws_access_key_id=AWS_ACCESS_KEY_ID,
                     aws_secret_access_key=AWS_SECRET_ACCESS_KEY)


openai.api_key = OPENAI_API_KEY
polly_voice = "Seoyeon"
