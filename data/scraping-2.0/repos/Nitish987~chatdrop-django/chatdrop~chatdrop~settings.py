import os
import firebase_admin
import openai

from os import getenv
from dotenv import load_dotenv
from pathlib import Path
from google.oauth2 import service_account
from firebase_admin import credentials


load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f'{Path(__file__).resolve().parent.parent}\\firebase_credentials.json'

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = getenv('SECRET_KEY')

DEBUG = True

ALLOWED_HOSTS = ['10.0.2.2', '127.0.0.1', getenv('HOST_IP')]


# Application definition
INSTALLED_APPS = [
    'daphne',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'account',
    'notifier',
    'post',
    'friends',
    'stories',
    'fanfollowing',
    'feeds',
    'search',
    'chat',
    'privacy',
    'reel',
    'web',
    'facility',
    'dbbackup',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'chatdrop.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

ASGI_APPLICATION = 'chatdrop.asgi.application'


# User Model
AUTH_USER_MODEL = 'account.User'


# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': getenv('DATABASE_NAME'),
        'USER': getenv('DATABASE_USER'),
        'PASSWORD': getenv('DATABASE_PASSWORD'),
        'HOST': getenv('DATABASE_HOST'),
        'PORT': getenv('DATABASE_PORT'),
        'OPTIONS': {
            'charset': 'utf8mb4',
        },
    }
}


# Cache
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/1",
        "OPTIONS": {
            "PASSWORD": getenv('REDIS_PASSWORD'),
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    }
}


# Channel Layers
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [(f"redis://:{getenv('REDIS_PASSWORD')}@127.0.0.1:6379/0")],
        },
    },
}


# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Google Cloud Storage Configurations
DEFAULT_FILE_STORAGE = 'utils.storages.CloudMediaStorage'
GS_MEDIA_BUCKET_NAME = getenv('GS_MEDIA_BUCKET_NAME')
GS_PROJECT_ID = getenv('GS_PROJECT_ID')
GS_CREDENTIALS = service_account.Credentials.from_service_account_file(str(BASE_DIR / 'gcloud_credentials.json'))


# Firebase Configurations
FB_CREDENTIALS = credentials.Certificate(str(BASE_DIR / 'firebase_credentials.json'))
firebase_admin.initialize_app(FB_CREDENTIALS)


# Rest API Framework configurations
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': (
        'rest_framework.renderers.JSONRenderer',
    ),
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'account.authentication.UserAuthentication',
    ),
    'DEFAULT_THROTTLE_RATES': {
        'signup': '10/min',
        'signup_verification': '100/min',
        'resent_signup_otp': '10/min',
        'login': '10/min',
        'password_recovery': '10/min',
        'password_recovery_verification': '100/min',
        'password_recovery_new_password': '10/min',
        'resent_password_recovery_otp': '10/min',
        'authenticated_user': '1000/hour',
        'change_names': '3/day',
        'chatgpt': '25/day',
        'logout': '10/min',
    },
    'EXCEPTION_HANDLER': 'utils.exceptions.ExceptionHandler'
}


# Jwt Config
JWT_SECRET = getenv('JWT_SECRET')


# Chatdrop api key
CHATDROP_API_KEY = getenv('CHATDROP_API_KEY')

# Account Creation Key
ACCOUNT_CREATION_KEY = getenv('ACCOUNT_CREATION_KEY')

# Chatdrop openai api key
OPENAI_API_KEY = getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# Encryption Key
SERVER_ENC_KEY = getenv('SERVER_ENC_KEY')


# Token expire time
SIGNUP_EXPIRE_SECONDS = 10 * 60 # 10 minute
PASSWORD_RECOVERY_EXPIRE_SECONDS = 10 * 60 # 10 minute
ONE_DAY_EXPIRE_SECONDS = 1 * 24 * 60 * 60 # 1 day
OTP_EXPIRE_SECONDS = 3 * 60 # 3 minute
RESENT_OTP_EXPIRE_SECONDS = 5 * 60 # 5 minute
PASSWORD_EXPIRE_SECONDS = 5 * 60 # 5 minute
AUTH_EXPIRE_SECONDS = 30 * 24 * 60 * 60 # 30 days


# Internationalization
LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'Asia/Kolkata'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'

# Media files
MEDIA_URL = f'https://storage.googleapis.com/{GS_MEDIA_BUCKET_NAME}/'

# CDN
CDN_MEDIA_URL = getenv("CDN_MEDIA_URL")

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Security
# CSRF_TRUSTED_ORIGINS = ['http://127.0.0.1', f'http://{getenv("HOST_IP")}']

# CSRF_COOKIE_SECURE = True

# SESSION_COOKIE_SECURE = True

# Email config
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = getenv('EMAIL_HOST')
EMAIL_USE_TLS = True
EMAIL_PORT = int(getenv('EMAIL_PORT'))
EMAIL_HOST_USER = getenv('EMAIL_USER')
EMAIL_HOST_PASSWORD = getenv('EMAIL_PASSWORD')

# Database Backup
DBBACKUP_STORAGE_OPTIONS = {
    'location': str(BASE_DIR.parent / 'backups'),
}