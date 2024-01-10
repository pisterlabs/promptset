from supabase import create_client
import openai

from webapp.server.ml.configsfile import *

client = create_client(config['supabase']['db_link'], config['supabase']['api_key'])