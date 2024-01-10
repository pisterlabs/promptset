from django.http.response import JsonResponse
from rest_framework.response import Response
import logging
from .serializer import *
from logtail import LogtailHandler
from datetime import datetime
import openai
import weaviate
import os
from memeModel import idl
from memeGeneration import handler as memeGenerationHandler
from memeMatching import handler as memeMatchingHandler

import requests
import base64

# handler = LogtailHandler(source_token="tvoi6AuG8ieLux2PbHqdJSVR")
# logger = logging.getLogger(__name__)
# logger.handlers = []
# logger.addHandler(handler)
# logger.setLevel(logging.INFO)

def meme_generation(generateTextForMemeRequest):
  """
    Generate text for meme image
  """

  # Match text to meme
  matchTextToMemeResponse = memeMatchingHandler.match_text_to_meme(
    MatchTextToMemeRequest(
      InputText=generateTextForMemeRequest.InputText,
    )
  )
  if matchTextToMemeResponse.Error != None:
    return MemeGenerationResponse(
      Memes= [],
      Error=str(matchTextToMemeResponse.Error )
    )

  # Generate text for meme
  generateTextForMemeResponse = memeGenerationHandler.generate_text_for_meme(
    GenerateTextForMemeRequest(
      MemeImage= matchTextToMemeResponse.Memes[0],
      InputText=generateTextForMemeRequest.InputText,
    )
  )
  if generateTextForMemeResponse.Error != None:
    return MemeGenerationResponse(
      Memes= [],
      Error=str(generateTextForMemeResponse.Error )
    )

  # Create meme image
  createMemeImageResponse = memeGenerationHandler.generate_meme_image(
    GenerateMemeImageRequest(
      MemeImage=matchTextToMemeResponse.Memes[0]['image'],
      MemeText= generateTextForMemeResponse.OutputText,
    )
  )


  template_url = 'https://hlhmmkpugruknefsttlr.supabase.co/storage/v1/object/public/meme-templates-public/Happy--Shock.png'
  r = requests.get(template_url, allow_redirects=True)
  with open(f'testing.png', 'wb') as f:
      f.write(r.content)

  return createMemeImageResponse



