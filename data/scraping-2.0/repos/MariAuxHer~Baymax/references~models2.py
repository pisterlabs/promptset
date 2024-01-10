from django.db import models
#from django.contrib.auth.models import User
from django.contrib.auth.models import AbstractUser
from django.utils import timezone

from django.conf import settings
import openai
from openai import OpenAI

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom User 
class CustomUser(AbstractUser):
    city = models.CharField(max_length=255)
    state = models.CharField(max_length=255)
    zipcode = models.CharField(max_length=10)

    # groups = models.ManyToManyField(
    #     'auth.Group',
    #     verbose_name='groups',
    #     blank=True,
    #     related_name="customuser_set",
    #     related_query_name="customuser",
    #     help_text=(
    #         'The groups this user belongs to. A user will get all permissions '
    #         'granted to each of their groups.'
    #     ),
    # )
    # user_permissions = models.ManyToManyField(
    #     'auth.Permission',
    #     verbose_name='user permissions',
    #     blank=True,
    #     related_name="customuser_set",
    #     related_query_name="customuser",
    #     help_text='Specific permissions for this user.',
    # )

# Holds Interactions
class Conversation(models.Model):
    owner = models.ForeignKey(CustomUser, on_delete=models.CASCADE, null=True)
    last_accessed = models.DateTimeField(null=True)
    creation_time = models.DateTimeField(null=True)
    name = models.CharField(max_length=100)

    def save(self, *args, **kwargs):
        time = timezone.now()
        # set the times on creation
        if not self.pk: # pk isn't assigned until after creation, so this checks for if a save is a creation
            self.creation_time = time
            self.last_accessed = time
        super(Conversation, self).save(*args, **kwargs)

    # updates the access time manually
    def update_access_time(self):
        self.last_accessed = timezone.now()
        self.save()

    def __str__(self):
        return self.name
    
# Holds prompt and responses
class Interaction(models.Model):
    creation_time = models.DateTimeField(null = True)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    owner = models.ForeignKey(CustomUser, on_delete=models.CASCADE, null = True)

    prompt : str =  models.CharField(max_length=1000, null = True)
    LLMresponse : str = models.CharField(max_length=1000)
    # maybe adjust the length here accordingly.......

    def generate_LLMResponse(self):
        #self.LLMresponse = "sample LLMResponse"
        # send self.prompt to the LLM
        # get the response back
        # set equal to the LLM Response
#        pass
        if self.prompt:
            if settings.OPENAI_API_KEY is not None:
                print("apikey: " + settings.OPENAI_API_KEY)
            else:
                print("OpenAI API key is not set.")
            try:
                client = OpenAI(api_key=settings.OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model='gpt-4',
                    messages=[{'role': 'user', 'content': self.prompt}]
                )
                self.LLMresponse = response.choices[0].message.content.strip()
                print(self.LLMresponse)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error("An error occurred in OpenAI API interaction: %s", str(e))
                self.LLMresponse = "Error: Unable to get response."

    def save(self, *args, **kwargs):
        if not self.pk: # pk isn't assigned until after creation, so this checks for if a save is a creation
            self.creation_time = timezone.now()
            if (self.prompt): 
                self.generate_LLMResponse()

        super(Interaction, self).save(*args, **kwargs)

#class Interaction(models.Model):
#    creation_time = models.DateTimeField(null = True)
#    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
#    owner = models.ForeignKey(CustomUser, on_delete=models.CASCADE, null = True)

#    prompt : str =  models.CharField(max_length=1000, null = True)
#    LLMresponse : str = models.CharField(max_length=1000)
    # maybe adjust the length here accordingly.......

#    def save(self, *args, **kwargs):
#        if not self.pk: # pk isn't assigned until after creation, so this checks for if a save is a creation
#            self.creation_time = timezone.now()
            #if (self.prompt): 
            #    self.LLMResponse

#        super(Interaction, self).save(*args, **kwargs)


# holds medical service information
# class Provider(models.Model):
#     prov : str = models.CharField(max_length=500) # provider name
#     location : str = models.CharField(max_length=500)
#     m_address : str = models.CharField(max_length=500) # mailing address
#     phone : str = models.CharField(max_length=500)
    
    
    # populate more fields or do it in code