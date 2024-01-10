from rest_framework import serializers
from core.user.models import User
from email_msg_generator.models import EmailGeneratorModel
from .models import EmailMessageModel
class EmailMessageSerializer(serializers.ModelSerializer):
    class Meta:
        fields=['title','message_body','maillist','site_url']
        model=EmailMessageModel

class EmailSerializer(serializers.ModelSerializer):
    number_of_mail = serializers.IntegerField()
    prompt = serializers.CharField(max_length=2000)
    
    # generated_emails = serializers.FileField()  # New field for generated emails
    # swapped_emails = serializers.FileField()  # New field for swapped emails

    class Meta:
        fields=['number_of_mail','prompt','access_token']
        model=EmailGeneratorModel

class OpenAiModelSerializers(serializers.ModelSerializer):
    custom_user_key_id=serializers.IntegerField()
    open_ai_key=serializers.CharField()


    class Meta:
        model=User
        fields=['custom_user_key_id','open_ai_key']

from rest_framework import serializers
from .models import OpenAiUserModel
class OpenAiUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = OpenAiUserModel
        fields = ['open_ai_key','custom_user_key_id','open_ai_key','time_of_assigning'
    ]