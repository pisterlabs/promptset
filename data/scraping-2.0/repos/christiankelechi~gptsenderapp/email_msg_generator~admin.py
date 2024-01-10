from django.contrib import admin
from email_msg_generator.models import OpenAiUserModel,OpenAiAdminModel

admin.site.register(OpenAiUserModel)
admin.site.register(OpenAiAdminModel)
# Register your models here.
