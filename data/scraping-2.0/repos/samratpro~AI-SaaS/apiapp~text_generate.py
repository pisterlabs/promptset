from userapp.models import AppUser
from dashapp.models import ApiList
from openai import OpenAI
from django.db.models import F


def text_render(previous_prompt, prompt, bulkmodel):
        try:
            api_count = ApiList.objects.all()
            i = 0
            while i < api_count:
                api= ApiList.objects.filter(filled_quota__lt=F('request_quota_limit')).first()
                print('bulk info , type(api) : ', type(api))
                print('bulk info , api : ', api)
                print('bulk info ,api.filled_quota : ', api.filled_quota)
                api.error_status = f'API filled quota : {str(api.filled_quota)} ,  request_quota_limit : {str(api.request_quota_limit)}' 
                api.save()
                if api != None:
                    api.filled_quota += 1

                if AppUser.credit > 100:
                    client = OpenAI(api_key=api.api_key)
                    print('prompts ', prompt)
                    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": previous_prompt},
                            {"role": "user","content": prompt}
                            ],
                            model="gpt-3.5-turbo",
                        )
                    print('bulk info ,api.filled_quota : ', api.filled_quota)
                    api.filled_quota -= 1
                    print('bulk info ,api.filled_quota : ', api.filled_quota)
                    words =  response.choices[0].message.content
                    AppUser.use_credit(words)
                    return words
                else:
                     'Please purchase credit'
        except Exception as oops:
            api.error_status = 'Error Message from OpenAI server : ' + str(oops)
            bulkmodel.eror = 'Error Message from OpenAI server : ' + str(oops)
            bulkmodel.save()
            api.save()
            pass
        return 'API Error From Server'