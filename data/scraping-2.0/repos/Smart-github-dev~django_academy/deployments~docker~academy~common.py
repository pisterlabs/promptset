"""
Django common for academy project.

Here all common classes and functions stored so we can reference withing each application
All commonly used fucntion or classes should be stored here
"""

from accounting.models import Subscriber
from django.contrib.auth.models import User
from videos.models import VideoFolder
from datetime import datetime, timedelta
from messenger import get_messenger
from django.utils import timezone
from django.conf import settings
from vimeo import VimeoClient
import openai


def is_user_need_payment(username):
    user = User.objects.get(username=username)
    if Subscriber.objects.filter(user=user):
        subscription = Subscriber.objects.get(user=user)
        if subscription.plan.name.lower() == 'basic' or subscription.plan.name.lower() == 'contributor':
            if not subscription.is_payment_confirmed():
                return True
        return subscription.is_expired()
    else:
        return True

def is_folder_has_new_videos(folder):
    query_date=getattr(settings, 'FLAG_VIDEOS_LESS_DATE', None)
    if VideoFolder.objects.filter(created_date__gte=timezone.now()-timedelta(days=query_date)):
        return True
    else:
        return False


def is_video_new(video):
    query_date=getattr(settings, 'FLAG_VIDEOS_LESS_DATE', None)
    if video in VideoFolder.objects.filter(created_date__gte=timezone.now()-timedelta(days=query_date)):
        return True
    else:
        return False


def is_new_video_uploaded():
    query_date=getattr(settings, 'FLAG_VIDEOS_LESS_DATE', None)
    if VideoFolder.objects.filter(created_date__gt=timezone.now()-timedelta(days=query_date)):
        return True
    else:
        return False


def get_new_videos():
    query_date=getattr(settings, 'FLAG_VIDEOS_LESS_DATE', None)
    videos = VideoFolder.objects.filter(created_date__gt=timezone.now()-timedelta(days=query_date))
    if videos.count() < 10:
        videos = VideoFolder.objects.filter(type='video').order_by('-created_date')[:10]
    return videos


def get_common_variables(username):
    return {
        'need_payment' : is_user_need_payment(username),
        'new_video_uploaded': is_new_video_uploaded(),
    }


def send_message(message):
    slack = get_messenger('slack')
    slack.send_messenge(message)


class Lucy():

    per_page = 25

    def __init__(self):
        self.vimeo = VimeoClient(
            token=getattr(settings, 'VIMEO_ACCESS_TOKEN', None),
            key=getattr(settings, 'VIMEO_CLIENT_ID', None),
            secret=getattr(settings, 'VIMEO_CLIENT_SECRET', None)
            )

        self.openai = openai
        self.openai_model_name = getattr(settings, 'OPENAI_MODEL_NAME', "gpt-3.5-turbo")
        self.openai.api_key = getattr(settings, 'OPEN_AI_TOKEN', None)

        self.academy_contents_folder = getattr(settings, 'ACADEMY_CONTENTS_FOLDER', None)
        self.academy_zoom_folder = getattr(settings, 'ACADEMY_ZOOM_FOLDER', None)
        self.academy_video_title_max_num = getattr(settings, 'ACADEMY_VIDEO_TITLE_MAX_NUM', None)

        ## Ones user provided the main folder name as string
        self.vimeo_account_folders = self.vimeo.get('/me/folders').json()

        ## Looping only first page of the vimeo folders
        for folder in self.vimeo_account_folders['data']:

            if self.academy_contents_folder.lower() == folder['name'].lower():

                ## Getting main folder uri to refernce later
                self.academy_contents_folder_uri = str(folder['uri'])
                self.academy_contents_folder_vimeo = folder

            if self.academy_zoom_folder.lower() == folder['name'].lower():

                ## Getting main folder uri to refernce later
                self.academy_zoom_folder_uri = str(folder['uri'])
                self.academy_zoom_folder_vimeo = folder

    def get_parent_folders(self):
        self.parent_folder_items = self.vimeo.get(f"{self.academy_contents_folder_uri}/items").json()['data']
        return self.parent_folder_items

    def get_child_items(self, parent_folder_uri):
        vimeo_child_items = {
            'videos' : [],
            'folders' : []
        }

        ## Getting all videos and returning data
        response_data = self.vimeo.get(f"{parent_folder_uri}/items").json()

        ## Checking other pages if videos are more
        while True:

            ## Filtering out all videos and folders
            for item in response_data['data']:
                if item not in vimeo_child_items['folders'] or item not in vimeo_child_items['videos']:
                    if item['type'] == 'folder':
                        vimeo_child_items['folders'].append(item)
                    else:
                        vimeo_child_items['videos'].append(item)

            ## If next page is exist
            if response_data['paging']['next']:
                ## Adding all videos to total list
                response_data = self.vimeo.get(response_data['paging']['next']).json()
            else:
                break


        return vimeo_child_items

    def get_folder_videos(self, folder_name):
        ## Function is resposible to find all videos from the folder
        folder_response = self.vimeo.get('/me/folders').json()

        ## Looping all folders in vimeo
        for folder in folder_response['data']:

            ## if folder is matching with <folder_name>
            if folder_name.lower() == folder['name'].lower():

                ## Getting link of the videos as
                link = str(folder['metadata']['connections']['videos']['uri'])
        if link:
            all_videos = []
            ## Getting all videos and returning data
            response_data = self.vimeo.get(link).json()

            ## Checking other pages if videos are more
            while True:

                ## If next page is exist
                if response_data['paging']['next']:

                    ## Adding all videos to total list
                    all_videos = all_videos + response_data['data']
                    response_data = self.vimeo.get(response_data['paging']['next']).json()

                else:
                    ## Before brakes the loop making sure all missing videos are there
                    if response_data['data'][0] not in all_videos:
                        all_videos = all_videos + response_data['data']
                    break

            return all_videos
        else:
            return {'message': 'function can not find link'}

    def get_folders(self):
        ## Function is resposible to find all videos from the folder
        response_data = self.vimeo.get('/me/folders').json()
        return response_data['data']

    def get_groups(self):
        ## Function to get all groups
        response_data = self.vimeo.get('/me/groups').json()
        return response_data['data']

    def get_date_time_format(self, string_date):
        date_time_format = getattr(settings, 'DATE_TIME_FORMAT', None)
        try:
            date_time_object = datetime.fromisoformat(string_date)
            return date_time_object.strftime(date_time_format)
        except:
            return None

    def rename_videos(self):
        ## Picking up the main zoom recording based folder
        parent_folder_items = self.vimeo.get(f"{self.academy_zoom_folder_uri}/items").json()['data']

        ## Going over the each videos and giving proper name
        for recorded_video in parent_folder_items:

            ## Find the link for the video transcripts
            texttrack_urls = self.vimeo.get(f"{recorded_video['video']['uri']}/texttracks").json()
            if texttrack_urls['data']:
                texttrack_url = texttrack_urls['data'][0]['link']

            print(recorded_video)

            ## Getting the multiline transcript and cleaning up
            multiline_texttrack = self.vimeo.get(texttrack_url).text
            cleaned_sctring = self.preprocess_transcript(multiline_texttrack)[0:self.academy_video_title_max_num]


            video_name = self.generate_video_title(cleaned_sctring).replace('"', '')

            data = {
                'name' : video_name
            }
            # [{"role": "system", "content": "Please generate a compelling and relevant title that accurately captures the essence of the video content. The title should be attention-grabbing and appealing to the audience. Feel free to get creative with your suggestions! Once you have the title, kindly provide it in your response."}, {"role": "user", "content": ''}]

            if self.vimeo.patch(recorded_video['video']['uri'], data=data).status_code == '200':
                print('The following video :', recorded_video['video']['name'])
                print('Updated to the following name:', video_name)


    def generate_video_title(self, transcript):

        # Generate the title using the GPT-3.5 model
        response = self.openai.ChatCompletion.create(
            model=self.openai_model_name,
            messages=[{"role": "system", "content": "Please generate a compelling and relevant title that accurately captures the essence of the video content. The title should be attention-grabbing and appealing to the audience. Feel free to get creative with your suggestions! Once you have the title, kindly provide it in your response."},
                    {"role": "user", "content": transcript}]
        )

        # Extract the generated title from the model's response
        generated_title = response['choices'][0]['message']['content']

        return generated_title


    def ask(self, messages):

        # Generate the title using the GPT-3.5 model
        response = self.openai.ChatCompletion.create(
            model=self.openai_model_name,
            messages=messages
        )

        # Extract the generated content from the model's response
        content = response['choices'][0]['message']['content']

        return content

    def preprocess_transcript(self, transcript):
        cleaned_transcript = transcript.replace('\n', ' ').strip()
        cleaned_transcript = cleaned_transcript.lower()
        return cleaned_transcript


    def update_video_for_repo(self, repo_name):
        pass
