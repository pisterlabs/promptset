import fbchat

import cookies
from src import sensitive
import openai
import people
import osrs
import generate_image
import crime
import bikes
import speech
import upload_fix
import command_handler

from itertools import islice


class MessageHandler(fbchat.Client):

    def __init__(self, email, password, session_cookies):
        super().__init__(email, password, session_cookies)
        self.osrs_items = osrs.OSRSItems()
        self.people_to_respond_to = []
        self.respond_to = people.RespondTo(self.people_to_respond_to)
        self.all_bikes = bikes.AllBikes()

    def onMessage(
            self,
            mid=None,
            author_id=None,
            message=None,
            message_object=None,
            thread_id=None,
            thread_type=fbchat.ThreadType.USER,
            ts=None,
            metadata=None,
            msg=None,
    ):
        if author_id == self.uid:
            return

        command, user_input = command_handler.extract_from(message_object)

        match command:
            case "genfill":
                image_url = generate_image.get_filled_url_from(user_input,"group_full.png", "group.png")
                self.reply_with_image_at_url(message_object, thread_id, thread_type, image_url)

            case "gen":
                image_url = generate_image.get_url_from(user_input)
                self.reply_with_image_at_url(message_object, thread_id, thread_type, image_url)

            case "say":
                self.reply_with_local_voice_clip_from(message_object, thread_id, thread_type, user_input)

            case "examine":
                examine_text = self.osrs_items.get_examine_text_from_item_name(user_input)
                self.reply_with_text(message_object, thread_id, thread_type, examine_text)

            case "echoimages":
                for image in islice(self.fetchThreadImages(thread_id), 100):
                    if not hasattr(image, 'thumbnail_url'):
                        continue
                    self.sendRemoteImage(image.thumbnail_url, self.message_object(), thread_id=thread_id,
                                         thread_type=thread_type)
            case "randomimage":
                image_count_in_thread = 0
                for count, image in enumerate(islice(self.fetchThreadImages(thread_id), 10000)):
                    image_count_in_thread = count
                print(image_count_in_thread)

            case "bike":
                bike_info = self.all_bikes.find(user_input)
                self.reply_with_text(message_object, thread_id, thread_type, bike_info)

            case "crime":
                lat, long = crime.get_lat_long_from_postcode(user_input)
                self.reply_with_location(message_object, thread_id, thread_type, self.create_location_from_lat_long_address(lat, long, user_input))
                crime.create_plot_from_postcode_at(user_input, "tmp_plot.png")
                self.reply_with_local_image_at(message_object, thread_id, thread_type, "tmp_plot.png")

            case _:
                if self.fetchUserInfo(author_id)[author_id].name not in self.people_to_respond_to:
                    return
                message = self.message_object()
                message.text = \
                    self.respond_to.get_response_from_name_and_message(
                        self.fetchUserInfo(author_id)[author_id].name, message_object.text)
                self.send(message, thread_id=thread_id, thread_type=thread_type)

    def _upload(self, files, voice_clip=False):
        return upload_fix.upload(self, files, voice_clip=voice_clip)

    @staticmethod
    def message_object(emoji_size=None, reply_to_id=None, sticker=None, text=None):
        return fbchat.Message(emoji_size=emoji_size, sticker=sticker, text=text, reply_to_id=reply_to_id)

    def reply_with_image_at_url(self, input_message, thread_id, thread_type, image_url):
        response_message = self.message_object()
        response_message.reply_to_id = input_message.uid
        self.sendRemoteImage(image_url, response_message, thread_id, thread_type)

    def reply_with_local_image_at(self, input_message, thread_id, thread_type, image_path):
        response_message = self.message_object()
        response_message.reply_to_id = input_message.uid
        self.sendLocalImage(image_path, response_message, thread_id, thread_type)

    def reply_with_local_voice_clip_from(self, input_message, thread_id, thread_type, text):
        response_message = self.message_object()
        response_message.reply_to_id = input_message.uid
        speech.create_audio_file_from_at(text, "tmp.mp3")
        self.sendLocalVoiceClips(["tmp.mp3"], response_message, thread_id, thread_type)

    def reply_with_text(self, input_message, thread_id, thread_type, text):
        response_message = self.message_object()
        response_message.reply_to_id = input_message.uid
        response_message.text = text
        self.send(response_message, thread_id, thread_type)

    def create_location_from_lat_long_address(self, lat, long, address):
        return fbchat.LocationAttachment(lat, long, address, self.uid)

    def reply_with_location(self, input_message, thread_id, thread_type, location):
        response_message = self.message_object()
        response_message.reply_to_id = input_message.uid
        self.sendLocation(location, response_message, thread_id, thread_type)


def start():
    openai.api_key = sensitive.api_key
    client = MessageHandler(sensitive.email, sensitive.password, session_cookies=cookies.read())
    cookies.store(client.getSession())
    client.listen()
