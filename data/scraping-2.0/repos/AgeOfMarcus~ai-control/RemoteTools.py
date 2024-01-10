from langchain.tools import BaseTool
from uuid import uuid4
from pydantic import BaseModel, Field, Extra
import requests
import json
import os

class BaseRemoteTool(BaseTool):
    url: str = Field(default_factory=lambda: os.getenv('TERMUX_AGENT_URL', 'http://localhost:8080'))

    def _send_cmd(self, cmd: str):
        return requests.post(f'{self.url}/run', json={'cmd': cmd}).json()

    class Config(BaseTool.Config):
        arbitrary_types_allowed = True
        extra = Extra.forbid

class BatteryStatusTool(BaseRemoteTool, BaseTool):
    name = 'BatteryStatus'
    description = (
        'Get the current battery status of the device.'
        'Useful for determining if the device is charging or not.'
        'Useful for getting the battery level.'
        'Does not accept any arguments.'
    )

    def _run(self, *args):
        return self._send_cmd('termux-battery-status')
    
    async def _arun(self, *args):
        return self._run(*args)

class BrightnessTool(BaseRemoteTool, BaseTool):
    name = 'Brightness'
    description = (
        'Set the screen brightness.'
        'Useful for setting the screen brightness.'
        'Accepts an integer between 0 and 255 OR the string "auto".'
    )

    def _run(self, level: int = 'auto'):
        return self._send_cmd(f'termux-brightness {level}')

    async def _arun(self, level: int = 0):
        return self._run(level)

class PhotoTool(BaseRemoteTool, BaseTool):
    name = 'TakePhoto'
    description = (
        'Take a photo with the device camera.'
        'Useful for taking a photo with the device camera.'
        'Accepts a single optional integer camera id. Default is 0.'
    )

    def _run(self, camera_id: int = 0):
        fn = f'photo-{str(uuid4())}.png'
        resp = self._send_cmd(f'termux-camera-photo -c {camera_id} {fn}')
        return {'file': fn, 'response': resp}

    async def _arun(self, camera_id: int = 0):
        return self._run(camera_id)

class ClipboardGetTool(BaseRemoteTool, BaseTool):
    name = 'GetClipboard'
    description = (
        'Get the current clipboard contents.'
        'Useful for getting the current clipboard contents.'
        'Does not accept any arguments.'
    )

    def _run(self, *args):
        return self._send_cmd('termux-clipboard-get')

    async def _arun(self, *args):
        return self._run(*args)

class ClipboardSetTool(BaseRemoteTool, BaseTool):
    name = 'SetClipboard'
    description = (
        'Set the current clipboard contents.'
        'Useful for setting the current clipboard contents.'
        'Accepts a single string.'
    )

    def _run(self, text: str):
        return self._send_cmd(f'termux-clipboard-set {text}')

    async def _arun(self, text: str):
        return self._run(text)

class FingerprintTool(BaseRemoteTool, BaseTool):
    name = 'Fingerprint'
    description = (
        'Asks the user to scan their fingerprint.'
        'Useful for getting authentication from the device owner.'
        'Does not accept any arguments.'
    )

    def _run(self, *args):
        return self._send_cmd('termux-fingerprint')

    async def _arun(self, *args):
        return self._run(*args)

class LocationTool(BaseRemoteTool, BaseTool):
    name = 'GetLocation'
    description = (
        'Get the current location of the device.'
        'Useful for getting the current location of the device.'
        'Does not accept any arguments.'
        'Returns a dict containing latitude and longitude.'
    )

    def _run(self, *args):
        return self._send_cmd('termux-location')

    async def _arun(self, *args):
        return self._run(*args)

class MediaPlayTool(BaseRemoteTool, BaseTool):
    name = 'PlayMedia'
    description = (
        'Play a media file.'
        'Useful for playing a media file.'
        'Accepts a single string file path.'
    )

    def _run(self, path: str):
        return self._send_cmd(f'termux-media-player play {path}')

    async def _arun(self, path: str):
        return self._run(path)

class MediaPauseTool(BaseRemoteTool, BaseTool):
    name = 'PauseMedia'
    description = (
        'Pause the currently playing media.'
        'Useful for pausing the currently playing media.'
    )

    def _run(self, *args):
        return self._send_cmd('termux-media-player pause')

    async def _arun(self, *args):
        return self._run(*args)

class RecordMicTool(BaseRemoteTool, BaseTool):
    name = 'RecordMicrophone'
    description = (
        'Record audio from the device microphone.'
        'Useful for recording audio from the device microphone.'
        'Accepts a single integer duration in seconds.'
        'Returns filename.'
    )

    def _run(self, duration: int):
        fn = f'record-{str(uuid4())}.mp3'
        resp = self._send_cmd(f'termux-microphone-record -d {duration} {fn}')
        return {'file': fn, 'response': resp}

    async def _arun(self, duration: int = 3):
        return self._run(duration)

class NotificationTool(BaseRemoteTool, BaseTool):
    name = 'CreateNotification'
    description = (
        'Send a notification to the device.'
        'Useful for sending a notification to the device.'
        'Accepts one argument, a dictionary in JSON format containing the keys "title" and "message".'
        'If message is not provided, it will be set to the same as title.'
    )

    def _run(self, arguments):
        if type(arguments) == dict:
            notif = arguments
        else:
            try:
                notif = json.loads(arguments)
            except json.JSONDecodeError:
                return {'error': 'Invalid JSON'}
        return self._send_cmd(f'termux-notification -t {notif["title"]} -c {notif.get("message", notif["title"])}')

    async def _arun(self, arguments):
        return self._run(arguments)

class ListNotificationsTool(BaseRemoteTool, BaseTool):
    name = 'ListNotifications'
    description = (
        'List the current notifications on the device.'
        'Useful for listing the current notifications on the device.'
        'Does not accept any arguments.'
        'Returns a list of dicts.'
    )

    def _run(self, *args):
        return self._send_cmd('termux-notification-list')

    async def _arun(self, *args):
        return self._run(*args)

class RemoveNotificationTool(BaseRemoteTool, BaseTool):
    name = 'RemoveNotification'
    description = (
        'Remove a notification from the device.'
        'Useful for removing a notification from the device.'
        'Accepts a single string id.'
    )

    def _run(self, id: str):
        return self._send_cmd(f'termux-notification-remove {id}')

    async def _arun(self, id: str):
        return self._run(id)

class URLOpenerTool(BaseRemoteTool, BaseTool):
    name = 'OpenURL'
    description = (
        'Open a URL in the default browser.'
        'Useful for opening a URL in the default browser.'
        'Accepts a single argument type string, "URL".'
    )

    def _run(self, url: str):
        return self._send_cmd(f'termux-open-url {url}')

    async def _arun(self, url: str):
        return self._run(url)

class TorchTool(BaseRemoteTool, BaseTool):
    name = 'Torch'
    description = (
        'Turn the device torch on or off.'
        'Useful for turning the device torch on or off.'
        'Accepts a single argument type string, "on" or "off".'
    )

    def _run(self, state: str):
        if not state in ('on', 'off'):
            return 'Error: argument must be "on" or "off"'
        return self._send_cmd(f'termux-torch {state}')

    async def _arun(self, state: str):
        return self._run(state)

class SpeakTool(BaseRemoteTool, BaseTool):
    name = 'Speak'
    description = (
        'Speak a string with TTS.'
        'Useful for speaking a string.'
        'Accepts a single argument type string.'
    )

    def _run(self, text: str):
        return self._send_cmd(f'termux-tts-speak {text}')

    async def _arun(self, text: str):
        return self._run(text)

class GetVolumeTool(BaseRemoteTool, BaseTool):
    name = 'GetVolume'
    description = (
        'Get the current volume.'
        'Useful for getting the current volume.'
        'Returns a dict.'
    )

    def _run(self, *args):
        return self._send_cmd('termux-volume')

    async def _arun(self, *args):
        return self._run(*args)

class SetVolumeTool(BaseRemoteTool, BaseTool):
    name = 'SetVolume'
    description = (
        'Set the current volume.'
        'Useful for setting the current volume.'
        'Accepts one argument, a dictionary in JSON format containing the keys "volume_type" (either: "music", "alarm", "notification", or "ring"), and key "value" containing an integer representing volume percentage (0-100).'
    )

    def _run(self, arguments):
        if type(arguments) == dict:
            args = arguments
        else:
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                return {'error': 'Invalid JSON'}
        if not args['volume_type'] in ('music', 'alarm', 'notification', 'ring'):
            return 'Error: type must be "music", "alarm", "notification", or "ring"'
        return self._send_cmd(f'termux-volume {args["volume_type"]} {args["value"]}')

    async def _arun(self, arguments):
        return self._run(arguments)

class WiFiInfoTool(BaseRemoteTool, BaseTool):
    name = 'WiFiInfo'
    description = (
        'Get information about the WiFi connection.'
        'Useful for getting information about the WiFi connection.'
        'Does not accept any arguments.'
        'Returns a dict.'
    )

    def _run(self, *args):
        return self._send_cmd('termux-wifi-connectioninfo')

    async def _arun(self, *args):
        return self._run(*args)

class WiFiScanTool(BaseRemoteTool, BaseTool):
    name = 'WiFiScan'
    description = (
        'Scan for WiFi networks.'
        'Useful for scanning for WiFi networks.'
        'Does not accept any arguments.'
        'Returns a list of dicts.'
    )

    def _run(self, *args):
        return self._send_cmd('termux-wifi-scaninfo')

    async def _arun(self, *args):
        return self._run(*args)

class VibratorTool(BaseRemoteTool, BaseTool):
    name = 'Vibrator'
    description = (
        'Vibrate the device.'
        'Useful for vibrating the device.'
        'Accepts a single argument type integer, duration in milliseconds.'
    )

    def _run(self, duration: int):
        return self._send_cmd(f'termux-vibrate -f -d {duration}')

    async def _arun(self, duration: int):
        return self._run(duration)

class SearchContactsTool(BaseRemoteTool, BaseTool):
    name = "SearchContacts"
    description = (
        "Search the contacts on the device."
        "Useful for finding phone numbers for people."
        "Accepts a single string as an argument, any name containing it will have their info returned."
        "Returns a list of dicts containing 'name' and 'number."
    )

    def _run(self, search: str):
        res = self._send_cmd('termux-contact-list')
        contacts = json.loads(res['output'])
        matches = [c for c in contacts if search.lower() in c['name'].lower()]
        return matches
    async def _arun(self, *args):
        return self._run(*args)

class ListSMSTool(BaseRemoteTool, BaseTool):
    name = "ListSMS"
    description = (
        "Lists SMS messages on the device."
        "Useful for reading SMS messages on the device."
        "Accepts a single argument, a dictionary in JSON format containing the keys 'box' (either: 'inbox', 'sent', 'draft', 'outbox', 'failed', 'queued', or 'all'), and 'limit' (an integer representing the maximum number of messages to return)."
        "Returns a list of dicts containing 'address', 'body', 'date', 'date_sent', 'read', 'status', 'type', 'thread_id', and 'person'."
    )

    def _run(self, arguments):
        if type(arguments) == dict:
            args = arguments
        else:
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                return {'error': 'Invalid JSON'}
        if not args['box'] in ('inbox', 'sent', 'draft', 'outbox', 'failed', 'queued', 'all'):
            return 'Error: box must be "inbox", "sent", "draft", "outbox", "failed", "queued", or "all"'
        return self._send_cmd(f'termux-sms-list -t {args["box"]} -l {args["limit"]}')

    async def _arun(self, arguments):
        return self._run(arguments)

class SendSMSTool(BaseRemoteTool, BaseTool):
    name = "SendSMS"
    description = (
        "Sends an SMS message."
        "Useful for sending an SMS message."
        "Accepts a single argument, a dictionary in JSON format containing the keys 'number' (a string representing the phone number to send the message to without spaces or symbols barring country code - seperated by commas for multiple numbers - use the SearchContactsTool to find numbers by name), and 'message' (a string representing the message to send)."
        "Does not return any response."
    )

    def _run(self, arguments):
        if type(arguments) == dict:
            args = arguments
        else:
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                return {'error': 'Invalid JSON'}
        numbers = args['number'].replace("-",'').replace(' ','').replace('(','').replace(")","")
        return self._send_cmd(f'termux-sms-send -n "{numbers}" "{args["message"]}"')

    async def _arun(self, arguments):
        return self._run(arguments)

class GetCellInfoTool(BaseRemoteTool, BaseTool):
    name = "GetCellInfo"
    description = (
        "Get information about the cellular connection."
        "Useful for getting information about the cellular connection."
        "Does not accept any arguments."
        "Returns two dicts."
    )

    def _run(self, *args):
        return self._send_cmd('termux-telephony-cellinfo && termux-telephony-deviceinfo')
    
    async def _arun(self, *args):
        return self._run(*args)

class StartCallTool(BaseRemoteTool, BaseTool):
    name = 'StartCall'
    description = (
        'Start a phone call.'
        'Useful for calling a phone number.'
        'Accepts a single string argument, the phone number to call.'
        'Does not return any response.'
    )

    def _run(self, number: str):
        return self._send_cmd(f'termux-telephony-call {number}')
    
    async def _arun(self, number: str):
        return self._run(number)

class ListSensorsTool(BaseRemoteTool, BaseTool):
    name = 'ListSensors'
    description = (
        'List the sensors on the device.'
        'Useful for getting all sensors on the device.'
        'Does not accept any arguments.'
        'Returns a list of dicts containing "name" and "vendor".'
    )

    def _run(self, *args):
        return self._send_cmd('termux-sensor -l')

    async def _arun(self, *args):
        return self._run(*args)

class ReadSensorTool(BaseRemoteTool, BaseTool):
    name = 'ReadSensor'
    description = (
        "Useful for reading the value(s) from a device sensor."
        "Accepts a single arumgnet, a dictionary in JSON format containing the key 'sensor' (a string representing the sensor to read from - seperated by commas for multiple sensors), the key 'limit' containing an integer representing the number of times to read from the sensor."
        "Returns a dict containing the sensor name as the key, and the value(s) as a list."
    )

    def _run(self, arguments):
        if type(arguments) == dict:
            args = arguments
        else:
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                return {'error': 'Invalid JSON'}
        return self._send_cmd(f'termux-sensor -s {args["sensor"]} -n {args["limit"]}')
    
    async def _arun(self, arguments):
        return self._run(arguments)

REMOTE_TOOLS = [
    BatteryStatusTool(),
    BrightnessTool(),
    PhotoTool(),
    ClipboardGetTool(),
    ClipboardSetTool(),
    FingerprintTool(),
    LocationTool(),
    MediaPauseTool(),
    RecordMicTool(),
    NotificationTool(),
    ListNotificationsTool(),
    RemoveNotificationTool(),
    URLOpenerTool(),
    TorchTool(),
    SpeakTool(),
    GetVolumeTool(),
    SetVolumeTool(),
    WiFiInfoTool(),
    WiFiScanTool(),
    VibratorTool(),
    MediaPlayTool(),
    SearchContactsTool(),
    ListSMSTool(),
    SendSMSTool(),
    GetCellInfoTool(),
    StartCallTool(),
    ListSensorsTool(),
    ReadSensorTool(),
]
