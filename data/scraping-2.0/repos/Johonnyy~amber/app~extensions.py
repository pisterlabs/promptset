import os
from flask_login import LoginManager
from flask_socketio import SocketIO
from app.deviceManager import DeviceManager
from app.moduleManager import ModuleManager
from app.openaiManager import OpenAIManager


class Extensions:
    def __init__(self):
        self.socket = SocketIO()
        self.login_manager = LoginManager()

        self.module_manager = ModuleManager()
        self.device_manager = DeviceManager()

        self.module_manager.load_functions()
        
        self.openai_manager = OpenAIManager(self.module_manager)

        self.login_manager.login_view = "dashboard.authBP.login"


extensions = Extensions()
# plugin_manager.load_plugins()
# device_manager.load_devices()
