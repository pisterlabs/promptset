"""
Command-line interface to launch app and create application object.
"""
import json
import os

import openai

import SVC.app as application


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
CONFIG_FILE = os.path.join(PROJECT_ROOT, '..','config.json')


def start_app():
    _Bootstrapper().start_app()


_HOST = '127.0.0.1'
_PORT = os.environ.get('FRIDAY_PORT', '3333') 

class _Bootstrapper(object):

    def start_app(self):
        self._create_application()
        self._run_app()



    def get_app(self):
        self._create_application()
        return self._application
    

    def _create_application(self):
        self._create_configuration()
        self._setup_openapi_key()
        self._create_application_object()


    def _create_configuration(self):
        config_file = open(CONFIG_FILE)
        self._configuration = json.load(config_file)


    def _setup_openapi_key(self):
        openai.api_key = self._configuration.get('OPENAI_API_KEY')

    def _create_application_object(self):
        self._application = application.create_app(self._configuration)


    def _run_app(self):
        self._application.run(
            host=_HOST,
            port=_PORT,
            debug=self._configuration.get('DEBUG', False),
            threaded=False,
            use_reloader=False
        )
