import sys
from .elevenlabs_text_to_speech.elevenlabs_text_to_speech import ElevenlabsTextToSpeech
from .azure_text_to_speech.azure_text_to_speech import AzureTextToSpeech
from.openai_text_to_speech.openai_text_to_speech import OpenAITextToSpeech
from src.utilities.logs.log_performance import PerformanceLogger

logger = PerformanceLogger()

class SpeechVerbalizer:
	"""
	A class that utilizes Azure's Cognitive Speech Service to verbalize the bot's response.
	"""
 
	def __init__(self, speech_objects:dict, api_keys:dict, setting_objects:dict):
		"""
		Initializes a new SpeechVerbalizer object
		"""
		self._load_in_settings(setting_objects)
		self._initilize_speech_engine(speech_objects, api_keys, setting_objects)
   
	@logger.log_operation
	def verbalize_speech(self, speech: str) -> str:
		"""
  		Verbalize the bot's response using the speech synthesizer.
    	"""
		self._reload_settings()
  
		# check whether the speech synthesizer needs to be reconfigured or if the bot is muted
		perform_text_to_speech = self._check_and_handle_preconditions(speech)
		if perform_text_to_speech:
			# Verbalize the response
			print('\nVerbalizing...')
			self.text_to_speech_engine.text_to_speech(speech, self.language_country_code)

		# Checks whether the following params are true and executed the appropriate actions
		self._check_and_handle_postconditions(self.reset_language, self.exit_status)
  
		return speech

	def _check_and_handle_preconditions(self, speech:str) -> bool:
		"""
		Initial flag check
		"""
		# check whether speech was given
		if not speech:
			print('No speech has been provided to verbalize.')
			return False
		
		# check if bot is muted
		if self.mute_status:
			return False

		# check if voice need to be reconfigured
		if self.reconfigure_voice:
			self.text_to_speech_engine.update_voice()
			self.master_settings.save_property('functions', False, 'reconfigure_verbalizer')
			return True
		return True
   
	def _check_and_handle_postconditions(self, reset_language, exit_status) -> None:
		"""
  		Post verbalization flag check
    	"""
		# check if language needs to be reset (this is done after one-shot speach translationions)
		if reset_language:
			old_language = self.profile_settings.retrieve_property('old_language')
			self.profile_settings.save_property('language', old_language)
			self.master_settings.save_property('functions', False, 'reset_language')

		# Exit the program needs to be exited
		if exit_status:
			self.master_settings.save_property('status', False, 'exit')
			sys.exit()
	
	def _initilize_speech_engine(self, speech_objects:dict, api_keys:dict, setting_objects:dict) -> str:
		"""
		Retrieves the speech engine
		"""
		# check which speech engine is used
		if self.engine_name.lower() == 'azure':
			self.text_to_speech_engine = AzureTextToSpeech(self.profile_name, speech_objects, setting_objects)
		elif self.engine_name.lower() == 'elevenlabs':
			self.text_to_speech_engine = ElevenlabsTextToSpeech(self.profile_name, api_keys, setting_objects)
		elif self.engine_name.lower() == 'openai':
			self.text_to_speech_engine = OpenAITextToSpeech(api_keys)
   
	def _load_in_settings(self, setting_objects:dict) -> None:
		"""
  		Loading in necessary data from 'master_settings.json'
    	"""
		self.master_settings = setting_objects['master_settings']
		self.profile_settings = setting_objects['profile_settings']
		self.voice_settings = setting_objects['voice_settings']
		self.profile_name = self.master_settings.retrieve_property('profile')
		self.engine_name = self.profile_settings.retrieve_property('tts', self.profile_name)

	def _reload_settings(self):
		self.master_settings.reload_settings()
		self.language  = self.profile_settings.retrieve_property('language')
		self.language_country_code = self.voice_settings.retrieve_language_country_code(self.language)
		self.bot_name = self.profile_settings.retrieve_property('name', profile_name=self.profile_name)
		self.mute_status = self.master_settings.retrieve_property('status', 'mute')
		self.exit_status = self.master_settings.retrieve_property('status', 'exit')
		self.reset_language = self.master_settings.retrieve_property('functions', 'reset_language')
		self.reconfigure_voice = self.master_settings.retrieve_property('functions', 'reconfigure_verbalizer')

   
  

   