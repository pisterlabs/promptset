import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import os
import speech_recognition as sr
from gtts import gTTS
import threading
from dexhand_llm_control.openai_session import OpenAISession
import json

class LLMControlNode(Node):
    def __init__(self):
        super().__init__('llm_control')

        # Retreive the OpenAI key from the environment
        openai_key = os.environ.get('OPENAI_API_KEY')
        if openai_key is None:
            self.get_logger().error('OPENAI_API_KEY environment variable not set. Cannot continue.')
            exit(1)
        self.openai_session = OpenAISession(openai_key, self.get_logger())

        # Default publishers for hand 
        self.extension_publisher = self.create_publisher(String, 'dexhand_finger_extension', 10)
        self.gesture_publisher = self.create_publisher(String, 'dexhand_gesture', 10)

        # Set up speech recognition
        self.recognizer = sr.Recognizer()

        # List the microphones - see if we can find a Respeaker
        self.microphone_index = 0  # Default to built in microphone
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            self.get_logger().debug("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))
            if (name.lower().find("respeaker") != -1):
                self.microphone_index = index
                self.get_logger().info("Respeaker detected - Using microphone with name \"{1}\" for `Microphone(device_index={0})`".format(index, name))
                break

        self.get_logger().info("Using microphone with index {0} named {1}".format(self.microphone_index, sr.Microphone.list_microphone_names()[self.microphone_index]))
        self.microphone = sr.Microphone(device_index=self.microphone_index)
        self.recognizer.energy_threshold = 4000
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(self.microphone)
        
        # Start audio processing loop in a thread
        self.audio_thread = threading.Thread(target=self.message_processing_loop)
        self.audio_thread.start()
        
    # Basic wrapper for Google Text to Speech. Takes a string, speaks it, and then deletes the file.
    def textToSpeech(self, text):
        tts = gTTS(text=text, lang='en')
        tts.save("speech.mp3")
        os.system("mpg123 speech.mp3")
        os.remove("speech.mp3")

    # Takes the JSON response from OpenAI and formats it into a command for the hand
    def formatROSMessages(self, jsonmessage):

        # Publish a dexhand_gesture message to return to base pose
        msg = String()
        msg.data = "reset"
        self.gesture_publisher.publish(msg)

        # Is this a gesture or a finger message?
        if "gesture" in jsonmessage:
            msg = String()
            msg.data = jsonmessage["gesture"]
            self.gesture_publisher.publish(msg)

            self._logger.debug("Published dexhand_gesture message: {0}".format(msg.data))
        else:
            # Convert each finger to a dexhand_finger_extension message and publish it
            for finger, value in jsonmessage.items():
                msg = String()
                msg.data = "{0}:{1}".format(finger, value)
                self.extension_publisher.publish(msg)

                self._logger.debug("Published dexhand_finger_extension message: {0}".format(msg.data))

    # This is our main message processor that records audio from the user and 
    # attempts to process it with OpenAI. It runs in a thread.
    def message_processing_loop(self):
        while rclpy.ok():
            with self.microphone as source:
                self.get_logger().info('Ready for audio input')
                audio = self.recognizer.listen(source)

                try:
                    # for testing purposes, we're just using the default API key
                    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
                    # instead of `r.recognize_google(audio)`
                    query = self.recognizer.recognize_google(audio)

                    # We have a few special commands that we handle here without needing AI

                    # Reset the hand
                    if query == "reset hand":
                        self.get_logger().info('Resetting hand')
                        msg = String()
                        msg.data = "reset"
                        self.gesture_publisher.publish(msg)
                        continue

                    # Reset the context
                    if query == "reset context":
                        self.get_logger().info('Resetting context')
                        self.openai_session.resetContext()
                        msg = String()
                        msg.data = "reset"
                        self.gesture_publisher.publish(msg)
                        
                        continue
                    
                    # Send the query to OpenAI
                    jsonmessage,textmessage = self.openai_session.processPrompt(query)

                    # If the we received a jsonmessage, we format that into a command and send it to the hand
                    if jsonmessage is not None:
                        self.get_logger().info('Sending to DexHand: "%s"' % jsonmessage)
                        self.formatROSMessages(json.loads(jsonmessage))

                    # If we received a text message, we just speak it
                    if textmessage is not None:
                        self.get_logger().info('Speaking to user: "%s"' % textmessage)
                        self.textToSpeech(textmessage)

                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                    #self.textToSpeech("I'm sorry, I didn't understand that. Try again.")

                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
                    self.textToSpeech("I'm sorry, there is an error with Google Speech Recognition.")

        

def main(args=None):
    rclpy.init(args=args)
    dexhand_node = LLMControlNode()
    rclpy.spin(dexhand_node)
    dexhand_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
