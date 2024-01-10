import rclpy
import openai
import json
import os
from gptros2_interfaces.srv import GPTPrompt
from rclpy.node import Node


class GPTService(Node):

    def __init__(self):
        super().__init__('gpt_service')
        self.srv = self.create_service(GPTPrompt, 'gpt_prompt', self.gpt_prompt_callback)
        self.clear_srv = self.create_service(GPTPrompt, 'gpt_clear', self.clear_history_callback)

    def save_historic(self, api_key, conversation_history):
        file_path = 'src/gptros2/tmp/historic_' + api_key + '.json'
        with open(file_path, 'w') as file:
            json.dump(conversation_history, file, indent=4)
    
    def load_historic(self, api_key):
        file_path = 'src/gptros2/tmp/historic_' + api_key + '.json'
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def clear_history_callback(self, request, response):
        try:
            api_key = request.api_key
            file_path = 'src/gptros2/tmp/historic_' + api_key + '.json'
            os.remove(file_path)  # Delete the conversation history file
            response.message = "History cleared successfully"
        except Exception as e:
            self.get_logger().info('Clear history failed %r' % (e,))
            response.message = 'Clear history failed %r' % (e,)
        return response
    
    def gpt_prompt_callback(self, request, response):

        openai.api_key = request.api_key
        conversation_history = self.load_historic(request.api_key)

        try:
            
            conversation_history.append({"role": "user", "content": request.prompt})

            completion = openai.chat.completions.create(
                model=request.model,
                messages=conversation_history
            )
            response.role = completion.choices[0].message.role
            response.message = completion.choices[0].message.content
            conversation_history.append({"role": "system", "content": response.message})
            self.save_historic(request.api_key, conversation_history)
            
        except Exception as e:
            self.get_logger().info('Service call failed %r' % (e,))
            response.message = 'Service call failed %r' % (e,)
            response.role = ''
            
        return response


def main(args=None):
    rclpy.init(args=args)

    gpt_service = GPTService()

    rclpy.spin(gpt_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()