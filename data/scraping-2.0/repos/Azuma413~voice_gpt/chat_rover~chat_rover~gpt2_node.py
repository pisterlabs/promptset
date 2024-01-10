import openai
import os
import rclpy
from rclpy.node import Node
from chatrover_msgs.srv import TextText

openai.api_key = os.getenv("OPENAI_API_KEY")
with open("/home/humble/ros2_ws/src/voice_gpt/chat_rover/chat_rover/prompt/prompt2.txt", "r") as f:
    prompt = f.read()
system_conf = {"role": "system", "content": prompt}
class GPTController(Node):
    def __init__(self):
        super().__init__('gpt2_node')
        self.server = self.create_service(TextText, "/gpt2_service", self.service_cb)
    def service_cb(self, request, response):
        response.text = self.chatProcess(request.text)
        self.get_logger().info('Publishing: "%s"' % response.text)
        return response
    def chatProcess(self, text):
        messages = [system_conf, {"role": "user", "content": text}]
        response = openai.ChatCompletion.create(model="gpt-4-1106-preview",
                                                response_format={"type":"json_object"},
                                                messages=messages)
        ai_response = response['choices'][0]['message']['content']
        return ai_response

def main(args=None):
    rclpy.init(args=args)
    controller = GPTController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()