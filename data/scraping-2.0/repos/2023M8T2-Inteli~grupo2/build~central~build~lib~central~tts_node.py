import rclpy
from rclpy.node import Node
from openai import OpenAI
from std_msgs.msg import String
from io import BytesIO
from playsound import playsound
from ament_index_python.packages import get_package_share_directory
import os

class TTSNode(Node):
    def __init__(self):
        super().__init__('tts_node')
        self.client = OpenAI()
        self.subscription = self.create_subscription(
            String,
            'move_robot',
            self.listener_callback,
            10)
        self.get_logger().info('TTS Node está rodando e esperando por comandos...')

    def generate_speech(self):
        try:
     #   response = self.client.audio.speech.create(
      #      model="tts-1",
       #     voice="alloy",
        #    input= text,
        #)
        

            package_name = 'central'

        # Construa o caminho para o diretório de compartilhamento do pacote
            package_share_directory = get_package_share_directory(package_name)

        # Construa o caminho para o seu arquivo de dados dentro do diretório de recursos
            data_file_path = os.path.join(package_share_directory, 'resource', 'audio.mp3')
            self.get_logger().info(data_file_path)
            playsound(data_file_path)
            self.get_logger().info("Audio reproduzido com sucesso!")
        except Exception as e:
            print(e, flush=True)
            self.get_logger().error(f"Error: {e}")


    def listener_callback(self, msg):
        self.generate_speech()
        self.get_logger().info(f"Received command: {msg.data}")

def main(args=None):

    rclpy.init(args=args)
    tts_node = TTSNode()

    try:
        rclpy.spin(tts_node)
    except Exception as e:
        tts_node.get_logger().error(f"Error: {e}")

    tts_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
