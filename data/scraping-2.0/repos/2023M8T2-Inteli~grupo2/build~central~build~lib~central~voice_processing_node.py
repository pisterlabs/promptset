import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from openai import OpenAI
import json
import os
from queue import Queue, Empty

class VoiceProcessingNode(Node):
    def __init__(self, open_api_key, organization_id):
        super().__init__("voice_processing_node")
        self.subscription = self.create_subscription(
            String, "voice_command", self.voice_command_callback, 10
        )
        self.publisher = self.create_publisher(String, "llm_command", 10)
        self.get_logger().info("Voice Processing Node has been started.")
        self.client = OpenAI(api_key=open_api_key, organization=organization_id)
        self.log_publisher = self.create_publisher(String, "log_register", 10)

        # Criando a fila de mensagens
        self.queue = Queue()

        # Configurando o timer do ROS para processar a fila
        self.timer = self.create_timer(0.1, self.process_queue)  # 0.1 segundos de intervalo

    def voice_command_callback(self, msg):
        # Adicionando mensagem à fila
        self.queue.put(msg)

    def process_queue(self):
        try:
            # Tentar obter uma mensagem da fila
            msg = self.queue.get_nowait()  # Obter sem bloquear
            self.process_message(msg)
            self.queue.task_done()
        except Empty:
            # Nenhuma mensagem na fila
            pass

    def process_message(self, msg):
        log_msg = 'Voice Processing Node ativado'
        self.log_publisher.publish(String(data=log_msg))
        self.get_logger().info(log_msg)

        telegram_message = json.loads(msg.data)

        voice_file_path = telegram_message['voice_file_path']
        transcript = self.transcribe_voice(voice_file_path)
        if transcript:
            self.log_publisher.publish(
                String(data=f'Transcript do Voice Processing Node: "{transcript}"')
            )
            telegram_message['text'] = transcript
            telegram_message_json = json.dumps(telegram_message)
            self.publisher.publish(String(data=telegram_message_json))

    def transcribe_voice(self, voice_file_path):
        try:
            # Carregue o arquivo de áudio para a memória
            audio_file = open(voice_file_path, "rb")

            # Transcreva o áudio para texto usando o Whisper
            model = "whisper-1"  # Especifique o modelo adequado do Whisper aqui
            transcript = self.client.audio.transcriptions.create(file=audio_file, response_format='text', model=model)
            self.get_logger().info(f"Transcription: {transcript}")
        except Exception as e:
            self.get_logger().error(f"Failed to transcribe voice file: {e}")
            return None
        finally:
            # Tente excluir o arquivo de áudio
            try:
                if os.path.exists(voice_file_path):
                    os.remove(voice_file_path)
                    self.get_logger().info(f"Audio file {voice_file_path} deleted successfully.")
            except Exception as e:
                self.get_logger().error(f"Failed to delete voice file: {e}")
        return transcript


def main(args=None):
    rclpy.init(args=args)
    voice_processing_node = VoiceProcessingNode(
        open_api_key=os.getenv("OPENAI_API_KEY"),
        organization_id=os.getenv("OPENAI_ORGANIZATION_ID"),
    )
    rclpy.spin(voice_processing_node)
    voice_processing_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
