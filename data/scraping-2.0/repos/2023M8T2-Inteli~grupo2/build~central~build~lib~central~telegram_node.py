import telebot
from rclpy.node import Node
from std_msgs.msg import String
import rclpy
import threading
import os
from ament_index_python.packages import get_package_share_directory
import json
from openai import OpenAI

class TelegramNode(Node):
    def __init__(self, api_key):
        super().__init__("telegram_node")
        self.bot = telebot.TeleBot(api_key)
        self.initialize_subscriptions_and_publishers()
        self.get_logger().info("Telegram Node is running and waiting for commands...")
        self.initialize_telegram_bot()
        self.chat_states = {}

    def initialize_subscriptions_and_publishers(self):
        self.subscription = self.create_subscription(String, "telegram_message", self.listener_callback, 10)
        self.publisher = self.create_publisher(String, "llm_command", 10)
        self.voice_publisher = self.create_publisher(String, "voice_command", 10)
        self.log_publisher = self.create_publisher(String, "log_register", 10)

    def initialize_telegram_bot(self):
        @self.bot.message_handler(content_types=['text', 'audio', 'voice'])
        def respond_to_message(message):
            chat_id = message.chat.id

            if self.chat_states.get(chat_id, False):
                self.process_response(message)
            else:
                intro_text = """
                Olá, eu sou o Bot do grupo BBB, no que posso te ajudar hoje?

                - Para pedir uma peça, digite ou mande áudio: "Quero um [...]", "Preciso de um [...]", etc. \n
                - Para saber mais informações sobre como usar uma peça, digite ou mande áudio: "Como eu uso [...]?", "Pra que serve [...]?", etc. \n
                - Para ter informações de segurança sobre uma peça, digite ou mande áudio: "Quais são os cuidados que eu tenho que ter usando [...]?", "É perigoso usar [...]?", etc. \n
                """
                self.bot.register_next_step_handler(message, self.process_response)
                self.bot.reply_to(message, intro_text)
                self.chat_states[chat_id] = True

        # Run the bot polling in a separate thread
        polling_thread = threading.Thread(target=self.bot.polling)
        polling_thread.start()

    def process_response(self, message):
        self.get_logger().info(f'Telegram recebeu mensagem: "{message.content_type}"')
        self.log_publisher.publish(
            String(data=f'Telegram recebeu mensagem do tipo: "{message.content_type}"')
        )

        if message.content_type == "voice":
            self.handle_voice_message(message)
        elif message.content_type == "text":
            self.handle_text_message(message)

    def handle_text_message(self, message):
        message_text = message.text.lower()
        self.log_publisher.publish(
            String(data=f'Telegram recebeu mensagem: "{message_text}"')
        )
        telegram_message = {"chat_id": message.chat.id, "text": message_text}
        telegram_message_dict = json.dumps(telegram_message)
        self.publisher.publish(String(data=telegram_message_dict))
        self.log_publisher.publish(
            String(data=f'Telegram enviou para a LLM: "{message_text}"')
        )

    def handle_voice_message(self, message):
        self.get_logger().info("Telegram recebeu mensagem de voz")
        self.log_publisher.publish(String(data="Telegram recebeu mensagem de voz"))

        # Obter informações do arquivo de voz
        file_info = self.bot.get_file(message.voice.file_id)

        # Encontre o diretório do pacote
        package_dir = get_package_share_directory("central")

        # Construa o caminho para o diretório 'resource/voice_files'
        voice_file_directory = os.path.join(package_dir, "resource", "voice_files")

        # Baixar o arquivo de voz
        downloaded_file = self.bot.download_file(file_info.file_path)
        voice_file_path = os.path.join(voice_file_directory, f"{file_info.file_id}.ogg")

        # Certifique-se de que o diretório existe
        os.makedirs(voice_file_directory, exist_ok=True)

        # Salve o arquivo de voz no diretório especificado
        with open(voice_file_path, "wb") as new_file:
            new_file.write(downloaded_file)

        self.get_logger().info("Telegram salvou mensagem de voz")

        # Enviar o caminho do arquivo para o nó de processamento de voz
        self.publish_voice_file_path(voice_file_path, message.chat.id)

    def publish_voice_file_path(self, file_path, chat_id):
        telegram_message = {"voice_file_path": file_path, "chat_id": chat_id}
        telegram_message_json = json.dumps(telegram_message)

        self.voice_publisher.publish(String(data=telegram_message_json))
        self.log_publisher.publish(
            String(data=f'Voice file path sent to voice processing node: "{file_path}"')
        )

    def audio_message_handler(self, message):
        client = OpenAI()
        response = client.audio.speech.create(
            model="tts-1",
            voice="echo",
            input= message,
        )
        return response

    def listener_callback(self, msg):
        self.log_publisher.publish(
            String(data=f'Telegram recebeu da LLM: "{msg.data}"')
        )
        telegram_message_after_llm = json.loads(msg.data)

        response = telegram_message_after_llm['llm_response']
        self.log_publisher.publish(
            String(data=f'Mensagem enviada via bot do telegram: "{response}"')
        )

        self.bot.send_audio(
            chat_id=telegram_message_after_llm['chat_id'],
            audio=self.audio_message_handler(response),
        )
        self.bot.send_message(telegram_message_after_llm['chat_id'], response)

def main(args=None):
    api_key = os.getenv('TELEGRAM_API_KEY')
    if not api_key:
        raise ValueError("API Key not set in environment variables")

    rclpy.init(args=args)

    try:
        telegram_node = TelegramNode(api_key=api_key)
        rclpy.spin(telegram_node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
