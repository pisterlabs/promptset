import discord
from discord.ext import commands
import sys
import tensorflow as tf
import numpy as np
import json
from openai import OpenAI

intents = discord.Intents.default()
intents.messages = True
bot = commands.Bot(command_prefix='!', intents=intents)

class ChatbotModel(tf.Module):
    def __init__(self, vocab_size, embed_dim, rnn_units):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    @tf.function
    def call(self, inputs, states=None):
        x = self.embed(inputs)
        if states is None:
            output, states = self.gru(x)
        else:
            output, states = self.gru(x, initial_state=states)
        return output, states

class ChatbotWithTensorFlowAndOpenAI:
    def __init__(self, api_key, model="gpt-3.5-turbo-1106", vocab_size=10000, embed_dim=256, rnn_units=1024):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chat_data = {'inputs': [], 'responses': []}

        # Initialize TensorFlow model
        self.tf_model = ChatbotModel(vocab_size, embed_dim, rnn_units)
        checkpoint_dir = './checkpoints'
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            self.tf_model.load_weights(latest_checkpoint)
        else:
            print("No checkpoint found. TensorFlow model is uninitialized. Continuing with an uninitialized model.")

    async def user_input(self, message):
        return await bot.wait_for('message', check=lambda m: m.author == message.author)

    def generate_response_with_tensorflow(self, input_text):
        # Implement the logic to generate a response using the TensorFlow model
        # For simplicity, let's assume the input is already tokenized
        input_tokens = np.array([1, 2, 3, 4, 5])  # Replace with actual tokenized input

        # Convert to TensorFlow format
        input_tensor = tf.convert_to_tensor(input_tokens, dtype=tf.int32)
        input_tensor = tf.expand_dims(input_tensor, 0)  # Add batch dimension

        # Initialize model state
        states = None

        # Generate response using TensorFlow model
        tf_response, states = self.tf_model(input_tensor, states)
        tf_response = tf.argmax(tf_response, axis=-1)  # Get indices of predicted tokens
        # Convert indices back to text (e.g., using a reverse tokenizer)
        # For now, let's just send a placeholder response
        response = "Generated response using TensorFlow model."
        return response

    def write_chat_data_to_file(self, filename='chat_data.json'):
        try:
            with open(filename, 'w') as file:
                json.dump(self.chat_data, file, indent=2)
            print(f"Chat data has been saved to {filename}")
        except Exception as e:
            print(f"Error saving chat data: {str(e)}")

    async def run_chatbot(self, message):
        user_input_text = await self.user_input(message)

        # Generate response using TensorFlow model
        tf_response = self.generate_response_with_tensorflow(user_input_text)
        await message.channel.send(f"TensorFlow_AI: {tf_response}")

        # Generate response using OpenAI model
        messages = [{"role": "user", "content": user_input_text},
                    {"role": "assistant", "content": tf_response}]
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model
        )
        openai_response = chat_completion.choices[0].message.content
        await message.channel.send(f"Greatest_Detective_AI: {openai_response}")

        self.chat_data['inputs'].append({"role": "user", "content": user_input_text})
        self.chat_data['responses'].append({"role": "TensorFlow_AI", "content": tf_response})
        self.chat_data['responses'].append({"role": "OpenAI_AI", "content": openai_response})

class SimpleChatbot:
    def __init__(self, api_key, model="gpt-3.5-turbo-1106"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chat_data = {'inputs': [], 'responses': []}

    async def on_message(self, message):
        if message.author == bot.user:
            return

        user_input_text = message.content

        messages = []

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo-1106"
        )

        assistant_response = chat_completion.choices[0].message.content
        await message.channel.send(f"Batman_AI: {assistant_response}")

        self.chat_data['inputs'].append({"role": "user", "content": user_input_text})
        self.chat_data['responses'].append({"role": "assistant", "content": assistant_response})

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if message.content.startswith('!gd'):
        chatbot_tf_openai = ChatbotWithTensorFlowAndOpenAI(api_key="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")
        await chatbot_tf_openai.run_chatbot(message)
        chatbot_tf_openai.write_chat_data_to_file()
        await bot.process_commands(message)  # Add this line to process commands
    else:
        simple_chatbot = SimpleChatbot(api_key="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")
        await simple_chatbot.on_message(message)
        await bot.process_commands(message)

bot.run('MTE2NDIzNjYzNjIzNDcxOTM2Mg.G-K34x.r0LDCFPbsHTtMmmNMIhVOa9eH8_3UTFOjT4zKA')
