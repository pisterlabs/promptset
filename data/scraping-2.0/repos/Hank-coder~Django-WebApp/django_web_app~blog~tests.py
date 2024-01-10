from django.test import TestCase

# Create your tests here.
from sympy import symbols, Eq, solve

from pathlib import Path
import openai

speech_texts = [
    "Good afternoon, boys and girls. Let's talk about computers.",
    "First, computers are great for chatting. You can send emails, talk with friends over the internet, and even see each other using video calls. It's like magic mail that sends your message in seconds!",
    "Second, they're our own entertainment center. You can play games, watch cartoons, and enjoy your favorite songs.",
    "Third, computers are like a giant book that never ends. They're perfect for schoolwork and learning on websites, watching educational videos, and even taking fun quizzes to test your knowledge!",
    "Fourth, you can use computers to draw, write essays, and make cool presentations for class.",
    "Fifth, computers help people in all kinds of jobs—from building skyscrapers to flying planes. Even doctors use them to figure out how to make us feel better!",
    "Last, just think—computers have become a big part of our lives. They're tools for talking, having fun, learning new stuff, and even helping us with our future jobs. Isn't that awesome? Keep exploring and who knows? Maybe you'll invent a new way to use computers!"
]


for i, text in enumerate(speech_texts):
    speech_file_path = Path(__file__).parent / f"part{i}.mp3"
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file(speech_file_path)
response.stream_to_file(speech_file_path)
