def get_welcome_message(first_name):
    return f"""Hello {first_name}! 👋 Welcome to Thought Keeper, your personal voice-to-text assistant.

I'm here to help you effortlessly convert your spoken words into written text. Just send me a voice message or an audio file, and I will transcribe it for you. It's that simple!

Feel free to start sending your audio messages whenever you're ready. If you need assistance or have any questions, just use the '/help' command.

Happy note-taking with Thought Keeper! 📝✨"""


def get_help_message():
    return """Thought Keeper Help Guide 📚

Need assistance? Here’s how you can use Thought Keeper to make your note-taking easy and efficient:

Sending Voice Messages: Simply record a voice message and send it to me. I'll transcribe it into text and reply with your note.

Uploading Audio Files: You can also upload an audio file (in supported formats), and I'll convert it to text for you.

Receiving Transcriptions: After processing your audio, I'll send back the transcribed text. You can then use it for your notes, reminders, or however you like!

Tips for Best Results:

Speak clearly and at a moderate pace.
Try to minimize background noise when recording voice messages.

Troubleshooting:
We are using Whisper from OpenAI for speech recognition. If you're facing issues with the transcription, check your 
audio clarity or check Whisper documentation. It supports multiple languages and has automatic language determination.

Here is the project codebase shared on GitHub: https://github.com/KoStard/PrivateWhisperNoteBot

Remember, Thought Keeper is here to simplify your note-taking. Start speaking, and leave the typing to me! 🎙️➡️📝"""


def get_access_denied_message():
    return """Access Denied ⚠️

I'm sorry, but it looks like you don't have permission to use Thought Keeper. This bot is restricted to authorized users only.

If you believe this is a mistake or if you would like to request access, please contact the administrator or the person who provided you with this bot's link.

Thank you for your understanding! 🛑"""
