from utils.logger import setup_logger
import openai
import time

client = openai.Client(api_key='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')
logger = setup_logger(__name__, 'bot.log')

my_assistant = client.beta.assistants.retrieve("asst_KgDFGq9GlACc8d2HVHDFjT0m")

def chat_with_openai(update, context):
    print(f'updating contexct ${context}, and update: ${update}')
    user_message = update.message.text
    chat_type = update.message.chat.type
    bot_username = "@TheometricsBot"

    # Check if the bot is mentioned in the message in group chats
    if chat_type in ['group', 'supergroup'] and bot_username not in user_message:
        return

    try:
        # Create a Thread
        thread = client.beta.threads.create()

        # Add a Message to a Thread
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message
        )
        print(f'Message Thread, ${message}')
        # Run the Assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=my_assistant.id,
        )

        # Poll for the completed run
        for _ in range(100):  # Try for a certain number of times
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            print(f'Run Status, ${run_status}')
            if run_status.status == 'completed':
                break
            time.sleep(1)  # Wait for a second before checking again

        if run_status.status != 'completed':
            update.message.reply_text("I'm still thinking, please wait a moment...")
            return

        assistant_messages = client.beta.threads.messages.list(thread_id=thread.id)
        if assistant_messages.data and assistant_messages.data[0].role == "assistant":
            # Correct way to access the 'value' attribute
            response_text = assistant_messages.data[0].content[0].text.value
            update.message.reply_text(response_text)
        

    except Exception as e:
        logger.error(f"Error with OpenAI API: {e}")
        update.message.reply_text("I'm having trouble processing that request.")