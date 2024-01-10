from settings import config
import openai
import re
import random
import sys
import time
import math
from utils.emote_utils import get_random_emote
from utils.emote_utils import remove_emotes_from_message
import utils.wiki_utils as wiki_utils
import utils.tokensArray as tokensArray
import string
from models.mathison_db import Database
from .text2speech import speak_text 

# This function logs that the bot is starting with also logs some configurations of th bot
# This also sends random emoticon to twitch chat room
def message_on_welcome(self, logger):
    logger.debug(
        "================================================STARTING BOT========================================")
    bot_mode = "BOT MODES \n"
    bot_mode += "TEST_MODE: " + str(config.TEST_MODE) + "\n"
    bot_mode += "TEST_MODE_SC2_CLIENT_JSON: " + \
        str(config.TEST_MODE_SC2_CLIENT_JSON) + "\n"
    bot_mode += "ANALYZE_REPLAYS_FOR_TEST: " + \
        str(config.USE_CONFIG_TEST_REPLAY_FILE) + "\n"
    bot_mode += "IGNORE_REPLAYS: " + \
        str(config.IGNORE_GAME_STATUS_WHILE_WATCHING_REPLAYS) + "\n"
    bot_mode += "IGNORE_PREVIOUS_GAME_RESULTS_ON_FIRST_RUN: " + \
        str(config.IGNORE_PREVIOUS_GAME_RESULTS_ON_FIRST_RUN) + "\n"
    bot_mode += "MONITOR_GAME_SLEEP_SECONDS: " + \
        str(config.MONITOR_GAME_SLEEP_SECONDS) + "\n"
    logger.debug(bot_mode)

    prefix = ""  # if any
    greeting_message = f'{prefix} {get_random_emote()}'
    msgToChannel(self, greeting_message, logger)

def clean_text_for_chat(msg):
    # Combine carriage return and line feed replacement with filtering non-printable characters
    msg = ''.join(filter(lambda x: x in set(string.printable), msg.replace('\r', '').replace('\n', '')))
    return msg

# This function sends and logs the messages sent to twitch chat channel
def msgToChannel(self, message, logger, text2speech=False):

    # Clean up the message
    message = clean_text_for_chat(message)

    # Calculate the size of the message in bytes
    message_bytes = message.encode()
    message_size = len(message_bytes)

    # Log the byte size of the message
    logger.debug(f"Message size in bytes: {message_size}")

    # Check if the message exceeds the 512-byte limit
    if message_size > 512:
        truncated_message_bytes = message_bytes[:512 - len(" ... more".encode())] + " ... more".encode()
    else:
        truncated_message_bytes = message_bytes

    # Convert the truncated message back to a string
    truncated_message_str = truncated_message_bytes.decode()

    self.connection.privmsg(self.channel, truncated_message_str)
    logger.debug(
        "---------------------MSG TO CHANNEL----------------------")
    logger.debug(truncated_message_str)
    logger.debug(
        "---------------------------------------------------------")
    
    if text2speech:
        # use text to speech capability to speak the response if enabled
        # try catch
        if not config.TEXT_TO_SPEECH:
            return
        try:
            logger.debug(f"Speaking")
            truncated_message_str = remove_emotes_from_message(truncated_message_str)          
            truncated_message_str = "add commas, period and other appropriate punctuation: " + truncated_message_str

            completion = send_prompt_to_openai(truncated_message_str)
            if completion.choices[0].message is not None:
                logger.debug(
                    "completion.choices[0].message.content: " + completion.choices[0].message.content)
            response = completion.choices[0].message.content
            truncated_message_str =  response

            speak_text(truncated_message_str, mode=1)
        except Exception as e:
            logger.debug(f"Error: {e}") 
        finally:
            logger.debug(f"Spoken")
        
# This function processes the message receive in twitch chat channel
# This will determine if the bot will reply base on dice roll
# And this will generate the response
def process_pubmsg(self, event, logger, contextHistory):

    logger.debug("processing pubmsg")

    # Get message from chat
    msg = event.arguments[0].lower()
    sender = event.source.split('!')[0]
    # tags = {kvpair["key"]: kvpair["value"] for kvpair in event.tags}
    # user = {"name": tags["display-name"], "id": tags["user-id"]}

    if 'commands' in msg.lower():
        response = f"{config.BOT_COMMANDS}"
        response = clean_text_for_chat(response)
        trimmed_msg = tokensArray.truncate_to_byte_limit(response, config.TWITCH_CHAT_BYTE_LIMIT)        
        msgToChannel(self, trimmed_msg, logger)
        return

    # Send response to direct msg or keyword which includes Mathison being mentioned
    if 'open sesame' in msg.lower() or any(sub.lower() == msg.lower() for sub in config.OPEN_SESAME_SUBSTITUTES):
        logger.debug("received open sesame: " + str(msg.lower()))
        processMessageForOpenAI(self, msg, self.conversation_mode, logger, contextHistory)
        return

    # search wikipedia
    if 'wiki' in msg.lower():
        logger.debug("received wiki command: /n" + msg)
        msg = wiki_utils.wikipedia_question(msg, self)
        logger.debug("wiki answer: /n" + msg)
        trimmed_msg = tokensArray.truncate_to_byte_limit(msg, config.TWITCH_CHAT_BYTE_LIMIT)        
        trimmed_msg = "restate all the info, do not ommit any details: " + trimmed_msg
        processMessageForOpenAI(self, trimmed_msg, self.conversation_mode, logger, contextHistory)
        return

    # search replays DB
    if 'career' in msg.lower():
        contextHistory.clear()
        logger.debug("received career record command: \n" + msg)
        player_name = msg.split(" ", 1)[1]
        career_record = self.db.get_player_overall_records(player_name)
        logger.debug("career overall record answer: \n" + career_record)          
        career2_record = self.db.get_player_race_matchup_records(player_name)        
        logger.debug("career matchups record answer: \n" + career_record)    
        career_record = career_record + " " + career2_record      

        # Check if there are any results
        if career_record:

            trimmed_msg = tokensArray.truncate_to_byte_limit(career_record, config.TWITCH_CHAT_BYTE_LIMIT)
            msg = f'''
                Review this example:

                    when given a player, DarkMenace the career records are:

                        Overall matchup records for darkmenace: 425 wins - 394 losses Race matchup records for darkmenace: Protoss vs Protoss: 15 wins - 51 lossesProtoss vs Terran: 11 wins - 8 lossesProtoss vs Zerg: 1 wins - 1 lossesTerran vs Protoss: 8 wins - 35 lossesTerran vs Terran: 3 wins - 1 lossesTerran vs Zerg: 4 wins - 3 lossesZerg vs Protoss: 170 wins - 137 lossesZerg vs Terran: 138 wins - 100 lossesZerg vs Zerg: 75 wins - 58 losses

                    From the above, say it exactly like this format:

                        overall: 425-394, each matchup: PvP: 15-51 PvT: 11-8 PvZ: 1-1 TvP: 8-35 TvT: 3-1 TvZ: 4-3 ZvP: 170-137 ZvT: 138-100 ZvZ: 75-58

                Now do the same but only using this data:

                    {player_name} : {trimmed_msg}.

                Then add a 10 word comment about the matchup, after.
            '''
            
        else:
            msg = f"Restate all of the info here: There is no career games that I know for {player_name} ."

        # Send the message for processing
        # processMessageForOpenAI(self, msg, self.conversation_mode, logger, contextHistory)
        
        # no mathison flavoring, just raw send to prompt
        completion = send_prompt_to_openai(msg)
        if completion.choices[0].message is not None:
            logger.debug(
                "completion.choices[0].message.content: " + completion.choices[0].message.content)
        response = completion.choices[0].message.content

        if len(response) >= 400:
            logger.debug(
                f"Chunking response since it's {len(response)} characters long")

            # Split the response into chunks of 400 characters without splitting words
            chunks = []
            temp_chunk = ''
            for word in response.split():
                if len(temp_chunk + ' ' + word) <= 400:
                    temp_chunk += ' ' + word if temp_chunk != '' else word
                else:
                    chunks.append(temp_chunk)
                    temp_chunk = word
            if temp_chunk:
                chunks.append(temp_chunk)

            # Send response chunks to chat
            for chunk in chunks:
                msgToChannel(self, chunk, logger)

        else:            
            msgToChannel(self, response, logger)
        return

    # search replays DB
    if 'history' in msg.lower():
        contextHistory.clear()
        logger.debug("received history command: /n" + msg)
        player_name = msg.split(" ", 1)[1]
        history_list = self.db.get_player_records(player_name)
        logger.debug("history answer: /n" + str(history_list))  

        # Process each record and format it as desired
        formatted_records = [f"{rec.split(', ')[0]} vs {rec.split(', ')[1]}, {rec.split(', ')[2].split(' ')[0]}-{rec.split(', ')[3].split(' ')[0]}" for rec in history_list]

        # Join the formatted records into a single string
        result_string = " and ".join(formatted_records)

        trimmed_msg = tokensArray.truncate_to_byte_limit(result_string, config.TWITCH_CHAT_BYTE_LIMIT)
        # if history_list is empty then msg is "no records found"
        if history_list == []:
            msg = (f"restate all of the info here: there are no game records in history for {player_name}")
        else:
            msg = (f"restate all of the info here and do not exclude anything: total win/loss record of {player_name} we know the results of so far {trimmed_msg}")
        #msgToChannel(self, msg, logger)
        processMessageForOpenAI(self, msg, self.conversation_mode, logger, contextHistory)
        return
  
    def chunk_list(lst, max_chunk_size):
        """Splits the list into smaller lists each having a length less than or equal to max_chunk_size."""
        for i in range(0, len(lst), max_chunk_size):
            yield lst[i:i + max_chunk_size]

    # Check if the message contains "games in" and "hours"
    if 'games in' in msg.lower() and 'hours' in msg.lower():
        logger.debug("Received command to fetch games in the last X hours")

        # Use regex to extract the number of hours from the message
        match = re.search(r'games in (the )?last (\d+) hours', msg.lower())
        if match:
            hours = int(match.group(2))  # Extract the number of hours
        else:
            hours = 4  # Default value if no number is found

        if hours > 72:
            hours = 72  # Max number of hours allowed is 72

        # Retrieve games for the last X hours
        recent_games = self.db.get_games_for_last_x_hours(hours)
        logger.debug(f"Games in the last {hours} hours: \n" + str(recent_games))

        # Process each game record and format it as desired
        formatted_records = [f"{game}" for game in recent_games]

        # Define chunk size based on an estimated average size of each record
        avg_record_size = 100  # This is an estimation; you might need to adjust it
        max_chunk_size = config.TWITCH_CHAT_BYTE_LIMIT // avg_record_size

        msg = f"Games played in the last {hours} hours are: "
        msgToChannel(self, msg, logger)
        
        # Split the formatted records into chunks
        for chunk in chunk_list(formatted_records, max_chunk_size):
            # Join the records in the chunk into a single string
            chunk_string = " and ".join(chunk)

            # Truncate the chunk string to byte limit
            trimmed_msg = tokensArray.truncate_to_byte_limit(chunk_string, config.TWITCH_CHAT_BYTE_LIMIT)

            # Send the chunk message
            msgToChannel(self, trimmed_msg, logger)
            # processMessageForOpenAI(self, msg, self.conversation_mode, logger, contextHistory)

    # Function to process the 'head to head' command
    if 'head to head' in msg.lower():
        contextHistory.clear()
        logger.debug(f"Received 'head to head' command: \n{msg}")

        # Use regular expression to extract player names
        match = re.search(r"head to head (\w+) (\w+)", msg.lower())
        if match:
            player1_name, player2_name = match.groups()

            # Retrieve head-to-head records
            head_to_head_list = self.db.get_head_to_head_matchup(player1_name, player2_name)
            logger.debug(f"Type of head_to_head_list: {type(head_to_head_list)}")
            logger.debug(f"Head to head answer: \n{str(head_to_head_list)}")

            # Check if there are any results
            if head_to_head_list:
                # Since the records are already formatted, join them into a single string
                result_string = ", ".join(head_to_head_list)

                trimmed_msg = tokensArray.truncate_to_byte_limit(result_string, config.TWITCH_CHAT_BYTE_LIMIT)
                msg = f'''
                    Review this example:

                        when given 2 player, DarkMenace vs KJ the records are:

                            ['DarkMenace (Terran) vs KJ (Zerg), 29 wins - 7 wins', 'DarkMenace (Protoss) vs KJ (Zerg), 9 wins - 12 wins', 'DarkMenace (Zerg) vs KJ (Zerg), 3 wins - 2 wins', 'DarkMenace (Protoss) vs KJ (Terran), 6 wins - 1 wins', 'DarkMenace (Terran) vs KJ (Terran), 1 wins - 0 wins', 'DarkMenace (Protoss) vs KJ (Protoss), 2 wins - 2 wins']

                        From the above, say it exactly like this format:

                            overall: 50-24, each matchup: TvZ 29-7, PvZ 9-12, ZvZ 3-2, PvT 6-1, TvT 1-0, PvP, 2-2.  Summary: <10 word comment about the matchup>

                    Now do the same but only using this data:

                        {player1_name} vs {player2_name}: {result_string}.

                    Then add a 10 word comment about the matchup, after.
                '''
                
            else:
                msg = f"Restate all of the info here: There are no head-to-head game records between {player1_name} and {player2_name} ."

            # Send the message for processing
            # processMessageForOpenAI(self, msg, self.conversation_mode, logger, contextHistory)
            
            # no mathison flavoring, just raw send to prompt
            completion = send_prompt_to_openai(msg)
            if completion.choices[0].message is not None:
                logger.debug(
                    "completion.choices[0].message.content: " + completion.choices[0].message.content)
            response = completion.choices[0].message.content

            if len(response) >= 400:
                logger.debug(
                    f"Chunking response since it's {len(response)} characters long")

                # Split the response into chunks of 400 characters without splitting words
                chunks = []
                temp_chunk = ''
                for word in response.split():
                    if len(temp_chunk + ' ' + word) <= 400:
                        temp_chunk += ' ' + word if temp_chunk != '' else word
                    else:
                        chunks.append(temp_chunk)
                        temp_chunk = word
                if temp_chunk:
                    chunks.append(temp_chunk)

                # Send response chunks to chat
                for chunk in chunks:
                    msgToChannel(self, chunk, logger)

            else:            
                msgToChannel(self, response, logger)
           
        else:
            logger.debug("Invalid 'head to head' command format or player names not found.")
            # Optionally send an error message to the channel or log it.
        return

    # ignore certain users
    logger.debug("checking user: " + sender + " against ignore list")
    if sender.lower() in [user.lower() for user in config.IGNORE]:
        logger.debug("ignoring user: " + sender)
        return
    else:
        logger.debug("allowed user: " + sender)

    if config.PERSPECTIVE_DISABLED:
        logger.debug("google perspective config is disabled")
        toxicity_probability = 0
    else:
        toxicity_probability = tokensArray.get_toxicity_probability(
            msg, logger)
    # do not send toxic messages to openAI
    if toxicity_probability < config.TOXICITY_THRESHOLD:

        # any user greets via config keywords will be responded to
        if any(greeting in msg.lower() for greeting in config.GREETINGS_LIST_FROM_OTHERS):
            response = f"Hi {sender}!"
            response = f'{response} {get_random_emote()}'
            msgToChannel(self, response, logger)
            # disable the return - sometimes it matches words so we want mathison to reply anyway
            # DO NOT return

        if 'bye' in msg.lower():
            response = f"byers {sender}!"
            msgToChannel(self, response, logger)
            return

        if 'gg' in msg.lower():
            response = f"HSWP"
            msgToChannel(self, response, logger)
            return

        if 'bracket' in msg.lower() or '!b' in msg.lower() or 'FSL' in msg.upper() or 'fsl' in msg.lower():
            msg = f"Restate this including the full URL: This is the tournament info {config.BRACKET}"
            processMessageForOpenAI(self, msg, self.conversation_mode, logger, contextHistory)
            return

        # will only respond to a certain percentage of messages per config
        diceRoll = random.randint(0, 100) / 100
        logger.debug("rolled: " + str(diceRoll) +
                        " settings: " + str(config.RESPONSE_PROBABILITY))
        if diceRoll >= config.RESPONSE_PROBABILITY:
            logger.debug("will not respond")
            return

        processMessageForOpenAI(self, msg, self.conversation_mode, logger, contextHistory)
    else:
        response = random.randint(1, 3)
        switcher = {
            1: f"{sender}, please refrain from sending toxic messages.",
            2: f"Woah {sender}! Strong language",
            3: f"Calm down {sender}. What's with the attitude?"
        }
        msgToChannel(self, switcher.get(response), logger)

def send_prompt_to_openai(msg):
    """
    Send a given message as a prompt to OpenAI and return the response.

    :param msg: The message to send to OpenAI as a prompt.
    :return: The response from OpenAI.
    """
    completion = openai.ChatCompletion.create(
        model=config.ENGINE,
        messages=[
            {"role": "user", "content": msg}
        ]
    )
    return completion

# This function will process the message for Open API
# This will connect to OpenAI and send the inquiry/message for AI to response
# Then calls the msgToChannel to send the response back to channel
# This also logs the what message was sent to OpenAI and the response
def processMessageForOpenAI(self, msg, conversation_mode, logger, contextHistory):

    # let's give these requests some breathing room
    time.sleep(config.MONITOR_GAME_SLEEP_SECONDS)

    # remove open sesame
    msg = msg.replace('open sesame', '')
    logger.debug(
        "----------------------------------------NEW MESSAGE FOR OPENAI-----------------------------------------")
    # logger.debug(msg)
    logger.debug(
        'msg omitted in log, to see it, look in: "sent to OpenAI"')
    # remove open sesame
    msg = msg.replace('open sesame', '')

    # remove quotes
    msg = msg.replace('"', '')
    msg = msg.replace("'", '')

    # add line break to ensure separation
    msg = msg + "\n"

    # TODO: redo this logic
    # if bool(config.STOP_WORDS_FLAG):
    #    msg, removedWords = tokensArray.apply_stop_words_filter(msg)
    #    logger.debug("removed stop words: %s" , removedWords)

    # check tokensize
    total_tokens = tokensArray.num_tokens_from_string(
        msg, config.TOKENIZER_ENCODING)
    msg_length = len(msg)
    logger.debug(f"string length: {msg_length}, {total_tokens} tokens")

    # This approach calculates the token_ratio as the desired token limit divided by the actual total tokens.
    # Then, it trims the message length based on this ratio, ensuring that the message fits within the desired token limit.
    # Additionally, the code adjusts the desired token limit by subtracting the buffer size before calculating the token ratio.
    # This ensures that the trimming process takes the buffer into account and helps prevent the message from
    # exceeding the desired token limit by an additional (BUFFER) of 200 tokens.

    # check tokensize
    total_tokens = tokensArray.num_tokens_from_string(
        msg, config.TOKENIZER_ENCODING)
    msg_length = len(msg)
    logger.debug(f"string length: {msg_length}, {total_tokens} tokens")
    if int(total_tokens) > config.CONVERSATION_MAX_TOKENS:
        divided_by = math.ceil(len(msg) // config.CONVERSATION_MAX_TOKENS)
        logger.debug(
            f"msg is too long so we are truncating it 1/{divided_by} of its length")
        msg = msg[0:msg_length // divided_by]
        msg = msg + "\n"  # add line break to ensure separation
        total_tokens = tokensArray.num_tokens_from_string(
            msg, config.TOKENIZER_ENCODING)
        msg_length = len(msg)
        logger.debug(
            f"new string length: {msg_length}, {total_tokens} tokens")

    # add User msg to conversation context if not replay nor last time played analysis
    if conversation_mode not in ["replay_analysis", "last_time_played"]:
        # add User msg to conversation context
        tokensArray.add_new_msg(
            contextHistory, 'User: ' + msg + "\n", logger)
        logger.debug("adding msg to context history")
    else:
        contextHistory.clear()

    if conversation_mode == "last_time_played":
        # no mood / perspective
        pass
    else:

        # add complete array as msg to OpenAI
        msg = msg + \
            tokensArray.get_printed_array("reversed", contextHistory)
        # Choose a random mood and perspective from the selected options
        mood = random.choice(self.selected_moods)

        if conversation_mode == "replay_analysis":
            # say cutoff is 4, then select indices 0-3
            perspective_indices = config.BOT_PERSPECTIVES[:config.PERSPECTIVE_INDEX_CUTOFF]
        else:
            # Select indices 4-onwards
            perspective_indices = config.BOT_PERSPECTIVES[config.PERSPECTIVE_INDEX_CUTOFF:]

        selected_perspectives = [
            config.PERSPECTIVE_OPTIONS[i] for i in perspective_indices]
        perspective = random.choice(selected_perspectives)

        if (conversation_mode == "normal"):
            # if contextHistory has > 15 tuples, clear it
            if len(contextHistory) > 15:
                logger.debug(f"contextHistory has more than 15 tuples, clearing it")  
                contextHistory.clear()
            else:
                pass
            # Add custom SC2 viewer perspective
            msg = (f"As a {mood} acquaintance of {config.STREAMER_NICKNAME}, {perspective}, "
                    + msg)
        else:
            if (conversation_mode == "in_game"):
                msg = (f"As a {mood} observer of matches in StarCraft 2, {perspective}, comment on this statement: "
                        + msg)
            else:
                msg = (f"As a {mood} observer of matches in StarCraft 2, {perspective}, "
                        + msg)

    logger.debug("CONVERSATION MODE: " + conversation_mode)

    logger.debug("sent to OpenAI: %s", msg)

    #msgToChannel(self, "chanchan", logger)

    completion = send_prompt_to_openai(msg)

    try:
        if completion.choices[0].message is not None:
            logger.debug(
                "completion.choices[0].message.content: " + completion.choices[0].message.content)
            response = completion.choices[0].message.content

            # add emote
            if random.choice([True, False]):
                response = f'{response} {get_random_emote()}'

            logger.debug('raw response from OpenAI:')
            logger.debug(response)

            # Clean up response
            # Remove carriage returns, newlines, and tabs
            response = re.sub('[\r\n\t]', ' ', response)
            # Remove non-ASCII characters
            response = re.sub('[^\x00-\x7F]+', '', response)
            response = re.sub(' +', ' ', response)  # Remove extra spaces
            response = response.strip()  # Remove leading and trailing whitespace

            # dont make it too obvious its a bot
            response = response.replace("As an AI language model, ", "")
            response = response.replace("User: , ", "")
            response = response.replace("Observer: , ", "")
            response = response.replace("Player: , ", "")

            logger.debug("cleaned up message from OpenAI:")
            # replace with ? all non ascii characters that throw an error in logger
            response = tokensArray.replace_non_ascii(response, replacement='?')

            logger.debug(response)

            if len(response) >= 400:
                logger.debug(
                    f"Chunking response since it's {len(response)} characters long")

                # Split the response into chunks of 400 characters without splitting words
                chunks = []
                temp_chunk = ''
                for word in response.split():
                    if len(temp_chunk + ' ' + word) <= 400:
                        temp_chunk += ' ' + word if temp_chunk != '' else word
                    else:
                        chunks.append(temp_chunk)
                        temp_chunk = word
                if temp_chunk:
                    chunks.append(temp_chunk)

                # Send response chunks to chat
                for chunk in chunks:
                    # Remove all occurrences of "AI: "
                    chunk = re.sub(r'\bAI: ', '', chunk)
                    msgToChannel(self, chunk, logger)

                    # Add AI response to conversation context
                    tokensArray.add_new_msg(
                        contextHistory, 'AI: ' + chunk + "\n", logger)

                    # Log relevant details
                    logger.debug(f'Sending openAI response chunk: {chunk}')
                    logger.debug(
                        f'Conversation in context so far: {tokensArray.get_printed_array("reversed", contextHistory)}')
            else:
                response = re.sub(r'\bAI: ', '', response)
                # if response is less than 150 characters
                if len(response) <= 150:
                    # really short messages get to be spoken
                    msgToChannel(self, response, logger, text2speech=True)
                else:
                    msgToChannel(self, response, logger)                                        

                # Add AI response to conversation context
                tokensArray.add_new_msg(
                    contextHistory, 'AI: ' + response + "\n", logger)

                # Log relevant details
                logger.debug(f'AI msg to chat: {response}')
                logger.debug(
                    f'Conversation in context so far: {tokensArray.get_printed_array("reversed", contextHistory)}')

        else:
            response = 'oops, I have no response to that'
            msgToChannel(self, response, logger)
            logger.debug('Failed to send response: %s', response)
    except SystemExit as e:
        logger.error('Failed to send response: %s', e)