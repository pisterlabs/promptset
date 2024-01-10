import discord
import openai
import textwrap
import mysql.connector
from transformers import GPT2Tokenizer


global CLIENT_ID
global SERVER_ID_CONTENT
global SYSTEM_ID
global PER

SYSTEM_ID = 0;

# Set the link to the source code
link = ""

# Set the intents for the Discord client
intents = discord.Intents.all()

# Create the Discord client object
client = discord.Client(intents=intents)

# Print out all the intents for debugging purposes
for Intents in discord.Intents.all():
    print(Intents)

# Set the API key for the OpenAI library
openai.api_key = ""

# ----------------------------------------------------------------------------------------------------------------------
def tval(x):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(x)
    return len(tokens)

# ----------------------------------------------------------------------------------------------------------------------

def calculate_token_percentage(x, j):
    try:

        # Connect to the MySQL database
        cnx = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='',
            database='chat'
        )

        # Create a cursor object to execute queries
        cursor = cnx.cursor()

        # Specify the ServerID and ClientID for filtering
        server_id = x  # Replace with your desired ServerID
        client_id = j  # Replace with your desired ClientID

        # Prepare the query with placeholders for ServerID and ClientID
        query = "SELECT SUM(TokenVal) FROM your_table WHERE ServerID = %s AND ClientID = %s AND Role IN ('user', 'assistant')"

        # Execute the query with the specified values
        cursor.execute(query, (server_id, client_id))

        # Fetch the result of the query
        sum_result = cursor.fetchone()[0]

        # Close the cursor and database connection
        cursor.close()
        cnx.close()

    except mysql.connector.Error as error:
            # Handle the database error
            print("An error occurred while querying the database:", error)

    return (sum_result / 4097) * 100

# ----------------------------------------------------------------------------------------------------------------------

def delate_chats(x, y):
    try:
        # Connect to MySQL
        cnx = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='',
            database='chat'
        )

        # Create a cursor
        cursor = cnx.cursor()

        # Define the query
        query = "DELETE FROM your_table WHERE ServerID = %s AND ClientID = %s"

        # Define the values to match
        value1 = x
        value2 = y

        # Execute the query
        cursor.execute(query, (value1, value2))

        # Commit the changes
        cnx.commit()

        # Close the cursor and connection
        cursor.close()
        cnx.close()

    except mysql.connector.Error as error:
        # Handle the database error
        print("An error occurred while querying the database:", error)

# ----------------------------------------------------------------------------------------------------------------------
def get_chats(x, y, z):
    try:
        # Connect to the database
        conn = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='',
            database='chat'
        )

        # Create a cursor to execute SQL queries
        cursor = conn.cursor()

        # Define the SQL query with a WHERE clause
        query = f"SELECT ServerID, Role, ClientID, Timestamp, TextContent FROM your_table WHERE ServerID = '{x}' AND Role IN ('{y}', 'system', 'assistant') AND ClientID IN ('{z}', '{SYSTEM_ID}') ORDER BY Timestamp"

        # Execute the query
        cursor.execute(query)

        # Fetch all the rows returned by the query
        rows = cursor.fetchall()

        # Define an array to store the dictionaries
        result = []

        # Iterate over the rows and create a dictionary for each row
        for row in rows:
            # Create a dictionary for the current row
            row_dict = {
                'role': row[1],
                'content': row[4]
            }

            # Add the dictionary to the result array
            result.append(row_dict)

        # Close the cursor and connection
        cursor.close()
        conn.close()

        return result

    except mysql.connector.Error as error:
        # Handle the database error
        print("An error occurred while querying the database:", error)


def put_chats(x, y, k, j, g):
    try:
        # Connect to the database
        conn = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='',
            database='chat'
        )

        # Create a cursor to execute SQL queries
        cursor = conn.cursor()

        # Define the data
        server_id = x;
        role = y;
        client_id = k
        text = j;
        tval = g;

        # Define the SQL query with the specific column names
        query = "INSERT INTO your_table (ServerID, Role, ClientID, TextContent, TokenVal)" \
                "VALUES (%s, %s, %s, %s, %s)"

        # Execute the query with the values
        cursor.execute(query, (server_id, role, client_id, text, tval))

        # Commit the changes to the database
        conn.commit()

        # Close the cursor and connection
        cursor.close()
        conn.close()

    except mysql.connector.Error as error:
        # Handle the database error
        print("An error occurred while querying the database:", error)


# ----------------------------------------------------------------------------------------------------------------------
def markdown_to_discord(text):
    # Split the text into lines
    lines = text.split("\n")

    # Process each line
    for i, line in enumerate(lines):
        # Check if the line starts with "#", "##", or "###"
        if line.startswith("# "):
            lines[i] = "**" + line[2:] + "**"
        elif line.startswith("## "):
            lines[i] = "**" + line[3:] + "**"
        elif line.startswith("### "):
            lines[i] = "**" + line[4:] + "**"
        else:
            # Check if the line contains "*" or "**"
            index = line.find("*")
            while index != -1:
                if line[index:index+2] == "**":
                    line = line[:index] + "**" + line[index+2:]
                    index += 2
                else:
                    line = line[:index] + "_" + line[index + 1:]
                # Find the next occurrence of "*" or "**"
                index = line.find("*", index + 1)
            lines[i] = line

    # Join the lines back into a single string
    return "\n".join(lines)


# This function takes a string and two delimiters as input
def escape_backticks_codeblocks(string, delimiter1):
    # Split the string on the first delimiter
    split_string = string.split(delimiter1)
    # If the delimiter was not found, the string is returned as-is
    if len(split_string) == 1:
        print("Delimiter1 not found in escape_backticks_codeblocks function")
        return string
    # Otherwise, the relevant part of the string (between the two delimiters) is extracted
    else:
        # Remove the delimiter from the beginning of the string
        sub_string = split_string[1]
        print(f"SUBSTRING PASSED TO THE FUNCTION:\n{sub_string}\n---------\n")
        # insert a backslash before each backtick found in the string
        modified_string = sub_string.replace("`", "\\`")
        # Re-insert the delimiters into the string
        final_string = delimiter1 + modified_string + delimiter1
        # Return the modified string
        return final_string


# Define the generate_response function
def generate_response(z, k):
    #~Create Messages array of dictionary using a for loop fetching the database TEXT_VALUES
    messagess = get_chats(k, 'user', z)
    print(f"MESSAGES DICT \n----------------\n\n{messagess}")
    ## Use the OpenAI Completion API to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = messagess
    )
    #print(response.choices)

    #Return the text of the first response choice
    return response.choices[0].message.content
    #return messagess


# ----------------------------------------------------------------------------------------------------------------------
# Initialize the response as a list containing str items(code blocks or normal text)
async def init_response(response, channel):
    messages = []
    before_code_block = ""  # Variable to track the text before the current code block
    i = 0  # Variable to track the current position in the response
    print(f"WHOLE_RESPONSE\n\n{response}\nEND\n---------------------\n\n\n")
    while i < len(response):
        # Find the start of the code block in the response
        start = response.find("\n```", i)
        if start == -1:
            # No more code blocks were found, so add the remaining response to the list of messages
            if before_code_block + response[i:]:  # Only add the remaining response if it is non-empty
                messages.append(before_code_block + response[i:])
            break
        else:
            print("Code block found")
            # A code block was found, so find the end of the code block
            end = response.find("\n```", start + 4)
            if end == -1:
                # The end of the code block was not found, so add the entire response as a single message
                print("'end' variable not found! Response from Open_AI formatted wrong")
                messages.append(before_code_block + response)
                break
            else:
                # Split the response into three parts: before the code block, the code block itself
                # And after the code block
                before = before_code_block + response[i:start - 1]  # Include the text before the previous code block
                code_block = response[start:end + 4]
                # after = response[end + 3:]

                # Add the before and code block parts to the list of messages
                if before:  # Only add before if it is non-empty
                    messages.append(before)
                else:
                    print("before variable is an empty string")
                print(f"Before: {before}\n//")
                print(f"This is the single code block\n{code_block}\n--------")
                messages.append(code_block)

                # Update the response to be the part after the code block
                before_code_block = ""  # Reset the text before the code block
                i = end + 4  # Update the current position to be after the end of the code block
    # Print the variable messages (list) for debugging purpose
    print("MESSAGES_VAR(LIST) ->>")
    for x in messages:
        print(f"{x}\n---------")
    print(f"END MESSAGE_VAR(LIST)-----------------\n")
    await send_response(messages, channel)


# The items inside the list passed as a parameter are checked for length
# (subdividing it in more messages if len(list[x]) > 1900) and sent over.
async def send_response(var, channel):
    # Initialize an empty list to store the formatted messages
    formatted_messages = []

    # Iterate through the messages
    for message in var:
        # Check if the message is a code block
        if message.endswith("\n```"):
            message = escape_backticks_codeblocks(message, "\n```")
            # Extract the language key from the message
            language_key = message[3:message.index('\n')]
            print(f"{language_key}")
            # Remove the triple backticks and the language key from the beginning of the message
            message = message[len(language_key) + 4:]

            # Check if the message is more than 2000 characters long
            if len(message) > 1990:
                print(f"code block longer than 1990: {len(message)}")
                # Initialize the list of chunks
                chunks = []

                # Split the message into chunks of 2000 characters or fewer, preserving the whitespace
                while len(message) > 1990:
                    chunk = message[:message.rindex('\n', 0, 1990)]
                    chunks.append(chunk)
                    message = message[len(chunk):]

                # Add the remaining message as the last chunk
                chunks.append(message)

                # Add the triple backticks, language key, and whitespace to the beginning and the end of each chunk
                # except the last one
                chunks = [f"```{language_key}\n{chunk}\n```" for chunk in chunks[:-1]]

                # Add the triple backticks, language key, and whitespace to the beginning of the last chunk
                chunks.append(f"```{language_key}\n{chunks[-1]}")
                for ch in chunks:
                    print(f"{len(ch)} line 118")
                    print(ch)

                # Add the chunks to the list of formatted messages
                formatted_messages.extend(chunks)
            else:
                # Add the triple backticks, language key, and whitespace to the beginning of the message
                message = f"```{language_key}{message}"
                print(f"code block no longer than 1990:\n{message}\n------")

                # Add the message to the list of formatted messages
                formatted_messages.append(message)

        else:
            print(f"text chunk: {message}\n--------\n\n")
            # Wrap the message and add the resulting lines to the list of formatted messages
            formatted_messages.extend(
                textwrap.wrap(markdown_to_discord(message), width=2000, replace_whitespace=False))

    # Send the formatted messages to the Discord channel
    print(f"{len(formatted_messages)} items in formatted_messages list (167)")
    for x in formatted_messages:
        print(
            f"-------------------------------\n-----------\n{len(x)} "
            f"char and {type(x)} in (message of formatted_messages)")
        print(f"MESSAGE IN FORMATTED_MESSAGES:\n{x}")
        await channel.send(x)


# ----------------------------------------------------------------------------------------------------------------------
# Create an event handler for when the Discord client is ready
@client.event
async def on_ready():
    print("Logged in!")
    print(client.user.name)
    print(client.user.id)
    print('------')
    # channel = client.get_channel (1067633857014267905)
    # await channel.send("I'm now Up and working!")


# Create an event handler for when a message is received
@client.event
async def on_message(message):
    # Check if the message was sent in the correct channel
    if message.channel.name == "chat-gpt3":
        # Check if the message was sent by the bot
        if message.author == client.user:
            return

        # Check if the message starts with the "/help" command
        if message.content.startswith("!help"):
            await message.channel.send(
                f"This bot has been written by Daxer, any pool request or issue can be uploaded here:"
                f"\n\n-Source code: {link}")
        # If the message is not the "/help" command, generate and send a response
        elif message.content.startswith("!clear"):
            await message.channel.purge()
        elif message.content:
            # Generate the response NO MESSAGE.CONTENT PASSED
            #~SEND MESSAGES to DB (SERVER_ID_CONTENT[0], ROLE, CLIENT_ID, TIMESTAMP, TEXT)
            put_chats(message.guild.id, 'user', message.author.id, message.content, tval(message.content))
            PER = calculate_token_percentage(message.guild.id, message.author.id)
            print(PER)
            #print(f"The response length is: {len(response)}")
            if PER < 83:
                #~Send messages DB with (SERVER_ID_CONTENT[0], ROLE, CLIENT_ID, TIMESTAMP, TEXT)
                response = generate_response(message.author.id, message.guild.id)
                put_chats(message.guild.id, 'assistant', message.author.id, response, tval(response))
                await init_response((response + f"({PER:.2f}%)") , message.channel)
            else:
                delate_chats(message.guild.id, message.author.id)
                put_chats(message.guild.id, 'user', message.author.id, message.content, tval(message.content))
                response = generate_response(message.author.id, message.guild.id)
                #~Send messages DB with (SERVER_ID_CONTENT[0], ROLE, CLIENT_ID, TIMESTAMP, TEXT)
                put_chats(message.guild.id, 'assistant', message.author.id, response, tval(response))
                await init_response(response , message.channel)

        else:
            pass


# Start the Discord client
client.run("")
