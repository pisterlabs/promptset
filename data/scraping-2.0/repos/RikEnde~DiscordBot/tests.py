from unittest.mock import MagicMock, patch

import asynctest
from asynctest import CoroutineMock
from discord import DMChannel, ClientUser
from discord.ext import commands
from openai.types import Image, ImagesResponse
import pytest

from discordbot import (
    bot,
    is_valid_input,
    call_openai_api,
    clear_history,
    forget,
    get_openai_image,
    image,
    main,
    on_command_error,
    on_ready,
    prompt,
    random_role,
    send_image,
    set_max_tokens,
    set_random_role,
    set_role,
    set_temperature,
    summarize_conversation,
    summarize_history,
    on_message
)


pytest_plugins = ('pytest_asyncio',)


@pytest.mark.asyncio
async def test_call_openai_api():
    """
    call_openai_api() should call openai.Completion.create with prompt_text, max_tokens, and temperature,
    and return the response text
    """
    with patch('discordbot.client.chat.completions.create') as mock_create:
        mock_choice = MagicMock()
        mock_choice.message.content = 'Test response'
        mock_create.return_value.choices = [mock_choice]

        result = await call_openai_api(
            prompt_text='Test prompt',
            max_tokens=10,
            temperature=0.7,
        )

        assert result == 'Test response'
        mock_create.assert_called_once()

        mock_create.side_effect = [Exception('Oh noes!')]
        with pytest.raises(Exception):
            await call_openai_api(
                prompt_text='Test prompt',
                max_tokens=10,
                temperature=0.7,
            )


@pytest.mark.asyncio
async def test_clear_history(ctx):
    """
    clear_history() should set bot.conversation_history for a user, and send a message
    """
    ctx.message.author.id = '123'
    bot.conversation_history['123'] = 'Hello bot'

    await clear_history(ctx)
    assert bot.conversation_history['123'] == ''


@pytest.mark.asyncio
@asynctest.patch('discordbot.clear_history', autospec=True)
@asynctest.patch('discord.ext.commands.Context.send', autospec=True)
async def test_forget(mock_context_send, mock_clear_history, ctx):
    """
    forget() should call clear_history() and send a message
    """
    await forget(ctx)

    mock_clear_history.assert_called_once_with(ctx)
    mock_clear_history.reset_mock()
    mock_context_send.assert_called_once()
    mock_context_send.reset_mock()


@pytest.mark.asyncio
@patch('discordbot.client.images.generate', autospec=True)
async def test_get_openai_image(mock_image_create):
    """
    get_openai_image() should call openai.Image.create() with a search_term
    and return url of the image
    """

    mock_image_create.return_value = ImagesResponse(created=1, data=[Image(url="https://www.example.com/image.jpg")])

    url = await get_openai_image('cuddly gray rat')
    assert url == 'https://www.example.com/image.jpg'


@pytest.mark.asyncio
@asynctest.patch('discordbot.send_image', autospec=True)
@asynctest.patch('discord.ext.commands.Context.send', autospec=True)
async def test_image(mock_context_send, mock_send_image, ctx):
    """
    image() should call send_image() with search terms and optional method arg
    """
    # If method_or_search_term is 'openai', image() should be called with AI-generated search terms
    # and 'openai' as method
    # Otherwise, method_search_term should be treated as search terms
    with asynctest.patch(
            'discordbot.call_openai_api', new=CoroutineMock(side_effect=['cuddly gray rat', 'cuddly gray rat'])
    ) as mock_call_openai_api:
        # If method_or_search_term is 'openai', call_openai_api() should be called, a message
        # sent with the search terms, and send_image() should be called with search terms and method
        for method_or_search_term in ['random']:
            await image(ctx, prompt=method_or_search_term)
            mock_call_openai_api.assert_awaited_once()
            mock_call_openai_api.reset_mock()
            mock_context_send.assert_called_with(ctx, 'cuddly gray rat')
            mock_context_send.reset_mock()
            mock_send_image.assert_called_with(ctx, 'cuddly gray rat')

        # If method_or_search_term is something else, send_image() should be called with that as search terms
        await image(ctx, prompt='cuddly gray rat')
        assert not mock_call_openai_api.call_count
        assert not mock_context_send.call_count
        mock_send_image.assert_called_with(ctx, 'cuddly gray rat')

        # If there's an Exception, send() should be called
        mock_send_image.side_effect = [Exception('Oh noes!')]
        await image(ctx, prompt='cuddly gray rat')
        mock_context_send.assert_called_once()
        mock_context_send.reset_mock()


def test_is_valid_input():
    """
    is_valid_input() should make sure input isn't > 4096 chars
    """
    assert is_valid_input('Hello')
    assert is_valid_input('a' * 4096)
    assert not is_valid_input('a' * 4097)


@patch('discordbot.bot.run', autospec=True)
@patch('discordbot.DISCORD_TOKEN', '123')
def test_main(mock_run):
    """
    main() should call bot.run()
    """
    main()
    mock_run.assert_called_once_with('123')


@pytest.mark.asyncio
@asynctest.patch('discord.ext.commands.Context.send', autospec=True)
async def test_on_command_error(mock_context_send, ctx):
    """
    test_on_command_error() should send a message appropriate to the error that's passed in
    """
    await on_command_error(ctx, commands.CommandNotFound())
    mock_context_send.assert_called_with(ctx, 'The command you entered does not exist. Please try again.')
    mock_context_send.reset_mock()

    await on_command_error(ctx, commands.MissingRequiredArgument(MagicMock()))
    mock_context_send.assert_called_with(
        ctx, 'A required argument is missing. Please check your command and try again.'
    )
    mock_context_send.reset_mock()

    await on_command_error(ctx, commands.BadArgument())
    mock_context_send.assert_called_with(ctx, 'Invalid argument provided. Please check your input and try again.')
    mock_context_send.reset_mock()

    await on_command_error(ctx, Exception('Oh noes!'))
    mock_context_send.assert_called_with(
        ctx, 'An error occurred while processing your command. Please try again later. Oh noes!'
    )
    mock_context_send.reset_mock()


@pytest.mark.asyncio
@patch('builtins.print')
async def test_on_ready(mock_print):
    """
    on_ready() should print something
    """
    await on_ready()
    assert mock_print.call_count == 1


@pytest.mark.asyncio
@asynctest.patch('discord.ext.commands.Context.send', autospec=True)
@asynctest.patch('discordbot.summarize_conversation', return_value='This is what we talked about')
@patch('discordbot.is_valid_input', autospec=True)
async def test_prompt(mock_is_valid_input, mock_summarize_conversation, mock_context_send, ctx):
    """
    prompt() should take a prompt from a user, call call_openai_api() with the prompt,
    and send the response in a message
    It should also replace history for that user with a summary if history gets too long
    """
    with asynctest.patch(
            'discordbot.call_openai_api', new=CoroutineMock(side_effect=['Test summary', 'Test summary'])
    ) as mock_call_openai_api:

        ctx.message.author.id = '123'
        bot.conversation_history['123'] = 'Hello bot'
        mock_is_valid_input.return_value = True

        # If conversation history > bot.max_history_tokens,
        # summarize_conversation() and call_openai_api() should be called
        bot.max_history_tokens = 1
        await prompt(ctx, text='Hello bot')
        mock_summarize_conversation.assert_awaited_once_with('Hello bot')
        mock_summarize_conversation.reset_mock()
        mock_call_openai_api.assert_awaited_once()
        mock_call_openai_api.reset_mock()

        # If conversation history <= bot.max_history_tokens, summarize_conversation() should not be called,
        # but call_openai_api() should be called
        bot.max_history_tokens = len(bot.conversation_history['123'])
        await prompt(ctx, text='Hello bot')
        mock_summarize_conversation.assert_not_called()
        mock_call_openai_api.assert_awaited_once()
        mock_call_openai_api.reset_mock()

        # If input not valid, message should be sent
        mock_is_valid_input.return_value = False
        await prompt(ctx, text='Hello bot')
        args, _ = mock_context_send.call_args
        # args are Context, text message
        assert 'Invalid input.' in args[1]
        mock_context_send.reset_mock()
        mock_call_openai_api.assert_not_called()

        # If answer <= MAX_DISCORD_TOKENS, send() should be called once
        mock_is_valid_input.return_value = True
        with patch('discordbot.MAX_DISCORD_TOKENS', 1):
            mock_call_openai_api.side_effect = ['a']
            await prompt(ctx, text='Hello bot')
            mock_call_openai_api.reset_mock()
            mock_context_send.assert_called_once()
            mock_context_send.reset_mock()

        # If answer > MAX_DISCORD_TOKENS, send() should be called multiple times
            mock_call_openai_api.side_effect = ['bc']
            await prompt(ctx, text='Hello bot')
            mock_call_openai_api.reset_mock()
            assert mock_context_send.call_count == 2
            mock_context_send.reset_mock()

        # If there's an Exception, send() should be called
        mock_call_openai_api.side_effect = [Exception('Oh noes!')]
        await prompt(ctx, text='Hello bot')
        mock_context_send.assert_called_once()
        mock_context_send.reset_mock()


@pytest.mark.asyncio
@asynctest.patch('discordbot.clear_history', autospec=True)
@asynctest.patch('discordbot.set_random_role', autospec=True, return_value='random role')
@asynctest.patch('discord.ext.commands.Context.send', autospec=True)
async def test_random_role(mock_context_send, mock_set_random_role, mock_clear_history, ctx):
    """
    random_role() should clear the history, set a random role, and send a message with the response
    """
    await random_role(ctx)
    mock_clear_history.assert_called_once_with(ctx)
    mock_clear_history.reset_mock()
    mock_set_random_role.assert_called_once()
    mock_set_random_role.reset_mock()
    mock_context_send.assert_called_once_with(ctx, mock_set_random_role.return_value)
    mock_context_send.reset_mock()

    # If there's an Exception, send() should be called
    mock_set_random_role.side_effect = [Exception('Oh noes!')]
    await random_role(ctx)
    mock_context_send.assert_called_once()
    mock_context_send.reset_mock()


@pytest.mark.asyncio
@patch('builtins.print', autospec=True)
@asynctest.patch('discordbot.call_openai_api', autospec=True, return_value='description')
async def test_set_random_role(mock_call_openai_api, mock_print):
    """
    set_random_role() should print a message, call call_openai_api(), and return the response
    """
    bot.role = 'old role'

    response = await set_random_role()

    mock_print.assert_called_once()
    mock_print.reset_mock()
    mock_call_openai_api.assert_called_once()
    mock_call_openai_api.reset_mock()
    assert response == mock_call_openai_api.return_value

    # If there's an Exception, print() should be called again
    mock_call_openai_api.side_effect = [Exception('Oh noes!')]
    await set_random_role()
    assert mock_print.call_count == 2
    mock_print.reset_mock()


@pytest.mark.asyncio
@asynctest.patch('discordbot.get_openai_image', autospec=True)
@patch('discord.Embed', autospec=True)
@asynctest.patch('discord.ext.commands.Context.send', autospec=True)
async def test_send_image(
        mock_context_send, mock_embed, mock_get_openai_image, ctx
):
    """
    send_random_image() should generate a random image for the search term or random
    and send it in a message
    """
    # Random image generated with OpenAI if method is 'openapi'
    await send_image(ctx=ctx, search_term='cuddly gray rat')
    mock_get_openai_image.assert_called_once()
    mock_get_openai_image.reset_mock()
    # Discord Embed.set_image() should've been called
    assert mock_embed.return_value.set_image.call_count == 1
    mock_embed.reset_mock()
    # message should've been sent
    mock_context_send.assert_called_once_with(ctx, embed=mock_embed.return_value)
    mock_context_send.reset_mock()

    # Random image generated with OpenAI if method is something else
    await send_image(ctx=ctx, search_term='cuddly gray bat')
    mock_get_openai_image.assert_called_once()
    mock_get_openai_image.reset_mock()
    # Discord Embed.set_image() should've been called
    assert mock_embed.return_value.set_image.call_count == 1
    mock_embed.reset_mock()
    # message should've been sent
    mock_context_send.assert_called_once_with(ctx, embed=mock_embed.return_value)
    mock_context_send.reset_mock()

    # If there's an Exception, send() should be called
    mock_get_openai_image.side_effect = [Exception('Oh noes!')]
    with pytest.raises(Exception):
        await send_image(ctx=ctx, search_term='cuddly gray rat')
    assert not mock_context_send.call_count


@pytest.mark.asyncio
@asynctest.patch('discord.ext.commands.Context.send', autospec=True)
async def test_set_max_tokens(mock_context_send, ctx):
    """
    set_max_tokens() should either set bot.max_tokens or not, and send a message accordingly
    """
    # 'tokens' is 0: error message should be sent
    await set_max_tokens(ctx, 0)
    args, _ = mock_context_send.call_args
    # args are Context, text message
    assert 'Invalid max tokens' in args[1]

    # 'tokens' is > 4096: error message should be sent
    await set_max_tokens(ctx, 4097)
    args, _ = mock_context_send.call_args
    # args are Context, text message
    assert 'Invalid max tokens' in args[1]

    # 'tokens' is > 1: success message should be sent
    assert bot.max_tokens != 123
    await set_max_tokens(ctx, 123)
    args, _ = mock_context_send.call_args
    # args are Context, text message
    assert args[1] == 'Max tokens set to 123.'
    assert bot.max_tokens == 123


@pytest.mark.asyncio
@asynctest.patch('discord.ext.commands.Context.send', autospec=True)
@asynctest.patch('discordbot.set_random_role', autospec=True)
async def test_set_role(mock_set_random_role, mock_context_send, ctx):
    """
    set_role() should update 'bot.role' and send a message
    """
    bot.role = 'You are Tim the Enchanter'

    new_role = 'Your mother was a hamster, and your father smelt of elderberries'
    await set_role(ctx, role=new_role)

    assert bot.role == new_role
    mock_context_send.assert_called_once()
    mock_context_send.reset_mock()

    # If new_role is "random", also call set_random_role
    await set_role(ctx, role='random')
    mock_set_random_role.assert_called_once()
    mock_context_send.assert_called_once()


@pytest.mark.asyncio
@asynctest.patch('discord.ext.commands.Context.send', autospec=True)
async def test_set_temperature(mock_context_send, ctx):
    """
    set_temperature() should either set 'bot.temperature' or not, and send a message accordingly
    """
    # temperature is < 0: error message should be sent
    await set_temperature(ctx, -.1)
    args, _ = mock_context_send.call_args
    # args are Context, text message
    assert 'Invalid temperature value' in args[1]

    # temperature is > 1: error message should be sent
    await set_temperature(ctx, 1.1)
    args, _ = mock_context_send.call_args
    # args are Context, text message
    assert 'Invalid temperature value' in args[1]

    # temperature is < 1: success message should be sent
    assert bot.temperature != .123
    await set_temperature(ctx, .123)
    args, _ = mock_context_send.call_args
    # args are Context, text message
    assert args[1] == 'Temperature set to 0.123.'
    assert bot.temperature == .123


@pytest.mark.asyncio
async def test_summarize_conversation():
    """
    summarize_conversation() should call call_openai_api() and return the response
    """
    with asynctest.patch(
            'discordbot.call_openai_api', new=CoroutineMock(side_effect=['Test summary'])
    ) as mock_call_openai_api:

        conversation = '''User: Hello\n
                       AI: Hi there\n
                       User: Tell me a joke\n
                       AI: Why did the chicken cross the road? To get to the other side!'''
        result = await summarize_conversation(conversation)
        assert result == 'Test summary'

        mock_call_openai_api.assert_awaited_once_with(
            prompt_text='Summarize the following conversation:\n' + conversation,
            max_tokens=1000,
            temperature=0.7,
        )


@pytest.mark.asyncio
@asynctest.patch('discord.ext.commands.Context.send', autospec=True)
@asynctest.patch('discordbot.summarize_conversation', autospec=True, return_value='This is what we talked about')
async def test_summarize_history(mock_summarize_conversation, mock_context_send, ctx):
    """
    summarize_history() should call summarize_conversation and send the summary
    """
    ctx.message.author.id = '123'
    bot.conversation_history['123'] = 'Hello bot'

    await summarize_history(ctx)

    # summarize_conversation() should've been called
    assert mock_summarize_conversation.call_count == 1
    args, _ = mock_summarize_conversation.call_args
    assert args[0] == 'Hello bot'

    # A message should've been sent
    assert mock_context_send.call_count == 1
    # args are Context, text message
    args, _ = mock_context_send.call_args
    assert args[1] == mock_summarize_conversation.return_value
    mock_context_send.reset_mock()

    # If there's an exception, an error message should be sent
    mock_summarize_conversation.side_effect = [Exception('Oh noes!')]
    await summarize_history(ctx)

    assert mock_context_send.call_count == 1
    args, _ = mock_context_send.call_args
    assert 'An unexpected error occurred' in args[1]


@pytest.mark.asyncio
@asynctest.patch('discord.ext.commands.Bot.process_commands', autospec=True)
async def test_on_message(mock_bot_process_commands, ctx):
    """
    on_message() should call prompt() if the message is from a DM channel
    """

    # Don't respond to bot own messages
    ctx.message.author = bot.user
    ctx.message.channel = MagicMock(DMChannel)
    ctx.message.content = 'hey there'
    await on_message(ctx.message)
    mock_bot_process_commands.assert_not_called()
    mock_bot_process_commands.reset_mock()

    ctx.message.author = MagicMock(ClientUser)
    ctx.message.content = '!image an image'
    await on_message(ctx.message)
    # how to specify this??
    args, kwargs = mock_bot_process_commands.call_args
    assert args[1].content == '!image an image'
    mock_bot_process_commands.assert_called_once()
    mock_bot_process_commands.reset_mock()

    ctx.message.author = MagicMock(ClientUser)
    ctx.message.content = 'hey there'
    await on_message(ctx.message)
    args, kwargs = mock_bot_process_commands.call_args
    assert args[1].content == '!prompt hey there'
    mock_bot_process_commands.assert_called_once()
