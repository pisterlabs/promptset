import telebot
from telebot import types
from ..utils import dbutils
import openai


def main_menu(bot: telebot.TeleBot, state: dict):
    '''
        This enclosure function generates callback for the main_manu handler.

        Callback desription: 
            Answers on corresponding main_menu handler.
    '''

    def callback(query: types.CallbackQuery):
        query_id = query.id
        user_id = query.from_user.id

        answer_create_group = '''🎅 Ho Ho Ho! Santa Needs Your Help! 🌟

Santa is on a quest to give this Secret Santa group a merry and magical name, and he needs your festive creativity! 🎁✨

What enchanting name shall we bestow upon this jolly gathering of Secret Santas? The more creative, the merrier! 🎄✨

Write down suggested name of the group (/cancel to cancel action):'''
        if state.get(user_id, None) is not None:
            bot.answer_callback_query(query_id, 'You are currently perform another action!')
            return
        
        if query.data == 'create_group':
            bot.answer_callback_query(query_id, text='You are trying to create a group!')

            bot.send_message(query.message.chat.id, answer_create_group)
            state[user_id] = {}
            state[user_id]['state'] = 'CREATING_GROUP'

        elif query.data == 'join_group':
            bot.answer_callback_query(query_id, text='You are trying to join group!')

            bot.send_message(query.message.chat.id, 'Please, write down your invitation hash:')
            state[user_id] = {}
            state[user_id]['state'] = 'JOINING_GROUP'

        elif query.data == 'list_groups':
            bot.answer_callback_query(query_id, text='You are trying to list all groups!')
            groups = dbutils.get_group_names(state['database'], user_id=user_id)
            group_names = [group[0] for group in groups]
            created_groups = dbutils.get_all_created_groups(state['database'], user_id)
            created_group_names = [group[0] for group in created_groups]

            answer = 'Behold, dear one! 🌟 Here is the enchanting and festive list of all the groups you are a cherished member of:'
            for group in group_names:
                answer += '\n - ' + group + (' (you are admin here)' if group in created_group_names else '')

            bot.send_message(chat_id=query.message.chat.id,
                             text=answer)
        elif query.data == 'list_presentee':
            bot.answer_callback_query(query_id, text='🌟 Searching for recipients!..')
            entries = dbutils.get_all_recipients(state['database'], user_id=user_id)
            template = '''🎅 <b>Santa's Present Recipient:</b>  @{recipient_name}

🎄 <b>Group Name:</b> {group_name}

📝 <b>About:</b>
{about}

🎁 <b>Desired Presents:</b>
{desired_presents}
'''
            def generate_markup_openai(user_id, group_id):
                markup = types.InlineKeyboardMarkup()
                markup.add(
                    types.InlineKeyboardButton(
                        'Get suggestion from 🎅',
                        callback_data='openai_' + str(user_id) + '_' + str(group_id)
                    )
                )

                return markup
            
            bot.send_message(chat_id=query.message.chat.id,
                             text='🌟 Here your recipients:')
            for entry in entries:
                message_text = template.format(group_name=entry[0],
                                               recipient_name=entry[1],
                                               about=entry[2],
                                               desired_presents=entry[3])
                print(entry)
                bot.send_message(chat_id=query.message.chat.id,
                                 text=message_text,
                                 parse_mode='HTML',
                                 reply_markup=generate_markup_openai(user_id=entry[4], group_id=entry[5]))
                
        elif query.data == 'start_randomization':
            groups = dbutils.get_all_created_groups(state['database'], user_id=user_id)

            # Generating keyboard markup for randomization
            def generate_markup_groups():
                markup = types.InlineKeyboardMarkup()
                for group in groups:
                    markup.add(
                        types.InlineKeyboardButton(
                            group[0],   # Name of the group as a label
                            callback_data='randomizing_group_' + group[0] + '_' + str(group[1])  # name and id of the group as a callback data
                        )
                    )

                return markup

            bot.send_message(chat_id=query.message.chat.id,
                             text='🎅 Please choose the group you want to start randomizing: ',
                             reply_markup=generate_markup_groups())
            
            if len(groups) > 0:
                state[user_id] = {}
                state[user_id]['state'] = 'RANDOMIZING_GROUP'

            bot.answer_callback_query(query_id, text='🌟 Randomization starts!..')

    return callback


def randomize_group(bot: telebot.TeleBot, state: dict=None):
    def generate_markup_openai(santa_id, group_id):
        markup = types.InlineKeyboardMarkup()
        callback_data = 'openai_{santa_id}_{group_id}'.format(santa_id=santa_id, group_id=group_id)
        markup.add(
            types.InlineKeyboardButton(
                'Get suggestions from 🎅',
                callback_data=callback_data
            )
        )
        return markup
    
    def callback(query: types.CallbackQuery):
        group_id = int(query.data.split('_')[-1])
        group_name = query.data.split('_')[-2]

        entries = dbutils.randomize_santas(state['database'], group_id=group_id)

        if len(entries) > 0:
            message_text = 'Ho ho ho! 🎅 All recipients have found their Santas in %s! 🎁🎉 Spread the joy and let the festive fun begin! 🌟 Merry Christmas to all! 🎄🎅' % group_name
            
            for entry in entries:
                bot.send_message(chat_id=entry[0],
                                text=message_text)
                template = '''🎅 <b>Santa's Present Recipient:</b>  @{recipient_name}

🎄 <b>Group Name:</b> {group_name}

📝 <b>About:</b>
    {about}

🎁 <b>Desired Presents:</b>
    {desired_presents}
'''
                bot.send_message(chat_id=entry[0],
                                text=template.format(recipient_name=entry[1], group_name=entry[2], about=entry[3], desired_presents=entry[4]),
                                parse_mode='HTML',
                                reply_markup=generate_markup_openai(entry[5], group_id))
            bot.answer_callback_query(query.id, text='Group has been randomized! 🎉')
        else:
            message_text = "😔🎁 Unfortunately, it seems there's been a hiccup in our matching process of the group %s. Try again later..." % group_name

            bot.send_message(chat_id=query.message.chat.id,
                             text=message_text)
            bot.answer_callback_query(query.id, text='Error 🎅')
            

        state.pop(query.from_user.id)

    return callback


def gpt_suggestion(bot: telebot.TeleBot, state: dict=None):
    def _send_request(info: str):
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': 'system', 'content': 'Imagine you are Santa and helping elf to choose present for his recipient.'},
                {"role": "user", "content": info}
            ],
            max_tokens=256, # Unfortunately, I can't afford more.
            temperature=0.1 # I am not concerned about truthfulness of statements, but rather trying to make bot as creative as possible. 
        )

        return completion.choices[0].message.content
        
    def callback(query: types.CallbackQuery):
        santa_id = int(query.data.split('_')[1])
        group_id = int(query.data.split('_')[2])

        entry = dbutils.get_gu_entry(state['database'], user_id=santa_id, group_id=group_id)

        request = 'Be friendly and write emojies. Відповідай коротко та яскраво. Відповідай українською. I have some troubles choosing present for my recipient. If recipient did not provide any more information just be creative! There is some information that he/she told me: ' + \
        'About: ' + entry[0] + '\n' + \
        'Desired presents (optional): ' + entry[1]

        if openai.api_key != '':
            bot.answer_callback_query(query.id,
                                    text='🎅 Pending request to Laplandia...')
        else:
            bot.answer_callback_query(query.id,
                                    text='🎅 Santa is busy right now. Try again later!')
        
        try:
            answer = _send_request(request)
            bot.reply_to(query.message,
                        text=answer)
            
        except:
            bot.reply_to(query.message,
                         text="Some troubles with Santa Claus's reindeer occured... 😔 Try again later!")

            
    return callback


