from telegram import *
from time import sleep
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from db.session import SessionLocal
from models.message import Message
from models.subscribe import Subscribe
from crud.subscribe import subscribe as crudSubscribe
from crud.user import user as crudUser
from crud.message import message as crudMessage
from crud.payment import payment as crudPayment
from crud.openai import API as crudAPI, StatusEnum
from crud.offer import OFF as crudOffer
from crud.used_offer import Used as crudUsed
from schema.payment import PaymentCreate
from schema.openai import OpenAICreate
from schema.offer import OfferCreate
from schema.used_offer import UsedOfferCreate

from .utils import (is_commands, try_again_exception,command_validate,
                    generate_invite_code, back_home, send_waiting_action)
from .commands import (chat_handler, send_message_to_all,
                       generate_picture, submit_prompt, PDF_to_word,
                       docx_translate, transcript_translate)
from .payment import send_request, verify
from .keyboards import  join_keyboard

session = SessionLocal()
pool = ThreadPoolExecutor(4)

CHANNELS = crudSubscribe.get_channels(db=session)
AUDIO_PATH = f'{Path(__file__).parent.resolve()}/audio/'
DOC_PATH = f'{Path(__file__).parent.resolve()}/document/'
JOIN_KEYBOARD = join_keyboard(channels=CHANNELS)

class GptBot():
    def __init__(self):
        
        api_key = crudAPI.get_first_active(db=session)
        if api_key:
            openai.api_key = api_key.api
            logger.info(f"email {api_key.email} with api {openai.api_key} selected")
        else:
            logger.warning(f"Not valid api key fund default api selected")
            openai.api_key = OPENAI_KEY
        self.BOT_TOKEN = BOT_TOKEN
        self.bot = None
    def bot_polling(self):
        logger.info(msg=f"{datetime.now()}: Starting bot polling now")
        while True:
            try:
                logger.info(msg=f"{datetime.now()}: New bot instance started")
                self.bot = TeleBot(self.BOT_TOKEN)  # Generate new bot instance
                self.bot_actions()  # If bot is used as a global variable, remove bot as an input param
                self.bot.polling(none_stop=True, interval=2, timeout=30)
            except Exception as ex:  # Error in polling
                logger.info("{}: Bot polling failed, restarting in {}sec. Error:\n{}".format(datetime.now(), 30, ex))
                # self.bot.stop_polling()
                sleep(2)
                continue
            else:  # Clean exit
                self.bot.stop_polling()

            logger.info(msg=f"{datetime.now()}: Bot polling loop finished")
            break  # End loop
    
    @staticmethod
    def check_subscribe(message, channels: List[Subscribe] = CHANNELS, token: int = 0) -> bool:
        return command_validate(message=message, db=session, channels=channels, token = token)
    
    def bot_actions(self):
        # start command
        @self.bot.message_handler(commands=['start'], func=self.check_subscribe)
        @try_again_exception(error_name='welcome')
        def welcome(message):
            self.bot.send_message(message.chat.id, trans('M_start_message'),
                                  reply_markup=START_KEYBOARD)

        # profile command
        @try_again_exception(error_name='profile')
        @self.bot.message_handler(regexp=trans("C_profile_status"),func=self.check_subscribe)
        def profile(message):
            user = crudUser.get_by_chat_id(db=session, chat_id=str(message.chat.id))
            profile_message = trans("M_profile").format(id=user.chat_id,
                                                        token=user.token,
                                                        requests=user.requests,
                                                        score=user.score)
            self.bot.send_message(message.chat.id,profile_message,
                                  reply_markup=START_KEYBOARD)

        # charge command for admin
        @self.bot.message_handler(commands=['charge'], func=lambda message: crudUser.is_admin(db=session, chat_id=str(message.chat.id)))
        @try_again_exception(error_name='charge admin')
        def charge(message):
            self.bot.send_message(message.chat.id,
                                  "Send me user id that you want to charge")
            self.bot.register_next_step_handler(message=message,
                                                callback=charge_account_step_1)

        @try_again_exception(error_name='charge_account_step_1')
        def charge_account_step_1(message):
            user_chat_id = message.text
            self.bot.send_message(message.chat.id, "How much you wana charge!")
            self.bot.register_next_step_handler(message=message,
                                                callback=charge_account_step_2,
                                                user_chat_id=user_chat_id)

        @try_again_exception(error_name='charge_account_step_2')
        def charge_account_step_2(message, user_chat_id):
            value = int(message.text)
            user = crudUser.charge_token(db=session, chat_id=user_chat_id, value=value)
            if user:
                profile_message = trans("M_profile").format(id=user.chat_id, token=user.token,
                                                            requests=user.requests,
                                                            score=user.score)
                self.bot.send_message(message.chat.id,
                                      profile_message,
                                      reply_markup=START_KEYBOARD)

            else:
                self.bot.send_message(message.chat.id,
                                      "not find user!",
                                      reply_markup=START_KEYBOARD)

        # add api command for admin
        @self.bot.message_handler(commands=['addapi'], func=lambda message: crudUser.is_admin(db=session, chat_id=str(message.chat.id)))
        @try_again_exception(error_name='add api admin')
        def add_api(message):
            self.bot.send_message(message.chat.id,
                                  "Send me api")
            self.bot.register_next_step_handler(message=message,
                                                callback=add_api_step_1)
            
        @try_again_exception(error_name='add_api_step_1')
        def add_api_step_1(message):
            api = message.text
            self.bot.send_message(message.chat.id, "send me email")
            self.bot.register_next_step_handler(message=message,
                                                callback=add_api_step_2,
                                                api=api)

        @try_again_exception(error_name='add_api_step_2')
        def add_api_step_2(message, api):
            email = message.text
            open_ai = OpenAICreate(api=api, email=email, chat_id=str(message.chat.id))
            api_created = crudAPI.create(db=session, obj_in=open_ai)
            if api_created:
                result = f"""API for email {api_created.email} created and api is {api_created.api} and 
                status {api_created.status}
                """
                self.bot.send_message(message.chat.id,
                                      result,
                                      reply_markup=START_KEYBOARD)

            else:
                self.bot.send_message(message.chat.id,
                                      "something wrong",
                                      reply_markup=START_KEYBOARD)
        @self.bot.message_handler(commands=['message'], func=lambda message: crudUser.is_admin(db=session, chat_id=str(message.chat.id)))
        @try_again_exception(error_name='send_message_to_users admin')
        def send_message_to_users(message):
            self.bot.send_message(message.chat.id,
                                  "Send me message you want forward to all")
            self.bot.register_next_step_handler(message=message,
                                                callback=forward_message_step_1)

        # get apis command for admin
        @self.bot.message_handler(commands=['getapi'], func=lambda message: crudUser.is_admin(db=session, chat_id=str(message.chat.id)))
        @try_again_exception(error_name='get api admin')
        def getapis(message):
            keyboard = InlineKeyboardMarkup(row_width=2)
            apis = crudAPI.get_multi(db=session)
            for api in apis:
                if api.status == StatusEnum.ACTIVE:
                    status = "🟢" 
                elif api.status == StatusEnum.INACTIVE:
                    status = "🔴"
                else:
                    status = "🟡"
                text = api.email + " " + status
                keyboard.add(InlineKeyboardButton(text=text, callback_data=f'api_{api.email}'))
            self.bot.send_message(message.chat.id, "Select email for delete", reply_markup=keyboard)

        @self.bot.callback_query_handler(func=lambda call: "api_" in call.data)
        def delete_api(call):
            try:
                data = str(call.data).split('_')
                email = data[1]
                api = crudAPI.remove(db=session, email=email)
                if api:
                    self.bot.send_message(call.message.chat.id, "🟢Success", reply_markup=START_KEYBOARD)
                else:
                    self.bot.send_message(call.message.chat.id, "🔴Something wrong", reply_markup=START_KEYBOARD)
            except Exception as e:
                self.bot.send_message(call.message.chat.id, trans("M_something_wrong"),
                                      reply_markup=START_KEYBOARD)
                
        @try_again_exception(error_name='forward_message_step_1 admin')
        def forward_message_step_1(message):
            key_markup = ReplyKeyboardMarkup(row_width=3, resize_keyboard=True)
            key_markup.add(KeyboardButton("Yes"),
                           KeyboardButton("No"))

            self.bot.send_message(message.chat.id,
                                f"your message is \n {message.text} \nare you sure?",
                                reply_markup=key_markup)

            self.bot.register_next_step_handler(message=message,
                                                callback=forward_message_step_2,
                                                text=message.text)

        @try_again_exception(error_name='forward_message_step_2 admin')
        def forward_message_step_2(message, text: str):
            if message.text == "Yes":
                pool.submit(send_message_to_all, message=text, db=session)
                self.bot.send_message(message.chat.id,"your message is sending!",
                                      reply_markup=START_KEYBOARD)
            elif message.text == "No":
                self.bot.send_message(message.chat.id,"your message is canceled!",
                                      reply_markup=START_KEYBOARD)
       
        @self.bot.message_handler(commands=['addoffer'],
                                  func=lambda message: crudUser.is_admin(db=session,
                                                                         chat_id=str(message.chat.id)))
        @try_again_exception(error_name='add offer admin')
        def add_offer(message):
            self.bot.send_message(message.chat.id,
                                  "How much is discount?(ex: 30)")
            self.bot.register_next_step_handler(message=message,
                                                callback=add_offer_step_1)
            
        @try_again_exception(error_name='add_offer_step_1')
        def add_offer_step_1(message):
            try:
                discount = float(message.text)
                self.bot.send_message(message.chat.id, "How many days does it expire?(ex: 20)")
                self.bot.register_next_step_handler(message=message,
                                                    callback=add_offer_step_2,
                                                    discount=discount)
            except Exception:
                self.bot.send_message(message.chat.id, "Not valid, try again",
                                      reply_markup=START_KEYBOARD)

        @try_again_exception(error_name='add_offer_step_2')
        def add_offer_step_2(message, discount):
            try:
                days = int(message.text)
            except Exception:
                self.bot.send_message(message.chat.id, "Not valid, try again",
                                      reply_markup=START_KEYBOARD)
                return
            
            obj_in = OfferCreate(days=days, discount=discount)
            offer = crudOffer.create(db=session, obj_in=obj_in)
            if offer:
                result = f"""
                Offer code {offer.code} created and is valid until \n 
                {offer.expire_date} and discount is {offer.discount}%
                """
                self.bot.send_message(message.chat.id,
                                      result,
                                      reply_markup=START_KEYBOARD)
            else:
                self.bot.send_message(message.chat.id,
                                      "something wrong",
                                      reply_markup=START_KEYBOARD)
       
        @self.bot.message_handler(regexp=trans("C_billings"))
        @try_again_exception(error_name='billings main handler')
        def billings(message):
            self.bot.send_message(message.chat.id,
                                  trans("M_billings_message").format(chat_value=CHAT_VALUE,
                                                                audio_value=AUDIO_VALUE,
                                                                summarize_value=SUMMARIZE_VALUE,
                                                                audio_score=AUDIO_SCORE,
                                                                image_value=PIC_VALUE,
                                                                buy_score=PURCHASE_PER_TOKEN_SCORE,
                                                                bonus=BONUS,
                                                                exchange_score=EXCHANGE_SCORE),
                                  parse_mode="MarkdownV2")
            
        @self.bot.message_handler(func=lambda message: message.text == trans("C_Help") or message.text == '/help')
        @try_again_exception(error_name='help_me')
        def help_me(message):
            self.bot.reply_to(message, trans("M_help"), parse_mode='Markdown')

        @self.bot.message_handler(func=lambda message: message.text == trans("C_contact") or message.text == '/contact')
        @try_again_exception(error_name='social_media')
        def social_media(message):
            self.bot.send_message(message.chat.id,trans("M_support_me"),
                                  reply_markup=social_keyboard())

        @self.bot.message_handler(func=lambda message: not is_commands(message=message) and self.check_subscribe(message=message, token=REPLY_VALUE))
        @try_again_exception(error_name='chat in main handler')
        def chat(message, user_messages: List[Message] = None):
            result = pool.submit(chat_handler, message=message, db=session, user_messages=user_messages).result()
            if result:
                value = REPLY_VALUE if user_messages else CHAT_VALUE
                crudUser.use_token(db=session, chat_id=str(message.chat.id),
                                   value=value, score=CHAT_SCORE)
                profile(message=message)
            elif not result:
                self.bot.send_message(message.chat.id, trans("M_bot_to_busy"),
                                      reply_markup=START_KEYBOARD)

        @self.bot.message_handler(func=lambda message: self.check_subscribe(message=message, token=AUDIO_VALUE), regexp=trans("C_Transcript"), content_types=['text'])
        @try_again_exception(error_name='Create_transcription main handler')
        def Create_transcription(message):
            self.bot.send_message(message.chat.id,
                                  trans("M_send_audio").format(message='رونویسی', docx=""),
                                  reply_markup=BACKHOME_KEYBOARD)
            self.bot.send_message(message.chat.id, trans("M_not_free_command").format(token=AUDIO_VALUE,
                                                                                      score=AUDIO_SCORE))
            self.bot.register_next_step_handler(message=message,
                                                callback=Create_transcription_step_1,
                                                is_transcription=True)

        @self.bot.message_handler(content_types=['audio', 'voice'], func=lambda message: self.check_subscribe(message=message, token=AUDIO_VALUE))
        @try_again_exception(error_name='Create_transcription_step_1 main handler')
        def Create_transcription_step_1(message, is_transcription: bool = True):
            chat_id = str(message.chat.id)
            if back_home(message=message, func=welcome):
                return
            content = message.content_type
            if content == 'document' and not is_transcription:
                if pool.submit(docx_translate, db=session, message=message, doc_path=DOC_PATH).result():
                    crudUser.use_token(db=session, chat_id=chat_id, value=AUDIO_VALUE, score=AUDIO_SCORE)
                    profile(message=message)
                else:
                    self.bot.send_message(chat_id, trans("M_bot_to_busy"), reply_markup=START_KEYBOARD)
                return

            else:
                if pool.submit(transcript_translate, db=session, message=message, doc_path=DOC_PATH, transcript=is_transcription).result():
                    crudUser.use_token(db=session, chat_id=chat_id, value=AUDIO_VALUE, score=AUDIO_SCORE)
                    profile(message=message)


        @self.bot.message_handler(func=lambda message: self.check_subscribe(message=message, token=AUDIO_VALUE), regexp=trans("C_translation"), content_types=['text'])
        @try_again_exception(error_name='Create_translate main handler')
        def Create_translate(message):
            chat_id = str(message.chat.id)
            self.bot.send_message(chat_id,trans("M_send_audio").format(message='ترجمه',
                                                                        docx="docx"),
                                  reply_markup=BACKHOME_KEYBOARD)
            self.bot.send_message(chat_id, trans("M_not_free_command").format(token=AUDIO_VALUE,
                                                                              score=AUDIO_SCORE))
            self.bot.register_next_step_handler(message=message,
                                                callback=Create_transcription_step_1,
                                                is_transcription=False)

        @self.bot.callback_query_handler(func=lambda call: call.data == 'buy_token')
        def buy_token_handler(call):
            buy_token(message=call.message)

        @self.bot.message_handler(func=self.check_subscribe, regexp=trans("C_buy_token"), content_types=['text'])
        @try_again_exception(error_name='buy_token main handler')
        def buy_token(message):
            self.bot.send_message(message.chat.id, trans("M_choose_amount"),
                                  reply_markup=BACKHOME_KEYBOARD)
            self.bot.register_next_step_handler(message=message,
                                                callback=buy_token_pre_step_1)
        
        @try_again_exception(error_name='buy_token_pre_step_1 main handler')
        def buy_token_pre_step_1(message):
            chat_id = message.chat.id
            if back_home(message=message, func=welcome):
                return
            
            try:
                amount = int(message.text)
                if amount < PAY_LIMIT:
                    self.bot.send_message(chat_id, trans("M_pay_limit").format(pay_limit=PAY_LIMIT))
                    self.bot.register_next_step_handler(message=message,
                                                            callback=buy_token_pre_step_1)
                    return
                    
            except Exception:
                self.bot.send_message(chat_id, trans("M_invalid_number"),
                                      reply_markup=START_KEYBOARD)
                return
            
            key_markup = ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
            key_markup.add(KeyboardButton(trans("M_no")),
                           KeyboardButton(trans("C_backhome")))

            self.bot.send_message(chat_id,
                                trans("M_send_offerCode"),
                                reply_markup=key_markup)
            
            self.bot.register_next_step_handler(message=message,
                                                callback=buy_token_step_1,
                                                amount=amount)
   
        @try_again_exception(error_name='buy_token_step_1 main handler')
        def buy_token_step_1(message, amount):
            code = None
            chat_id = message.chat.id
            
            if back_home(message=message, func=welcome):
                return
            
            if not message.content_type == 'text':
                self.bot.send_message(chat_id, trans("M_invalid_format"))
                self.bot.register_next_step_handler(message=message,
                                                callback=buy_token_pre_step_1)
                
            if message.text == trans("M_no"):
                self.bot.send_message(chat_id, trans("M_no_problem"))
            else:
                code = message.text
                used_offer = crudUsed.get(db=session, chat_id=str(chat_id), code=code)
                print(used_offer)
                if not used_offer:
                    code = crudOffer.get(db=session, code=code)
                    if code:
                        self.bot.send_message(chat_id,
                                            trans("M_your_offer").format(discount=code.discount))
                    else:
                        self.bot.send_message(chat_id, trans("M_no_offer"))
                else:
                    code = None
                    self.bot.send_message(chat_id, trans("M_no_offer"))
                
            
            discount = 0 if code is None else code.discount
            amount_rial = amount * TOKEN_PRICE
            amount_rial = amount_rial - (amount_rial * discount/100) if discount > 0 else amount_rial
            amount_rial = int(amount_rial)
            response = send_request(amount=amount_rial)
            if response['status']:
                code = code.code if code is not None else None
                pay_option = pay_keyboard(url=response['url'], amount_rial=amount_rial,
                                        authority=response['authority'],
                                        amount_token=amount, code=code)
                        
                self.bot.send_message(chat_id=chat_id,
                                      text=trans("M_shop").format(amount_rial=amount_rial,
                                                                score=PURCHASE_PER_TOKEN_SCORE * amount),
                                                                reply_markup=pay_option,
                                                                parse_mode="MarkdownV2")
                        
                self.bot.send_message(chat_id, trans("M_pay_message"),
                                        reply_markup=START_KEYBOARD, parse_mode="MarkdownV2")
            else:
                self.bot.send_message(chat_id, trans("M_something_wrong"),
                                        reply_markup=START_KEYBOARD)


        @self.bot.callback_query_handler(func=lambda call: "payed_" in call.data)
        def payed(call):
            try:
                data = str(call.data).split('_')
                amount_rial = float(data[1])
                auth = data[2]
                number = int(data[3])
                code = data[4]
                
                chat_id = str(call.message.chat.id)
                response = verify(amount=amount_rial, authority=auth)
                if response['status']:
                    user = crudUser.charge_token(db=session,
                                                 chat_id=chat_id, value=number)
                    crudUser.update_score(db=session,
                                          chat_id=chat_id,
                                          score=(number*PURCHASE_PER_TOKEN_SCORE))
                    payment = PaymentCreate(chat_id=chat_id,
                                            amount=amount_rial,
                                            authority=auth)
                    crudPayment.create(db=session, obj_in=payment)
                    if user:
                        if code != "None":
                            obj_in = UsedOfferCreate(chat_id=str(chat_id), code=code)
                            crudUsed.create(db=session, obj_in=obj_in)
                        profile_message = trans("M_profile").format(id=user.chat_id,
                                                                    token=user.token,
                                                                    requests=user.requests,
                                                                    score=user.score)
                        self.bot.send_message(chat_id, profile_message,
                                              reply_markup=START_KEYBOARD)
                        self.bot.send_message(chat_id, trans("M_success_pay"),
                                              reply_markup=START_KEYBOARD)
                        self.bot.edit_message_reply_markup(chat_id, call.message.message_id)

            except Exception as e:
                try:
                    self.bot.send_message(chat_id, trans("M_something_wrong"),
                                          reply_markup=START_KEYBOARD)
                except Exception as t:
                    logger.warning(
                        f"{datetime.now()}: sending message to user {chat_id} warning is {t}")
                logger.error(
                    f"{datetime.now()}: pay callback query handler user : {chat_id}. error is :\n{e}")

        @self.bot.callback_query_handler(func=lambda call: "reply_" in call.data and self.check_subscribe(message=call.message, token=CHAT_VALUE))
        def reply_message_handler(call):
            chat_id=str(call.message.chat.id)
            try:
                message_id = str(call.data).split('_')[1]
                messages = crudMessage.get_by_message_id(db=session,
                                                         message_id=message_id,
                                                         chat_id=chat_id)
                self.bot.send_message(chat_id, trans("M_send_reply"),
                                      reply_markup=BACKHOME_KEYBOARD)
                self.bot.send_message(chat_id, trans("M_not_free_command").format(token=REPLY_VALUE,
                                                                                  score=CHAT_SCORE))
                self.bot.register_next_step_handler(message=call.message,
                                                    callback=reply_chat_step_1,
                                                    user_messages=messages)
            except Exception as e:
                try:
                    self.bot.send_message(chat_id, trans("M_something_wrong"),
                                          reply_markup=START_KEYBOARD)
                except Exception as t:
                    logger.warning(f"{datetime.now()}: sending message to user {chat_id} warning is {t}")
                logger.error(f"{datetime.now()}: reply callback query handler user : {chat_id}. error is :\n{e}")

        @try_again_exception(error_name='reply_chat_step_1 main handler')
        def reply_chat_step_1(message, user_messages: List[Message]):
            if back_home(message=message, func=welcome):
                return
            chat(message=message, user_messages=user_messages)

        @try_again_exception(error_name='invite main handler')
        @self.bot.message_handler(regexp=trans("C_invite_code"), func=self.check_subscribe)
        def invite(message):
            chat_id=str(message.chat.id)
            user = crudUser.get_by_chat_id(db=session, chat_id=chat_id)
            if user.referral_code is None:
                code, link = generate_invite_code()
                crudUser.update_link(db=session,
                                     chat_id=chat_id,
                                     referral_code=code,
                                     invite_link=link)

            self.bot.send_message(chat_id, trans("M_your_invite_link").format(link=user.invite_link),
                                  reply_markup=START_KEYBOARD)

        @self.bot.message_handler(func=lambda message: self.check_subscribe(message=message, token=1), regexp=trans("C_OCR"), content_types=['text'])
        @try_again_exception(error_name='ocr main handler')
        def ocr(message):
            self.bot.send_message(message.chat.id, trans("M_send_document"),
                                  reply_markup=BACKHOME_KEYBOARD)
            self.bot.register_next_step_handler(message=message, callback=ocr_step_1)

        @self.bot.message_handler(content_types=['document'], func=lambda message: self.check_subscribe(message=message, token=1))
        @try_again_exception(error_name='ocr_step_1 main handler')
        def ocr_step_1(message):
            chat_id = str(message.chat.id)
            if back_home(message=message, func=welcome):
                return
            if pool.submit(PDF_to_word, message=message, doc_path=DOC_PATH).result():
                crudUser.use_token(db=session, chat_id=chat_id, value=DOC_VALUE)
            else:
                self.bot.send_message(chat_id, trans("M_something_wrong"), reply_markup=START_KEYBOARD)

        @self.bot.message_handler(func=lambda message: self.check_subscribe(message=message, token=SUMMARIZE_VALUE), regexp=trans("C_summarize"), content_types=['text'])
        @try_again_exception(error_name='summarize main handler')
        def summarize(message):
            self.bot.send_message(message.chat.id, trans("M_select_summarize_lang"),
                                  reply_markup=LANGUAGE_KEYBOARD)

        @try_again_exception(error_name='summarize_query main handler')
        @self.bot.callback_query_handler(func=lambda call: "summarize_" in call.data and self.check_subscribe(message=call.message))
        def summarize_handler(call):
            chat_id  = str(call.message.chat.id)
            
            try:
                self.bot.edit_message_reply_markup(chat_id=call.message.chat.id,
                                                message_id=call.message.message_id,
                                                reply_markup=None)
            except apihelper.ApiTelegramException as e:
                logger.error(str(e))
                
            lang = str(call.data).split('_')[1]
            self.bot.send_message(chat_id, trans("M_send_your_text_summ"), reply_markup=BACKHOME_KEYBOARD)
            self.bot.send_message(chat_id, trans("M_not_free_command").format(token=SUMMARIZE_VALUE,
                                                                             score=SUMMARIZE_SCORE))
            self.bot.register_next_step_handler(message=call.message,
                                                callback=summarize_step_1, lang=lang)

        @try_again_exception(error_name='summarize_step_1 main handler')
        def summarize_step_1(message, lang: str):
            if back_home(message=message, func=welcome):
                return
            chat_id  = str(message.chat.id)
            prompt = f"Summarize message bellow and translate summarize to {lang}:\n\n {message.text}"
            send_waiting_action(chat_id=chat_id, token_usage=SUMMARIZE_VALUE, score=SUMMARIZE_SCORE)
            result = pool.submit(submit_prompt, db=session, prompt=prompt, chat_id=chat_id).result()
            
            if result:
                crudUser.use_token(db=session, chat_id=chat_id, value=SUMMARIZE_VALUE,
                                   score=SUMMARIZE_SCORE)
                profile(message=message)
            else:
                self.bot.send_message(message.chat.id,
                                      trans("M_bot_to_busy"),
                                      reply_markup=START_KEYBOARD)

        
        @self.bot.message_handler(func=self.check_subscribe, regexp=trans("C_exchange"), content_types=['text'])
        @try_again_exception(error_name='exchange main handler')
        def exchange(message):
            
            self.bot.send_message(message.chat.id, trans("M_excahnge_amount"),
                                  reply_markup=BACKHOME_KEYBOARD)
            self.bot.register_next_step_handler(message=message,
                                                callback=exchange_step_1)

        @try_again_exception(error_name='exchange_step_1 main handler')
        def exchange_step_1(message):
            content = message.content_type
            if not content == 'text':
                self.bot.send_message(message.chat.id,
                                      trans("M_invalid_format"),
                                      reply_markup=START_KEYBOARD)
                return
            if back_home(message=message, func=welcome):
                return

            try:
                score = int(message.text)
                if score < EXCHANGE_SCORE:
                    self.bot.send_message(message.chat.id,
                                         trans("M_exchange_amount_error").format(score=EXCHANGE_SCORE))
                    self.bot.register_next_step_handler(message=message,
                                                        callback=exchange_step_1)

                else:
                    user = crudUser.exchange_score(db=session, chat_id=str(message.chat.id), score=score)
                    if user:
                        self.bot.send_message(message.chat.id,
                                              trans("M_exchange_success").format(score=int(score // EXCHANGE_SCORE) * EXCHANGE_SCORE,
                                                                                token=int(score//EXCHANGE_SCORE)*TOKEN_REWARD),
                                              reply_markup=START_KEYBOARD)
                        profile(message=message)
                    else:
                        self.bot.send_message(message.chat.id,
                                                trans("M_exchange_field"),
                                                reply_markup=START_KEYBOARD)
            except ValueError:
                self.bot.send_message(message.chat.id, trans("M_invalid_number"),
                                      reply_markup=START_KEYBOARD)
        
        @self.bot.message_handler(func=lambda message: self.check_subscribe(message=message, token=PIC_VALUE), regexp=trans("C_generate_picture"), content_types=['text'])
        @try_again_exception(error_name='design_pic main handler')
        def design_pic(message):
            self.bot.send_message(message.chat.id, trans("M_describe"),
                                  reply_markup=BACKHOME_KEYBOARD)
            self.bot.send_message(message.chat.id,
                                          trans("M_not_free_command").format(token=PIC_VALUE, score=PIC_SCORE))
            self.bot.register_next_step_handler(message=message,
                                                callback=design_pic_step_1)

        @try_again_exception(error_name='design_pic_step_1 main handler')
        def design_pic_step_1(message):
            chat_id = int(message.chat.id) 
            if back_home(message=message, func=welcome):
                return
            send_waiting_action(chat_id=chat_id, token_usage=PIC_VALUE, score=PIC_SCORE, action="upload_photo")
            if generate_picture(db=session, prompt=message.text, chat_id=chat_id):
                crudUser.use_token(db=session, chat_id=str(chat_id), value=PIC_VALUE, score=PIC_SCORE)
                profile(message=message)
            else:
                self.bot.send_message(message.chat.id,
                                      trans("M_bot_to_busy"),
                                      reply_markup=START_KEYBOARD)
                
        @self.bot.inline_handler(func=lambda query: 50 > len(query.query) > 0)
        def query_text(query):
            chat_id = query.from_user.id
            join_link = 'https://t.me/GPT_fa_bot'
            join_button = InlineKeyboardButton(text='GPT فارسی', url=join_link)
            join_markup = InlineKeyboardMarkup([[join_button]])
            
            user = crudUser.get_by_chat_id(db=session, chat_id=str(chat_id))
            if not user:
                
                results = [InlineQueryResultArticle(id='1', title='Join First!',
                                                          reply_markup=join_markup,
                                                          thumb_url='https://pasteboard.co/Oi34tLVJI4VU.jpg',
                                                          thumb_height=50, thumb_width=50,
                                                          url = join_link,
                                                          input_message_content=InputTextMessageContent(message_text='https://t.me/GPT_fa_bot'))]
                self.bot.answer_inline_query(query.id, results)
                return
            else:
                try:
                    res = "Hello! How can I assist you today?" 
                    description = "برای پردازش سریع تر و سوالات نامحدود تر به ربات مراجعه کنید."
                    results = [InlineQueryResultArticle(id='1',description=description , reply_markup=join_markup,
                                                              title=res,
                                                              thumb_url='https://pasteboard.co/Oi34tLVJI4VU.jpg',
                                                              thumb_height=50, thumb_width=50,
                                                              input_message_content=InputTextMessageContent(message_text=res))]
                    self.bot.answer_inline_query(query.id, results=results)
                except Exception as e:
                    logger.error(f"{datetime.now()}: {e} ")

