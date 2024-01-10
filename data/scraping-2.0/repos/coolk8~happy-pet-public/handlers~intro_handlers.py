from aiogram import Router, F, Bot
from aiogram.types import Message, InputMediaPhoto
from aiogram.filters import Command, CommandStart, StateFilter, Text
from aiogram.filters.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import default_state
from aiogram.types import (CallbackQuery, InlineKeyboardButton,
                           InlineKeyboardMarkup, Message, PhotoSize)
from aiogram.methods import SendChatAction

from aiogram.types.input_file import FSInputFile

from locales.loc_utils import i18n, _
from amplitude.analytics import post_analytic_event

from openai_actions.text_answer import create_short_answer

import database.mongodb_model as mongodb_model
from image_generations.stable_diffusion import generate_pet_images


router: Router = Router()


#images_media_group_object: list[Message] = []

class FSMFillIntro(StatesGroup):
    choosing_pet_type: State = State()
    choosing_image: State = State()
    writing_name: State = State()
    writing_character: State = State()
    ready_to_chat: State = State()



#handler for /start command and no state
@router.message(CommandStart(), StateFilter(default_state))
async def process_start_command(message: Message, state: FSMContext, users_collection):

    current_user = await mongodb_model.get_or_create_user(users_collection,
                                    _id = message.from_user.id,                             
                                    user_id=message.from_user.id,
                                    username=message.from_user.username,
                                    user_first_name=message.from_user.first_name,
                                    user_lastname=message.from_user.last_name,
                                    user_lang=message.from_user.language_code,)
    
    await state.update_data(user_id=current_user.user_id,
                      user_lang=current_user.user_lang)
    
    fsm_user = await state.get_data()

    #creating buttons for pet type choice
    cats_button = InlineKeyboardButton(text = _(fsm_user["user_lang"], "cat"), callback_data = "cat")
    dogs_button = InlineKeyboardButton(text = _(fsm_user["user_lang"], "dog"), callback_data = "dog")
    keyboard: list[list[InlineKeyboardButton]] = [[cats_button, dogs_button]]
    markup = InlineKeyboardMarkup(inline_keyboard = keyboard)
    await message.answer(text = _(fsm_user["user_lang"], "start_message"), reply_markup=markup)

    await post_analytic_event({"user_id": fsm_user["user_id"],
                               "event_type": "start_command",
                               "language": fsm_user["user_lang"]
                                })


    #setting first state    
    await state.set_state(FSMFillIntro.choosing_pet_type)

#handler for choosing pet type
@router.callback_query(StateFilter(FSMFillIntro.choosing_pet_type),
                   Text(text=['cat', 'dog']))
async def process_pet_type_callback(callback: CallbackQuery, state: FSMContext, users_collection):
    #first time using Redis for storing data
    #await state.update_data(user_pet_type=callback.data)
    current_user = await mongodb_model.update_user(users_collection, 
                                    _id=callback.from_user.id, 
                                    user_pet_type=callback.data)
    
    await state.update_data(user_pet_type=callback.data)


    fsm_user = await state.get_data()
    await post_analytic_event({"user_id": fsm_user["user_id"],
                               "event_type": "type_choice_done",
                               "user_properties": {"pet_type": fsm_user["user_pet_type"]},
                                })
    await callback.message.answer(text=_(fsm_user["user_lang"], "pet_choice_done_message"))

    #list for 3 images    
    pets_images = []

    #TODO убрать хардкод в 3 картинки
    #TODO засекать за сколько генерятся картинки и если больше 10 секунд, то писать что долго генерится
    #а сначала засекать и класть в аналитику в event_properties
    if current_user.user_sd_images is None:
        pets_images_files = await generate_pet_images(callback.from_user.id, callback.data, samples=3)
        pets_images = [InputMediaPhoto(media=FSInputFile(f'out/{pets_images_files[0]}')),
                    InputMediaPhoto(media=FSInputFile(f'out/{pets_images_files[1]}')),
                    InputMediaPhoto(media=FSInputFile(f'out/{pets_images_files[2]}'))]

    else:
        pets_images = [InputMediaPhoto(media=current_user.user_sd_images[0]), 
                    InputMediaPhoto(media=current_user.user_sd_images[1]), 
                    InputMediaPhoto(media=current_user.user_sd_images[2])]

    #TODO Добавить await callback.answer() чтобы не было часиков

    #TODO Добавить к картинкам подписи

    pet_image_1_button = InlineKeyboardButton(text = _(fsm_user["user_lang"], "image_1_button"), callback_data = "image_1_pressed")
    pet_image_2_button = InlineKeyboardButton(text = _(fsm_user["user_lang"], "image_2_button"), callback_data = "image_2_pressed")
    pet_image_3_button = InlineKeyboardButton(text = _(fsm_user["user_lang"], "image_3_button"), callback_data = "image_3_pressed")
    
    choose_img_keyboard: list[list[InlineKeyboardButton]] = [[pet_image_1_button, pet_image_2_button, pet_image_3_button]]
    markup_choose_img = InlineKeyboardMarkup(inline_keyboard = choose_img_keyboard)

    #saving media group object for deleting images later
    result = await callback.message.answer_media_group(pets_images)

    await state.update_data(messages_to_delete=[[result[0].chat.id, result[0].message_id],
                             [result[1].chat.id, result[1].message_id],
                             [result[2].chat.id, result[2].message_id]])

    #saving images ids for future use
    sd_images = [result[0].photo[-1].file_id, 
                 result[1].photo[-1].file_id,
                 result[2].photo[-1].file_id]
    await state.update_data(user_sd_images=sd_images)
    await mongodb_model.update_user(users_collection,
                                    _id=callback.from_user.id, 
                                    user_sd_images=sd_images)
                        
    await post_analytic_event({"user_id": fsm_user["user_id"],
                               "event_type": "images_generation_done"
                                })

    await callback.message.answer(text=_(fsm_user["user_lang"],"choose_img_message"), reply_markup=markup_choose_img)
    await state.set_state(FSMFillIntro.choosing_image)
 
#handler for any wrong input in choose_pet_type state
@router.message(StateFilter(FSMFillIntro.choosing_pet_type))
async def warning_wrong_type(message: Message, state: FSMContext):
    fsm_user = await state.get_data()
    await post_analytic_event({"user_id": fsm_user["user_id"],
                               "event_type": "wrong_type_message",
                               "event_properties": {"state": await state.get_state()}
                                })    
    await message.answer(text = _(fsm_user["user_lang"], "wrong_type_message"))

#handler for image choice 
@router.callback_query(StateFilter(FSMFillIntro.choosing_image),
                   Text(text=['image_1_pressed', 'image_2_pressed', 'image_3_pressed']))
async def process_choice_img_callback(callback: CallbackQuery, 
                                      state: FSMContext, 
                                      bot: Bot,
                                      users_collection):
    #TODO добавить подтверждение выбора, так как можно случайно нажать кнопку

    fsm_user = await state.get_data()

    await post_analytic_event({"user_id": fsm_user["user_id"],
                               "event_type": "image_choice_done",
                               "event_properties": {"image_number_str": callback.data}
                                })    

    #TODO не удалять картинку с выбранным животным и не присылать её потом собственно
    for msg in fsm_user["messages_to_delete"]:
       #await bot.delete_message(msg.chat.id, msg.message_id)
       await bot.delete_message(msg[0], msg[1])

    #delete mesessage with buttons
    #TODO: где-то тут падает, если пройти путь, удалить FSM, а потом пройти путь ещё раз
    await bot.delete_message(callback.message.chat.id, callback.message.message_id)

    if callback.data == 'image_1_pressed':
        user_pet_image = fsm_user["user_sd_images"][0]        
        await state.update_data(user_pet_image=user_pet_image)
        await mongodb_model.update_user(users_collection,
                                        _id=callback.from_user.id, 
                                        user_pet_image=user_pet_image)

        await callback.message.answer_photo(user_pet_image)
    if callback.data == 'image_2_pressed':
        user_pet_image = fsm_user["user_sd_images"][1]
        
        await state.update_data(user_pet_image=user_pet_image)
        await mongodb_model.update_user(users_collection,
                                        _id=callback.from_user.id, 
                                        user_pet_image=user_pet_image)

        await callback.message.answer_photo(user_pet_image)
    if callback.data == 'image_3_pressed':
        user_pet_image = fsm_user["user_sd_images"][2]

        
        await state.update_data(user_pet_image=user_pet_image)
        await mongodb_model.update_user(users_collection,
                                        _id=callback.from_user.id, 
                                        user_pet_image=user_pet_image)

        await callback.message.answer_photo(user_pet_image)

    await callback.message.answer(text=_(fsm_user["user_lang"], "name_pet_message"))
    await state.set_state(FSMFillIntro.writing_name)

#handler for wrong input during image choice
@router.callback_query(StateFilter(FSMFillIntro.choosing_image))
async def process_wrong_choice_img_callback(callback: CallbackQuery, state: FSMContext):
    fsm_user = await state.get_data()
    await post_analytic_event({"user_id": fsm_user["user_id"],
                               "event_type": "wrong_type_message",
                               "event_properties": {"state": await state.get_state()}
                                })      
    await callback.message.answer(text=_(fsm_user["user_lang"], "wrong_input_image_choice"))

#handler for name input
@router.message(StateFilter(FSMFillIntro.writing_name), F.text)
async def writing_pet_name(message: Message, state: FSMContext, users_collection):
    #TODO положить в БД
    #await state.update_data(user_pet_name=message.text)
    fsm_user = await state.get_data()
    await post_analytic_event({"user_id": fsm_user["user_id"],
                               "event_type": "name_input_done"
                                })    
    await mongodb_model.update_user(users_collection,
                                    _id = message.from_user.id, 
                                    user_pet_name=message.text)
    await state.update_data(user_pet_name=message.text)
    await state.set_state(FSMFillIntro.writing_character)
    await message.answer(text = _(fsm_user["user_lang"], "describe_pet_message"))

@router.message(StateFilter(FSMFillIntro.writing_name))
async def writing_pet_name_wrong_input(message: Message, state: FSMContext):
    fsm_user = await state.get_data()
    await post_analytic_event({"user_id": fsm_user["user_id"],
                               "event_type": "wrong_type_message",
                               "event_properties": {"state": await state.get_state()}
                                })        
    await message.answer(text = _(fsm_user["user_lang"], "wrong_name_message"))

#this handler for name input
@router.message(StateFilter(FSMFillIntro.writing_character), F.text)
async def writing_pet_character(message: Message, state: FSMContext, users_collection):
    #TODO F.text пропускает эмодзи, надо бы проверить, что только текст
    #await state.update_data(user_pet_character=message.text)
    fsm_user = await state.get_data()

    await post_analytic_event({"user_id": fsm_user["user_id"],
                               "event_type": "character_input_done"
                                })
    await mongodb_model.update_user(users_collection,
                                    _id=message.from_user.id, 
                                    user_pet_character=message.text)
    await state.update_data(user_pet_character=message.text)
     

    await state.set_state(FSMFillIntro.ready_to_chat)
    await message.answer(text = _(fsm_user["user_lang"], "ready_to_chat_message"))

@router.message(StateFilter(FSMFillIntro.writing_character))
async def writing_pet_character_wrong_input(message: Message, state: FSMContext):
    fsm_user = await state.get_data()
    await post_analytic_event({"user_id": fsm_user["user_id"],
                               "event_type": "wrong_type_message",
                               "event_properties": {"state": await state.get_state()}
                                })      
    await message.answer(text = _(fsm_user["user_lang"], "wrong_character_message"))

@router.message(StateFilter(FSMFillIntro.ready_to_chat), F.text)
async def chating_with_pet(message: Message, state: FSMContext, bot: Bot):
    #TODO делаем запрос в OpenAI
    #TODO F.text пропускает эмодзи, надо бы проверить, что только текст и какой это текст, а то сломаем запрос в openAI
  
    fsm_user = await state.get_data()
    #TODO: Показывать раз в 5 сообщений изображения 

    await post_analytic_event({"user_id": fsm_user["user_id"],
                               "event_type": "answer_generation_started"
                                })
    
    await bot.send_chat_action(message.chat.id, "typing")

    system_message = _(fsm_user["user_lang"], 
                       "open_ai_system_prompt",
                       pet_type = _(fsm_user["user_lang"], fsm_user["user_pet_type"]),
                       pet_name = fsm_user["user_pet_name"],
                       pet_character = fsm_user["user_pet_character"])

    #TODO: добавить знак "печатаю" пока генерим ответ
    #TODO: хендлить, когда послали два сообщения подряд
    short_answer = await create_short_answer(message.text, system_message)

    await post_analytic_event({"user_id": fsm_user["user_id"],
                               "event_type": "answer_generation_done"
                                })

    if len(short_answer) > 1000:           
        await message.answer_photo(fsm_user["user_pet_image"])
        await message.answer(text = short_answer)
    else:
        await message.answer_photo(fsm_user["user_pet_image"], caption=short_answer)
        


@router.message(StateFilter(FSMFillIntro.ready_to_chat))
async def chating_with_pet_wrong_input(message: Message, state: FSMContext):
    fsm_user = await state.get_data()
    await post_analytic_event({"user_id": fsm_user["user_id"],
                               "event_type": "wrong_type_message",
                               "event_properties": {"state": await state.get_state()}
                                })      
    await message.answer(text = _(fsm_user["user_lang"], "wrong_input_message"))




#TODO добавить обработку неправильного ввода во всех состояниях
# @dp.message(StateFilter(default_state))
# async def send_echo(message: Message):
#     await message.reply(text='Извините, моя твоя не понимать')    

#@router.message(StateFilter(FSMFillIntro.choose_img))
