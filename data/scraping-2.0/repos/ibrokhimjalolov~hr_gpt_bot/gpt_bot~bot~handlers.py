from django.conf import settings
from telegram.ext import CallbackQueryHandler, MessageHandler, Filters, ConversationHandler
from telegram.ext.commandhandler import CommandHandler
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from gpt_bot.models import TelegramUser, FlowProcess, Question, Specialization, Region, UserLimit
from .loader import updater
from django.db.models import F
from django.core.cache import cache
import datetime
import openai


class State:
    START_POINT = 1
    ENTER_LANGUAGE = 2
    ENTER_FULL_NAME = 3
    ENTER_PHONE_NUMBER = 4
    ENTER_BIRTH_DATE = 5
    ENTER_REGION = 6
    ENTER_GENDER = 7
    ENTER_CV = 8

    ENTER_CATEGORIES = 9
    ENTER_QUESTION_ANSWER = 10

    CHAT = 18
    UNKNOWN_TEXT = 19
    INLINE_CALLBACK = 20


def init_user(func):
    def wrapper(update: Update, context: CallbackContext):
        chat_id = update.effective_chat.id
        TelegramUser.objects.get_or_create(
            user_id=chat_id,
            defaults={
                "username": update.effective_chat.username,
            }
        )
        return func(update, context)

    return wrapper


openai.api_key = settings.OPENAI_API_KEY


def ask_gpt(prompt):
    question = openai.Completion.create(
        model="text-davinci-003",
        temperature=0.7,
        prompt=prompt,
        max_tokens=1000,
        top_p=1,
        n=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return question.choices[0].text


def get_iq_questions() -> list:
    prompt = "Give 10 questions for testing my IQ. Question in russian."
    question = openai.Completion.create(
        model="text-davinci-003",
        temperature=0.7,
        prompt=prompt,
        max_tokens=2000,
        top_p=1,
        n=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    iq_questions = []
    for q in question.choices[0].text.split("\n"):
        if q:
            iq_questions.append(q)
    return iq_questions


phone_request_button = ReplyKeyboardMarkup(
    [[KeyboardButton("Отправить номер📲", request_contact=True)]],
    resize_keyboard=True,
    one_time_keyboard=True
)
gender_choices = ReplyKeyboardMarkup(
    [["Мужской", "Женский"]],
    resize_keyboard=True,
    one_time_keyboard=True
)


@init_user
def send_welcome(update: Update, context: CallbackContext):
    update.message.reply_text(
        "👨‍💼Добро пожаловать! Пожалуйста, укажите свой номер телефона:",
        reply_markup=phone_request_button
    )
    return State.ENTER_PHONE_NUMBER


def get_user_conv_data(user_id) -> dict:
    return cache.get(f"conv_data_{user_id}", {})


def set_user_conv_data(user_id, data) -> None:
    cache.set(f"conv_data_{user_id}", data, timeout=60 * 60 * 24)


def save_user_conv_data(user_id, data) -> None:
    def get_file_from_id(file_id):
        file = updater.bot.get_file(file_id)
        return file.file_path

    process = FlowProcess.objects.create(
        telegram_user_id=user_id,
        full_name=data["full_name"],
        phone_number=data["phone_number"],
        birth_date=data["birth_date"],
        gender="male" if data["gender"] == "Мужской" else "famale",
        region_id=data["region"],
        cv=get_file_from_id(data["cv_file_id"]),
    )
    process.specialization.set(data["categories"])
    data["process_id"] = process.id
    set_user_conv_data(user_id, data)


def get_cur_question_state(user_id) -> dict:
    return cache.get(f"question_state_{user_id}")


def set_cur_question_state(user_id, data):
    return cache.set(f"question_state_{user_id}", data, timeout=60 * 60 * 24)


@init_user
def get_user_contact(update: Update, context: CallbackContext):
    phone_number = update.message.contact.phone_number
    phone_number = "".join(phone_number.split())
    print(phone_number)
    data = {
        "phone_number": phone_number
    }
    allow = UserLimit.objects.filter(used__lt=F("limit"), phone_number=phone_number).first()
    # if not allow:
    #     update.message.reply_text(
    #         "👨‍💼К сожалению, вы не можете пройти тестирование. Пожалуйста, обратитесь к администратору.")
    #     return ConversationHandler.END
    allow.used = F("used") + 1
    allow.save(update_fields=["used"])
    set_user_conv_data(update.message.chat.id, data)
    update.message.reply_text("👨‍💼Спасибо! Пожалуйста введите свое полное имя:")
    return State.ENTER_FULL_NAME


@init_user
def get_user_full_name(update: Update, context: CallbackContext):
    full_name = update.message.text
    data = get_user_conv_data(update.message.chat.id)
    data["full_name"] = full_name
    set_user_conv_data(update.message.chat.id, data)
    update.message.reply_text("👨‍💼Спасибо! Пожалуйста введите свою дату рождения (dd.mm.yyyy):")
    return State.ENTER_BIRTH_DATE


def get_regions_board():
    regions = [[]]
    for r in Region.objects.all():
        name = r.name
        if len(regions[-1]) < 2:
            regions[-1].append(InlineKeyboardButton(name, callback_data=str(r.id)))
        else:
            regions.append([InlineKeyboardButton(name, callback_data=str(r.id))])
    regions_board = InlineKeyboardMarkup(regions)
    return regions_board


@init_user
def get_user_birth_date(update: Update, context: CallbackContext):
    birth_date = update.message.text
    print(birth_date)
    date = datetime.datetime.strptime(birth_date, "%d.%m.%Y")
    data = get_user_conv_data(update.message.chat.id)
    data["birth_date"] = date
    set_user_conv_data(update.message.chat.id, data)
    update.message.reply_text("👨‍💼Спасибо! Пожалуйста введите свой регион:", reply_markup=get_regions_board())
    return State.ENTER_REGION


@init_user
def get_user_region(update: Update, context: CallbackContext):
    update.callback_query.message.edit_reply_markup(reply_markup=None)
    region = update.callback_query.data
    update.callback_query.message.edit_text(
        f"👨‍💼Спасибо! Пожалуйста введите свой регион:\n{Region.objects.get(id=region).name}")
    user_id = update.effective_chat.id
    data = get_user_conv_data(user_id)
    data["region"] = region
    set_user_conv_data(user_id, data)
    update.callback_query.message.reply_text("👨‍💼Спасибо! Пожалуйста введите свой пол:", reply_markup=gender_choices)
    return State.ENTER_GENDER


@init_user
def get_user_gender(update: Update, context: CallbackContext):
    gender = update.message.text
    if gender not in ["Мужской", "Женский"]:
        return State.ENTER_GENDER
    data = get_user_conv_data(update.message.chat.id)
    data["gender"] = gender
    set_user_conv_data(update.message.chat.id, data)
    # send cv
    update.message.reply_text("👨‍💼Спасибо! Пожалуйста отправьте свое резюме:")
    return State.ENTER_CV


def get_user_category_board(user_id, selected_categories=None):
    selected_categories = selected_categories or []
    categories = [[]]
    for c in Specialization.objects.all():
        name = c.name
        if str(c.id) in selected_categories:
            name = f"✅{name}"
        if len(categories[-1]) < 2:
            categories[-1].append(InlineKeyboardButton(name, callback_data=str(c.id)))
        else:
            categories.append([InlineKeyboardButton(name, callback_data=str(c.id))])
    if len(categories[-1]) < 2:
        categories[-1].append(InlineKeyboardButton("Сохранить", callback_data="save"))
    else:
        categories.append([InlineKeyboardButton("Сохранить", callback_data="save")])
    categories_board = InlineKeyboardMarkup(categories)
    return categories_board


@init_user
def get_user_cv(update: Update, context: CallbackContext):
    cv = update.message.document or update.message.photo[-1]
    data = get_user_conv_data(update.message.chat.id)
    data["cv_file_id"] = cv.file_id
    set_user_conv_data(update.message.chat.id, data)
    categories_board = get_user_category_board(update.message.chat.id)
    update.message.reply_text("👨‍💼Спасибо! Пожалуйста выберите категории, в которых вы хотите работать:",
                              reply_markup=categories_board)
    return State.ENTER_CATEGORIES


def get_user_category(update: Update, context: CallbackContext):
    query = update.callback_query
    user_id = update.callback_query.from_user.id
    data = get_user_conv_data(user_id)
    data.setdefault("categories", list())
    if query.data != "save":
        if query.data in data["categories"]:
            data["categories"].remove(query.data)
        else:
            data["categories"].append(query.data)
        set_user_conv_data(user_id, data)
        categories_board = get_user_category_board(user_id, data["categories"])
        update.callback_query.message.edit_reply_markup(reply_markup=categories_board)
        return State.ENTER_CATEGORIES
    if not data["categories"]:
        context.bot.answer_callback_query(query.id, "Выберите хотя бы одну категорию", show_alert=True)
        return State.ENTER_CATEGORIES
    set_user_conv_data(user_id, data)
    save_user_conv_data(user_id, data)
    update.callback_query.message.reply_text(
        "📝Спасибо! Ваша информация сохранена.\n"
        "🔄Генерация вопросов..."
    )
    iq_questions = get_iq_questions()
    set_cur_question_state(user_id, {"questions": iq_questions, "index": 0, "question_type": "iq_test"})
    update.callback_query.message.reply_text(iq_questions[0])
    return State.ENTER_QUESTION_ANSWER


def analize_user_answers(process_id):
    iq_tests = Question.objects.filter(process_id=process_id, question_type="iq_test")
    result = ask_gpt(
        "\n".join(["%s\nОтвет: %s" % (q.question, q.answer) for q in iq_tests]) +
        "\n\nосноваясь выщеуказанным ответам дай суммарный балл по шкале 1 - 250 для "
        "определения IQ интеллект и верни только цифру."
    )

    def parse_int(st):
        try:
            return int("".join([c for c in st if c.isdigit()]))
        except TypeError:
            return 0

    iq_score = parse_int(result)
    soft_skill_tests = Question.objects.filter(process_id=process_id, question_type="soft_skill")
    prompt = (
            "\n".join(["%s\nОтвет: %s" % (q.question, q.answer) for q in soft_skill_tests]) +
            "\n\nПроанализируй ответы кандидата на вопросы по софт-скиллам и верни результат в виде текста."
    )
    soft_skill_main_result = ask_gpt(prompt).strip()
    if soft_skill_main_result.startswith("Ответ:"):
        soft_skill_main_result = soft_skill_main_result[len("Ответ:"):]
    prompt = (
            "\n".join(["%s\nОтвет: %s" % (q.question, q.answer) for q in soft_skill_tests]) +
            "\n\nПроанализируй ответы кандидата на вопросы по софт-скиллам и рекомендации по улучшению."
    )
    soft_skill_recommendations = ask_gpt(prompt).strip()
    if soft_skill_recommendations.startswith("Ответ:"):
        soft_skill_recommendations = soft_skill_recommendations[len("Ответ:"):]

    tech_tests = Question.objects.filter(process_id=process_id, question_type="professional_test")
    prompt = (
            "\n".join(["%s\nОтвет: %s" % (q.question, q.answer) for q in tech_tests]) +
            "\n\nПроанализируй ответы кандидата на вопросы по техническим навыкам и верни результат в виде текста."
    )
    tech_main_result = ask_gpt(prompt).strip()
    if tech_main_result.startswith("Ответ:"):
        tech_main_result = tech_main_result[len("Ответ:"):]
    prompt = (
            "\n".join(["%s\nОтвет: %s" % (q.question, q.answer) for q in tech_tests]) +
            "\n\nПроанализируй ответы кандидата на вопросы по техническим навыкам и рекомендации по улучшению."
    )
    tech_recommendations = ask_gpt(prompt).strip()
    if tech_recommendations.startswith("Ответ:"):
        tech_recommendations = tech_recommendations[len("Ответ:"):]
    FlowProcess.objects.filter(id=process_id).update(
        iq_test_score=iq_score,
        soft_skill_main_result=soft_skill_main_result,
        soft_skill_recommendation=soft_skill_recommendations,
        professional_test_main_result=tech_main_result,
        professional_test_recommendation=tech_recommendations
    )
    process = FlowProcess.objects.get(id=process_id)
    process.generate_resume()
    return process


@init_user
def get_user_question_answer(update: Update, context: CallbackContext):
    answer = update.message.text

    if len(answer) > 200:
        update.message.reply_text("Слишком длинный ответ. Пожалуйста, ответьте на вопрос в одном сообщении.")
        return State.ENTER_QUESTION_ANSWER

    data = get_cur_question_state(update.message.chat.id)
    conv_data = get_user_conv_data(update.message.chat.id)
    question = data["questions"][data["index"]]
    question_type = data["question_type"]
    Question.objects.update_or_create(
        process_id=conv_data["process_id"],
        index=data["index"], question_type=question_type,
        defaults={"question": question, "answer": answer}
    )
    if data["index"] == len(data["questions"]) - 1:
        if question_type == "iq_test":
            update.message.reply_text("🔄Генерация вопросов...")
            questions = ask_gpt(
                "Сформулируйте 10 вопрос, связанный с софт-навыком <<коммуникация>>, <<руководство>>,"
                "<<решение проблем>>, <<адаптивность>>."
            )
            soft_skill_questions = []
            for q in questions.split("\n"):
                if q:
                    soft_skill_questions.append(q)

            set_cur_question_state(update.message.chat.id,
                                   {"questions": soft_skill_questions, "index": 0, "question_type": "soft_skill"})
            update.message.reply_text(soft_skill_questions[0])
            return State.ENTER_QUESTION_ANSWER
        elif question_type == "soft_skill":
            techs = list(Specialization.objects.filter(id__in=conv_data["categories"]).values_list("name", flat=True))
            for i in range(len(techs)):
                techs[i] = f"<<{techs[i]}>>"
            update.message.reply_text("🔄Генерация вопросов...")
            questions = ask_gpt(
                "Сформулируйте 10 вопрос, связанный с технологиями " + ", ".join(techs) + "."
            )
            tech_skill_questions = []
            for q in questions.split("\n"):
                if q:
                    tech_skill_questions.append(q)
            set_cur_question_state(update.message.chat.id, {"questions": tech_skill_questions, "index": 0,
                                                            "question_type": "professional_test"})
            update.message.reply_text(tech_skill_questions[0])
            return State.ENTER_QUESTION_ANSWER
        else:  # professional_test
            update.message.reply_text("📝Спасибо! Ваши ответы сохранены.")
            update.message.reply_text("🔄Подготовка резюме...")
            process = analize_user_answers(conv_data["process_id"])
            update.message.reply_document(
                document=open(process.generated_resume.path, "rb"),
                caption="📄Ваше резюме готово!"
            )
    else:
        data["index"] += 1
        update.message.reply_text(data["questions"][data["index"]])
        set_cur_question_state(update.message.chat.id, data)
        return State.ENTER_QUESTION_ANSWER


question_conv_handler = ConversationHandler(
    entry_points=[
        CommandHandler('start', send_welcome)
    ],
    states={
        State.ENTER_PHONE_NUMBER: [MessageHandler(Filters.contact, get_user_contact)],
        State.ENTER_FULL_NAME: [MessageHandler(Filters.text, get_user_full_name)],
        State.ENTER_BIRTH_DATE: [MessageHandler(Filters.text, get_user_birth_date)],
        State.ENTER_REGION: [CallbackQueryHandler(get_user_region)],
        State.ENTER_GENDER: [MessageHandler(Filters.text, get_user_gender)],
        State.ENTER_CV: [MessageHandler(Filters.document | Filters.photo, get_user_cv)],
        State.ENTER_CATEGORIES: [CallbackQueryHandler(get_user_category)],
        State.ENTER_QUESTION_ANSWER: [MessageHandler(Filters.text, get_user_question_answer)],
    },
    fallbacks=[]
)

updater.dispatcher.add_handler(question_conv_handler)
