from bot.bot import router_chat as router, BotStates
from aiogram import filters, types, F
from aiogram.fsm.context import FSMContext
from gpt_openai.price_embeding import price_db
from bot.keyboard import make_keyboard
from langchain.docstore.document import Document
import json
from aiogram.enums.parse_mode import ParseMode
from html import escape

@router.message(BotStates.price)
async def price_chosen(message: types.Message, state: FSMContext):
    await message.answer("ищу...")
    text = message.text
    docs_s = price_db.search_with_score(text)
    await price_chosen1(message, state, docs_s)

async def price_chosen1(message: types.Message, state: FSMContext, docs_s):
    docs =  price_db.parse_searched(docs_s)
    await state.update_data(docs=docs)
    if len(docs)==0:
        await message.answer("Ничего не найдено")
        return
    elif len(docs)==1:
        await price_show_doc(message, state, docs[0])
        return
    l=[]
    h1_current=""
    for i, d in enumerate(docs):
        md = d.metadata
        h1 = md["H1"]
        if h1_current!=h1:
            h1_current=h1
            l.append(f"<b>{escape(h1)}</b>")
        l.append(f"{i}. {escape(d.page_content)}")

    from bot.keyboard import create_choice_keyboard
    kb = create_choice_keyboard(len(docs),"pricechoice" )
    #kb = make_keyboard(l)
    await message.answer(
        text="Выберите раздел для ответа:\n"+"\n".join(l),
        reply_markup=kb,
        parse_mode=ParseMode.HTML
    )
    await state.set_state(BotStates.choosing_price)

@router.message(BotStates.choosing_price)
async def price_show(message: types.Message, state:FSMContext):
    await message.answer("отображаю...", reply_markup=types.ReplyKeyboardRemove())
    text:str = message.text
    user_data = await state.get_data()
    docs = user_data["docs"]
    select_doc = None
    if text.isnumeric():
        try:
            num = int(text)
            if num>=0 and num<len(docs):
                select_doc = docs[num]
        except:
            pass
    else:    
        for doc in docs:
            if text == doc.page_content:
                select_doc = doc
                break
    selected = False
    if select_doc:
        selected = await price_show_doc(message, state, select_doc)
    if not selected:
        await price_chosen(message, state)
    pass

async def price_show_doc(message: types.Message, state:FSMContext, doc:Document):
    selected = False
    t = doc.metadata["type"]
    if t =="H2":
        selected = True
    elif t=="H1":
        await price_chosen1(message, state, [doc])
        return
    table_str = doc_table(doc)
    # parse_mode = aiogram.enums.parse_mode.ParseMode...
    import aiogram
    # print( table_str)
    await message.answer(text=table_str, parse_mode = aiogram.enums.parse_mode.ParseMode.HTML, reply_markup=types.ReplyKeyboardRemove())
    return selected

def doc_table(doc:Document):
    t = doc.metadata["type"]
    if t =="H2":
        content = json.loads( doc.metadata["content"])
        selected = True
    elif t=="H1":
        return 
    h1 = doc.metadata["H1"]
    h = content["header"]
    le = len(h)
    b = content["body"]
    data =[i for i in b if len(i)>1]
    table =[]
    for i in h:
        table.append([i])
    for row in data:
        for i, r in enumerate(table):
            if i>=len(row):
                break
            r.append(row[i])
    s = [f"<b>{h1}</b>\n"]
    # s.append("<table>")
    for r in table:
        # s.append("\n<tr>")
        if len(r)==1:
            continue
        s.append(f"{str_to_html(r[0])}")
        for d in r[1:]:
            s.append(f" |  <b>{str_to_html(d)}</b>")
        # s.append("</tr>")
        s.append("\n")
    # s.append("</table>")
    return  "".join(s)

def str_to_html(s:str):
    return escape(s)


@router.callback_query(lambda c: c.data.startswith('pricechoice_'))
async def handle_rating(callback_query: types.CallbackQuery, state:FSMContext):
    message = callback_query.message
    
    data = callback_query.data.split('_') # Разделяем данные в callback_data
    await message.answer(data[1])
    text:str = data[1]
    user_data = await state.get_data()
    docs = user_data["docs"]
    select_doc = None
    if text.isnumeric():
        try:
            num = int(text)
            if num>=0 and num<len(docs):
                select_doc = docs[num]
        except:
            pass
    selected = False
    if select_doc:
        selected = await price_show_doc(message, state, select_doc)
    if not selected:
        await message.answer("Ничего не найдено")
    pass    
    
