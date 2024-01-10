# import gpt3
from utils import *
import base64
from time import sleep
from asyncio import run
from langchain.prompts import PromptTemplate
import streamlit as st
from pathlib import Path
from streamlit_option_menu import option_menu
import g4f

question_prompt_template = """
            الدور: مساعد الصحة الشخصي الافتراضي (PVHA)

المهمة الرئيسية: تخيل أنك مدرب للياقة البدنية بخبرة 30 عامًا وتقدم نصائح صحية قابلة للتنفيذ وتوجيهات استنادًا إلى ملفي الشخصي.

ملفي الشخصي:
    مستوى اللياقة البدنية الحالي: {cfl} (0 هو الأدنى ، 10 هو الأعلى)
    أهداف الصحية: {hg}
    استخدام المكملات البروتينية: {psu}
    الأمراض المزمنة / الحالات الصحية: {cd}
    الإعاقات الجسدية : {pd}
    العمر: {age} سنة
    وقت الفراغ للنشاط البدني: {ft}
    الجنس: {g}
    حالة الحمل إذا كنت أنثى: {ps}

    يمكنني إعطاؤك معلومات أخرى إضافة للمذكورة. خذها أيضًا في اعتبارك أثناء إعداد الرد.

استعلامي هو: {q}

الإجابة السابقة هي: {prev_answer}

إذا كان استعلامي عبارة تحية أو كلمات عامة مثل "شكرًا" وما شابه ذلك، قدم إجابة مناسبة مثل "على الرحب والسعة!" وتجاهل الملف الشخصي حينئذ

استنادًا إلى استعلامي أولا ثم ملفي الشخصي،
1- ساعدني أولاً في تحقيق هدفي الصحي.
2- ثانيًا، قدم إرشادًا بنظام غذائي مفصل مع أمثلة على الطعام والخضروات والفواكه المناسبة لهدفي مع تحديد مواعيد.
3- ثالثًا
1. حدد الرياضات الهوائية المناسبة لي وأقترح الأوقات المناسبة لممارستها.
2. قدم إرشادات حول زيادة فعالية نشاطي البدني.
3. توصية بأنواع الأنشطة الرياضية المناسبة لي.
4. اقترح وسائل لدمج النشاط البدني في روتيني اليومي.
5. حدد الحد الأدنى للنشاط البدني المناسب لي بناء وقت فراغي.
6. قدم نصائح عامة لنمط حياة صحية.

قيود: تأكد من تقديم الردود بتنسيق واضح وفهم سهل. احترم خصوصية المستخدم ولوائح حماية البيانات.

بالإضافة إلى ذلك، قدم إجابتك مع روابط المصادر التي ساعدتك في الرد في الإجابة.

يرجى عدم الإشارة إلى هويتك في الإجابة، على سبيل المثال، لا تقل إنك GPT 3.5 أو GPT4 أو أي شيء آخر، بل تصرف كمدرب للياقة البدنية.
        """

prompt = PromptTemplate(input_variables=["cfl","hg", "psu", "cd", "pd", "age", "ft", "g", "ps", 'q', 'prev_answer'], template=question_prompt_template)

# async def get_answer_from_chatgpt(question):
#     try:
#         resp = await gpt3.Completion().create(question)
#         return resp
#     except:
#         st.info('Service may be stopped or you are disconnected with internet. Feel free to open an issue here "https://github.com/Mohamed01555/Waai_physicalHealth"')
#         st.stop()

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def main():
    # setup streamlit page
    st.set_page_config(
        page_title="VitaLink Pro",
        page_icon="logo.jpeg")
    
    option = option_menu(
    menu_title=None,
    options=["Home", "FAQs", "Contact"],
    icons=["house-check", "patch-question-fill", "envelope"],
    orientation='horizontal',
    styles={
        "container": {"padding": "0!important", "background-color": "#333"},        
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#ff9900"},
        "nav-link-selected": {"background-color": "#6c757d"},
    }
    )   

    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown(html_code, unsafe_allow_html=True)

    # initialize responses.
    if "responses" not in st.session_state:
        st.session_state.responses = []
    
    if "question" not in st.session_state:
        st.session_state.question = None

    with st.sidebar:
        title = st.markdown("""**مرحبًا، أهلاً بك، أنا "فيتالينك برو!" مساعدك الشخصي في مجال اللياقة البدنية والصحة العامة . أقدم لك كيف تحسن لياقتك البدنية،
نود جمع بعض المعلومات من أجل ذلك. يرجى الإجابة على الأسئلة التالية:**""")
        
        selected_value = st.slider("من فضلك، قيم مستوى لياقتك البدنية الحالي (0 تعني مستوي منخفض و 10 تعني ممتاز )", min_value=0, max_value=10, value=5, step=1)
        
        # Define a list of health goal options
        health_goal_options = ["فقدان الوزن", "بناء العضلات", "الصحة العامة"]

        # Display a multiselect for users to choose multiple health goals
        selected_health_goals = st.multiselect("ما هي أهدافك الرئيسية في مجال الصحة واللياقة؟", health_goal_options)
    
        selected_protein_supplements = st.radio('هل تتناول مكملات البروتين؟', ['لا','نعم'])
        
        selected_chronic_disease = st.radio('هل تعاني من أمراض مزمنة، لا قدر الله؟',  ['لا','نعم'])
        if selected_chronic_disease == 'نعم':
            selected_chronic_disease = st.text_input('من فضلك، ذكر الأمراض المزمنة التي تعاني منها.')
        
        selected_disability = st.radio('هل تعاني من إعاقة؟', ['لا','نعم'])
        if selected_disability == 'نعم':
            selected_disability = st.text_input('من فضلك، ذكر الإعاقة التي تعاني منها.')

        age = st.number_input('من فضلك، أدخل عمرك.')

        free_time = st.text_input('متى يكون وقت فراغك لممارسة النشاط البدني؟')

        gender = st.radio('من فضلك، أدخل جنسك.', ['ذكر', 'أنثي'])
        
        is_pregnant = None
        if gender == 'أنثي':
            is_pregnant = st.radio('هل أنت حامل؟', ['نعم أنا حامل.', 'لا لست بحامل.'])

    if option == 'Home':
        for response in st.session_state.responses:
            with st.chat_message(response['role']):
                st.markdown(response['content'], unsafe_allow_html=True)
        
        st.session_state.question = st.chat_input('اطلب إرشادا غذائيا، وتمارين مناسبة لك، واطلب النصائح ، واسأل حول الصحة بشكل عام', key = 'giving a question')
        if st.session_state.question:
            with st.chat_message('user'):
                st.markdown(st.session_state.question, unsafe_allow_html=True)

            st.session_state.responses.append({'role':"user", 'content': st.session_state.question})
            with st.spinner("فضلًا، لا تقم بإدخال سؤال جديد أو تغيير أي شيء في الشريط الجانبي أثناء إعداد الإجابة!"):
                with st.chat_message('assistant'):
                    st.session_state.message_placeholder = st.empty()

                    query = prompt.format(cfl = selected_value, hg = selected_health_goals, psu = selected_protein_supplements, cd = selected_chronic_disease,
                                        pd = selected_disability, age = age, ft = free_time, g = gender, ps = is_pregnant,
                                        q = st.session_state.question, prev_answer = st.session_state.responses[-2]['content'] if len(st.session_state.responses) != 0 else '')
                    print(query)
                    # response = g4f.ChatCompletion.create(
                    #     model=g4f.models.gpt_35_turbo_0613,
                    #     messages=[{"role": "user", "content": query}],
                    # )

                    # ai_response = run(get_answer_from_chatgpt(query))     

                    response = g4f.ChatCompletion.create(model=g4f.models.gpt_35_turbo, messages=[{"role": "user", "content": query}], stream=True)  # Alternative model setting
                    res = ''
                    for r in response:
                        res += r
                        st.session_state.message_placeholder.markdown(res, unsafe_allow_html=True)                   
            
            st.session_state.responses.append({'role' : 'assistant', 'content' : res})
           
    elif option == 'FAQs':
        FAQs()
    elif option == 'Contact':
        contact()
    else:
        donate()

if __name__ == '__main__':
    main()