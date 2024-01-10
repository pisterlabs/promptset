import streamlit as st
from streamlit_quill import st_quill
import openai
import requests
from google.cloud import storage
from PIL import Image
import io
from imgix import UrlBuilder

username = "test"
password = "1234"
openai.api_key = st.secrets["OPENAI_KEY"]
url = UrlBuilder("captiondemo.imgix.net", include_library_param=False)

if 'login_' not in st.session_state:
    st.session_state.login_ = False

if 'page_placeholder_' not in st.session_state:
    st.session_state.page_placeholder_ = {}

if 'curr_tool_' not in st.session_state:
    st.session_state.curr_tool_ = ""

if 'title_' not in st.session_state:
    st.session_state.title_ = ""

if 'product_desc_list_' not in st.session_state:
    st.session_state.product_desc_list_ = []

if 'tagline_list_' not in st.session_state:
    st.session_state.tagline_list_ = []

if 'hashtag_list_' not in st.session_state:
    st.session_state.hashtag_list_ = []

if 'post_list_' not in st.session_state:
    st.session_state.post_list_ = []

if 'post_select_' not in st.session_state:
    st.session_state.post_select_ = ""

if 'edit_val_' not in st.session_state:
    st.session_state.edit_val_ = ""

if 'curr_img_' not in st.session_state:
    st.session_state.curr_img_ = 2

font_options = [
        "Arial,BoldMT",
		"serif",
		"sans-serif",
		"monospace",
		"cursive",
		"fantasy",
		"serif,bold",
		"sans-serif,bold",
		"monospace,bold",
		"fantasy,bold",
		"serif,italic",
		"sans-serif,italic",
		"monospace,italic",
		"serif,bold,italic",
		"sans-serif,bold,italic",
		"monospace,bold,italic",
		"American Typewriter",
		"American Typewriter Condensed",
		"American Typewriter Condensed Light",
		"American Typewriter Condensed,Bold",
		"American Typewriter Light",
		"American Typewriter,Bold",
		"AndaleMono",
		"Arial Narrow",
		"Arial Narrow,Bold",
		"Arial Narrow,BoldItalic",
		"Arial Narrow,Italic",
		"Arial Rounded MT,Bold",
		"Arial UnicodeMS",
		"Arial,BoldItalicMT",
		"Arial,ItalicMT",
		"Arial-Black",
		"ArialMT",
		"Athelas,Bold",
		"Athelas,BoldItalic",
		"Athelas,Italic",
		"Athelas-Regular",
		"Avenir Next Condensed Demi,Bold",
		"Avenir Next Condensed Demi,BoldItalic",
		"Avenir Next Condensed Heavy",
		"Avenir Next Condensed Heavy,Italic",
		"Avenir Next Condensed Medium",
		"Avenir Next Condensed Medium,Italic",
		"Avenir Next Condensed Regular",
		"Avenir Next Condensed Ultra Light",
		"Avenir Next Condensed Ultra Light,Italic",
		"Avenir Next Condensed,Bold",
		"Avenir Next Condensed,BoldItalic",
		"Avenir Next Condensed,Italic",
		"Avenir Next Demi,Bold",
		"Avenir Next Demi,BoldItalic",
		"Avenir Next Heavy",
		"Avenir Next Heavy,Italic",
		"Avenir Next Medium",
		"Avenir Next Medium,Italic",
		"Avenir Next Regular",
		"Avenir Next Ultra Light",
		"Avenir Next Ultra Light,Italic",
		"Avenir Next,Bold",
		"Avenir Next,BoldItalic",
		"Avenir Next,Italic",
		"Avenir-Black",
		"Avenir-BlackOblique",
		"Avenir-Book",
		"Avenir-BookOblique",
		"Avenir-Heavy",
		"Avenir-HeavyOblique",
		"Avenir-Light",
		"Avenir-LightOblique",
		"Avenir-Medium",
		"Avenir-MediumOblique",
		"Avenir-Oblique",
		"Avenir-Roman",
		"Baskerville",
		"Baskerville,Bold",
		"Baskerville,BoldItalic",
		"Baskerville,Italic",
		"BigCaslon-Medium",
		"BrushScriptMT",
		"Chalkboard",
		"Chalkboard SE Light",
		"Chalkboard SE Regular",
		"Chalkboard SE,Bold",
		"Chalkboard,Bold",
		"Chalkduster",
		"CharcoalCY",
		"Charter Black,Italic",
		"Charter,Bold",
		"Charter,BoldItalic",
		"Charter,Italic",
		"Charter-Black",
		"Charter-Roman",
		"Cochin",
		"Cochin,Bold",
		"Cochin,BoldItalic",
		"Cochin,Italic",
		"Comic Sans MS,Bold",
		"ComicSansMS",
		"Copperplate",
		"Copperplate,Bold",
		"Copperplate-Light",
		"Courier",
		"Courier New,Bold",
		"Courier New,BoldItalic",
		"Courier New,Italic",
		"Courier,Bold",
		"Courier-Oblique",
		"CourierNewPSMT",
		"DIN Alternate,Bold",
		"DIN Condensed,Bold",
		"Didot",
		"Didot,Bold",
		"Didot,Italic",
		"Futura Medium,Italic",
		"Futura-CondensedMedium",
		"Futura-Medium",
		"Geneva",
		"GenevaCyr",
		"Georgia",
		"Georgia,Bold",
		"Georgia,BoldItalic",
		"Georgia,Italic",
		"Gill Sans",
		"Gill Sans Light,Italic",
		"Gill Sans,Bold",
		"Gill Sans,BoldItalic",
		"Gill Sans,UltraBold",
		"GillSans,Italic",
		"GillSans-Light",
		"Helvetica",
		"Helvetica CY,Bold",
		"Helvetica Neue",
		"Helvetica Neue Condensed Black",
		"Helvetica Neue Condensed,Bold",
		"Helvetica Neue Light",
		"Helvetica Neue Light,Italic",
		"Helvetica Neue Medium",
		"Helvetica Neue Medium,Italic",
		"Helvetica Neue Thin",
		"Helvetica Neue Thin,Italic",
		"Helvetica Neue UltraLight",
		"Helvetica Neue UltraLight,Italic",
		"Helvetica Neue,Bold",
		"Helvetica Neue,BoldItalic",
		"Helvetica Neue,Italic",
		"Helvetica,Bold",
		"Helvetica-Light",
		"Helvetica-LightOblique",
		"Helvetica-Oblique",
		"HelveticaCY-Oblique",
		"HelveticaCY-Plain",
		"Herculanum",
		"Hoefler Text Black,Italic",
		"Hoefler Text,Italic",
		"HoeflerText-Black",
		"HoeflerText-Ornaments",
		"HoeflerText-Regular",
		"Impact",
		"Iowan Old Style Black,Italic",
		"Iowan Old Style,Bold",
		"Iowan Old Style,BoldItalic",
		"Iowan Old Style,Italic",
		"IowanOldStyle-Black",
		"IowanOldStyle-Roman",
		"IowanOldStyle-Titling",
		"Lucida Grande",
		"Lucida Grande,Bold",
		"Marion,Bold",
		"Marion,Italic",
		"Marion-Regular",
		"Marker Felt Thin",
		"Marker Felt Wide",
		"Menlo,Bold",
		"Menlo,BoldItalic",
		"Menlo,Italic",
		"Menlo-Regular",
		"Monaco",
		"Noteworthy,Bold",
		"Noteworthy-Light",
		"Optima,Bold",
		"Optima,BoldItalic",
		"Optima,Italic",
		"Optima-ExtraBlack",
		"Optima-Regular",
		"PT Mono,Bold",
		"PT Sans Caption,Bold",
		"PT Sans Narrow,Bold",
		"PT Sans,Bold",
		"PT Sans,BoldItalic",
		"PT Sans,Italic",
		"PT Serif Caption,Italic",
		"PT Serif,Bold",
		"PT Serif,BoldItalic",
		"PT Serif,Italic",
		"PTMono-Regular",
		"PTSans-Caption",
		"PTSans-Narrow",
		"PTSans-Regular",
		"PTSerif-Caption",
		"PTSerif-Regular",
		"Palatino,Bold",
		"Palatino,BoldItalic",
		"Palatino,Italic",
		"Palatino-Roman",
		"Papyrus",
		"Papyrus-Condensed",
		"PlantagenetCherokee",
		"STBaoli-SC-Regular",
		"STYuanti-SC-Light",
		"STYuanti-SC-Regular",
		"SavoyeLetPlain",
		"Seravek",
		"Seravek ExtraLight, Italic",
		"Seravek Light,Italic",
		"Seravek Medium,Italic",
		"Seravek,Bold",
		"Seravek,BoldItalic",
		"Seravek,Italic",
		"Seravek-ExtraLight",
		"Seravek-Light",
		"Seravek-Medium",
		"Skia-Regular",
		"Skia-Regular_Black",
		"Skia-Regular_Black-Condensed",
		"Skia-Regular_Black-Extended",
		"Skia-Regular_Condensed",
		"Skia-Regular_Extended",
		"Skia-Regular_Light",
		"Skia-Regular_Light-Condensed",
		"Skia-Regular_Light-Extended",
		"Snell Roundhand,Bold",
		"SnellRoundhand",
		"SnellRoundhand-Black",
		"Superclarendon Black,Italic",
		"Superclarendon Light,Italic",
		"Superclarendon,Bold",
		"Superclarendon,BoldItalic",
		"Superclarendon,Italic",
		"Superclarendon-Black",
		"Superclarendon-Light",
		"Superclarendon-Regular",
		"Tahoma",
		"Tahoma,Bold",
		"Times New Roman,Bold",
		"Times New Roman,BoldItalic",
		"Times New Roman,Italic",
		"Times,Bold",
		"Times,BoldItalic",
		"Times,Italic",
		"Times-Roman",
		"TimesNewRomanPSMT",
		"Trebuchet MS,Bold",
		"Trebuchet MS,BoldItalic",
		"Trebuchet MS,Italic",
		"TrebuchetMS",
		"Verdana",
		"Verdana,Bold",
		"Verdana,BoldItalic",
		"Verdana,Italic",
		"Waseem",
		"WaseemLight",
		"Webdings",
		"Wingdings-Regular",
		"Wingdings2",
		"Wingdings3",
		"Yuanti SC,Bold",
		"YuppySC-Regular",
		"Zapf Dingbats",
		"Zapfino"
	]

def imgix_url(image="~text", param={}):
        return url.create_url(image, param)

def product_description(description, name, num_responses=1):
        desc = []
        for i in range(num_responses):
            response = openai.Completion.create(
                engine="davinci-instruct-beta-v3",
                prompt="Write a fake marketting for a product called \""+name+"\" \n\""+description+"\".",
                temperature=1.0,
                max_tokens=200,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            desc.append(response.choices[0].get("text").strip("\n"))
        return desc

def tagline(description, name, num_responses=1):
        taglines = []
        for i in range(num_responses):
            response = openai.Completion.create(
              engine="davinci-instruct-beta-v3",
              prompt="Write a tagline for the business named \"" + name + "\" using the description given below.\n\""+description+"\"",
              temperature=1,
              max_tokens=64,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
            )
            taglines.append( "\"" + response.choices[0].get("text").strip("\n") + "\"")
        return taglines

def hashtag(description, num_responses=1):
    hashtags = []
    for i in range(num_responses):
        response = openai.Completion.create(
            engine="davinci-instruct-beta-v3",
            prompt="Write 10 hashtags for social media using the description given below.\n\""+description+"\"",
            temperature=0.7,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        hashtags.append(response.choices[0].get("text").strip("\n"))
    return hashtags

def header(description, demography, intent, tone, name):

    if intent[0] == "Convince": intent = "convincing"
    elif intent[0] == "Inform": intent = "informative"
    elif intent[0] == "Describe": intent = "descriptive"

    response = openai.Completion.create(
        engine="davinci-instruct-beta-v3",
        prompt="Write a 60 word"+ intent +" advertisement for "+ demography[0] +" with an "+ tone[0] +" tone using the description given below for the business named \"" + name + "\".\n\""+description+"\"",
        temperature=0.7,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return "\"" + response.choices[0].get("text").strip("\n") + "\""

def login_page():
    login_placeholder = st.empty()
    st.session_state.page_placeholder_["login_page"] = login_placeholder
    with login_placeholder.container():
        c1,c2,c3 = st.columns([1,1,1])
        c1.text("")
        c3.text("")
        c2.title("Caption.AI")
        c2.subheader("Login")
        user_name = c2.text_input(label="Enter Username", key = 1)
        pass_word = c2.text_input(label="Enter Password", type="password", key = 2)
        login = c2.button(label="Login")
        if login:
            if user_name == username and pass_word == password:
                st.session_state.login_ = True
                c2.success("Succesful")
            else:
                c2.error("Wrong username or password")

def main_page():
    main_placeholder = st.empty()
    st.session_state.page_placeholder_["main_page"] = main_placeholder

    with main_placeholder.container():
        with st.container():
            c1,c2,c3 = st.columns([1,1,1])
            c1.text("")
            c3.text("")
            c2.title("Caption.AI")
            c2.markdown("""---""")
            c2.header("Get Started")
            title = c2.text_input(label="Enter Title")
            tool = c2.selectbox(label="Select Tool", options=("Product Description", "Tagline Generator", "Hashtag Generator", "Post Generator"))
            create_project = c2.button(label="Create Project")
            st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
            if create_project:
                st.session_state.curr_tool_ = tool
                st.session_state.title_ = title
        with st.container():
            st.header("Tools")
            st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns([1,1,1,1])
            c1.subheader("Product Description")
            c1.caption("Create multiline product description.")
            with c1.expander("Example"):
                st.subheader("Input:")
                st.caption("A flying scooter for dogs!")
                st.subheader("Output:")
                st.caption("The sky is the limit for your pup with the Doggy Flyer! This flying scooter for dogs is sure to give your pup a new perspective on life. It's lightweight, durable, and easy to assemble. Plus, it's easy to clean and it has a low profile, so your pup can fly under the radar. The Doggy Flyer is the perfect gift for any pup who's always dreamed of soaring through the sky like a bird. Soar with your pup today!")


            c2.subheader("Tagline Generator")
            c2.caption("Create tagline/motto.")
            with c2.expander("Example"):
                st.subheader("Input:")
                st.caption("The Good Gift Co was founded by an inspirational mother of three, Holly, who was passionate about the gift of giving. Holly had an eye for superior detail and sourcing premium products, which when gifted left a lasting impression. In July 2021, Holly stepped back from the business and The Good Gift Co began a new chapter in Queensland.")
                st.subheader("Output:")
                st.caption("\"The Good Gift Co: For the gift that keeps on giving.\"")

            c3.subheader("Hashtag Generator")
            c3.caption("Generate relevant tags for a given description.")
            with c3.expander("Example"):
                st.subheader("Input:")
                st.caption("Healthcorp is one of Australia's leading First Aid Training and Workplace Safety Specialists.  As one of the country’s leading First Aid and Workplace Safety Specialists, we deliver public courses as well as company training or group bookings and guarantee that all courses are delivered with unrivalled standards of professionalism, flexibility and technical expertise.")
                st.subheader("Output:")
                st.caption("#healthcorp #firstaidtraining #workplacesafety #australia #professional #flexible #technicalexpertise")

            c4.subheader("Post Generator")
            c4.caption("Create social media posts using description.")
            with c4.expander("Example"):
                st.caption("This tool helps you create social media posts. A header, caption, and relevant tags are generated and the provided image/text editor allows you to edit and customize the output.")
            st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

            

def product_description_():
    product_description_placeholder = st.empty()
    st.session_state.page_placeholder_["product_description_page"] = product_description_placeholder
    with product_description_placeholder.container():
        c1,c2,c3 = st.columns([1,2,2])
        st.sidebar.title("Caption.AI")
        if st.sidebar.button(label="Home"):
            st.session_state.curr_tool_ = ""
            st.experimental_rerun()
        st.sidebar.markdown("""---""")
        st.sidebar.header("Project Title: "+st.session_state.title_)
        with st.sidebar.expander("Tools"):

            if st.checkbox(label="Tagline Generator"):
                st.session_state.curr_tool_ = "Tagline Generator"
                st.experimental_rerun()

            if st.checkbox(label="Hashtag Generator"):
                st.session_state.curr_tool_ = "Hashtag Generator"
                st.experimental_rerun()

            if st.checkbox(label="Post Generator"):
                st.session_state.curr_tool_ = "Post Generator"
                st.experimental_rerun()

        c2.title("Product Description Generator")
        c2.caption("Create multiline product description.")
        c2.markdown("""---""")
        c2.subheader("Enter Product Name")
        name = c2.text_input(label="")
        c2.subheader("Enter Description")
        desc = c2.text_area(label="", height = 429)
        generate = c2.button(label="Generate")
        
        editor = c3.empty()
        with editor.container():
            st.title("Editor")
            st.caption("Make changes and customize your output here.")
            st.markdown("""---""")
            st_quill(
                html=False,
                readonly=False,
                value = st.session_state.edit_val_
            )
            #content = st_ace(height=641, theme="ambiance", language="text")
            with st.expander("Example"):
                st.markdown("""---""")
                st.subheader("Input:")
                st.caption("A flying scooter for dogs!")
                st.subheader("Output:")
                st.caption("The sky is the limit for your pup with the Doggy Flyer! This flying scooter for dogs is sure to give your pup a new perspective on life. It's lightweight, durable, and easy to assemble. Plus, it's easy to clean and it has a low profile, so your pup can fly under the radar. The Doggy Flyer is the perfect gift for any pup who's always dreamed of soaring through the sky like a bird. Soar with your pup today!")
        st.markdown("""---""")
        if generate:
            with c2.container():
                with st.spinner('Processing...'):
                    st.session_state.product_desc_list_ += product_description(desc,name,2)
        with st.container():
            c1,c2,c3 = st.columns([1,1,1])
            c1.text("")
            c3.text("")
            edit = False
            for index, i in enumerate(st.session_state.product_desc_list_):
                with c2.container():
                    st.subheader("Description "+ str(index+1))
                    st.caption(i)
                    edit_btn = st.button(label="Edit", key=index)
                    if edit_btn:
                        st.session_state.edit_val_ = i
                        edit = True
                    st.markdown("""---""")
            if edit:
                editor.empty()
                st.experimental_rerun()
            if st.session_state.product_desc_list_:
                more = c2.button(label="More")
                if more:
                    with c2.container():
                        with st.spinner('Processing...'):
                            st.session_state.product_desc_list_ += product_description(desc,name,2)
                        st.experimental_rerun()

        st.markdown("""---""")

def tagline_generator():
    tagline_generator_placeholder = st.empty()
    st.session_state.page_placeholder_["tagline_generator_page"] = tagline_generator_placeholder
    with st.session_state.page_placeholder_["tagline_generator_page"].container():
        c1,c2,c3 = st.columns([1,2,2])
        st.sidebar.title("Caption.AI")
        if st.sidebar.button(label="Home"):
            st.session_state.curr_tool_ = ""
            st.experimental_rerun()
        st.sidebar.markdown("""---""")
        st.sidebar.header("Project Title: "+st.session_state.title_)
        with st.sidebar.expander("Tools"):

            if st.checkbox(label="Product Description"):
                st.session_state.curr_tool_ = "Product Description"
                st.experimental_rerun()

            if st.checkbox(label="Hashtag Generator"):
                st.session_state.curr_tool_ = "Hashtag Generator"
                st.experimental_rerun()

            if st.checkbox(label="Post Generator"):
                st.session_state.curr_tool_ = "Post Generator"
                st.experimental_rerun()

        c2.title("Tagline Generator")
        c2.caption("Create tagline/motto.")
        c2.markdown("""---""")
        c2.subheader("Enter Product/Service Name")
        name = c2.text_input(label="")
        c2.subheader("Enter Description")
        desc = c2.text_area(label="", height = 429)
        generate = c2.button(label="Generate")
        
        editor = c3.empty()
        with editor.container():
            st.title("Editor")
            st.caption("Make changes and customize your output here.")
            st.markdown("""---""")
            st_quill(
                html=False,
                readonly=False,
                value = st.session_state.edit_val_
            )
 
            with st.expander("Example"):
                st.subheader("Input:")
                st.caption("The Good Gift Co was founded by an inspirational mother of three, Holly, who was passionate about the gift of giving. Holly had an eye for superior detail and sourcing premium products, which when gifted left a lasting impression. In July 2021, Holly stepped back from the business and The Good Gift Co began a new chapter in Queensland.")
                st.subheader("Output:")
                st.caption("\"The Good Gift Co: For the gift that keeps on giving.\"")
        st.markdown("""---""")
        if generate:
            with c2.container():
                with st.spinner('Processing...'):
                    st.session_state.tagline_list_ += tagline(desc,name,2)
        with st.container():
            c1,c2,c3 = st.columns([1,1,1])
            c1.text("")
            c3.text("")
            edit = False
            for index, i in enumerate(st.session_state.tagline_list_):
                with c2.container():
                    st.subheader("Tagline "+ str(index+1))
                    st.caption(i)
                    edit_btn = st.button(label="Edit", key=index)
                    if edit_btn:
                        st.session_state.edit_val_ = i
                        edit = True
                    st.markdown("""---""")
            if edit:
                editor.empty()
                st.experimental_rerun()
            if st.session_state.tagline_list_:
                more = c2.button(label="More")
                if more:
                    with c2.container():
                        with st.spinner('Processing...'):
                            st.session_state.tagline_list_ +=  tagline(desc,name,2)
                        st.experimental_rerun()

        st.markdown("""---""")

def hashtag_generator():
    hashtag_generator_placeholder = st.empty()
    st.session_state.page_placeholder_["hashtag_generator_page"] = hashtag_generator_placeholder
    with st.session_state.page_placeholder_["hashtag_generator_page"].container():
        c1,c2,c3 = st.columns([1,2,2])
        st.sidebar.title("Caption.AI")
        if st.sidebar.button(label="Home"):
            st.session_state.curr_tool_ = ""
            st.experimental_rerun()
        st.sidebar.markdown("""---""")
        st.sidebar.header("Project Title: "+st.session_state.title_)
        with st.sidebar.expander("Tools"):

            if st.checkbox(label="Product Description"):
                st.session_state.curr_tool_ = "Product Description"
                st.experimental_rerun()

            if st.checkbox(label="Tagline Generator"):
                st.session_state.curr_tool_ = "Tagline Generator"
                st.experimental_rerun()

            if st.checkbox(label="Post Generator"):
                st.session_state.curr_tool_ = "Post Generator"
                st.experimental_rerun()


        c2.title("Hashtag Generator")
        c2.caption("Suggests tags for a given description.")
        c2.markdown("""---""")
        c2.subheader("Enter Description")
        desc = c2.text_area(label="", height = 524)
        generate = c2.button(label="Generate")
        
        editor = c3.empty()
        with editor.container():
            st.title("Editor")
            st.caption("Make changes and customize your output here.")
            st.markdown("""---""")
            st_quill(
                html=False,
                readonly=False,
                value = st.session_state.edit_val_
            )
 
            with st.expander("Example"):
                st.subheader("Input:")
                st.caption("Healthcorp is one of Australia's leading First Aid Training and Workplace Safety Specialists.  As one of the country’s leading First Aid and Workplace Safety Specialists, we deliver public courses as well as company training or group bookings and guarantee that all courses are delivered with unrivalled standards of professionalism, flexibility and technical expertise.")
                st.subheader("Output:")
                st.caption("#healthcorp #firstaidtraining #workplacesafety #australia #professional #flexible #technicalexpertise")
        st.markdown("""---""")
        if generate:
            with c2.container():
                with st.spinner('Processing...'):
                    st.session_state.hashtag_list_ += hashtag(desc, 2)
        with st.container():
            c1,c2,c3 = st.columns([1,1,1])
            c1.text("")
            c3.text("")
            edit = False
            for index, i in enumerate(st.session_state.hashtag_list_):
                with c2.container():
                    st.subheader("Hashtags "+ str(index+1))
                    st.caption(i)
                    edit_btn = st.button(label="Edit", key=index)
                    if edit_btn:
                        st.session_state.edit_val_ = i
                        edit = True
                    st.markdown("""---""")
            if edit:
                editor.empty()
                st.experimental_rerun()
            if st.session_state.hashtag_list_:
                more = c2.button(label="More")
                if more:
                    with c2.container():
                        with st.spinner('Processing...'):
                            st.session_state.hashtag_list_ += hashtag(desc, 2)
                        st.experimental_rerun()

        st.markdown("""---""")

def post_generator():
    post_generator_placeholder = st.empty()
    st.session_state.page_placeholder_["post_generator_page"] = post_generator_placeholder
    with st.session_state.page_placeholder_["post_generator_page"].container():
        st.sidebar.title("Caption.AI")
        if st.sidebar.button(label="Home"):
            st.session_state.curr_tool_ = ""
            st.experimental_rerun()
        st.sidebar.markdown("""---""")
        st.sidebar.header("Project Title: "+st.session_state.title_)
        with st.sidebar.expander("Tools"):

            if st.checkbox(label="Product Description"):
                st.session_state.curr_tool_ = "Product Description"
                st.experimental_rerun()

            if st.checkbox(label="Hashtag Generator"):
                st.session_state.curr_tool_ = "Hashtag Generator"
                st.experimental_rerun()

            if st.checkbox(label="Tagline Generator"):
                st.session_state.curr_tool_ = "Tagline Generator"
                st.experimental_rerun()

        with st.container():
            st.title("Post Generator")
            st.caption("Creates social media posts")
            st.markdown("""---""")

        c1,c2,c3 = st.columns([1,3,3])
        c1.subheader("Set Goals")
        demography = c1.selectbox(label="Demography", options=("Children", "Youth", "Adults", "General Audience"))
        intent = c1.selectbox(label="Intent", options=("Inform", "Describe", "Convince"))
        tone = c1.selectbox(label="Tone", options=("Neutral", "Confident", "Joyful", "Optimistic", "Friendly", "Urgent"))
        p_name = c1.text_input(label="Product/Company Name")
        c2.subheader("Enter Description")
        desc = c2.text_area(label="", height = 337)
        
        with st.container():
            st.markdown("""---""")
            generate = st.button(label="Generate")
            st.markdown("""---""")

        if generate:
            with c1.container():
                with st.spinner('Processing...'):
                    st.session_state.post_list_.append({"d":header(desc,demography,intent,tone,p_name),"t":tagline(desc,p_name)[0],"h":hashtag(desc)[0]})

        with st.container():
            c1,c2,c3 = st.columns([1,1,1])
            c1.text("")
            c3.text("")

            
            for index, i in enumerate(st.session_state.post_list_):
                with c2.container():
                    st.subheader("Suggestion " + str(index+1))
                    st.caption("Header:")
                    st.info(i["d"])
                    st.caption("Tagline:")
                    st.info(i["t"])
                    st.caption("Tags:")
                    st.info(i["h"])
                    if st.button(label="Select", key=index):
                        st.session_state.post_select_ = i
                    st.markdown("""---""")

            if st.session_state.post_list_:
                more = c2.button(label="More")
                if more:
                    with c2.container():
                        with st.spinner('Processing...'):
                            st.session_state.post_list_.append({"d":header(desc,demography,intent,tone,p_name),"t":tagline(desc,p_name)[0],"h":hashtag(desc)[0]})
                        st.experimental_rerun()

                st.markdown("""---""")

            if st.session_state.post_select_ != "":
                editor = st.empty()
                with editor.container():
                    st.header("Header Editor")
                    st.write("Header")
                    st_quill(
                    html=False,
                    readonly=False,
                    value = st.session_state.post_select_["d"]
                    )
                    st.text("Tags")
                    st.text_area(label="", value = st.session_state.post_select_["h"])
                    st.markdown("""---""")
                    with st.container():
                        st.header("Customize Tagline")
                        c1,c2,c3 = st.columns([1,1,2])
                        c1.subheader("Text Customization")
                        c2.subheader("Text Allignment")
                        c3.subheader("Preview")
                        tagline_ = c1.text_area(label="Tagline", value = st.session_state.post_select_["t"].upper())
                        font = c1.selectbox(label="Select Font", options=font_options)
                        align = c1.multiselect(label="Text Align", options=["top","middle","bottom","left","center","right"], default=["middle","center"])
                        font_size = c1.slider(label="Font Size", min_value=30, max_value=150, value = 100)
                        text_line_size = c1.slider(label="Outline Width", min_value=0, max_value=10, value = 2)
                        text_shadow = c1.slider(label="Text Shadow", min_value=0, max_value=10, value = 0)
                        outline_color = c1.color_picker('Outline Color', '#ffffff')   
                        text_color = c1.color_picker('Text Color', '#000000')
                        #bg = c1.color_picker('Background Color', '#000000')
                        x_pos = c2.slider(label="X Position", min_value=-1000, max_value=1000, value=0)
                        y_pos = c2.slider(label="Y Position", min_value=-1000, max_value=1000, value=0)
                        rot = c2.slider(label="Rotation", min_value=0, max_value=359, value=0)
                        scale = c2.slider(label="Scale", min_value=0, max_value=100, value=50)
                        alpha = c2.slider(label="Alpha", min_value=0, max_value=100, value=100)
                        param = {"txt":tagline_,
                                 "txt-size":font_size,
                                 "txt-fit":"max",
                                 "txt-font":font,
                                 "txt-align":",".join(align),
                                 "txt-line":text_line_size,
                                 "txt-line-color":outline_color,
                                 "txt-shad":text_shadow,
                                 "txt-color":text_color,
                                 "w":1080,
                                 "h":1080,
                                 #"bg":bg,
                                 #"rot":rot
                                 }
                                 
                        text_url = imgix_url(param=param)
                        img_param = {"mark":text_url,
                                    #"blend-mode":"normal",
                                    "mark-x":x_pos,
                                    "mark-y":y_pos,
                                    "mark-alpha":alpha,
                                    "mark-rot":rot,
                                    "mark-scale":scale}

                        curr_im = "d"+str(st.session_state.curr_img_)+".png"
                        c3.image(imgix_url(image=curr_im, param=img_param), width = 900)
                        next_template = c3.button(label="Next Template")
                        if next_template:
                            st.session_state.curr_img_ += 1
                            if st.session_state.curr_img_ == 7:
                                st.session_state.curr_img_ = 2
                            st.experimental_rerun()
                    st.markdown("""---""")
                    with st.container():
                        st.header("Preset Templates")
                        c1,c2,c3 = st.columns([2,4,2.2])
                        c1.text("")
                        c3.text("")
                        text_param_1 = {"txt":tagline_,
                                 "txt-size":120,
                                 "txt-fit":"max",
                                 "txt-font":"Trebuchet MS,Bold",
                                 "txt-align":"middle,center",
                                 "txt-color":"#000000",
                                 "w":1080,
                                 "h":1080,
                                 }
                        text_param_2 = {"txt":tagline_,
                                 "txt-size":110,
                                 "txt-fit":"max",
                                 "txt-font":"Helvetica Neue,BoldItalic",
                                 "txt-align":"middle,left",
                                 "txt-line":5,
                                 "txt-line-color":"#FFFFFF",
                                 "txt-color":"#611DB3",
                                 "w":900,
                                 "h":1080,
                                 }
                        text_param_3 = {"txt":tagline_,
                                 "txt-size":120,
                                 "txt-fit":"max",
                                 "txt-font":"Verdana,Bold",
                                 "txt-align":"middle,center",
                                 "txt-color":"#FFFFFF",
                                 "w":1080,
                                 "h":1080,
                                 }
                        text_param_4 = {"txt":tagline_,
                                 "txt-size":124,
                                 "txt-fit":"max",
                                 "txt-font":"Helvetica Neue,BoldItalic",
                                 "txt-align":"middle",
                                 "txt-line":6,
                                 "txt-line-color":"#FF9800",
                                 "txt-color":"#FFFFFF",
                                 "w":1080,
                                 "h":1080,
                                 }
                        text_param_5 = {"txt":tagline_,
                                 "txt-size":110,
                                 "txt-fit":"max",
                                 "txt-font":"Futura-Medium",
                                 "txt-align":"middle",
                                 "txt-color":"#000000",
                                 "w":1000,
                                 "h":1080,
                                 }
                        text_url_1 = imgix_url(param=text_param_1)
                        text_url_2 = imgix_url(param=text_param_2)
                        text_url_3 = imgix_url(param=text_param_3)
                        text_url_4 = imgix_url(param=text_param_4)
                        text_url_5 = imgix_url(param=text_param_5)

                        img_param_1 = {"blend":text_url_1,
                                    "blend-mode":"normal",
                                    "blend-x":40,
                                    "blend-y":-50}

                        img_param_2 = {"blend":text_url_2,
                                    "blend-mode":"normal",
                                    "blend-x":1000,
                                    "blend-y":-50}

                        img_param_3 = {"blend":text_url_3,
                                    "blend-mode":"normal",
                                    "blend-x":40,
                                    "blend-y":-50}

                        img_param_4 = {"blend":text_url_4,
                                    "blend-mode":"normal"}

                        img_param_5 = {"blend":text_url_5,
                                    "blend-mode":"normal",
                                    "blend-x":40,
                                    "blend-y":-50}

                        img_param_list = [img_param_1, img_param_2, img_param_3, img_param_4, img_param_5]
                        for i in range(5):
                            with c2.container():
                                st.markdown("""---""")
                                st.subheader("Template "+str(i+1))
                                image = "d"+str(i+2)+".png"
                                st.image(imgix_url(image=image, param=img_param_list[i]), width = 1000)
                        c2.markdown("""---""")



if __name__ == "__main__":
    st.set_page_config(layout="wide")
    if not st.session_state.login_:
        login_page()
    if st.session_state.login_:
        st.session_state.page_placeholder_["login_page"].empty()
        if st.session_state.curr_tool_ == "":
            main_page()
        if st.session_state.curr_tool_ == "Product Description":
            st.session_state.page_placeholder_["main_page"].empty()
            product_description_()
        elif st.session_state.curr_tool_ == "Tagline Generator":
            st.session_state.page_placeholder_["main_page"].empty()
            tagline_generator()
        elif st.session_state.curr_tool_ == "Hashtag Generator":
            st.session_state.page_placeholder_["main_page"].empty()
            hashtag_generator()
        elif st.session_state.curr_tool_ == "Post Generator":
            st.session_state.page_placeholder_["main_page"].empty()
            post_generator()
   
