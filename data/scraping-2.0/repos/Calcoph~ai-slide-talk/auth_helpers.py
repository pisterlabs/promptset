import openai
import streamlit as st
import pandas as pd
import os, string, random, smtplib
import bcrypt
from cryptography.fernet import Fernet
from dictclasses import UserRegister
from history_helpers import load_history
from database import Database
import shutil
import stat

def render_login_register():
    """Show the login and register forms"""
    ##

    ## LOGIN
    ##
    login_tab, register_tab = st.tabs(["**Login**", "**Register**"])
    with login_tab:
        with st.form("login", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password",type="password")
            login = st.form_submit_button("Login")
        if login:
            login_user(username=username,password=password)
        ##
        ## RESTORE PASSWORD
        ##
        with st.expander("Forgot Password"):
            with st.form("restore_pw",clear_on_submit=True):
                email = st.text_input("Enter your E-Mail")
                restore_pw  = st.form_submit_button("Restore Password")

            if restore_pw:
                with st.spinner("Sending new password."):
                    send_new_password(email)
    ##
    ## REGISTER
    ##
    with register_tab:
        with st.form("register", clear_on_submit=False):
            st.subheader("Register")
            email = st.text_input("E-Mail")
            st.warning("Double check your E-Mail-Address, it is the only way to restore your account.")
            username = st.text_input("Username")
            apikey = st.text_input("OPENAI-API KEY",type="password")
            password = st.text_input("Password",type="password")
            register = st.form_submit_button("Register")
        if register:
            salt = bcrypt.gensalt()
            email = email.lower()
            password = bcrypt.hashpw(password=password.encode(),salt=salt)
            open_api_key = encrypt_api_key(apikey)
            userinfo = UserRegister(email, username, password, open_api_key)
            with st.spinner("Registering User"):
                create_new_user(userinfo, check_key=True)

def check_api_key(key: str) -> bool:
    """Checks if the key is a valid openai key"""

    try:
        openai.api_key = key
        openai.Completion.create(
        prompt="Test",
        model = "davinci",
            max_tokens=5)
        return True
    except Exception as e:
        #st.write(e)
        return False

def create_new_user(userinfo: UserRegister, check_key=True):
    """Validates `userinfo` and adds a new user to the database

    Args:
        userinfo (UserRegister)
        check_key (bool, optional): whether to check if the api key is valid. Defaults to True.
    """

    db = Database()
    #check if api key is valid, can be disabled for development purposes, set "check_key" to False
    if not check_api_key(decrypt_api_key(userinfo.open_api_key)) and check_key:
        st.error("Your OPENAI API-KEY is faulty. Try again or use a different Key.")
        st.stop()
    # ensure username to be unique
    try:
        db.query("SELECT * FROM users WHERE username = %s",(userinfo.username,))
    except:
        db = Database()
        db.query("SELECT * FROM users WHERE username = %s",(userinfo.username,))
    if len(db.query("SELECT * FROM users WHERE username = %s",(userinfo.username,))) != 0:
        st.error("Username already taken. Choose another one.")
        st.stop()
    if len(db.query("SELECT * FROM users WHERE email = %s",(userinfo.email,))) != 0:
        st.error("Email already taken. Reset your password in the 'Login' tab.")
        st.stop()
    if not userinfo.username.isalnum():
        st.error("Your username must only contain letters and numbers.")
        st.stop()

    #add user to database
    db.add_user(userinfo)
    st.success("You registered succesfully. Login with your credentials.")

def logout_user():
    """Clean up user user-related variables and {workspace}/tmp directory"""

    os.environ["OPENAI_API_KEY"] = ""
    for k in st.session_state:
        del st.session_state[k]

    if os.path.isdir("tmp"):
        delete_files("tmp")
        delete_empty_folder("tmp")
    st.rerun()

def login_user(username: str, password: str):
    """Checks if username and password are correct.

    If they are, log in.
    """

    db = Database()
    try:
        userinfo = db.query(f"SELECT * FROM users WHERE username = %s",(username,))[0]
    except IndexError:
        st.warning("Username not correct.")
        return
    if bcrypt.checkpw(password.encode(),userinfo[3].encode()):
        st.session_state["authentication_status"] = True
        st.session_state["username"] = username
        st.session_state["userhistory"] = load_history()
        
        os.environ["OPENAI_API_KEY"] = decrypt_api_key(userinfo[4])
        st.rerun()
    else:
        st.error("Password is wrong.")

def decrypt_api_key(encrypted_api_key: bytes):
    encryption_key = st.secrets["encryption_key"]
    fernet = Fernet(encryption_key)
    return fernet.decrypt(encrypted_api_key).decode()

def encrypt_api_key(raw_api_key: str) -> bytes:
    encryption_key = st.secrets["encryption_key"]
    fernet = Fernet(encryption_key)
    return fernet.encrypt(raw_api_key.encode())

class EmailReceiver:
    def __init__(self, email, username) -> None:
        self.email = email
        self.username = username

def send_email(recipient: EmailReceiver, generated_pw: str) -> bool:
    """Sends the password recovery email

    Args:
        recipient (EmailReceiver)

    Returns:
        bool: True if the email was sent, False otherwise
    """
    try:

        subject = "Your new Slidechatter Password"
        text = f"""

        Hey {recipient.username},

        here is your new slidechatter password, make sure to change it after login in:

        Username: {recipient.username}
        New Password: {generated_pw}

        Regards,
        The Slidechatter Team
        """
        message = 'Subject: {}\n{}'.format(subject, text)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(st.secrets["gmail"]["gmail_email"], st.secrets["gmail"]["gmail_pw"])
            smtp_server.sendmail(st.secrets["gmail"]["gmail_email"], recipient.email, message)
        return True
    except:
        return None

# function from streamlit authenticator (https://github.com/mkhorasani/Streamlit-Authenticator)
def generate_random_pw(length: int=16) -> str:
    """Generate random alphanumeric password"""

    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(length)).replace(' ','')

def send_new_password(email: str) -> bool:
    """Send a new password

    Returns:
        bool: True if it was sent, False if otherwise
    """

    db = Database()
    try:
        userinfo = db.query(f"SELECT * FROM users WHERE email = %s",(email,))[0]
    except IndexError:
        st.error("E-Mail not correct.")
        st.stop()

    recipient = EmailReceiver(userinfo[1], userinfo[2])

    new_pw = generate_random_pw()
    if send_email(recipient=recipient, generated_pw=new_pw):
        salt = bcrypt.gensalt()
        new_pw_enctrypted = bcrypt.hashpw(password=new_pw.encode(),salt=salt)
        db.execute_query(
            """UPDATE users SET password = %s WHERE email = %s""",
            (new_pw_enctrypted, email)
        )
        st.success("Succesfully send new password.")
        return True
    else:
        print("Error sending new password.")
        return None

class PasswordChangeData:
    def __init__(self, old_pw: str, new_pw: str, new_pw_repeat: str) -> None:
        self.old_pw = old_pw
        self.new_pw = new_pw
        self.new_pw_repeat = new_pw_repeat

def change_password(change_info: PasswordChangeData):
    """Changes the password

    Args:
        change_info (PasswordChangeData)
    """

    db = Database()
    user_pw = db.query(f"SELECT password FROM users WHERE username = %s",(st.session_state["username"],))[0][0]
    if change_info.new_pw != change_info.new_pw_repeat:
        st.warning("New passwords are not equal.")
        st.stop()
    if bcrypt.checkpw(change_info.old_pw.encode(),user_pw.encode()):
        salt = bcrypt.gensalt()
        new_pw_encrypted = bcrypt.hashpw(password=change_info.new_pw.encode(),salt=salt)
        db.execute_query(
            """UPDATE users SET password = %s WHERE username = %s""",
            (new_pw_encrypted,st.session_state["username"])
        )
        st.success("Password changed succesfully.")
    else:
        st.warning("Old Password not correct.")


class APIKeyChangeData:
    def __init__(self, password: str, new_api_key: str) -> None:
        self.password = password
        self.new_api_key = new_api_key

def change_openai_apikey(change_info: APIKeyChangeData):
    """Changes the openai key

    Args:
        change_info (APIKeyChangeData)
    """

    db = Database()
    user_pw = db.query("SELECT password FROM users WHERE username = %s",(st.session_state["username"],))[0][0]
    if bcrypt.checkpw(change_info.password.encode(),user_pw.encode()):
        if check_api_key(change_info.new_api_key):
            db.execute_query(
                "UPDATE users SET openai_api_key = %s WHERE username = %s",
                (encrypt_api_key(change_info["newapikey"]), st.session_state["username"])
            )
            st.success("OPENAI API KEY changed successfully.")
            os.environ["OPENAI_API_KEY"] = change_info.new_api_key
        else:
            st.warning("New OPENAI API KEY is wrong.")
    else:
        st.warning("Old Password not correct.")

def check_login(render_login_template=False) -> bool:
    """Check if user is logged in. Can show login form. Adds logout button if loged in.

    Args:
        render_login_template (bool, optional): whether to show login screen. Defaults to False.

    Returns:
        bool: True if user is logged in, false otherwise
    """

    if st.session_state["authentication_status"]:
        logout = st.sidebar.button("Logout")
        if logout:
            logout_user()
        return True
    else:
        if render_login_template:
            render_login_register()
        else:
            st.warning("Login on the 'ai_slide_talk' page.")
        return False

def delete_files(path):
    """Recursively deletes all files"""
    if os.path.isdir(path):
        dir = os.listdir(path)
        for item in dir:
            delete_files(f"{path}/{item}")
    else:
        os.remove(path)

def delete_empty_folder(path):
    def removeReadOnly(func, path, excinfo):
    # Using os.chmod with stat.S_IWRITE to allow write permissions
        os.chmod(path, stat.S_IWRITE)
        func(path)
    shutil.rmtree(path,onerror=removeReadOnly)
