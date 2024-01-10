import base64
import random
from datetime import datetime
import os
import re
import tempfile
import time
from abc import ABC, abstractmethod
import streamlit as st
from langchain.llms.openai import AzureOpenAI

from components.AvatarLoader import AvatarLoader
from components.ListBuilder import ListBuilder
from dashboards.NotificationsDashboard import NotificationsDashboard
from enums.ActivityType import ActivityType
from enums.Badges import UserBadges, TrackBadges
from enums.Settings import Settings, SettingType
from enums.SoundEffect import SoundEffect
from enums.UserType import UserType
from repositories.AssignmentRepository import AssignmentRepository
from repositories.DatabaseManager import DatabaseManager
from repositories.FeatureToggleRepository import FeatureToggleRepository
from repositories.MessageRepository import MessageRepository
from repositories.PortalRepository import PortalRepository
from repositories.RagaRepository import RagaRepository
from repositories.RecordingRepository import RecordingRepository
from repositories.SettingsRepository import SettingsRepository
from repositories.StorageRepository import StorageRepository
from repositories.TenantRepository import TenantRepository
from repositories.TrackRepository import TrackRepository
from repositories.UserAchievementRepository import UserAchievementRepository
from repositories.UserActivityRepository import UserActivityRepository
from repositories.UserAssessmentRepository import UserAssessmentRepository
from repositories.UserPracticeLogRepository import UserPracticeLogRepository
from repositories.UserRepository import UserRepository
from repositories.OrganizationRepository import OrganizationRepository
from repositories.UserSessionRepository import UserSessionRepository
from repositories.ResourceRepository import ResourceRepository


class BasePortal(ABC):
    def __init__(self):
        self.tenant_repo = None
        self.org_repo = None
        self.user_repo = None
        self.user_session_repo = None
        self.user_activity_repo = None
        self.user_achievement_repo = None
        self.portal_repo = None
        self.settings_repo = None
        self.recording_repo = None
        self.track_repo = None
        self.storage_repo = None
        self.feature_repo = None
        self.raga_repo = None
        self.user_practice_log_repo = None
        self.resource_repo = None
        self.assignment_repo = None
        self.message_repo = None
        self.assessment_repo = None
        self.avatar_loader = None
        self.notifications_dashboard = None
        self.set_env()
        self.database_manager = DatabaseManager()
        self.init_repositories()

    def init_repositories(self):
        self.tenant_repo = TenantRepository(self.get_connection())
        self.org_repo = OrganizationRepository(self.get_connection())
        self.user_repo = UserRepository(self.get_connection())
        self.user_session_repo = UserSessionRepository(self.get_connection())
        self.user_activity_repo = UserActivityRepository(self.get_connection())
        self.user_achievement_repo = UserAchievementRepository(self.get_connection())
        self.user_practice_log_repo = UserPracticeLogRepository(self.get_connection())
        self.feature_repo = FeatureToggleRepository(self.get_connection())
        self.settings_repo = SettingsRepository(self.get_connection())
        self.raga_repo = RagaRepository(self.get_connection())
        self.track_repo = TrackRepository(self.get_connection())
        self.recording_repo = RecordingRepository(self.get_connection())
        self.portal_repo = PortalRepository(self.get_connection())
        self.resource_repo = ResourceRepository(self.get_connection())
        self.assignment_repo = AssignmentRepository(self.get_connection())
        self.message_repo = MessageRepository(self.get_connection())
        self.assessment_repo = UserAssessmentRepository(self.get_connection())
        self.storage_repo = StorageRepository('melodymaster')
        self.avatar_loader = AvatarLoader(self.storage_repo, self.user_repo)
        self.notifications_dashboard = NotificationsDashboard(
            self.user_session_repo, self.portal_repo)

    @abstractmethod
    def get_portal(self):
        pass

    def get_connection(self):
        return self.database_manager.connection

    def close_connection(self):
        self.database_manager.close()

    def start(self, register=False):
        self.init_session()
        self.cache_badges()
        self.cache_sound_effects()
        self.avatar_loader.cache_avatars()
        self.set_app_layout()
        if self.user_logged_in():
            user = self.user_repo.get_user(self.get_user_id())
            st.session_state['group_id'] = user['group_id']
            last_activity_time = self.user_session_repo.get_last_activity_time(
                self.get_session_id())
            st.session_state["last_activity_time"] = last_activity_time
            self.show_notifications(last_activity_time)
            self.user_session_repo.update_last_activity_time(self.get_session_id())
        else:
            self.show_introduction()

        if not self.user_logged_in():
            if register:
                self.register_and_login_user()
            else:
                self.login_user()
        else:
            self.build_tabs()
        self.show_copyright()
        self.clean_up()

    def show_notifications(self, last_activity_time):
        messages = self.notifications_dashboard.notify(
            self.get_user_id(), self.get_group_id(), self.get_org_id(), last_activity_time)
        if len(messages) > 0:
            self.play_sound_effect(SoundEffect.NOTIFICATION)
            for message in messages:
                st.toast(message)

    def play_sound_effect(self, effect_type: SoundEffect):
        with open(self.get_sound_effect(effect_type), "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay>
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)

    def show_avatar(self, avatar):
        st.image(self.avatar_loader.get_avatar(avatar), width=75)

    def get_sound_effect(self, sound_effect: SoundEffect):
        # Directory where badges are stored locally
        local_directory = self.get_sound_effects_bucket()

        # Construct the local file path for the badge
        effect = random.choice(sound_effect.effects)
        local_file_path = os.path.join(local_directory, effect)

        # Check if the badge exists locally
        if os.path.exists(local_file_path):
            return local_file_path

        # If badge not found locally, attempt to download from remote
        remote_path = f"{self.get_sound_effects_bucket()}/{sound_effect}"
        success = self.storage_repo.download_blob_and_save(remote_path, local_file_path)

        if success:
            return local_file_path

    def cache_sound_effects(self):
        # Directory where sound effects are stored locally
        local_sound_effects_directory = self.get_sound_effects_bucket()

        # Ensure the directory exists
        if not os.path.exists(local_sound_effects_directory):
            os.makedirs(local_sound_effects_directory)

        effects = SoundEffect.get_all_effects()
        for effect in effects:
            self._download_if_not_exist(
                self.get_sound_effects_bucket(), effect, local_sound_effects_directory)

    def cache_badges(self):
        # Directory where badges are stored locally
        local_badges_directory = self.get_badges_bucket()

        # Ensure the directory exists
        if not os.path.exists(local_badges_directory):
            os.makedirs(local_badges_directory)

        # Create a set to store all badge values (user and track)
        all_badges = set(badge.value for badge in UserBadges) | \
                     set(badge.value for badge in TrackBadges)

        # Now, iterate over all badges and download them if they don't exist
        for badge in all_badges:
            self._download_if_not_exist(
                self.get_badges_bucket(), f"{badge}.png", local_badges_directory)

    def _download_if_not_exist(self, bucket, filename, directory):
        # Construct the local file path
        local_path = os.path.join(directory, filename)

        # Check if the badge exists locally
        if not os.path.exists(local_path):
            # If the badge doesn't exist locally, download it
            remote_path = f"{bucket}/{filename}"
            self.storage_repo.download_blob_and_save(remote_path, local_path)

    def get_badge(self, badge_name):
        # Directory where badges are stored locally
        local_badges_directory = 'badges'

        # Construct the local file path for the badge
        local_file_path = os.path.join(local_badges_directory, f"{badge_name}.png")

        # Check if the badge exists locally
        if os.path.exists(local_file_path):
            return local_file_path

        # If badge not found locally, attempt to download from remote
        remote_path = f"{self.get_badges_bucket()}/{badge_name}.png"
        success = self.storage_repo.download_blob_and_save(remote_path, local_file_path)

        if success:
            return local_file_path
        else:
            print(f"Failed to download badge named '{badge_name}' from remote location.")
            return None

    def set_app_layout(self):
        self.set_background_color()
        st.set_page_config(
            page_title=self.get_title(),
            page_icon=self.get_icon(),
            layout="wide"
        )
        custom_css = """
                <style>
                    #MainMenu {visibility: hidden;}
                    header {visibility: hidden;}
                    footer {visibility: hidden;}   
                </style>
            """
        st.markdown(custom_css, unsafe_allow_html=True)

        css = f"""
        <style>
            [data-testid="stForm"] {{
                background: white;
            }}
        </style>
        """
        st.write(css, unsafe_allow_html=True)
        css = f"""
            <style>
                div[data-testid="stExpander"] > div[role="button"] p {{
                    font-size: 20px;  
                }}
                [data-testid="stExpander"] {{
                    background: #AED6F1;
                }}
            </style>
        """
        st.write(css, unsafe_allow_html=True)

        if self.user_logged_in():
            self.show_app_header(150)
            col1, col2, col3 = st.columns([1.5, 8.5, 1.5])
            with col1:
                if self.is_avatar_assigned():
                    self.show_avatar(st.session_state["avatar"])
            with col2:
                self.show_app_title()
            with col3:
                self.show_user_menu()
        else:
            col1, col2 = st.columns([8.5, 1.5])

            with col1:
                self.show_app_header()

            self.show_app_title()

        st.write("")

    def set_background_color(self):
        if "background_color" not in st.session_state:
            st.session_state["background_color"] = self.settings_repo.get_setting(
                self.get_org_id(), Settings.TAB_BACKGROUND_COLOR)

    def show_app_header(self, width=0):
        st.markdown("""
                <style>
                       .block-container {
                            padding-top: 0rem;
                            padding-bottom: 0rem;
                            padding-left: 5rem;
                            padding-right: 5rem;
                        }
                </style>
                """, unsafe_allow_html=True)

        image = self.storage_repo.download_blob_by_name(f"logo/{self.get_app_name()}.png")
        if width == 0:
            left_column, center_column, right_column = st.columns([5.5, 10, 2.5])
            with center_column:
                st.image(image, use_column_width=True)
        else:
            left_column, center_column, right_column = st.columns([10.2, 10, 2.5])
            with center_column:
                st.image(image, width=width)

    @staticmethod
    def show_app_title():
        st.markdown("""
            <style>
                .cursive-font {
                    font-family: 'Comic Sans MS', cursive, sans-serif;
                    font-size: 30px;
                    color: #853507; 
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* Text shadow effect */
                }
            </style>
            <div class="cursive-font" style="text-align: center">
                Welcome to <span class="bold-text">GuruShishya</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""<div style="text-align: center; font-size: 17px; "> Embark on a musical journey at GuruShishya, 
        where the age-old guru-shishya tradition fosters endless artistic exploration.</div>""", unsafe_allow_html=True)

        st.write("")
        st.write("")

    def show_user_menu(self):
        col2_1, col2_2 = st.columns([1, 3])
        with col2_2:
            user_options = st.selectbox("", ["", "Logout"], index=0,
                                        format_func=lambda
                                            x: f"👤\u2003{self.get_username()}" if x == "" else x)

            if user_options == "Logout":
                self.logout_user()

    def login_user(self):
        st.write("### Login")
        # Build form
        form_key = 'login_form'
        field_names = ['Username', 'Password']
        button_label = 'Login'
        submitted, form_data = self.build_form(form_key, field_names, button_label, False)
        # Process form data
        username = form_data['Username']
        password = form_data['Password']
        if submitted:
            if not username or not password:
                st.error("Both username and password are required.")
                return
            success, user_id, org_id, group_id = self.user_repo.authenticate_user(username, password)
            if success:
                self.set_session_state(user_id, org_id, username, group_id)
                st.session_state['session_id'], previous_session_id, previous_session_open = \
                    self.user_session_repo.open_session(user_id)
                self.user_activity_repo.log_activity(user_id, self.get_session_id(), ActivityType.LOG_IN)
                if previous_session_id:
                    # If the previous session was open and was closed then log activity
                    if previous_session_open:
                        self.user_activity_repo.log_activity(
                            self.get_user_id(), previous_session_id, ActivityType.LOG_OUT)
                    # Check for notifications after the last session
                    last_activity_time = self.user_session_repo.get_last_activity_time(
                        previous_session_id)
                    st.session_state["last_activity_time"] = last_activity_time
                    self.show_notifications(last_activity_time)
                st.rerun()
            else:
                st.error("Invalid username or password.")

    def register_and_login_user(self):
        """
        Handle student login and registration through the sidebar.
        """
        is_authenticated = False
        # Login
        if not self.register_user() and not self.change_password():
            st.header("Login")
            is_authenticated = False
            password, username = self.show_login_screen()
            col1, col2, col3, col4 = st.columns([4, 4, 20, 32])
            if col1.button("Login", type="primary"):
                if username and password:
                    is_authenticated, user_id, org_id, group_id = \
                        self.user_repo.authenticate_user(username, password)
                    if is_authenticated:
                        self.set_session_state(user_id, org_id, username, group_id)
                        st.session_state['session_id'], previous_session_id, previous_session_open = \
                            self.user_session_repo.open_session(user_id)
                        self.user_activity_repo.log_activity(
                            user_id, self.get_session_id(), ActivityType.LOG_IN)
                        if previous_session_id:
                            # If the previous session was open and was closed then log activity
                            if previous_session_open:
                                self.user_activity_repo.log_activity(
                                    self.get_user_id(), previous_session_id, ActivityType.LOG_OUT)
                            # Check for notifications after the last session
                            last_activity_time = self.user_session_repo.get_last_activity_time(
                                previous_session_id)
                            st.session_state["last_activity_time"] = last_activity_time
                            self.show_notifications(last_activity_time)
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.error("Both username and password are required")

            if col2.button("Register", type="primary"):
                st.session_state["show_register_section"] = True
                st.rerun()

            if col3.button("Change Password", type="primary"):
                st.session_state["change_password_section"] = True
                st.rerun()

        # Change password
        if self.change_password():
            st.subheader("Change Password")
            email, username, new_password, confirm_new_password = \
                self.show_change_password_screen()

            col1, col2, col3 = st.columns([3, 4, 40])

            if col1.button("Submit", type="primary", key='cp_submit'):
                if username and email and new_password and confirm_new_password:
                    # Validate user entry
                    validated, message = self.validate(username, new_password, confirm_new_password, email)
                    # Change password
                    if validated:
                        is_password_changed, message = self.user_repo.change_password(username, email, new_password)
                        # Successful?
                        if is_password_changed:
                            st.success(message)
                            st.session_state["change_password_section"] = False
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(message)

            if col2.button("Cancel", type="primary"):
                st.session_state["change_password_section"] = False
                st.rerun()

        # User registration
        if self.register_user():
            st.subheader("Register")
            reg_email, reg_name, reg_username, reg_password, confirm_password, join_code = \
                self.show_user_registration_screen()

            # Initialize a variable for selected avatar id
            if 'selected_avatar_id' not in st.session_state:
                st.session_state['selected_avatar_id'] = None

            # Display avatars for selection
            st.write("Choose an avatar")
            avatar_columns = st.columns(10)
            available_avatars = self.user_repo.get_available_avatars()

            for index, avatar in enumerate(available_avatars):
                with avatar_columns[index % 10]:
                    # Display the avatar using the URL
                    st.image(self.avatar_loader.get_avatar(
                        avatar['name']), width=100, caption="")
                    # Create a button for selecting an avatar
                    if st.button(f"Select", key=f"avatar_{index}"):
                        st.session_state['selected_avatar_id'] = avatar['id']

            col1, col2, col3 = st.columns([3, 4, 40])

            if col1.button("Submit", type="primary"):
                if reg_name and reg_username and reg_email and reg_password and confirm_password and join_code:
                    # Validate user entry
                    validated, message = self.validate(reg_username, reg_password, confirm_password, reg_email)
                    # Register user
                    if validated:
                        org_found, org_id, message = self.org_repo.get_org_id_by_join_code(join_code)
                        if org_found:
                            if st.session_state['selected_avatar_id']:
                                is_registered, message, user_id = self.user_repo.register_user(
                                    reg_name, reg_username, reg_email, reg_password, org_id,
                                    UserType.STUDENT.value, st.session_state['selected_avatar_id']
                                )
                                # Successful?
                                if is_registered:
                                    st.success(message)
                                    st.session_state["show_register_section"] = False
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error(message)
                            else:
                                st.error("Please select an avatar.")
                        else:
                            st.error(message)
                    else:
                        st.error(message)
                else:
                    st.error("All fields are required for registration")

            if col2.button("Cancel", type="primary"):
                st.session_state["show_register_section"] = False
                st.rerun()

        return is_authenticated

    def validate(self, username, password, confirm_password, email):
        if not self.is_valid_username(username):
            return False, "Invalid username. It should be at least 5 characters and only " \
                          "contain alphanumeric characters. "

        if not self.is_valid_password(password):
            return False, "Invalid password. It should be at least 8 characters, contain at least one " \
                          "digit, one lowercase and one uppercase. "

        if not self.is_valid_email(email):
            return False, "Invalid email. Please enter a valid email address."

        if password != confirm_password:
            return False, "Passwords do not match."

        return True, ""

    @staticmethod
    def is_valid_username(username):
        # The username should be at least 5 characters
        if len(username) < 5:
            return False

        # The username can contain alphanumeric characters and special characters
        if not re.match("^[a-zA-Z0-9_!@#$%^&*()+=-]*$", username):
            return False

        return True

    @staticmethod
    def is_valid_password(password):
        if len(password) < 8:
            return False
        if not re.search("[a-z]", password):
            return False
        if not re.search("[A-Z]", password):
            return False
        if not re.search("[0-9]", password):
            return False
        return True

    @staticmethod
    def is_valid_email(email):
        # Regular expression for validating an Email
        email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

        if re.search(email_regex, email):
            return True
        else:
            return False

    @staticmethod
    def show_login_screen():
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        return password, username

    @staticmethod
    def show_user_registration_screen():
        reg_name = st.text_input("Name")
        reg_email = st.text_input("Email")
        reg_username = st.text_input(key="registration_username", label="User")
        reg_password = st.text_input(key="registration_password", type="password", label="Password")
        confirm_password = st.text_input("Confirm Password", type="password")  # Add this line
        join_code = st.text_input("Join Code")

        return reg_email, reg_name, reg_username, reg_password, confirm_password, join_code

    @staticmethod
    def show_change_password_screen():
        email = st.text_input("Email")
        username = st.text_input(key="cp_username", label="User")
        new_password = st.text_input(key="cp_password", type="password", label=" New Password")
        confirm_new_password = st.text_input("Confirm New Password", type="password")

        return email, username, new_password, confirm_new_password

    @staticmethod
    def ok():
        return st.button("Ok", type="primary")

    @staticmethod
    def login():
        return st.button("Login", type="primary")

    @staticmethod
    def register():
        return st.button("Register", type="primary")

    @staticmethod
    def cancel():
        return st.button("Cancel", type="primary")

    def init_session(self):
        if 'user_logged_in' not in st.session_state:
            st.session_state['user_logged_in'] = False
        if 'user_id' not in st.session_state:
            st.session_state['user_id'] = None
        if 'org_id' not in st.session_state:
            st.session_state['org_id'] = None
        if 'join_code' not in st.session_state:
            st.session_state['join_code'] = None
        if 'tenant_id' not in st.session_state:
            st.session_state['tenant_id'] = None
        if 'username' not in st.session_state:
            st.session_state['username'] = None
        if 'show_register_section' not in st.session_state:
            st.session_state['show_register_section'] = None
        if 'change_password_section' not in st.session_state:
            st.session_state['change_password_section'] = None
        if 'last_selected_track' not in st.session_state:
            st.session_state['last_selected_track'] = None

        self.load_all_feature_toggles_into_session()
        self.load_settings()

    def load_all_feature_toggles_into_session(self):
        features = self.feature_repo.get_all_features()
        if 'feature_toggles' not in st.session_state:
            st.session_state['feature_toggles'] = {}

        for feature in features:
            feature_name = feature.get('feature_name', 'Unknown')
            is_enabled = feature.get('is_enabled', False)
            st.session_state['feature_toggles'][feature_name] = is_enabled

    def load_settings(self):
        self.set_tab_heading_font_color()
        self.set_limit()

    def set_limit(self):
        if "limit" not in st.session_state:
            st.session_state["limit"] = self.settings_repo.get_setting(
                self.get_org_id(), Settings.MAX_ROW_COUNT_IN_LIST)

    def set_tab_heading_font_color(self):
        if "tab_heading_font_color" not in st.session_state:
            st.session_state["tab_heading_font_color"] = self.settings_repo.get_setting(
                self.get_org_id(), Settings.TAB_HEADING_FONT_COLOR)

    def set_session_state(self, user_id, org_id, username, group_id):
        st.session_state['user_logged_in'] = True
        st.session_state['user_id'] = user_id
        st.session_state['org_id'] = org_id
        st.session_state['username'] = username
        st.session_state['group_id'] = group_id
        success, organization = self.org_repo.get_organization_by_id(org_id)
        if success:
            st.session_state['join_code'] = organization['join_code']
            st.session_state['tenant_id'] = organization['tenant_id']
        st.session_state["avatar"] = None
        avatar = self.user_repo.get_avatar(user_id)
        if avatar:
            st.session_state["avatar"] = avatar["name"]

    @staticmethod
    def clear_session_state():
        st.session_state['user_logged_in'] = False
        st.session_state['user_id'] = None
        st.session_state['org_id'] = None
        st.session_state['tenant_id'] = None
        st.session_state['username'] = None
        st.session_state['feature_toggles'] = {}

    def logout_user(self):
        session_id = st.session_state.get('session_id')
        if session_id:
            session_duration_seconds = self.user_session_repo.close_session(session_id)
            if session_duration_seconds > 0:
                session_duration_minutes = session_duration_seconds // 60  # Convert seconds to minutes
                additional_params = {
                    "Session Duration": f"{session_duration_minutes}m",
                }
            else:
                additional_params = {}
            self.user_activity_repo.log_activity(
                self.get_user_id(), self.get_session_id(), ActivityType.LOG_OUT, additional_params)
        self.clear_session_state()
        st.rerun()

    @staticmethod
    def show_copyright():
        st.write("")
        st.write("")
        st.write("")
        footer_html = """
            <div style="text-align: center; color: gray;">
                <p style="font-size: 14px;">© 2023 KA Academy of Indian Music and Dance. All rights reserved.</p>
            </div>
            """
        st.markdown(footer_html, unsafe_allow_html=True)

    def build_tabs(self):
        st.markdown(f"""
                <style>
                    .stTabs [data-baseweb="tab-list"] {{
                        gap: 2px;
                    }}

                    .stTabs [data-baseweb="tab"] {{
                        height: 30px;
                        white-space: pre-wrap;               
                        background-color: {self.get_tab_background_color()};
                        border-radius: 6px 6px 0px 0px;
                        gap: 5px;
                        padding-top: 10px;
                        padding-bottom: 10px;
                        padding-left: 10px;
                        padding-right: 10px;
                        font-weight: bold;
                    }}

                    .stTabs [aria-selected="true"] {{
                        background-color: #FFFFFF;
                    }}
                </style>""", unsafe_allow_html=True)
        tab_dict = self.get_tab_dict()
        tab_names = list(tab_dict.keys())
        tab_functions = list(tab_dict.values())
        tabs = st.tabs(tab_names)
        for tab, tab_function in zip(tabs, tab_functions):
            with tab:
                tab_function()

    def activities(self):
        user_id = self.get_user_id()  # Get the current user ID

        # Fetch user activities data for the current user
        user_activities_data = self.user_activity_repo.get_user_activities(user_id)

        if not user_activities_data:
            st.info("No user activity data available.")
            return

        # Build the header for the user activities listing
        column_widths = [33.33, 33.33, 33.33]
        list_builder = ListBuilder(column_widths)
        list_builder.build_header(
            column_names=['Activity Type', 'Time', 'Metadata'])

        # Build rows for the user activities listing
        for activity in user_activities_data:
            activity_type = activity['activity_type']
            timestamp = activity['timestamp'].strftime('%-I:%M %p | %b %d') \
                if isinstance(activity['timestamp'], datetime) else activity['timestamp']
            additional_params_dict = activity.get('additional_params', '{}')
            additional_params_str = None
            if additional_params_dict:
                additional_params_str = ', '.join(
                    f'{key}: {value}' for key, value in additional_params_dict.items() if key != 'session_id'
                )

            if not additional_params_str or not additional_params_dict:
                additional_params_str = 'N/A'

            list_builder.build_row(row_data={
                'Activity Type': activity_type,
                'Timestamp': timestamp,
                'Additional Parameters': additional_params_str
            })

    def sessions(self):
        user_id = self.get_user_id()  # Get the current user ID

        # Fetch time series data for the current user
        time_series_data = self.user_session_repo.get_time_series_data(self.get_user_id())

        if not time_series_data:
            st.info("No session data available.")
            return

        # Prepare data for visualization
        dates = [entry['date'] for entry in time_series_data]
        total_sessions = [entry['total_sessions'] for entry in time_series_data]
        total_duration = [entry['total_duration'] for entry in time_series_data]

        # Create a DataFrame for visualization
        import pandas as pd
        df = pd.DataFrame({
            'Date': pd.to_datetime(dates),
            'Total Sessions': total_sessions,
            'Total Duration': total_duration
        })

        # Draw a line chart
        st.line_chart(df.set_index('Date'))

        # Fetch session details for the current user
        session_details = self.user_session_repo.get_user_sessions(user_id)

        if not session_details:
            st.info("No session details available.")
            return

        # Build the header for the session listing
        column_widths = [25, 25, 25, 25]
        list_builder = ListBuilder(column_widths)
        list_builder.build_header(
            column_names=['Session Start', 'Last Activity', 'Session End', 'Duration (minutes)'])

        # Build rows for the session listing
        for session in session_details:
            open_time = session['open_session_time'].strftime('%-I:%M %p | %b %d') \
                if isinstance(session['open_session_time'], datetime) else session['open_session_time']
            last_activity_time = session['last_activity_time'].strftime('%-I:%M %p | %b %d') \
                if isinstance(session['last_activity_time'], datetime) else session['last_activity_time']
            close_time = session['close_session_time'].strftime('%-I:%M %p | %b %d') \
                if isinstance(session['close_session_time'], datetime) else session['close_session_time']
            # Convert the session duration from seconds to minutes
            duration_minutes = session['session_duration'] / 60
            list_builder.build_row(
                row_data={
                    'Session Start': open_time,
                    'Last Activity': last_activity_time,
                    'Session End': close_time,
                    'Duration (minutes)': f'{duration_minutes:.2f}'
                })

    def settings(self):
        settings = self.settings_repo.get_all_settings_by_portal(self.get_portal())
        if len(settings.keys()) == 0:
            st.info("No settings found")
            return

        new_settings = {}

        for setting_name, setting_value in settings.items():
            cols = st.columns(5)  # Create two columns
            cols[0].write("")
            cols[0].markdown(
                f"<div style='padding-top:15px;color:black;font-size:14px;text-align:left'>{setting_name}</div>",
                unsafe_allow_html=True)

            setting_enum = Settings.get_by_description(setting_name)
            data_type = setting_enum.data_type

            # Display input field in the second column based on data type
            if data_type == SettingType.INTEGER:
                new_value = cols[1].number_input(key=setting_name, label='Value', value=int(setting_value))
            elif data_type == SettingType.FLOAT:
                new_value = cols[1].number_input(key=setting_name, label='Value', value=float(setting_value),
                                                 format="%f")
            elif data_type == SettingType.BOOL:
                new_value = cols[1].checkbox(key=setting_name, label='Value', value=bool(setting_value))
            elif data_type == SettingType.COLOR:
                new_value = cols[1].color_picker(key=setting_name, label='Value', value=str(setting_value))
            else:
                new_value = cols[1].text_input(key=setting_name, label='Value', value=str(setting_value))

            new_settings[setting_name] = new_value

        st.write("")
        st.write("")
        submit = st.button('Update Settings', type="primary")
        if submit:
            self.update_settings(new_settings)
            st.success("Settings updated successfully")

    def update_settings(self, new_settings):
        for setting_name, new_value in new_settings.items():
            setting_enum = Settings.get_by_description(setting_name)
            if setting_enum is not None:
                self.settings_repo.upsert_setting(
                    self.get_org_id(),
                    setting_enum,
                    new_value,
                    self.get_portal()
                )

    @staticmethod
    def build_form(form_key, field_names, button_label='Submit', clear_on_submit=True):
        # Custom CSS to remove form border and adjust padding and margin
        css = r'''
                <style>
                    [data-testid="stForm"] {
                        border: 0px;
                        padding: 0px;
                        margin: 0px;
                    }
                </style>
            '''
        st.markdown(css, unsafe_allow_html=True)

        form_data = {}
        with st.form(key=form_key, clear_on_submit=clear_on_submit):
            for field in field_names:
                if field.lower() == 'password':
                    form_data[field] = st.text_input(label=field.capitalize(), key=field, type='password')
                else:
                    form_data[field] = st.text_input(label=field.capitalize(), key=field)

            submitted = st.form_submit_button(label=button_label, type="primary")

        return submitted, form_data

    @staticmethod
    def build_form_with_select_boxes(form_key, select_box_options, button_label='Submit', clear_on_submit=True):
        # Custom CSS to remove form border and adjust padding and margin
        css = r'''
                <style>
                    [data-testid="stForm"] {
                        border: 0px;
                        padding: 0px;
                        margin: 0px;
                    }
                </style>
            '''
        st.markdown(css, unsafe_allow_html=True)

        form_data = {}
        with st.form(key=form_key, clear_on_submit=clear_on_submit):
            for label, options in select_box_options.items():
                default_option = f"--Select a {label.lower()}--"
                form_data[label] = st.selectbox(label, [default_option] + list(options.keys()))

            button = st.form_submit_button(label=button_label, type="primary")

        return button, form_data

    def download_to_temp_file_by_url(self, blob_url):
        """Download a blob to a temporary file and return the file's path."""
        blob_data = self.storage_repo.download_blob_by_url(blob_url)
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as temp_file:
            temp_file.write(blob_data)
            return temp_file.name

    def download_to_temp_file_by_name(self, blob_name):
        """Download a blob to a temporary file and return the file's path."""
        blob_data = self.storage_repo.download_blob_by_name(blob_name)
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as temp_file:
            temp_file.write(blob_data)
            return temp_file.name

    @staticmethod
    def divider(height=2):
        divider = f"<hr style='height:{height}px; margin-top: 0; border-width:0; background: lightblue;'>"
        st.markdown(f"{divider}", unsafe_allow_html=True)

    def clean_up(self):
        self.tenant_repo = None
        self.org_repo = None
        self.user_repo = None
        self.user_session_repo = None
        self.user_activity_repo = None
        self.user_achievement_repo = None
        self.portal_repo = None
        self.settings_repo = None
        self.recording_repo = None
        self.track_repo = None
        self.storage_repo = None
        self.feature_repo = None
        self.raga_repo = None
        self.user_practice_log_repo = None
        self.resource_repo = None
        self.close_connection()
        self.database_manager = None

    @staticmethod
    def is_feature_enabled(feature):
        return st.session_state['feature_toggles'][feature.name]

    @staticmethod
    def user_logged_in():
        return st.session_state['user_logged_in']

    @staticmethod
    def is_avatar_assigned():
        return st.session_state["avatar"]

    @staticmethod
    def get_org_id():
        return st.session_state['org_id']

    @staticmethod
    def get_user_id():
        return st.session_state['user_id']

    @staticmethod
    def get_group_id():
        return st.session_state['group_id']

    @staticmethod
    def get_session_id():
        return st.session_state['session_id']

    @staticmethod
    def get_username():
        return st.session_state['username']

    @staticmethod
    def get_tenant_id():
        return st.session_state['tenant_id']

    @staticmethod
    def register_user():
        return st.session_state["show_register_section"]

    @staticmethod
    def change_password():
        return st.session_state["change_password_section"]

    def get_org_dir_bucket(self):
        return f'{self.get_tenant_id()}/{self.get_org_id()}'

    def get_tracks_bucket(self):
        return f'{self.get_org_dir_bucket()}/tracks'

    def get_resources_bucket(self):
        return f'{self.get_org_dir_bucket()}/resources'

    def get_recordings_bucket(self):
        return f'{self.get_org_dir_bucket()}/recordings'

    def get_models_bucket(self):
        return f'{self.get_org_dir_bucket()}/models'

    @staticmethod
    def get_badges_bucket():
        return 'badges'

    @staticmethod
    def get_sound_effects_bucket():
        return 'sound effects'

    @staticmethod
    def get_logo_bucket():
        return 'logo'

    @staticmethod
    def get_tab_heading_font_color():
        return st.session_state["tab_heading_font_color"]

    @staticmethod
    def get_tab_background_color():
        return st.session_state["background_color"]

    @staticmethod
    def get_limit():
        return st.session_state["limit"]

    @staticmethod
    def set_env():
        env_vars = ['ROOT_USER', 'ROOT_PASSWORD', 'ADMIN_PASSWORD',
                    'SQL_SERVER', 'SQL_DATABASE', 'SQL_USERNAME', 'SQL_PASSWORD',
                    'MYSQL_CONNECTION_STRING', 'EMAIL_ID', 'EMAIL_PASSWORD',
                    'OPENAI_API_TYPE', 'OPENAI_API_BASE', 'OPENAI_API_KEY',
                    'OPENAI_API_VERSION', 'MODEL_NAME', 'DEPLOYMENT_NAME']
        for var in env_vars:
            os.environ[var] = st.secrets[var]
        os.environ["GOOGLE_APP_CRED"] = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]

    @staticmethod
    def get_app_name():
        return "Guru Shishya"

    @abstractmethod
    def get_title(self):
        pass

    @abstractmethod
    def get_icon(self):
        pass

    @abstractmethod
    def show_introduction(self):
        pass

    @abstractmethod
    def get_tab_dict(self):
        pass

    @staticmethod
    def load_llm(temperature):
        os.environ["OPENAI_API_TYPE"] = st.secrets["OPENAI_API_TYPE"]
        os.environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        os.environ["DEPLOYMENT_NAME"] = st.secrets["DEPLOYMENT_NAME"]
        os.environ["OPENAI_API_VERSION"] = st.secrets["OPENAI_API_VERSION"]
        os.environ["MODEL_NAME"] = st.secrets["MODEL_NAME"]
        return AzureOpenAI(temperature=temperature,
                           deployment_name=os.environ["DEPLOYMENT_NAME"],
                           model_name=os.environ["MODEL_NAME"])
