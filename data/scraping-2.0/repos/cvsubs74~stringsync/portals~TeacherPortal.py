import base64
import hashlib
import os
import time
from abc import ABC
from datetime import datetime

from langchain.llms.openai import AzureOpenAI

from components.AudioProcessor import AudioProcessor
from components.BadgeAwarder import BadgeAwarder
from components.ListBuilder import ListBuilder
from components.RecordingUploader import RecordingUploader
from components.RecordingsAndTrackScoreTrendsDisplay import RecordingsAndTrackScoreTrendsDisplay
from components.ScorePredictor import ScorePredictor
from components.TrackRecommender import TrackRecommender
from components.TrackScoringTrendsDisplay import TrackScoringTrendsDisplay
from dashboards.AssignmentDashboard import AssignmentDashboard
from dashboards.BadgesDashboard import BadgesDashboard
from dashboards.HallOfFameDashboard import HallOfFameDashboard
from dashboards.MessageDashboard import MessageDashboard
from dashboards.ModelGenerationDashboard import ModelGenerationDashboard
from dashboards.NotesDashboard import NotesDashboard
from dashboards.PracticeDashboard import PracticeDashboard
from dashboards.SkillsDashboard import SkillsDashboard
from dashboards.ResourceDashboard import ResourceDashboard
from dashboards.StudentAssessmentDashboard import StudentAssessmentDashboard
from dashboards.TeamDashboard import TeamDashboard
from dashboards.TrackRecommendationDashboard import TrackRecommendationDashboard
from enums.ActivityType import ActivityType
from enums.Badges import TrackBadges
from enums.Features import Features
from enums.LearningModels import LearningModels
from enums.Settings import Portal
from enums.TimeFrame import TimeFrame
from portals.BasePortal import BasePortal
import streamlit as st
import pandas as pd

from enums.UserType import UserType
from repositories.ModelPerformanceRepository import ModelPerformanceRepository
from repositories.NotesRepository import NotesRepository
from repositories.ScorePredictionModelRepository import ScorePredictionModelRepository


class TeacherPortal(BasePortal, ABC):
    def __init__(self):
        super().__init__()
        self.model_performance_repo = ModelPerformanceRepository(self.get_connection())
        self.audio_processor = AudioProcessor()
        self.track_recommender = TrackRecommender(self.recording_repo, self.user_repo)
        self.badge_awarder = BadgeAwarder(
            self.settings_repo, self.recording_repo,
            self.user_achievement_repo, self.user_practice_log_repo,
            self.portal_repo, self.storage_repo, self.track_recommender)
        self.notes_repo = NotesRepository(self.get_connection())
        self.score_prediction_model_repo = ScorePredictionModelRepository(self.get_connection())
        self.student_assessment_dashboard_builder = StudentAssessmentDashboard(
            self.user_repo, self.recording_repo, self.user_activity_repo, self.user_session_repo,
            self.user_practice_log_repo, self.user_achievement_repo, self.assessment_repo,
            self.portal_repo)
        self.hall_of_fame_dashboard_builder = HallOfFameDashboard(
            self.portal_repo, self.badge_awarder, self.avatar_loader)
        self.resource_dashboard_builder = ResourceDashboard(
            self.resource_repo, self.storage_repo)

    def get_badges_dashboard(self):
        return BadgesDashboard(
            self.settings_repo, self.user_achievement_repo, self.user_session_repo, self.storage_repo)

    def get_skills_dashboard(self):
        return SkillsDashboard(
            self.settings_repo, self.recording_repo, self.user_achievement_repo,
            self.user_practice_log_repo, self.track_repo, self.assignment_repo, self.user_repo)

    def get_practice_dashboard(self):
        return PracticeDashboard(self.user_practice_log_repo)

    def get_team_dashboard(self):
        return TeamDashboard(
            self.portal_repo, self.user_repo,
            self.user_achievement_repo, self.badge_awarder, self.avatar_loader)

    def get_assignment_dashboard(self):
        return AssignmentDashboard(
            self.resource_repo, self.track_repo, self.recording_repo,
            self.assignment_repo, self.storage_repo,
            self.resource_dashboard_builder, self.get_recording_uploader())

    def get_message_dashboard(self):
        return MessageDashboard(
            self.message_repo, self.user_activity_repo, self.avatar_loader)

    def get_student_assessment_dashboard(self):
        return StudentAssessmentDashboard(
            self.user_repo, self.recording_repo, self.user_activity_repo, self.user_session_repo,
            self.user_practice_log_repo, self.user_achievement_repo, self.assessment_repo,
            self.portal_repo)

    def get_hall_of_fame_dashboard(self):
        return HallOfFameDashboard(
            self.portal_repo, self.badge_awarder, self.avatar_loader)

    def get_model_generation_dashboard(self):
        return ModelGenerationDashboard(
            self.track_repo, self.recording_repo, self.portal_repo, self.storage_repo,
            self.score_prediction_model_repo, self.model_performance_repo, self.audio_processor,
            self.get_models_bucket())

    def get_score_predictor(self):
        return ScorePredictor(self.score_prediction_model_repo, self.track_repo,
                              self.model_performance_repo, self.get_models_bucket())

    def get_notes_dashboard(self):
        return NotesDashboard(self.notes_repo)

    def get_recording_uploader(self):
        return RecordingUploader(
            self.recording_repo, self.track_repo, self.raga_repo, self.user_activity_repo,
            self.user_session_repo, self.score_prediction_model_repo, self.model_performance_repo,
            self.storage_repo, self.badge_awarder, AudioProcessor(), self.get_models_bucket())

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

    def get_portal(self):
        return Portal.TEACHER

    def get_title(self):
        return f"{self.get_app_name()} Teacher Portal"

    def get_icon(self):
        return "🎶"

    def get_tab_dict(self):
        tabs = [
            # ("👥 Create a Team", self.create_team),
            # ("👩‍🎓 Students", self.list_students),
            # ("🔀 Team Assignments", self.team_assignments),
            ("📚 Resources", self.resource_management),
            ("🎵 Create Track", self.create_track),
            ("🎵 List Tracks", self.list_tracks),
            ("🧠 Scoring Models", self.generate_prediction_models),
            ("📝 Assignments", self.assignment_management),
            ("🎵 Recordings", self.list_recordings) if self.is_feature_enabled(
                Features.TEACHER_PORTAL_RECORDINGS) else None,
            ("📥 Submissions", self.submissions),
            ("📊 Skills Dashboard", self.progress_dashboard),
            ("📋 Assessments", self.assessments),
            ("👥 Team Dashboard", self.team_dashboard),
            ("🏆 Hall of Fame", self.hall_of_fame),
            ("🔗 Team Connect", self.team_connect),
            ("🗒️ Notes", self.notes_dashboard),
            ("⚙️ Settings", self.settings) if self.is_feature_enabled(
                Features.TEACHER_PORTAL_SETTINGS) else None,
            ("🗂️ Sessions", self.sessions) if self.is_feature_enabled(
                Features.TEACHER_PORTAL_SHOW_USER_SESSIONS) else None,
            ("📊 Activities", self.activities) if self.is_feature_enabled(
                Features.TEACHER_PORTAL_SHOW_USER_ACTIVITY) else None
        ]
        return {tab[0]: tab[1] for tab in tabs if tab}

    def show_introduction(self):
        st.write("""
            ### **Teacher Portal**

            **Empowering Music Educators with Comprehensive Tools**

            Dive into a platform tailored for the needs of progressive music educators. With the StringSync Teacher Portal, manage your classroom with precision and efficiency. Here's what you can do directly from the dashboard:
            - 👥 **Group Management**: Craft student groups for efficient class structures with the "Create Group" feature.
            - 👩‍🎓 **Student Overview**: Browse through your students' profiles and details under the "Students" tab.
            - 🔀 **Student Assignments**: Directly assign students to specific groups using the "Assign Students to Groups" functionality.
            - 🎵 **Track Creation**: Introduce new tracks for practice or teaching via the "Create Track" feature.
            - 🎵 **Recording Review**: Listen, evaluate, and manage student recordings under the "Recordings" tab.
            - 📥 **Submission Insights**: Monitor and manage student submissions through the "Submissions" section.

            Tap into the tabs, explore the features, and elevate your teaching methods. Together, let's redefine music education!
        """)

    def create_team(self):
        st.markdown(
            f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
            f"-size: 28px;'> 🛠️ Create Teams 🛠️️ </h2>", unsafe_allow_html=True)
        self.divider()
        with st.form(key='create_team_form', clear_on_submit=True):
            group_name = st.text_input("Team Name")
            if st.form_submit_button("Create Team", type='primary'):
                if group_name:
                    success, message = self.user_repo.create_user_group(group_name, self.get_org_id())
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.warning("Team name cannot be empty.")

        st.write("")
        col1, col2, col3 = st.columns([2.6, 2, 1])
        with col2:
            load = st.button("Load Teams", type="primary")

        if load:
            self.teams()

    def teams(self):
        # Fetch all groups
        groups = self.user_repo.get_all_groups(self.get_org_id())

        # No groups?
        if not groups:
            st.info("No teams found. Create a new team to get started.")
            return

        # Define the column widths for three columns
        column_widths = [33, 33, 33]
        list_builder = ListBuilder(column_widths)
        list_builder.build_header(column_names=["Team ID", "Team Name", "Member Count"])

        # Display each team and its member count in a row
        for group in groups:
            row_data = {
                "Team ID": group['group_id'],
                "Team Name": group['group_name'],
                "Member Count": group['member_count']
            }
            list_builder.build_row(row_data=row_data)

    def list_students(self):
        st.markdown(
            f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
            f"-size: 28px;'> 📋 Students Listing 📋️ </h2>", unsafe_allow_html=True)
        self.divider()
        col1, col2, col3 = st.columns([2.6, 2, 1])
        with col2:
            if not st.button("Load Students", type="primary"):
                return

        students = self.user_repo.get_users_by_org_id_and_type(self.get_org_id(), UserType.STUDENT.value)

        if not students:
            st.info(f"Please ask new members to join the team using join code: {st.session_state['join_code']}")
            return

        column_widths = [20, 20, 20, 20, 20]
        list_builder = ListBuilder(column_widths)
        list_builder.build_header(column_names=["Name", "Username", "Email", "Team", "Join Code"])

        avatar_image_html = ""
        for student in students:
            avatar_file_path = self.avatar_loader.get_avatar(student['avatar'])
            if os.path.isfile(avatar_file_path):
                with open(avatar_file_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()
                    avatar_image_html = f'<img src="data:image/png;base64,{encoded_string}" alt="avatar" ' \
                                        f'style="width: 60px; ' \
                                        f'height: 60px; border-radius: 50%; margin-right: 10px;"> '
            row_data = {
                "Name": student['name'],
                "Username": student['username'],
                "Email": student['email'],
                "Team": student['group_name'] if 'group_name' in student else 'N/A',
                "Join Code": st.session_state['join_code']
            }
            list_builder.build_row(row_data, f""" <div style='display: flex; align-items: center; border-radius: 10px; padding: 10px; 
                        margin-bottom: 5px;'> {avatar_image_html} </div>""")

    def team_assignments(self):
        st.markdown(
            f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
            f"-size: 28px;'> 🗂️ Team Management 🗂️ </h2>", unsafe_allow_html=True)
        self.divider()

        groups = self.user_repo.get_all_groups(self.get_org_id())
        if not groups:
            st.info("Please create a team to get started.")
            return

        group_options = ["--Select a Team--"] + [group['group_name'] for group in groups]
        group_ids = [None] + [group['group_id'] for group in groups]
        group_name_to_id = {group['group_name']: group['group_id'] for group in groups}

        students = self.user_repo.get_users_by_org_id_and_type(
            self.get_org_id(), UserType.STUDENT.value)

        if not students:
            st.info(f"Please ask new members to join the team using join code: {st.session_state['join_code']}")
            return

        # Column headers
        list_builder = ListBuilder(column_widths=[33.33, 33.33, 33.33])
        list_builder.build_header(column_names=["Name", "Email", "Team"])

        for student in students:
            st.markdown("<div style='border-top:1px solid #AFCAD6; height: 1px;'>", unsafe_allow_html=True)
            with st.container():
                current_group_id = self.user_repo.get_group_by_user_id(student['id'])['group_id']
                if current_group_id is None:
                    current_group_index = 0
                else:
                    current_group_index = group_ids.index(current_group_id)

                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.write("")
                    st.markdown(
                        f"<div style='padding-top:14px;color:black;font-size:14px;'>{student['name']}</div>",
                        unsafe_allow_html=True)
                with col2:
                    st.write("")
                    st.markdown(
                        f"<div style='padding-top:14px;color:black;font-size:14px;'>{student['email']}</div>",
                        unsafe_allow_html=True)
                with col3:
                    selected_group = col3.selectbox(
                        "Select a Team", group_options, index=current_group_index, key=student['id'],
                    )

                    if selected_group != "--Select a Team--":
                        selected_group_id = group_name_to_id[selected_group]
                        if selected_group_id != current_group_id:
                            self.user_repo.assign_user_to_group(student['id'], selected_group_id)
                            st.rerun()

    def resource_management(self):
        st.markdown(
            f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
            f"-size: 28px;'> 📚 Resources Management 📚</h2>", unsafe_allow_html=True)
        self.divider()

        # Part for uploading new resources
        with st.form("resource_upload"):
            resource_title = st.text_input("Title", key='resource_title')
            resource_description = st.text_area("Description", key='resource_description')
            resource_file = st.file_uploader("Upload Resource", type=['pdf', 'mp3', 'mp4'], key='resource_file')
            resource_type = st.selectbox("Type", ["PDF", "Audio", "Video", "Link"], key='resource_type')
            resource_link = st.text_input("Resource Link (if applicable)", key='resource_link')
            submit_button = st.form_submit_button("Upload Resource")

            if submit_button:
                self.handle_resource_upload(
                    title=resource_title,
                    description=resource_description,
                    file=resource_file,
                    rtype=resource_type,
                    link=resource_link
                )

        col1, col2, col3 = st.columns([2.6, 2, 1])
        with col2:
            load = st.button("Load Resources", type="primary")

        if load:
            self.list_resources()

    def assignment_management(self):
        st.markdown(
            f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
            f"-size: 28px;'> 📚 Assignments Management 📚</h2>", unsafe_allow_html=True)
        self.divider()
        assignment_title = st.text_input("Assignment Title", key="assignment_title")
        assignment_description = st.text_input("Assignment Description", key="assignment_desc")
        due_date = st.date_input("Due Date", key="assignment_due_date")

        all_tracks = self.track_repo.get_all_tracks()
        all_resources = self.resource_repo.get_all_resources()

        track_options = {track['name']: track['id'] for track in all_tracks}
        resource_options = {resource['title']: resource['id'] for resource in all_resources}

        selected_track_names = st.multiselect(
            "Select Tracks", list(track_options.keys()), key='assignment_tracks')
        selected_resource_titles = st.multiselect(
            "Select Resources", list(resource_options.keys()), key='assignment_resources')

        track_descriptions = {}
        for track_name in selected_track_names:
            track_descriptions[track_name] = st.text_input(
                f"Instructions for {track_name}")

        resource_descriptions = {}
        for resource_title in selected_resource_titles:
            resource_descriptions[resource_title] = st.text_input(
                f"Instructions for {resource_title}")

        # Fetch all teams
        all_teams = self.user_repo.get_all_groups(self.get_org_id())
        team_options = {team['group_name']: team['group_id'] for team in all_teams}
        selected_team_ids = [team_options[team_name] for team_name in
                             st.multiselect("Select Teams", list(team_options.keys()),
                                            key='selected_teams')]

        # Fetch all individual users
        all_users = self.user_repo.get_users_by_org_id_and_type(
            self.get_org_id(), UserType.STUDENT.value)
        user_options = {user['name']: user['id'] for user in all_users}
        selected_user_ids = [user_options[username] for username in
                             st.multiselect("Select Individual Users", list(user_options.keys()),
                                            key='selected_users')]

        if st.button("Create Assignment", type='primary'):
            if assignment_title:
                assignment_id = self.assignment_repo.add_assignment(
                    assignment_title, assignment_description, due_date
                )

                for track_name in selected_track_names:
                    track_id = track_options[track_name]
                    self.assignment_repo.add_assignment_detail(
                        assignment_id, track_descriptions[track_name], track_id=track_id)

                for resource_title in selected_resource_titles:
                    resource_id = resource_options[resource_title]
                    self.assignment_repo.add_assignment_detail(
                        assignment_id, resource_descriptions[resource_title], resource_id=resource_id)

                # Combine users from selected teams with individually selected users
                users_to_assign = set(selected_user_ids)
                for team_id in selected_team_ids:
                    team_members = self.user_repo.get_users_by_org_id_group_and_type(
                        self.get_org_id(), team_id, UserType.STUDENT.value)
                    users_to_assign.update(member['id'] for member in team_members)
                    for user_id in users_to_assign:
                        additional_params = {
                            "user_id": user_id,
                            "assignment": assignment_title,
                        }
                        self.user_activity_repo.log_activity(self.get_user_id(),
                                                             self.get_session_id(),
                                                             ActivityType.CREATE_ASSIGNMENT,
                                                             additional_params)

                # Deduplicate and assign the assignment to each user
                self.assignment_repo.assign_to_users(assignment_id, list(users_to_assign))

                st.success("Assignment created and assigned successfully!")
            else:
                st.error("Please provide a title for the assignment.")

        self.list_assignments()

    def list_assignments(self):
        st.markdown(
            f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
            f"-size: 28px;'> 📚 Assignments 📚</h2>", unsafe_allow_html=True)
        self.divider()

        groups = self.user_repo.get_all_groups(self.get_org_id())
        group_options = {group['group_name']: group['group_id'] for group in groups}
        selected_group_name = st.selectbox(key=f"assignments-group", label="Select a team:",
                                           options=['--Select a team--'] + list(group_options.keys()))

        # Filter users by the selected group
        selected_group_id = None
        if selected_group_name == '--Select a team--':
            st.info("Please select a team to view assignments..")
            return

        selected_group_id = group_options[selected_group_name]
        self.get_assignment_dashboard().build_by_group(selected_group_id)

    def handle_resource_upload(self, title, description, file, rtype, link):
        if rtype != "Link" and not file:
            st.error("Please upload a file.")
            return

        if rtype == "Link" and not link:
            st.error("Please provide a resource link.")
            return

        if title:
            if rtype == "Link":
                # Save the link to the database
                self.resource_repo.add_resource(self.get_user_id(), title, description, rtype, None, link)
                st.success("Resource link added successfully!")
            else:
                # Save the file to storage and get the URL
                file_url = self.upload_resource_to_storage(file, file.getvalue())
                # Save the file information to the database
                self.resource_repo.add_resource(self.get_user_id(), title, description, rtype, file_url, None)
                st.success("File uploaded successfully!")
        else:
            st.error("Title is required.")

    def list_resources(self):
        # Fetch resources from the DB
        resources = self.resource_repo.get_all_resources()
        if resources:
            for resource in resources:
                with st.expander(f"{resource['title']}"):
                    st.write(resource['description'])
                    if resource['type'] == "Link":
                        st.markdown(f"[Resource Link]({resource['link']})")
                    else:
                        data = self.storage_repo.download_blob_by_url(resource['file_url'])
                        col1, col2, col3 = st.columns([1, 3, 10])
                        with col1:
                            st.download_button(
                                label="Download",
                                data=data,
                                file_name=resource['title'],
                                mime='application/octet-stream'
                            )
                        with col2:
                            if st.button(f"Delete", key=f"delete_{resource['id']}"):
                                self.delete_resource(resource['id'])
        else:
            st.info("No resources found. Upload a resource to get started.")

    def delete_resource(self, resource_id):
        # Delete resource from the database and storage
        resource = self.resource_repo.get_resource_by_id(resource_id)
        if resource:
            if resource['file_url']:
                # Delete the file from storage
                self.storage_repo.delete_blob(resource['file_url'])
            # Delete the resource from the database
            self.resource_repo.delete_resource(resource_id)
            st.success("Resource deleted successfully!")
            # Refresh the page to show the updated list
            st.rerun()
        else:
            st.error("Resource not found.")

    def create_track(self):
        st.markdown(f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; "
                    "font-size: 28px;'> 🔊 Create Audio Tracks 🔊 </h2>", unsafe_allow_html=True)
        self.divider()
        with st.form(key='create_track_form', clear_on_submit=True):
            track_name = st.text_input("Track Name",
                                       help="Enter the name of the track. This will be used to identify the track in "
                                            "the system.")

            track_file = st.file_uploader("Choose an audio file", type=["m4a", "mp3"],
                                          help="Upload the audio file for the track. Acceptable formats: m4a, mp3.")

            ref_track_file = st.file_uploader("Choose a Reference Audio File", type=["m4a", "mp3"],
                                              help="Upload a reference audio file for the track. This audio should "
                                                   "demonstrate how a student would ideally play along with the main "
                                                   "track. It serves as a practical example for comparison and "
                                                   "learning.")

            description = st.text_input("Description",
                                        help="Provide a brief description of the track. This could include its "
                                             "significance, style, or any other relevant information.")

            recommendation_threshold_score = st.number_input(
                "Recommendation Threshold Score", min_value=5.0, max_value=10.0, step=0.1,
                help="Set the score threshold for track recommendations. Tracks with average scores below this "
                     "threshold will be recommended for further practice. Choose a value between 5.0 and 10.0 to "
                     "tailor recommendations to your skill level.")

            ragas = self.raga_repo.get_all_ragas()
            ragam_options = {raga['name']: raga['id'] for raga in ragas}
            selected_ragam = st.selectbox("Select Ragam", ['--Select a Ragam--'] + list(ragam_options.keys()),
                                          help="Choose a Ragam (melodic framework) for the track. This defines the "
                                               "melodic structure and feel of the track.")

            # Existing tags
            tags = self.track_repo.get_all_tags()
            selected_tags = st.multiselect("Select tags:", tags,
                                           help="Choose from existing tags to categorize the track. Tags help in "
                                                "organizing and searching for tracks.")

            new_tags = st.text_input("Add new tags (comma-separated):",
                                     help="Add new tags for the track. Separate multiple tags with commas. Tags are "
                                          "keywords or labels that help in categorizing and finding tracks.")
            if new_tags:
                new_tags = [tag.strip() for tag in new_tags.split(",")]
                selected_tags.extend(new_tags)

            level = st.selectbox("Select Level", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                 help="Select the difficulty level of the track. Level 1 is the easiest, and the "
                                      "levels increase in difficulty.")

            if st.form_submit_button("Submit", type="primary"):
                if self.validate_inputs(track_name, track_file, ref_track_file):
                    ragam_id = ragam_options[selected_ragam]
                    track_data = track_file.getbuffer()
                    track_hash = self.calculate_file_hash(track_data)
                    if self.track_repo.is_duplicate(track_hash):
                        st.error("You have already uploaded this track.")
                        return
                    track_url = self.upload_track_to_storage(track_file, track_data)
                    ref_track_data = ref_track_file.getbuffer()
                    ref_track_url = self.upload_track_to_storage(ref_track_file, ref_track_data)
                    self.storage_repo.download_blob(track_url, track_file.name)
                    self.storage_repo.download_blob(ref_track_url, ref_track_file.name)
                    offset = self.audio_processor.compare_audio(track_file.name, ref_track_file.name)
                    os.remove(track_file.name)
                    os.remove(ref_track_file.name)
                    self.track_repo.add_track(
                        name=track_name,
                        track_path=track_url,
                        track_ref_path=ref_track_url,
                        level=level,
                        recommendation_threshold_score=recommendation_threshold_score,
                        ragam_id=ragam_id,
                        tags=selected_tags,
                        description=description,
                        offset=offset,
                        track_hash=track_hash
                    )
                    additional_params = {
                        "track_name": track_name,
                    }
                    self.user_activity_repo.log_activity(self.get_user_id(),
                                                         self.get_session_id(),
                                                         ActivityType.CREATE_TRACK,
                                                         additional_params)
                    st.success("Track added successfully!")

    def list_tracks(self):
        st.markdown(f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; "
                    "font-size: 28px;'> 🎶 Track Listing 🎶</h2>", unsafe_allow_html=True)
        self.divider()

        ragas = self.raga_repo.get_all_ragas()
        filter_options = self.fetch_filter_options(ragas)

        # Create three columns
        col1, col2, col3 = st.columns(3)

        level = col1.selectbox("Filter by Level", ["All"] + filter_options["Level"])
        raga = col2.selectbox("Filter by Ragam", ["All"] + filter_options["Ragam"])
        tags = col3.multiselect("Filter by Tags", ["All"] + filter_options["Tags"])

        tracks = self.track_repo.search_tracks(
            raga=None if raga == "All" else raga,
            level=None if level == "All" else level,
            tags=None if tags == ["All"] else tags
        )
        if not tracks:
            st.info("No tracks found. Create a track to get started.")
            return

        selected_tracks = self.get_selected_tracks(tracks)

        if not selected_tracks:
            return

        list_builder = ListBuilder(column_widths=[16.66, 16.66, 16.66, 16.66, 16.66, 16.66])
        list_builder.build_header(
            column_names=["Track Name", "Audio", "Ref Audio", "Ragam", "Level", "Description"])

        for track in selected_tracks:
            col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
            row_data = {
                "Track Name": track['track_name'],
                "Ragam": track['ragam'],
                "Level": track['level'],
                "Description": track['description']
            }
            col1.write("")
            col1.markdown(
                f"<div style='padding-top:12px;color:black;font-size:14px;text-align:left'>{row_data['Track Name']}</div>",
                unsafe_allow_html=True)

            col2.write("")
            blob_url = track['track_path']
            audio_file_path = self.storage_repo.download_blob_by_url(blob_url)
            col2.audio(audio_file_path, format='dashboards/m4a')

            col3.write("")
            blob_url = track['track_ref_path']
            audio_ref_file_path = self.storage_repo.download_blob_by_url(blob_url)
            col3.audio(audio_ref_file_path, format='dashboards/m4a')

            col4.write("")
            col4.markdown(
                f"<div style='padding-top:12px;color:black;font-size:14px;text-align:left'>{row_data['Ragam']}</div>",
                unsafe_allow_html=True)

            col5.write("")
            col5.markdown(
                f"<div style='padding-top:12px;color:black;font-size:14px;text-align:left'>{row_data['Level']}</div>",
                unsafe_allow_html=True)

            col6.write("")
            col6.markdown(
                f"<div style='padding-top:12px;color:black;font-size:14px;text-align:left'>{row_data['Description']}</div>",
                unsafe_allow_html=True)

    def generate_prediction_models(self):
        st.markdown(f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; "
                    "font-size: 28px;'> 🎵 Scoring Models 🎵</h2>", unsafe_allow_html=True)
        self.divider()
        self.get_model_generation_dashboard().build()

        submission_ids = self.model_performance_repo.get_recent_influential_ids(
            LearningModels.RandomForestRegressorScorePredictionModel.name)
        self.display_submissions_by_track(submission_ids)

    def display_submissions_by_track(self, submission_ids):
        if submission_ids:
            submissions = self.recording_repo.get_recordings_by_ids(submission_ids)

            # Group submissions by track name
            submissions_by_track = {}
            for submission in submissions:
                track_name = submission['track_name']
                if track_name not in submissions_by_track:
                    submissions_by_track[track_name] = []
                submissions_by_track[track_name].append(submission)

            # Display selectbox for track names
            track_names = list(submissions_by_track.keys())
            selected_track = st.selectbox("Select a Track", track_names)

            # Display submissions for the selected track
            if selected_track:
                for submission in submissions_by_track[selected_track]:
                    self.show_submission(submission)

    def fetch_filter_options(self, ragas):
        return {
            "Level": self.track_repo.get_all_levels(),
            "Ragam": [raga['name'] for raga in ragas],
            "Tags": self.track_repo.get_all_tags()
        }

    @staticmethod
    def get_selected_tracks(tracks):
        # Create a mapping from track names to track objects
        track_options = {track['track_name']: track for track in tracks}

        # Use a multiselect widget to let the user select tracks by name
        selected_track_names = st.multiselect("Select Tracks",
                                              list(track_options.keys()),
                                              key="track_selection",
                                              placeholder='Select Tracks')

        # Filter the tracks list to only include the tracks with names that were selected
        selected_tracks = [track_options[name] for name in selected_track_names if name in track_options]

        return selected_tracks

    def validate_inputs(self, track_name, track_file, ref_track_file):
        if not track_name:
            st.warning("Please provide a name for the track.")
            return False
        if not track_file:
            st.error("Please upload an audio file.")
            return False
        if not ref_track_file:
            st.error("Please upload a reference audio file.")
            return False
        if self.track_repo.get_track_by_name(track_name):
            st.error(f"A track with the name '{track_name}' already exists.")
            return False
        return True

    def upload_track_to_storage(self, file, data):
        blob_path = f'{self.get_tracks_bucket()}/{file.name}'
        return self.storage_repo.upload_blob(data, blob_path)

    def upload_resource_to_storage(self, file, data):
        blob_path = f'{self.get_resources_bucket()}/{file.name}'
        return self.storage_repo.upload_blob(data, blob_path)

    def remove_track(self):
        st.markdown(f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; "
                    "font-size: 28px;'> 🗑️ Remove Tracks 🗑️ </h2>", unsafe_allow_html=True)
        self.divider()
        # Fetch all tracks
        all_tracks = self.track_repo.get_all_tracks()

        if not all_tracks:
            st.info("No tracks found.")
            return

        track_options = {track['name']: track['id'] for track in all_tracks}

        # Dropdown to select a track to remove
        selected_track_name = st.selectbox("Select a track to remove:",
                                           ["--Select a track--"] + list(track_options.keys()))

        # Button to initiate the removal process
        if st.button("Remove", type="primary"):
            if selected_track_name and selected_track_name != "--Select a track--":
                selected_track_id = track_options[selected_track_name]

                # Check if recordings for the track exist
                if self.recording_repo.recordings_exist_for_track(selected_track_id):
                    st.error(
                        f"Cannot remove track '{selected_track_name}' as there are recordings associated with it.")
                    return

                # Get the track details
                track_details = self.track_repo.get_track_by_id(selected_track_id)
                files_to_remove = [track_details['track_path'], track_details.get('track_ref_path'),
                                   track_details.get('notation_path')]

                # Remove the track and associated files from storage
                for file_path in files_to_remove:
                    if file_path and not self.storage_repo.delete_file(file_path):
                        st.warning(f"Failed to remove file '{file_path}' from storage.")
                        return

                # Remove the track from database
                if self.track_repo.remove_track_by_id(selected_track_id):
                    st.success(f"Track '{selected_track_name}' removed successfully!")
                    st.rerun()
                else:
                    st.error("Error removing track from database.")

    @staticmethod
    def save_audio(audio, path):
        with open(path, "wb") as f:
            f.write(audio)

    def list_students_and_tracks(self, source):
        # Show groups in a dropdown
        groups = self.user_repo.get_all_groups(self.get_org_id())
        group_options = {group['group_name']: group['group_id'] for group in groups}
        selected_group_name = st.selectbox(key=f"{source}-group", label="Select a team:",
                                           options=['--Select a team--'] + list(group_options.keys()))

        # Filter users by the selected group
        selected_group_id = None
        if selected_group_name != '--Select a team--':
            selected_group_id = group_options[selected_group_name]
            users = self.user_repo.get_users_by_org_id_group_and_type(
                self.get_org_id(), selected_group_id, UserType.STUDENT.value)
        else:
            users = self.user_repo.get_users_by_org_id_and_type(
                self.get_org_id(), UserType.STUDENT.value)

        user_options = {user['name']: user['id'] for user in users}
        options = ['--Select a student--'] + list(user_options.keys())
        selected_username = st.selectbox(key=f"{source}-user", label="Select a student to view their recordings:",
                                         options=options)
        selected_user_id = None
        if selected_username != '--Select a student--':
            selected_user_id = user_options[selected_username]

        selected_track_id = None
        track_path = None
        if selected_user_id is not None:
            track_ids = self.recording_repo.get_unique_tracks_by_user(selected_user_id)
            if track_ids:
                # Fetch track names by their IDs
                tracks = self.track_repo.get_tracks_by_ids(track_ids)
                # Create a mapping for the dropdown
                track_options = {tracks[id]['name']: id for id in track_ids if id in tracks}
                selected_track_name = st.selectbox(key=f"{source}-track", label="Select a track:",
                                                   options=['--Select a track--'] + list(track_options.keys()))
                if selected_track_name != '--Select a track--':
                    selected_track_id = track_options[selected_track_name]
                    track = tracks[selected_track_id]
                    track_path = track['track_path']

        return selected_group_id, selected_username, selected_user_id, selected_track_id, track_path

    def display_track_files(self, url):
        if url is None:
            return

        st.write("")
        st.write("")
        audio_data = self.storage_repo.download_blob_by_url(url)
        st.audio(audio_data, format='audio/mp4')

    def list_recordings(self):
        st.markdown(f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; "
                    "font-size: 28px;'> 🎙️ Recordings 🎙️️ </h2>", unsafe_allow_html=True)
        self.divider()

        group_id, username, user_id, track_id, track_name = self.list_students_and_tracks("R")
        if user_id is None or track_id is None:
            return

        self.display_track_files(track_name)
        recordings = self.recording_repo.get_recordings_by_user_id_and_track_id(user_id, track_id)
        if not recordings:
            st.info("No recordings found.")
            return

        for recording in recordings:
            self.check_and_update_distance_and_score(recording)
            with st.expander(
                    f"Recording ID {recording['id']} - {recording['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                if recording['blob_url']:
                    filename = self.storage_repo.download_blob_by_url(recording['blob_url'])
                    st.audio(filename, format='dashboards/m4a')
                else:
                    st.write("No dashboards data available.")

                # Create a form for score, remarks, and timestamp
                with st.form(f"recording_form_{recording['id']}"):
                    score = st.text_input("Score", value=recording['score'],
                                          key=f"recording_score_{recording['id']}")
                    remarks = st.text_area("Remarks", value=recording.get('remarks', ''),
                                           key=f"recording_remarks_{recording['id']}")

                    # Checkbox for using the recording for model training
                    use_for_training = st.checkbox("Use this recording for model training?",
                                                   key=f"recording_training_{recording['id']}",
                                                   value=recording["is_training_data"])

                    # Display the timestamp, but it's not editable
                    timestamp = recording['timestamp'].strftime('%-I:%M %p | %b %d')
                    st.write(timestamp)

                    # Submit button for the form
                    if st.form_submit_button("Update", type="primary"):
                        self.handle_remarks_and_badges(
                            score, recording, remarks, 'N/A', use_for_training)
                        if use_for_training:
                            self.track_repo.flag_model_rebuild(track_id)
                        st.success("Remarks/Score updated successfully.")

        RecordingsAndTrackScoreTrendsDisplay(self.recording_repo).show(user_id, track_id)

    def submissions(self):
        st.markdown(
            f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
            f"-size: 28px;'> ✅ Review Your Students' Submissions & Provide Feedback ✅ </h2>",
            unsafe_allow_html=True)
        self.divider()
        # Show submissions summary
        self.show_submissions_summary()
        self.divider()
        # Filter criteria
        group_id, username, user_id, track_id, track_name = self.list_students_and_tracks("S")
        if group_id or user_id:
            # Fetch and sort recordings
            submissions = self.portal_repo.get_recordings(group_id, user_id, track_id)
            if not submissions:
                st.info("No submissions found.")
            else:
                df = pd.DataFrame(submissions)

                # Display each recording in an expander
                for index, recording in df.iterrows():
                    self.show_submission(recording)

        self.divider()
        st.markdown(
            f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
            f"-size: 20px;'> ✅ Recently Reviewed Submissions ✅ </h2>",
            unsafe_allow_html=True)
        # Show recently reviewed submissions
        self.show_recently_reviewed_submissions()

    def show_submission(self, submission):
        expander_label = f"**{submission.get('user_name', 'N/A')} - " \
                         f"{submission.get('track_name', 'N/A')} - " \
                         f"{submission.get('timestamp', 'N/A')}**"
        self.check_and_update_distance_and_score(submission)
        with st.expander(expander_label):
            form_key = f"submission_form_{submission['id']}"
            with st.form(key=form_key):
                if submission['blob_url']:
                    filename = self.storage_repo.download_blob_by_name(submission['blob_name'])
                    st.markdown("<span style='font-size: 15px;'>Submission:</span>", unsafe_allow_html=True)
                    st.audio(filename, format='dashboards/m4a')
                else:
                    st.write("No dashboards data available.")

                score = st.text_input("Score", key=f"submission_score_{submission['id']}",
                                      value=submission['score'])
                remarks = st.text_area("Remarks",
                                       key=f"submission_remarks_{submission['id']}",
                                       value=submission["remarks"])

                badge_options = [badge.value for badge in TrackBadges]

                # Find the index of the badge name in the options list
                default_index = next((index for index, badge in enumerate(TrackBadges)
                                      if badge.description == submission["badge"]), -1)
                if default_index >= 0:
                    # Add 2 to the index to account for the '--Select a badge--' and 'N/A' options
                    default_index += 2
                else:
                    # If no badge is found, default to '--Select a badge--'
                    default_index = 0

                selected_badge = st.selectbox("Select a badge", ['--Select a badge--', 'N/A'] + badge_options,
                                              key=f"badge_{submission['id']}",
                                              index=default_index)

                # Checkbox for using the recording for model training
                use_for_training = st.checkbox("Use this recording for model training?",
                                               key=f"submission_training_{submission['id']}",
                                               value=submission["is_training_data"])
                # Submit button for the form
                if st.form_submit_button("Submit", type="primary"):
                    # Check for required fields
                    if not remarks:
                        st.error("Please provide remarks.")
                        return
                    if selected_badge == '--Select a badge--':
                        st.error("Please select a badge (or N/A).")
                        return

                    # Update logic
                    self.handle_remarks_and_badges(
                        score, submission, remarks, selected_badge, use_for_training)
                    if use_for_training:
                        self.track_repo.flag_model_rebuild(submission['track_id'])
                    st.success("Remarks/Score/Badge updated successfully.")

            RecordingsAndTrackScoreTrendsDisplay(self.recording_repo).show(
                submission['user_id'], submission['track_id'])

    def show_reviewed_submission(self, submission):
        expander_label = f"**{submission.get('user_name', 'N/A')} - " \
                         f"{submission.get('track_name', 'N/A')} - " \
                         f"{submission.get('timestamp', 'N/A')}**"
        self.check_and_update_distance_and_score(submission)
        with st.expander(expander_label):
            form_key = f"reviewed_submission_form_{submission['id']}"
            with st.form(key=form_key):
                score = st.text_input("Score", key=f"reviewed_submission_score_{submission['id']}",
                                      value=submission['score'])
                remarks = st.text_area("Remarks",
                                       key=f"reviewed_submission_remarks_{submission['id']}",
                                       value=submission["remarks"])

                badge_options = [badge.value for badge in TrackBadges]

                # Find the index of the badge name in the options list
                default_index = next((index for index, badge in enumerate(TrackBadges)
                                      if badge.description == submission["badge"]), -1)
                if default_index >= 0:
                    # Add 2 to the index to account for the '--Select a badge--' and 'N/A' options
                    default_index += 2
                else:
                    # If no badge is found, default to '--Select a badge--'
                    default_index = 0

                selected_badge = st.selectbox("Select a badge", ['--Select a badge--', 'N/A'] + badge_options,
                                              key=f"reviewed_badge_{submission['id']}",
                                              index=default_index)

                # Checkbox for using the recording for model training
                use_for_training = st.checkbox("Use this recording for model training?",
                                               key=f"reviewed_submission_training_{submission['id']}",
                                               value=submission["is_training_data"])
                # Submit button for the form
                if st.form_submit_button("Submit", type="primary"):
                    # Check for required fields
                    if not remarks:
                        st.error("Please provide remarks.")
                        return
                    if selected_badge == '--Select a badge--':
                        st.error("Please select a badge (or N/A).")
                        return

                    # Update logic
                    self.handle_remarks_and_badges(
                        score, submission, remarks, selected_badge, use_for_training)
                    if use_for_training:
                        self.track_repo.flag_model_rebuild(submission['track_id'])
                    st.success("Remarks/Score/Badge updated successfully.")

            RecordingsAndTrackScoreTrendsDisplay(self.recording_repo).show(
                submission['user_id'], submission['track_id'])

    def check_and_update_distance_and_score(self, submission):
        if submission['distance'] and submission['score']:
            return

        id = submission['id']
        track_name = submission['track_name']
        level = submission['level']
        offset = submission['offset']
        duration = submission['duration']
        track_path = submission['track_path']
        recording_path = submission['blob_url']
        recording_name = f"recording_{id}"
        self.storage_repo.download_blob(track_path, track_name)
        self.storage_repo.download_blob(recording_path, recording_name)
        distance, score = self.get_recording_uploader().analyze_recording_by_track(
            track_name, level, offset, duration, track_name, recording_name)
        self.recording_repo.update_score_distance_analysis(id, distance, score)
        os.remove(track_name)
        os.remove(recording_name)
        submission['distance'] = distance
        submission['score'] = score

    def handle_remarks_and_badges(self, score, submission, remarks, badge, use_for_training):
        self.recording_repo.update_score_remarks_training(
            submission["id"], score, remarks, use_for_training)
        additional_params = {
            "user_id": submission["user_id"],
            "submission_id": submission["id"],
        }
        self.user_activity_repo.log_activity(self.get_user_id(),
                                             self.get_session_id(),
                                             ActivityType.REVIEW_SUBMISSION,
                                             additional_params)
        if badge != 'N/A':
            self.user_achievement_repo.award_track_badge(submission['user_id'],
                                                         submission['id'],
                                                         TrackBadges.from_value(badge),
                                                         submission['timestamp'])

    def show_submissions_summary(self):
        submissions = self.portal_repo.get_unremarked_submissions()
        list_builder = ListBuilder(column_widths=[33.33, 33.33, 33.33])
        list_builder.build_header(
            column_names=["Name", "Group", "Tracks"])
        # Display recent submission summary
        for submission in submissions:
            list_builder.build_row(submission)

    def show_recently_reviewed_submissions(self):
        submissions = self.recording_repo.get_recently_reviewed_submissions()
        for submission in submissions:
            self.show_reviewed_submission(submission)

    def progress_dashboard(self):
        st.markdown(
            f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
            f"-size: 28px;'> 📊 Track Your Students' Progress & Development 📊 </h2>", unsafe_allow_html=True)
        self.divider()
        st.markdown(
            f"<h2 style='text-align: left; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
            f"-size: 24px;'> Skills Progress Tracker </h2>", unsafe_allow_html=True)
        users = self.user_repo.get_users_by_org_id_and_type(
            self.get_org_id(), UserType.STUDENT.value)

        user_options = {user['name']: user['id'] for user in users}
        options = ['--Select a student--'] + list(user_options.keys())
        selected_username = st.selectbox(key=f"user_select_progress_dashboard",
                                         label="Select a student to view their skills progress report:",
                                         options=options)

        selected_user_id = None
        if selected_username != '--Select a student--':
            selected_user_id = user_options[selected_username]
        else:
            return

        user_group = self.user_repo.get_group_by_user_id(selected_user_id)
        group_id = user_group['group_id']
        avatar = self.user_repo.get_avatar(selected_user_id)
        avatar_name = avatar['name']
        self.show_avatar(avatar_name)
        self.get_skills_dashboard().build(selected_user_id, group_id)
        st.write("")
        st.write("")
        st.write("")
        st.markdown(
            f"<h2 style='text-align: left; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
            f"-size: 24px;'> Track Recommendations </h2>", unsafe_allow_html=True)
        selected_track_id, selected_track_name, _ = TrackRecommendationDashboard(
            self.recording_repo, self.user_repo).display_recommendations(selected_user_id, False)
        if selected_track_id:
            st.markdown(
                f"<h2 style='text-align: left; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
                f"-size: 24px;'> Remarks & Scores </h2>", unsafe_allow_html=True)
            st.write(f"Track: {selected_track_name}")
            RecordingsAndTrackScoreTrendsDisplay(self.recording_repo).show(selected_user_id, selected_track_id)
        st.markdown(
            f"<h2 style='text-align: left; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
            f"-size: 24px;'> Practice Logs </h2>", unsafe_allow_html=True)
        self.get_practice_dashboard().build(selected_user_id)
        st.markdown(
            f"<h2 style='text-align: left; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
            f"-size: 24px;'> Badges Won </h2>", unsafe_allow_html=True)
        self.get_badges_dashboard().show_badges_won(selected_user_id, False)

    def assessments(self):
        st.markdown(
            f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
            f"-size: 28px;'> 📋 Student Assessments 📋 </h2>", unsafe_allow_html=True)
        self.divider()

        groups = self.user_repo.get_all_groups(self.get_org_id())

        if not groups:
            st.info("Please create a team to get started.")
            return

        group_options = ["--Select a Team--"] + [group['group_name'] for group in groups]
        group_name_to_id = {group['group_name']: group['group_id'] for group in groups}

        selected_group = st.selectbox(
            "Select a Team", group_options, key="assessments_group_selector")
        if selected_group != "--Select a Team--":
            llm = self.load_llm(0)
            self.student_assessment_dashboard_builder.publish_assessments(
                self.get_user_id(), self.get_session_id(), group_name_to_id[selected_group], llm)
        else:
            st.info("Please select a group to continue..")

    def team_dashboard(self):
        st.markdown(f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; "
                    "font-size: 28px;'> 🤝 Team Performance & Collaboration 🤝 </h2>", unsafe_allow_html=True)
        self.divider()
        groups = self.user_repo.get_all_groups(self.get_org_id())

        if not groups:
            st.info("Please create a team to get started.")
            return

        group_options = [group['group_name'] for group in groups]
        group_name_to_id = {group['group_name']: group['group_id'] for group in groups}

        col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 0.3, 1, 2, 1.5])

        selected_group_ids = None
        with col1:
            selected_groups = st.multiselect(
                "Select Teams", group_options, key="team_dashboard_group_selector")
            if selected_groups:
                selected_group_ids = [group_name_to_id[group] for group in selected_groups]

        with col2:
            options = [time_frame for time_frame in TimeFrame]
            default_index = next((i for i, time_frame in enumerate(TimeFrame)
                                  if time_frame == TimeFrame.CURRENT_WEEK), 0)
            timeframe = st.selectbox(
                'Select a time frame:',
                options,
                index=default_index,
                format_func=lambda x: x.value
            )
        error_message = None
        # Apply actions to all selected groups
        if selected_group_ids:
            with col4:
                st.write("")
                st.write("")
                if st.button("Award Badges", type='primary'):
                    if timeframe not in (
                            TimeFrame.PREVIOUS_WEEK,
                            TimeFrame.PREVIOUS_MONTH,
                            TimeFrame.PREVIOUS_YEAR):
                        error_message = "Badges cannot be awarded for the current Week, Month or Year."
                    else:
                        with st.spinner("Please wait.."):
                            for group_id in selected_group_ids:
                                self.badge_awarder.auto_award_badges(group_id, timeframe)
                                self.log_activity(self.get_activity_type(timeframe), group_id)
                    self.hall_of_fame_dashboard_builder.clear_cache()

            with col5:
                st.write("")
                st.write("")
                if st.button("GenerateAssessments", type='primary'):
                    if timeframe not in (
                            TimeFrame.PREVIOUS_WEEK,
                            TimeFrame.PREVIOUS_MONTH,
                            TimeFrame.PREVIOUS_YEAR):
                        error_message = "Assessments cannot be generated for the current Week, Month or Year."
                    else:
                        llm = self.load_llm(0)
                        with st.spinner("Please wait.."):
                            for group_id in selected_group_ids:
                                self.student_assessment_dashboard_builder.generate_assessments(
                                    group_id, llm, timeframe)
            if error_message:
                st.error(error_message)
            st.write("")
            self.get_team_dashboard().build(selected_group_ids, timeframe)
        else:
            st.info("Please select a team to continue..")

    @staticmethod
    def get_activity_type(timeframe):
        if timeframe == TimeFrame.PREVIOUS_WEEK:
            return ActivityType.AWARD_WEEKLY_BADGES
        elif timeframe == TimeFrame.PREVIOUS_MONTH:
            return ActivityType.AWARD_MONTHLY_BADGES
        elif timeframe == TimeFrame.PREVIOUS_YEAR:
            return ActivityType.AWARD_YEARLY_BADGES

    def log_activity(self, activity_type, group_id):
        additional_params = {"group_id": group_id}
        self.user_activity_repo.log_activity(self.get_user_id(), self.get_session_id(),
                                             activity_type, additional_params)

    def hall_of_fame(self):
        st.markdown(
            f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; font"
            f"-size: 28px;'> 🏆 Hall of Fame 🏆️ </h2>", unsafe_allow_html=True)
        groups = self.user_repo.get_all_groups(self.get_org_id())

        if not groups:
            st.info("Please create a team to get started.")
            return

        group_options = ["--Select a Team--"] + [group['group_name'] for group in groups]
        group_name_to_id = {group['group_name']: group['group_id'] for group in groups}
        selected_group = st.selectbox(
            "Select a Team", group_options, key="hall_of_fame_group_selector")
        if selected_group != "--Select a Team--":
            selected_group_id = group_name_to_id[selected_group]
            self.hall_of_fame_dashboard_builder.build(selected_group_id, TimeFrame.PREVIOUS_WEEK)
            st.write("")
            self.divider(3)
            self.hall_of_fame_dashboard_builder.build(selected_group_id, TimeFrame.PREVIOUS_MONTH)

    def team_connect(self):
        st.markdown(f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; "
                    "font-size: 28px;'> 💼 Team Engagement & Insights 💼</h2>", unsafe_allow_html=True)
        self.divider()

        # Fetch all groups
        all_groups = self.user_repo.get_all_groups(self.get_org_id())
        group_options = ["Select a Team"] + [group['group_name'] for group in all_groups]
        group_ids = [None] + [group['group_id'] for group in all_groups]
        group_name_to_id = {group['group_name']: group['group_id'] for group in all_groups}

        # Dropdown to select a group
        selected_group = st.selectbox("Choose a Team to Interact With:", group_options)

        # Only show the message dashboard if a group is selected
        if selected_group != "Select a Team":
            selected_group_id = group_name_to_id[selected_group]
            self.get_message_dashboard().build(
                self.get_user_id(), selected_group_id, self.get_session_id())

    def notes_dashboard(self):
        st.markdown(f"<h2 style='text-align: center; font-weight: bold; color: {self.get_tab_heading_font_color()}; "
                    "font-size: 28px;'> 📝 Notes 📝</h2>", unsafe_allow_html=True)

        self.divider()
        self.get_notes_dashboard().notes_dashboard(self.get_user_id())

    @staticmethod
    def calculate_file_hash(audio_data):
        return hashlib.md5(audio_data).hexdigest()

    @staticmethod
    def ordinal(n):
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
        if 11 <= (n % 100) <= 13:
            suffix = 'th'
        return str(n) + suffix
