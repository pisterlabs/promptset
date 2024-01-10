import os
import sys
import json
import re
import httpx
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QMessageBox, QRadioButton, QCheckBox,
    QTextEdit, QLabel, QComboBox, QSpinBox, QAction, QMenuBar, QDialog, QGroupBox
)
from PyQt5.QtGui import QPalette, QColor, QFont, QPixmap, QIcon, QFontMetrics, QDesktopServices
import openai
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5 import QtGui
from bs4 import BeautifulSoup
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from icon_binary_code import ICON_DATA
from pyqt5switch import PyQtSwitch
from generate_html_diff import generate_html_diff

class AboutDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.base_resolution = (1920, 1080)
        self.screen_resolution = QApplication.desktop().screenGeometry()

        if self.screen_resolution.width() > self.screen_resolution.height():
            self.window_width_percentage = 25
            self.window_height_percentage = 20
            self.scaling_factor_width = self.screen_resolution.width() / self.base_resolution[0]
            self.scaling_factor_height = self.screen_resolution.height() / self.base_resolution[1]
        else:
            self.window_width_percentage = 20
            self.window_height_percentage = 25
            self.scaling_factor_width = self.screen_resolution.width() / self.base_resolution[1]
            self.scaling_factor_height = self.screen_resolution.height() / self.base_resolution[0]

        self.window_width = int(self.screen_resolution.width() * (self.window_width_percentage / 100))
        self.window_height = int(self.screen_resolution.height() * (self.window_height_percentage / 100))
        self.setWindowTitle('About ChatGPT UI')
        self.setGeometry(100, 100, self.window_width, self.window_height)

        layout = QVBoxLayout()

        description_label = QLabel('This is version 3.0 of ChatGPT UI.', self)
        description_label.setFont(QFont('Arial', 11))  # Set font size to 12 points
        layout.addWidget(description_label)

        link_label = QLabel(
            'Find the latest version on our GitHub: <a href="https://github.com/AITwinMinds/ChatGPT-UI">GitHub</a>',
            self)
        link_label.setFont(QFont('Arial', 11))  # Set font size to 12 points
        link_label.setOpenExternalLinks(True)
        layout.addWidget(link_label)

        x_label = QLabel(
            'Follow us on X (Twitter): <a href="https://twitter.com/AITwinMinds">https://twitter.com/AITwinMinds</a>',
            self)
        x_label.setFont(QFont('Arial', 11))  # Set font size to 12 points
        x_label.setOpenExternalLinks(True)
        layout.addWidget(x_label)

        youtube_label = QLabel(
            'Subscribe to our YouTube Channel: <a href="https://www.youtube.com/@AITwinMinds"> AITwinMinds Channel</a> for video tutorials and updates',
            self)
        youtube_label.setFont(QFont('Arial', 11))  # Set font size to 12 points
        youtube_label.setOpenExternalLinks(True)
        layout.addWidget(youtube_label)

        self.setLayout(layout)

class GPTUI(QWidget):
    CONFIG_FILE_PATH = "config.json"

    def __init__(self):
        super().__init__()
        self.client = None
        self.api_key_fixed = False
        self.stop_generation = False
        self.base_resolution = (1920, 1080)
        self.screen_resolution = QApplication.desktop().screenGeometry()

        if self.screen_resolution.width() > self.screen_resolution.height():
            self.window_width_percentage = 31
            self.window_height_percentage = 56
            self.scaling_factor_width = self.screen_resolution.width() / self.base_resolution[0]
            self.scaling_factor_height = self.screen_resolution.height() / self.base_resolution[1]
        else:
            self.window_width_percentage = 56
            self.window_height_percentage = 31
            self.scaling_factor_width = self.screen_resolution.width() / self.base_resolution[1]
            self.scaling_factor_height = self.screen_resolution.height() / self.base_resolution[0]

        self.window_width = int(self.screen_resolution.width() * (self.window_width_percentage / 100))
        self.window_height = int(self.screen_resolution.height() * (self.window_height_percentage / 100))

        self.init_ui()
        self.load_config_from_file()
    def init_ui(self):
        self.setWindowTitle('ChatGPT')
        self.setGeometry(100, 100, self.window_width, self.window_height)

        self.color_API = "#222529"
        self.color_API_text = "white"
        self.color_API_border = "#565856"

        self.color_API_button_background_active = '#073a43'
        self.color_API_button_background = "#757777"

        self.color_toggle_always_on_top_button_background = "#757777"
        self.color_toggle_always_on_top_button_background_active = "#073a43"

        self.color_output_text_background = "#222529"
        self.color_output_text_border = "#565856"
        self.color_output_text_font = "#ffffff"

        self.color_stop_button_background = "#073a43"

        self.color_copy_button_background = "#073a43"
        self.color_copy_button_background_active = "#007a5a"

        self.color_generate_button = "#073a43"
        self.color_generate_button_active = "#4b4b4b"

        self.color_output_text_responce_font = "#ffffff"

        self.color_email_details_background = "#222529"
        self.color_email_details_border = "#565856"
        self.color_email_details_font = "#ffffff"

        self.color_rewrite_level_selector_font = "#FFFFFF"

        self.color_mode_detail_background = "#132b34"
        self.color_mode_style_font = "#ffffff"

        self.color_rephrase_options_dropdown_background = "#073a43"
        self.color_rephrase_options_dropdown_font = "#FFFFFF"

        self.rewrite_level_selector_background = "#1a4850"
        self.rewrite_level_selector_border = "#000000"
        self.rewrite_level_selector_font = "#FFFFFF"

        self.color_input_text_background = "#222529"
        self.color_input_text_font = "#ffffff"
        self.color_input_text_border = "#565856"

        self.color_text_editor_background = "#222529"
        self.color_text_editor_font = "#ffffff"
        self.color_text_editor_border = "#565856"

        self.color_proxy_server_input_background = "#222529"
        self.color_proxy_server_input_font = "#ffffff"
        self.color_proxy_server_input_border = "#565856"

        self.color_proxy_port_input_background = "#222529"
        self.color_proxy_port_input_font = "#ffffff"
        self.color_proxy_port_input_border = "#565856"

        self.save_proxy_button_background = "#073a43"

        self.color_language_dropdown_background = "#073a43"


        icon_path = os.path.join(os.path.dirname(sys.executable), "icon.ico")
        if not os.path.exists(icon_path):
            with open(icon_path, "wb") as icon_file:
                icon_file.write(ICON_DATA)

        self.setWindowIcon(QIcon(icon_path))

        layout = QVBoxLayout()

        layout_menu = QHBoxLayout()

        # Create a menu bar
        menu_bar = QMenuBar(self)

        # Create a "File" menu
        file_menu = menu_bar.addMenu('Help')

        # Create actions for the "File" menu
        updates_action = QAction('Check for updates', self)
        star_action = QAction('Star the repository on GitHub ✨', self)
        about_action = QAction('About', self)
        exit_action = QAction('Exit', self)

        updates_action.triggered.connect(self.update_program)
        star_action.triggered.connect(self.give_star)
        about_action.triggered.connect(self.about_app)
        exit_action.triggered.connect(self.close)  # Connect the action to the close method

        # Add actions to the "File" menu
        file_menu.addAction(updates_action)
        file_menu.addAction(star_action)
        file_menu.addAction(about_action)
        file_menu.addSeparator()  # Add a separator between items
        file_menu.addAction(exit_action)

        layout_menu.addWidget(menu_bar)


        self.proxy_checkbox_label = QLabel('HTTP Proxy', self)


        layout_menu.addWidget(self.proxy_checkbox_label)

        self.proxy_checkbox = PyQtSwitch()
        self.proxy_checkbox.setAnimation(True)
        self.proxy_checkbox.setCircleDiameter(int(24 * self.scaling_factor_height))
        layout_menu.addWidget(self.proxy_checkbox)

        self.proxy_checkbox.toggled.connect(self.configure_proxy)
        self.proxy_checkbox_isEnabled = False

        layout.addLayout(layout_menu)

        self.proxy_input_layout = QVBoxLayout()
        self.proxy_server_input = QLineEdit(self)
        self.proxy_server_input.setFixedHeight(int(34 * self.scaling_factor_height))
        self.proxy_server_input.setPlaceholderText('Enter server address (e.g., 127.0.0.1)')
        self.proxy_server_input.setStyleSheet(
            f"""
                    QLineEdit {{
                        background-color: {self.color_proxy_server_input_background};
                        color: {self.color_proxy_server_input_font};
                        font-size: 11pt;
                        border: {1 * self.scaling_factor_width}px solid {self.color_proxy_server_input_border};
                        border-radius: {2 * self.scaling_factor_width}px;
                    }}
                    """
        )

        self.proxy_server_input.setHidden(True)

        self.proxy_input_layout.addWidget(self.proxy_server_input)

        self.proxy_port_input = QLineEdit(self)
        self.proxy_port_input.setFixedHeight(int(34 * self.scaling_factor_height))

        self.proxy_port_input.setStyleSheet(
            f"""
                    QLineEdit {{
                        background-color: {self.color_proxy_port_input_background};
                        color: {self.color_proxy_port_input_font};
                        font-size: 11pt;
                        border: {1 * self.scaling_factor_width}px solid {self.color_proxy_port_input_border};
                        border-radius: {2 * self.scaling_factor_width}px;
                    }}
                    """
        )
        self.proxy_port_input.setPlaceholderText('Enter port number (e.g., 10809)')
        self.proxy_port_input.setHidden(True)
        self.proxy_input_layout.addWidget(self.proxy_port_input)

        layout.addLayout(self.proxy_input_layout)

        self.save_proxy_button = QPushButton('Save proxy settings', self)
        self.save_proxy_button.clicked.connect(self.save_proxy_settings)
        self.save_proxy_button.setStyleSheet(
            f"""
                    QPushButton {{
                        background-color: {self.save_proxy_button_background};
                        border-color: #127287;
                        color: #FFFFFF;
                        min-height: {27 * self.scaling_factor_height}px;
                        font: 11.5pt Arial;
                    }}
                    """
        )
        self.save_proxy_button.setFixedHeight(int(34 * self.scaling_factor_height))
        self.save_proxy_button.setHidden(True)
        layout.addWidget(self.save_proxy_button)

        self.switch_control_label = QLabel('Dark Mode', self)
        self.switch_control_label.setStyleSheet(
            f"""
            QLabel {{
                color: #ffffff;
                font-size: 10pt;
            }}
            """
        )

        layout_menu.addWidget(self.switch_control_label)

        switch_control = PyQtSwitch()
        switch_control.setAnimation(True)
        switch_control.setCircleDiameter(int(24 * self.scaling_factor_height))
        switch_control.initial_toggle()
        layout_menu.addWidget(switch_control)

        switch_control.toggled.connect(self.theme_switch_button_changed)

        layout.addLayout(layout_menu)

        self.scrollbar_stylesheet_dark = f"""
            QScrollBar:vertical {{
                border: 1px solid #041b20;
                background: #222529;
                width: {15 * self.scaling_factor_width}px;
                margin: {5 * self.scaling_factor_height}px 0 {5 * self.scaling_factor_height}px 0;
            }}

            QScrollBar::handle:vertical {{
                background: #0b4654;
                min-height: {30 * self.scaling_factor_height}px;  /* Adjust min-height for the handle */
                max-height: {30 * self.scaling_factor_height}px;  /* Adjust max-height for the handle */
            }}

            QScrollBar::add-line:vertical {{
                border: 1px solid #041b20;
                background: #073a43;  /* Change color for the add-line (down arrow) */
                height: {5 * self.scaling_factor_height}px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
            }}

            QScrollBar::sub-line:vertical {{
                border: 1px solid #041b20;
                background: #073a43;  /* Change color for the sub-line (up arrow) */
                height: {5 * self.scaling_factor_height}px;
                subcontrol-position: top;
                subcontrol-origin: margin;
            }}

            QScrollBar:horizontal {{
                border: 1px solid #041b20;
                background: #222529;
                height: {15 * self.scaling_factor_height}px;
                margin: 0 {5 * self.scaling_factor_width}px 0 {5 * self.scaling_factor_width}px;
            }}

            QScrollBar::handle:horizontal {{
                background: #0b4654;
                min-width: {30 * self.scaling_factor_width}px;  /* Adjust min-width for the handle */
                max-width: {30 * self.scaling_factor_width}px;  /* Adjust max-width for the handle */
            }}

            QScrollBar::add-line:horizontal {{
                border: 1px solid #041b20;
                background: #073a43;  /* Change color for the add-line (right arrow) */
                width: {5 * self.scaling_factor_width}px;
                subcontrol-position: right;
                subcontrol-origin: margin;
            }}

            QScrollBar::sub-line:horizontal {{
                border: 1px solid #041b20;
                background: #073a43;  /* Change color for the sub-line (left arrow) */
                width: {5 * self.scaling_factor_width}px;
                subcontrol-position: left;
                subcontrol-origin: margin;
            }}
        """

        self.scrollbar_stylesheet_white = f"""
                    QScrollBar:vertical {{
                        border: 1px solid #bcbcbc;
                        background: #f8f8f8;
                        width: {15 * self.scaling_factor_width}px;
                        margin: {5 * self.scaling_factor_height}px 0 {5 * self.scaling_factor_height}px 0;
                    }}

                    QScrollBar::handle:vertical {{
                        background: #999999;
                        min-height: {30 * self.scaling_factor_height}px;  /* Adjust min-height for the handle */
                        max-height: {30 * self.scaling_factor_height}px;  /* Adjust max-height for the handle */
                    }}

                    QScrollBar::add-line:vertical {{
                        border: 1px solid #bcbcbc;
                        background: #5b5b5b;  /* Change color for the add-line (down arrow) */
                        height: {5 * self.scaling_factor_height}px;
                        subcontrol-position: bottom;
                        subcontrol-origin: margin;
                    }}

                    QScrollBar::sub-line:vertical {{
                        border: 1px solid #bcbcbc;
                        background: #5b5b5b;  /* Change color for the sub-line (up arrow) */
                        height: {5 * self.scaling_factor_height}px;
                        subcontrol-position: top;
                        subcontrol-origin: margin;
                    }}

                    QScrollBar:horizontal {{
                        border: 1px solid #bcbcbc;
                        background: #222529;
                        height: {15 * self.scaling_factor_height}px;
                        margin: 0 {5 * self.scaling_factor_width}px 0 {5 * self.scaling_factor_width}px;
                    }}

                    QScrollBar::handle:horizontal {{
                        background: #f8f8f8;
                        min-width: {30 * self.scaling_factor_width}px;  /* Adjust min-width for the handle */
                        max-width: {30 * self.scaling_factor_width}px;  /* Adjust max-width for the handle */
                    }}

                    QScrollBar::add-line:horizontal {{
                        border: 1px solid #bcbcbc;
                        background: #5b5b5b;  /* Change color for the add-line (right arrow) */
                        width: {5 * self.scaling_factor_width}px;
                        subcontrol-position: right;
                        subcontrol-origin: margin;
                    }}

                    QScrollBar::sub-line:horizontal {{
                        border: 1px solid #bcbcbc;
                        background: #5b5b5b;  /* Change color for the sub-line (left arrow) */
                        width: {5 * self.scaling_factor_width}px;
                        subcontrol-position: left;
                        subcontrol-origin: margin;
                    }}
                """

        # Create a container to hold the input layout
        input_layout = QHBoxLayout()

        self.api_key_input = QLineEdit(self)
        self.api_key_input.setPlaceholderText('Enter API key...')
        self.api_key_input.setFixedHeight(int(34 * self.scaling_factor_height))
        self.api_key_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.color_API};
                color: {self.color_API_text};
                font-size: 10pt;
                min-height: {27 * self.scaling_factor_height}px;
                border: {1 * self.scaling_factor_width}px solid {self.color_API_border};
                border-radius: {2 * self.scaling_factor_width}px;
            }}
        """)
        input_layout.addWidget(self.api_key_input)
        self.fix_api_key_button = QPushButton('Fix/Unfix API Key', self)
        self.fix_api_key_button.setFixedHeight(int(34 * self.scaling_factor_height))
        self.fix_api_key_button.clicked.connect(self.toggle_api_key)
        self.fix_api_key_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.color_API_button_background};
                border-color: #0f1c19;
                color: #000000;
                min-height: {27 * self.scaling_factor_height}px;
                font: 10.5pt Arial;
            }}
            """
        )
        input_layout.addWidget(self.fix_api_key_button)
        self.toggle_always_on_top_button = QPushButton('Always On Top', self)
        self.toggle_always_on_top_button.setFixedHeight(int(34 * self.scaling_factor_height))
        self.toggle_always_on_top_button.clicked.connect(self.toggle_always_on_top)
        self.toggle_always_on_top_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.color_toggle_always_on_top_button_background};
                border-color: #0f1c19;
                color: #000000;
                min-height: {27 * self.scaling_factor_height}px;
                font: 10.5pt Arial;
            }}
            """
        )
        input_layout.addWidget(self.toggle_always_on_top_button)
        layout.addLayout(input_layout)

        # Create a single group box
        self.group_box = QGroupBox(self)
        self.group_box.setAlignment(Qt.AlignLeft)  # Set alignment if needed


        # Create two separate layouts
        prompt_layout = QHBoxLayout()
        prompt_layout2 = QHBoxLayout()



        self.radio_rephrase = QRadioButton('Paraphrase', self)
        self.stylesheet_radio_buttons = f"""QRadioButton {{ 
                                          font-size: 10.5pt;
                                          color: #ffffff;}}
     
                                          QRadioButton::indicator::unchecked {{
                                          background-color: #1c4e56;
                                          width: {int(6 * self.scaling_factor_height)}px;
                                          height: {int(6 * self.scaling_factor_height)}px;
                                          border-radius: {int(6 * self.scaling_factor_height)}px;
                                          border: {int(3 * self.scaling_factor_height)}px solid #1c4e56;
                                          }}
                                          
                                          QRadioButton::indicator::checked {{
                                          background-color: #dddddd;
                                          width: {int(6 * self.scaling_factor_height)}px;
                                          height: {int(6 * self.scaling_factor_height)}px;
                                          border-radius: {int(6 * self.scaling_factor_height)}px;
                                          border: {int(3 * self.scaling_factor_height)}px solid #1c4e56;
                                          }}"""


        self.radio_rephrase.setChecked(True)
        prompt_layout.addWidget(self.radio_rephrase)

        self.radio_debug_code = QRadioButton('Debug code', self)
        prompt_layout.addWidget(self.radio_debug_code)

        self.radio_summarize = QRadioButton('Summarize', self)
        prompt_layout.addWidget(self.radio_summarize)

        self.radio_translate = QRadioButton('Translate', self)
        prompt_layout.addWidget(self.radio_translate)

        self.radio_email = QRadioButton('Reply to email', self)
        prompt_layout.addWidget(self.radio_email)

        self.radio_explain = QRadioButton('Explain', self)
        prompt_layout.addWidget(self.radio_explain)
        prompt_layout.setAlignment(Qt.AlignLeft)


        self.radio_manual_prompts = QRadioButton('Custom prompt', self)
        prompt_layout2.addWidget(self.radio_manual_prompts)

        self.radio_grammer_checker = QRadioButton('Check grammar', self)
        prompt_layout2.addWidget(self.radio_grammer_checker)
        prompt_layout2.setAlignment(Qt.AlignLeft)

        # Add both layouts to the group box
        self.group_box.setLayout(QVBoxLayout())  # QVBoxLayout for stacking two rows
        self.group_box.layout().addLayout(prompt_layout)
        self.group_box.layout().addLayout(prompt_layout2)

        layout.addWidget(self.group_box)

        # Add another layout (prompt_layout2) if needed

        self.setLayout(layout)



        rephrase_options_layout = QHBoxLayout()
        rephrase_options_layout.setAlignment(Qt.AlignLeft)

        self.rephrase_options_dropdown = QComboBox(self)
        self.rephrase_options_dropdown.addItems(["Standard", "Fluency", "Formal", "Academic", "Simple", "Creative", "Expand", "Shorten"])
        self.rephrase_options_dropdown.setFixedHeight(int(27 * self.scaling_factor_height))
        self.rephrase_options_dropdown.setCurrentIndex(0)


        self.rephrase_options_dropdown.setStyleSheet(
            f"""
            background-color: {self.color_rephrase_options_dropdown_background};
            border-color: #000000;
            color: {self.color_rephrase_options_dropdown_font};
            min-height: {27 * self.scaling_factor_height}px;
            font: 10.5pt Arial;
            """
        )

        rephrase_options_layout.addWidget(self.rephrase_options_dropdown)

        self.mode_detail = QLabel('  Rewrites text in a reliable manner to maintain meaning  ', self)
        self.mode_detail.setStyleSheet(
            f"""
            QLabel {{
                background-color: {self.color_mode_detail_background};
                color: {self.color_mode_style_font};
                font: 10.5pt Arial;
                border-radius: {4 * self.scaling_factor_width}px;
                }}
            """
        )

        rephrase_options_layout.addWidget(self.mode_detail)
        layout.addLayout(rephrase_options_layout)

        rewrite_level_layout = QHBoxLayout()
        self.rewrite_level_selector_label = QLabel("How many rewrites do you want?", self)
        self.rewrite_level_selector_label.setStyleSheet(
            f"""
            font: 10.5pt Arial;
            color: {self.color_rewrite_level_selector_font};
            """
        )
        rewrite_level_layout.addWidget(self.rewrite_level_selector_label)

        self.rewrite_level = 1  # Default rewrite level

        # Create number selector widget
        self.rewrite_level_selector = QSpinBox(self)
        self.rewrite_level_selector.setMinimum(1)
        self.rewrite_level_selector.setMaximum(20)
        self.rewrite_level_selector.setValue(self.rewrite_level)


        self.rewrite_level_selector.setStyleSheet(
            f"""
            background-color: {self.rewrite_level_selector_background};
            border-color: {self.rewrite_level_selector_border};
            color: {self.rewrite_level_selector_font};
            min-height: {17 * self.scaling_factor_height}px;
            font: 10.5pt Arial;
            padding-left: 4px;
            """
        )
        self.rewrite_level_selector.setToolTip(
            "Choose how many different rewrites you want to receive (max 20)"
        )
        self.rewrite_level_selector.setFixedHeight(int(27 * self.scaling_factor_height))
        self.rewrite_level_selector.valueChanged.connect(self.update_rewrite_level)

        rewrite_level_layout.addWidget(self.rewrite_level_selector)
        rewrite_level_layout.setAlignment(Qt.AlignLeft)
        layout.addLayout(rewrite_level_layout)

        language_layout = QHBoxLayout()
        language_layout.setAlignment(Qt.AlignLeft)

        self.from_language_dropdown = QComboBox(self)
        self.from_language_dropdown.setStyleSheet(
            f"""
                background-color: {self.color_language_dropdown_background};
                border-color: #000000;
                color: #FFFFFF;
                min-height: {20 * self.scaling_factor_height}px;
                font: 10.5pt Arial;
            """
        )
        self.to_language_dropdown = QComboBox(self)
        self.to_language_dropdown.setStyleSheet(
            f"""
                background-color: {self.color_language_dropdown_background};
                border-color: #000000;
                color: #FFFFFF;
                min-height: {20 * self.scaling_factor_height}px;
                font: 10.5pt Arial;
            """
        )
        self.from_language_dropdown.setHidden(True)
        self.to_language_dropdown.setHidden(True)
        self.from_language_label = QLabel('From Language:', self)
        self.to_language_label = QLabel('To Language:', self)
        self.from_language_label.setHidden(True)
        self.to_language_label.setHidden(True)

        language_layout.addWidget(self.from_language_label)
        language_layout.addWidget(self.from_language_dropdown)
        language_layout.addWidget(self.to_language_label)
        language_layout.addWidget(self.to_language_dropdown)

        layout.addLayout(language_layout)

        self.radio_translate.toggled.connect(self.toggle_language_dropdowns)

        available_languages = [
            "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Azerbaijani", "Basque", "Belarusian", "Bengali",
            "Bosnian", "Bulgarian", "Catalan", "Cebuano", "Chichewa", "Chinese (Simplified)", "Chinese (Traditional)",
            "Corsican", "Croatian",
            "Czech", "Danish", "Dutch", "English", "Esperanto", "Estonian", "Filipino", "Finnish", "French", "Frisian",
            "Galician", "Georgian", "German", "Greek", "Gujarati", "Haitian Creole", "Hausa", "Hawaiian", "Hebrew",
            "Hindi",
            "Hmong", "Hungarian", "Icelandic", "Igbo", "Indonesian", "Irish", "Italian", "Japanese", "Javanese",
            "Kannada",
            "Kazakh", "Khmer", "Kinyarwanda", "Korean", "Kurdish (Kurmanji)", "Kyrgyz", "Lao", "Latin", "Latvian",
            "Lithuanian",
            "Luxembourgish", "Macedonian", "Malagasy", "Malay", "Malayalam", "Maltese", "Maori", "Marathi", "Mongolian",
            "Myanmar (Burmese)",
            "Nepali", "Norwegian", "Pashto", "Persian", "Polish", "Portuguese", "Punjabi", "Romanian", "Russian",
            "Samoan",
            "Scots Gaelic", "Serbian", "Sesotho", "Shona", "Sindhi", "Sinhala", "Slovak", "Slovenian", "Somali",
            "Spanish",
            "Sundanese", "Swahili", "Swedish", "Tajik", "Tamil", "Telugu", "Thai", "Turkish", "Ukrainian", "Urdu",
            "Uzbek", "Vietnamese", "Welsh", "Xhosa", "Yiddish", "Yoruba", "Zulu"
        ]

        self.from_language_dropdown.addItems(available_languages)
        self.to_language_dropdown.addItems(available_languages)

        clipboard_email_detail_layout = QHBoxLayout()

        self.checkbox_clipboard = QCheckBox('Use last clipboard text', self)
        self.checkbox_clipboard.setChecked(True)

        clipboard_email_detail_layout.addWidget(self.checkbox_clipboard)

        self.keep_previous_checkbox = QCheckBox("Keep previous responses")
        self.keep_previous_checkbox.setChecked(True)
        self.keep_previous_checkbox.stateChanged.connect(self.on_keep_previous_changed)
        clipboard_email_detail_layout.addWidget(self.keep_previous_checkbox)

        self.checkbox_reply_to_email = QCheckBox('Email response details', self)
        self.checkbox_reply_to_email.setHidden(True)

        clipboard_email_detail_layout.addWidget(self.checkbox_reply_to_email)

        clipboard_email_detail_layout.setAlignment(Qt.AlignLeft)

        layout.addLayout(clipboard_email_detail_layout)

        editor_email_details_layout = QHBoxLayout()

        self.editor_email_details = QTextEdit(self)
        self.editor_email_details.setPlaceholderText('Add specific information to enhance the email prompt...')

        self.editor_email_details.setStyleSheet(
            f"""
                        QTextEdit {{
                            background-color: {self.color_email_details_background};
                            color: {self.color_email_details_font};
                            font-size: 11.5pt;
                            border: {1 * self.scaling_factor_width}px solid {self.color_email_details_border};
                            border-radius: {4 * self.scaling_factor_width}px;
                            padding: {5 * self.scaling_factor_height}px;
                        }}
                        """
        )
        self.editor_email_details.setFixedHeight(int(100 * self.scaling_factor_height))


        self.editor_email_details.setHidden(True)
        editor_email_details_layout.addWidget(self.editor_email_details)

        layout.addLayout(editor_email_details_layout)

        self.checkbox_reply_to_email.stateChanged.connect(self.toggle_email_details_checkbox)
        self.radio_email.toggled.connect(self.toggle_email_details)
        self.toggle_email_details(False)

        self.input_text = QTextEdit(self)
        self.input_text.setPlaceholderText('Enter prompt manually...')
        self.input_text.setFixedHeight(int(70 * self.scaling_factor_height))
        self.input_text.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: {self.color_input_text_background};
                color: {self.color_input_text_font};
                font-size: 11.5pt;
                border: {1 * self.scaling_factor_width}px solid {self.color_input_text_border};
                border-radius: {4 * self.scaling_factor_width}px;
                padding: {5 * self.scaling_factor_height}px;
            }}
            """
        )


        self.input_text.setHidden(True)
        layout.addWidget(self.input_text)

        self.text_editor = QTextEdit(self)
        self.text_editor.setPlaceholderText('Enter text manually...')

        self.text_editor.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: {self.color_text_editor_background};
                color: {self.color_text_editor_font};
                font-size: 11.5pt;
                border: {1 * self.scaling_factor_width}px solid {self.color_text_editor_border};
                border-radius: {4 * self.scaling_factor_width}px;
                padding: {5 * self.scaling_factor_height}px;
            }}
            """
        )
        self.text_editor.setFixedHeight(int(150 * self.scaling_factor_height))

        self.text_editor.setHidden(True)
        layout.addWidget(self.text_editor)

        self.checkbox_clipboard.stateChanged.connect(self.toggle_clipboard_text)

        self.radio_manual_prompts.toggled.connect(self.toggle_manual_prompt_input)

        self.output_text = QTextEdit(self)
        self.output_text.setPlaceholderText('Response will appear here...')
        self.output_text.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: {self.color_output_text_background};
                color: {self.color_output_text_font};
                font-size: 11.5pt;
                border: {1 * self.scaling_factor_width}px solid {self.color_output_text_border};
                border-radius: {4 * self.scaling_factor_width}px;
                padding: {5 * self.scaling_factor_height}px;
            }}
            """
        )

        layout.addWidget(self.output_text)

        button_layout = QHBoxLayout()

        self.run_regenerate_button = QPushButton('Generate', self)
        self.run_regenerate_button.clicked.connect(self.run_regenerate_text)
        self.run_regenerate_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.color_generate_button};
                border-color: #127287;
                color: #FFFFFF;
                min-height: {27 * self.scaling_factor_height}px;
                font: 11.5pt Arial;
            }}
            """
        )
        self.run_regenerate_button.setFixedHeight(int(34 * self.scaling_factor_height))
        button_layout.addWidget(self.run_regenerate_button)


        self.copy_button = QPushButton('Copy response', self)
        self.copy_button.clicked.connect(self.copy_text)
        self.copy_button.setStyleSheet(
            f"""
            background-color: {self.color_copy_button_background};
            border-color: #127287;
            color: #FFFFFF;
            min-height: {27 * self.scaling_factor_height}px;
            font: 11.5pt Arial;
            """)
        self.copy_button.setFixedHeight(int(34 * self.scaling_factor_height))
        button_layout.addWidget(self.copy_button)


        self.clear_button = QPushButton('Clear response', self)
        self.clear_button.clicked.connect(self.clear_text)
        self.clear_button.setStyleSheet(
            f"""
            background-color: {self.color_copy_button_background};
            border-color: #127287;
            color: #FFFFFF;
            min-height: {27 * self.scaling_factor_height}px;
            font: 11.5pt Arial;
            """)
        self.clear_button.setFixedHeight(int(34 * self.scaling_factor_height))
        button_layout.addWidget(self.clear_button)


        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_generation_process)
        self.stop_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.color_stop_button_background};
                border-color: #127287;
                color: #FFFFFF;
                min-height: {27 * self.scaling_factor_height}px;
                font: 11.5pt Arial;
            }}
            """
        )
        self.stop_button.setFixedHeight(int(34 * self.scaling_factor_height))
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        font = QFont()
        font.setFamily("Arial")
        font.setPointSizeF(11.5)
        self.setFont(font)

        self.from_language_dropdown.currentIndexChanged.connect(self.save_config_to_file)
        self.to_language_dropdown.currentIndexChanged.connect(self.save_config_to_file)

        self.load_config_from_file()
        if self.proxy_checkbox_isEnabled:
            self.proxy_checkbox_isEnabled = False
            self.proxy_checkbox.initial_toggle()

        if self.api_key_input.text() == "":
            self.api_key_fixed = True
            self.api_key_input.setReadOnly(False)
            self.fix_api_key_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.color_API_button_background};
                    border-color: #0f1c19;
                    color: #000000;
                    min-height: {27 * self.scaling_factor_height}px;
                    font: 10.5pt Arial;
                }}
            """)
        else:
            self.api_key_fixed = False
            self.api_key_input.setReadOnly(True)

            self.fix_api_key_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.color_API_button_background_active};
                    border-color: #127287;
                    color: #FFFFFF;
                    min-height: {27 * self.scaling_factor_height}px;
                    font: 10.5pt Arial;
                }}
            """)

        self.rephrase_options_dropdown.currentIndexChanged.connect(self.set_rephrase_prompt)
        self.radio_rephrase.toggled.connect(self.toggle_radio_rephrase)
        self.set_rephrase_prompt(0)
        self.radio_debug_code.toggled.connect(self.toggle_debug_code)
        self.radio_manual_prompts.toggled.connect(self.toggle_custom_prompt)

        self.radio_grammer_checker.toggled.connect(self.toggle_grammer_checker)



        self.theme_switch_button_changed(1)


    def theme_switch_button_changed(self, state):
        if state == 1:
            self.theme_flag = 1
            # Apply dark theme
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(26, 29, 33))
            dark_palette.setColor(QPalette.WindowText, Qt.white)
            dark_palette.setColor(QPalette.Base, QColor(8, 68, 49))
            dark_palette.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
            dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
            dark_palette.setColor(QPalette.ToolTipText, Qt.white)
            dark_palette.setColor(QPalette.Text, Qt.white)
            dark_palette.setColor(QPalette.Button, QColor(26, 82, 118))
            dark_palette.setColor(QPalette.ButtonText, Qt.white)
            dark_palette.setColor(QPalette.BrightText, Qt.red)
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, Qt.black)
            self.setPalette(dark_palette)

            self.editor_email_details.verticalScrollBar().setStyleSheet(self.scrollbar_stylesheet_dark)
            self.editor_email_details.horizontalScrollBar().setStyleSheet(self.scrollbar_stylesheet_dark)
            self.input_text.verticalScrollBar().setStyleSheet(self.scrollbar_stylesheet_dark)
            self.input_text.horizontalScrollBar().setStyleSheet(self.scrollbar_stylesheet_dark)
            self.text_editor.verticalScrollBar().setStyleSheet(self.scrollbar_stylesheet_dark)
            self.text_editor.horizontalScrollBar().setStyleSheet(self.scrollbar_stylesheet_dark)
            self.output_text.verticalScrollBar().setStyleSheet(self.scrollbar_stylesheet_dark)
            self.output_text.horizontalScrollBar().setStyleSheet(self.scrollbar_stylesheet_dark)

            self.color_API = "#222529"
            self.color_API_text = "white"
            self.color_API_border = "#565856"
            self.api_key_input.setStyleSheet(f"""
                  QLineEdit {{
                      background-color: {self.color_API};
                      color: {self.color_API_text};
                      font-size: 10pt;
                      min-height: {27 * self.scaling_factor_height}px;
                      border: {1 * self.scaling_factor_width}px solid {self.color_API_border};
                      border-radius: {2 * self.scaling_factor_width}px;
                  }}
              """)


            self.color_API_button_background = "#757777"
            self.color_API_button_background_active = "#073a43"
            if not self.api_key_fixed:
                self.color_API_temp = self.color_API_button_background_active
                self.colour_API_temp2 = "#ffffff"
            else:
                self.color_API_temp = self.color_API_button_background
                self.colour_API_temp2 = "#000000"
            self.fix_api_key_button.setStyleSheet(f"""
                            QPushButton {{
                                background-color: {self.color_API_temp};
                                border-color: #0f1c19;
                                color: {self.colour_API_temp2};
                                min-height: {27 * self.scaling_factor_height}px;
                                font: 10.5pt Arial;
                            }}
                        """)

            self.color_toggle_always_on_top_button_background = "#757777"
            self.color_toggle_always_on_top_button_background_active = "#073a43"
            if self.windowFlags() & Qt.WindowStaysOnTopHint:
                self.color_toggle_always_on_top_button_background_temp = self.color_toggle_always_on_top_button_background_active
                self.colour_toggle_always_on_top_button_background_temp2 = "#ffffff"
            else:
                self.color_toggle_always_on_top_button_background_temp = self.color_toggle_always_on_top_button_background
                self.colour_toggle_always_on_top_button_background_temp2 = "#000000"
            self.toggle_always_on_top_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.color_toggle_always_on_top_button_background_temp};
                    border-color: #0f1c19;
                    color: {self.colour_toggle_always_on_top_button_background_temp2};
                    min-height: {27 * self.scaling_factor_height}px;
                    font: 10.5pt Arial;
                }}
                """
            )

            self.color_output_text_background = "#222529"
            self.color_output_text_border = "#565856"
            self.color_output_text_font = "#ffffff"
            self.output_text.setStyleSheet(
                f"""
                        QTextEdit {{
                            background-color: {self.color_output_text_background};
                            color: {self.color_output_text_font};
                            font-size: 11.5pt;
                            border: {1 * self.scaling_factor_width}px solid {self.color_output_text_border};
                            border-radius: {4 * self.scaling_factor_width}px;
                            padding: {5 * self.scaling_factor_height}px;
                        }}
                        """
            )

            self.color_stop_button_background = "#073a43"
            self.stop_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.color_stop_button_background};
                    border-color: #127287;
                    color: #FFFFFF;
                    min-height: {27 * self.scaling_factor_height}px;
                    font: 11.5pt Arial;
                }}
                """
            )

            self.color_copy_button_background = "#073a43"
            self.color_copy_button_background_active = "#04262c"
            self.copy_button.setStyleSheet(
                f"""
                        background-color: {self.color_copy_button_background};
                        border-color: #127287;
                        color: #FFFFFF;
                        min-height: {27 * self.scaling_factor_height}px;
                        font: 11.5pt Arial;
                        """)

            self.color_generate_button = "#073a43"
            self.color_generate_button_active = "#4b4b4b"
            self.run_regenerate_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.color_generate_button};
                    border-color: #127287;
                    color: #FFFFFF;
                    min-height: {27 * self.scaling_factor_height}px;
                    font: 11.5pt Arial;
                }}
                """
            )

            self.color_output_text_responce_font = "#ffffff"

            self.color_email_details_background = "#222529"
            self.color_email_details_border = "#565856"
            self.color_email_details_font = "#ffffff"
            self.editor_email_details.setStyleSheet(
                f"""
                            QTextEdit {{
                                background-color: {self.color_email_details_background};
                                color: {self.color_email_details_font};
                                font-size: 11.5pt;
                                border: {1 * self.scaling_factor_width}px solid {self.color_email_details_border};
                                border-radius: {4 * self.scaling_factor_width}px;
                                padding: {5 * self.scaling_factor_height}px;
                            }}
                            """
            )

            self.color_rewrite_level_selector_font = "#FFFFFF"
            self.rewrite_level_selector_label.setStyleSheet(
                f"""
                        font: 10.5pt Arial;
                        color: {self.color_rewrite_level_selector_font};
                        """
            )

            self.color_mode_detail_background = "#132b34"
            self.color_mode_style_font = "#ffffff"
            self.mode_detail.setStyleSheet(
                f"""
                QLabel {{
                    background-color: {self.color_mode_detail_background};
                    color: {self.color_mode_style_font};
                    font: 10.5pt Arial;
                    border-radius: {4 * self.scaling_factor_width}px;
                    }}
                """
            )

            self.color_rephrase_options_dropdown_background = "#073a43"
            self.color_rephrase_options_dropdown_font = "#FFFFFF"

            self.rephrase_options_dropdown.setStyleSheet(
                f"""
                background-color: {self.color_rephrase_options_dropdown_background};
                border-color: #000000;
                color: {self.color_rephrase_options_dropdown_font};
                min-height: {27 * self.scaling_factor_height}px;
                font: 10.5pt Arial;
                """
            )

            self.rewrite_level_selector_background = "#1a4850"
            self.rewrite_level_selector_border = "#000000"
            self.rewrite_level_selector_font = "#FFFFFF"
            self.rewrite_level_selector.setStyleSheet(
                f"""
                background-color: {self.rewrite_level_selector_background};
                border-color: {self.rewrite_level_selector_border};
                color: {self.rewrite_level_selector_font};
                min-height: {17 * self.scaling_factor_height}px;
                font: 10.5pt Arial;
                padding-left: 4px;
                """
            )

            self.color_input_text_background = "#222529"
            self.color_input_text_font = "#ffffff"
            self.color_input_text_border = "#565856"
            self.input_text.setStyleSheet(
                f"""
                QTextEdit {{
                    background-color: {self.color_input_text_background};
                    color: {self.color_input_text_font};
                    font-size: 11.5pt;
                    border: {1 * self.scaling_factor_width}px solid {self.color_input_text_border};
                    border-radius: {4 * self.scaling_factor_width}px;
                    padding: {5 * self.scaling_factor_height}px;
                }}
                """
            )

            self.color_text_editor_background = "#222529"
            self.color_text_editor_font = "#ffffff"
            self.color_text_editor_border = "#565856"
            self.text_editor.setStyleSheet(
                f"""
                        QTextEdit {{
                            background-color: {self.color_text_editor_background};
                            color: {self.color_text_editor_font};
                            font-size: 11.5pt;
                            border: {1 * self.scaling_factor_width}px solid {self.color_text_editor_border};
                            border-radius: {4 * self.scaling_factor_width}px;
                            padding: {5 * self.scaling_factor_height}px;
                        }}
                        """
            )

            self.color_proxy_server_input_background = "#222529"
            self.color_proxy_server_input_font = "#ffffff"
            self.color_proxy_server_input_border = "#565856"
            self.proxy_server_input.setStyleSheet(
                f"""
                        QLineEdit {{
                            background-color: {self.color_proxy_server_input_background};
                            color: {self.color_proxy_server_input_font};
                            font-size: 11pt;
                            border: {1 * self.scaling_factor_width}px solid {self.color_proxy_server_input_border};
                            border-radius: {2 * self.scaling_factor_width}px;
                        }}
                        """
            )

            self.color_proxy_port_input_background = "#222529"
            self.color_proxy_port_input_font = "#ffffff"
            self.color_proxy_port_input_border = "#565856"
            self.proxy_port_input.setStyleSheet(
                f"""
                      QLineEdit {{
                          background-color: {self.color_proxy_port_input_background};
                          color: {self.color_proxy_port_input_font};
                          font-size: 11pt;
                          border: {1 * self.scaling_factor_width}px solid {self.color_proxy_port_input_border};
                          border-radius: {2 * self.scaling_factor_width}px;
                      }}
                      """
            )

            self.save_proxy_button_background = "#073a43"
            self.save_proxy_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.save_proxy_button_background};
                    border-color: #127287;
                    color: #FFFFFF;
                    min-height: {27 * self.scaling_factor_height}px;
                    font: 11.5pt Arial;
                }}
                """
            )

            self.color_language_dropdown_background = "#073a43"
            self.from_language_dropdown.setStyleSheet(
                f"""
                    background-color: {self.color_language_dropdown_background};
                    border-color: #000000;
                    color: #FFFFFF;
                    min-height: {20 * self.scaling_factor_height}px;
                    font: 10.5pt Arial;
                """
            )
            self.to_language_dropdown.setStyleSheet(
                f"""
                    background-color: {self.color_language_dropdown_background};
                    border-color: #000000;
                    color: #FFFFFF;
                    min-height: {20 * self.scaling_factor_height}px;
                    font: 10.5pt Arial;
                """
            )
            self.switch_control_label.setStyleSheet(
                f"""
                      QLabel {{
                          color: #ffffff;
                          font-size: 10pt;
                      }}
                      """
            )
            self.clear_button.setStyleSheet(
                f"""
                background-color: {self.color_copy_button_background};
                border-color: #127287;
                color: #FFFFFF;
                min-height: {27 * self.scaling_factor_height}px;
                font: 11.5pt Arial;
                """)
            self.group_box.setStyleSheet(
                f"""QGroupBox {{ 
                 background-color: #1e2626; 
                 border: {int(1 * self.scaling_factor_height)}px solid #565856; 
                 border-radius: {int(4 * self.scaling_factor_height)}px; 
                 font-size: 10pt;}}""")

            self.stylesheet_radio_buttons = f"""QRadioButton {{ 
                                                 font-size: 10.5pt;
                                                 color: #ffffff;}}

                                                 QRadioButton::indicator::unchecked {{
                                                 background-color: #1c4e56;
                                                 width: {int(6 * self.scaling_factor_height)}px;
                                                 height: {int(6 * self.scaling_factor_height)}px;
                                                 border-radius: {int(6 * self.scaling_factor_height)}px;
                                                 border: {int(3 * self.scaling_factor_height)}px solid #1c4e56;
                                                 }}

                                                 QRadioButton::indicator::checked {{
                                                 background-color: #dddddd;
                                                 width: {int(6 * self.scaling_factor_height)}px;
                                                 height: {int(6 * self.scaling_factor_height)}px;
                                                 border-radius: {int(6 * self.scaling_factor_height)}px;
                                                 border: {int(3 * self.scaling_factor_height)}px solid #1c4e56;
                                                 }}"""
            self.radio_rephrase.setStyleSheet(self.stylesheet_radio_buttons)
            self.radio_debug_code.setStyleSheet(self.stylesheet_radio_buttons)
            self.radio_summarize.setStyleSheet(self.stylesheet_radio_buttons)
            self.radio_translate.setStyleSheet(self.stylesheet_radio_buttons)
            self.radio_email.setStyleSheet(self.stylesheet_radio_buttons)
            self.radio_explain.setStyleSheet(self.stylesheet_radio_buttons)
            self.radio_manual_prompts.setStyleSheet(self.stylesheet_radio_buttons)
            self.radio_grammer_checker.setStyleSheet(self.stylesheet_radio_buttons)

            self.proxy_checkbox_label.setStyleSheet(
                f"""
                        QLabel {{
                            color: #ffffff;
                            font-size: 10pt;
                        }}
                        """
            )



        else:
            self.theme_flag = 0
            # Apply white theme
            white_palette = QPalette()
            white_palette.setColor(QPalette.Window, Qt.white)
            white_palette.setColor(QPalette.WindowText, Qt.black)
            white_palette.setColor(QPalette.Base, QColor(231, 249, 244))
            white_palette.setColor(QPalette.AlternateBase, Qt.lightGray)
            white_palette.setColor(QPalette.ToolTipBase, Qt.white)
            white_palette.setColor(QPalette.ToolTipText, Qt.black)
            white_palette.setColor(QPalette.Text, Qt.black)
            white_palette.setColor(QPalette.Button, Qt.lightGray)
            white_palette.setColor(QPalette.ButtonText, Qt.black)
            self.setPalette(white_palette)

            self.editor_email_details.verticalScrollBar().setStyleSheet(self.scrollbar_stylesheet_white)
            self.editor_email_details.horizontalScrollBar().setStyleSheet(self.scrollbar_stylesheet_white)
            self.input_text.verticalScrollBar().setStyleSheet(self.scrollbar_stylesheet_white)
            self.input_text.horizontalScrollBar().setStyleSheet(self.scrollbar_stylesheet_white)
            self.text_editor.verticalScrollBar().setStyleSheet(self.scrollbar_stylesheet_white)
            self.text_editor.horizontalScrollBar().setStyleSheet(self.scrollbar_stylesheet_white)
            self.output_text.verticalScrollBar().setStyleSheet(self.scrollbar_stylesheet_white)
            self.output_text.horizontalScrollBar().setStyleSheet(self.scrollbar_stylesheet_white)

            self.color_API = "#f8f8f8"
            self.color_API_text = "#1d1c1d"
            self.color_API_border = "#b6b5b6"
            self.api_key_input.setStyleSheet(f"""
                  QLineEdit {{
                      background-color: {self.color_API};
                      color: {self.color_API_text};
                      font-size: 10pt;
                      min-height: {27 * self.scaling_factor_height}px;
                      border: {1 * self.scaling_factor_width}px solid {self.color_API_border};
                      border-radius: {2 * self.scaling_factor_width}px;
                  }}
              """)


            self.color_API_button_background =  "#ffffff"
            self.color_API_button_background_active = "#006b4f"
            if not self.api_key_fixed:
                self.color_API_temp_white = self.color_API_button_background_active
                self.colour_API_temp2_white = "#ffffff"
            else:
                self.color_API_temp_white = self.color_API_button_background
                self.colour_API_temp2_white = "#000000"
            self.fix_api_key_button.setStyleSheet(f"""
                            QPushButton {{
                                background-color: {self.color_API_temp_white};
                                border-color: #0f1c19;
                                color: {self.colour_API_temp2_white};
                                min-height: {27 * self.scaling_factor_height}px;
                                font: 10.5pt Arial;
                            }}
                        """)

            self.color_toggle_always_on_top_button_background = "#ffffff"
            self.color_toggle_always_on_top_button_background_active = "#006b4f"
            if self.windowFlags() & Qt.WindowStaysOnTopHint:
                self.color_toggle_always_on_top_button_background_temp_white = self.color_toggle_always_on_top_button_background_active
                self.colour_toggle_always_on_top_button_background_temp2_white = "#ffffff"
            else:
                self.color_toggle_always_on_top_button_background_temp_white = self.color_toggle_always_on_top_button_background
                self.colour_toggle_always_on_top_button_background_temp2_white = "#000000"
            self.toggle_always_on_top_button.setStyleSheet(
                f"""
                          QPushButton {{
                              background-color: {self.color_toggle_always_on_top_button_background_temp_white};
                              border-color: #0f1c19;
                              color: {self.colour_toggle_always_on_top_button_background_temp2_white};
                              min-height: {27 * self.scaling_factor_height}px;
                              font: 10.5pt Arial;
                          }}
                          """
            )

            self.color_output_text_background = "#f8f8f8"
            self.color_output_text_border = "#b6b5b6"
            self.color_output_text_font = "#1d1c1d"
            self.output_text.setStyleSheet(
                f"""
                        QTextEdit {{
                            background-color: {self.color_output_text_background};
                            color: {self.color_output_text_font};
                            font-size: 11.5pt;
                            border: {1 * self.scaling_factor_width}px solid {self.color_output_text_border};
                            border-radius: {4 * self.scaling_factor_width}px;
                            padding: {5 * self.scaling_factor_height}px;
                        }}
                        """
            )
            self.color_stop_button_background_white= "#006b4f"
            self.stop_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.color_stop_button_background_white};
                    border-color: #127287;
                    color: #FFFFFF;
                    min-height: {27 * self.scaling_factor_height}px;
                    font: 11.5pt Arial;
                }}
                """
            )

            self.color_copy_button_background = "#006b4f"
            self.color_copy_button_background_active = "#004734"
            self.copy_button.setStyleSheet(
                f"""
                                    background-color: {self.color_copy_button_background};
                                    border-color: #127287;
                                    color: #FFFFFF;
                                    min-height: {27 * self.scaling_factor_height}px;
                                    font: 11.5pt Arial;
                                    """)

            self.color_generate_button = "#006b4f"
            self.color_generate_button_active = "#004734"
            self.run_regenerate_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.color_generate_button};
                    border-color: #127287;
                    color: #FFFFFF;
                    min-height: {27 * self.scaling_factor_height}px;
                    font: 11.5pt Arial;
                }}
                """
            )

            self.color_output_text_responce_font = "#000000"

            self.color_email_details_background = "#f8f8f8"
            self.color_email_details_border = "#b6b5b6"
            self.color_email_details_font = "#1d1c1d"
            self.editor_email_details.setStyleSheet(
                f"""
                            QTextEdit {{
                                background-color: {self.color_email_details_background};
                                color: {self.color_email_details_font};
                                font-size: 11.5pt;
                                border: {1 * self.scaling_factor_width}px solid {self.color_email_details_border};
                                border-radius: {4 * self.scaling_factor_width}px;
                                padding: {5 * self.scaling_factor_height}px;
                            }}
                            """
            )

            self.color_rewrite_level_selector_font = "#000000"
            self.rewrite_level_selector_label.setStyleSheet(
                f"""
                                   font: 10.5pt Arial;
                                   color: {self.color_rewrite_level_selector_font};
                                   """
            )

            self.color_mode_detail_background = "#c2e3d7"
            self.color_mode_style_font = "#000000"
            self.mode_detail.setStyleSheet(
                f"""
                QLabel {{
                    background-color: {self.color_mode_detail_background};
                    color: {self.color_mode_style_font};
                    font: 10.5pt Arial;
                    border-radius: {4 * self.scaling_factor_width}px;
                    }}
                """
            )

            self.color_rephrase_options_dropdown_background = "#006b4f"
            self.color_rephrase_options_dropdown_font = "#FFFFFF"

            self.rephrase_options_dropdown.setStyleSheet(
                f"""
                background-color: {self.color_rephrase_options_dropdown_background};
                border-color: #000000;
                color: {self.color_rephrase_options_dropdown_font};
                min-height: {27 * self.scaling_factor_height}px;
                font: 10.5pt Arial;
                """
            )

            self.rewrite_level_selector_background = "#259476"
            self.rewrite_level_selector_border = "#000000"
            self.rewrite_level_selector_font = "#FFFFFF"
            self.rewrite_level_selector.setStyleSheet(
                f"""
                background-color: {self.rewrite_level_selector_background};
                border-color: {self.rewrite_level_selector_border};
                color: {self.rewrite_level_selector_font};
                min-height: {17 * self.scaling_factor_height}px;
                font: 10.5pt Arial;
                padding-left: 4px;
                """
            )

            self.color_input_text_background = "#f8f8f8"
            self.color_input_text_font = "#000000"
            self.color_input_text_border = "#b6b5b6"
            self.input_text.setStyleSheet(
                f"""
                QTextEdit {{
                    background-color: {self.color_input_text_background};
                    color: {self.color_input_text_font};
                    font-size: 11.5pt;
                    border: {1 * self.scaling_factor_width}px solid {self.color_input_text_border};
                    border-radius: {4 * self.scaling_factor_width}px;
                    padding: {5 * self.scaling_factor_height}px;
                }}
                """
            )

            self.color_text_editor_background = "#f8f8f8"
            self.color_text_editor_font = "#000000"
            self.color_text_editor_border = "#b6b5b6"
            self.text_editor.setStyleSheet(
                f"""
                        QTextEdit {{
                            background-color: {self.color_text_editor_background};
                            color: {self.color_text_editor_font};
                            font-size: 11.5pt;
                            border: {1 * self.scaling_factor_width}px solid {self.color_text_editor_border};
                            border-radius: {4 * self.scaling_factor_width}px;
                            padding: {5 * self.scaling_factor_height}px;
                        }}
                        """
            )

            self.color_proxy_server_input_background = "#f8f8f8"
            self.color_proxy_server_input_font = "#000000"
            self.color_proxy_server_input_border = "#b6b5b6"
            self.proxy_server_input.setStyleSheet(
                f"""
                        QLineEdit {{
                            background-color: {self.color_proxy_server_input_background};
                            color: {self.color_proxy_server_input_font};
                            font-size: 11pt;
                            border: {1 * self.scaling_factor_width}px solid {self.color_proxy_server_input_border};
                            border-radius: {2 * self.scaling_factor_width}px;
                        }}
                        """
            )

            self.color_proxy_port_input_background = "#f8f8f8"
            self.color_proxy_port_input_font = "#000000"
            self.color_proxy_port_input_border = "#b6b5b6"
            self.proxy_port_input.setStyleSheet(
                f"""
                      QLineEdit {{
                          background-color: {self.color_proxy_port_input_background};
                          color: {self.color_proxy_port_input_font};
                          font-size: 11pt;
                          border: {1 * self.scaling_factor_width}px solid {self.color_proxy_port_input_border};
                          border-radius: {2 * self.scaling_factor_width}px;
                      }}
                      """
            )

            self.save_proxy_button_background = "#006b4f"
            self.save_proxy_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.save_proxy_button_background};
                    border-color: #127287;
                    color: #FFFFFF;
                    min-height: {27 * self.scaling_factor_height}px;
                    font: 11.5pt Arial;
                }}
                """
            )

            self.color_language_dropdown_background = "#006b4f"
            self.from_language_dropdown.setStyleSheet(
                f"""
                    background-color: {self.color_language_dropdown_background};
                    border-color: #000000;
                    color: #FFFFFF;
                    min-height: {20 * self.scaling_factor_height}px;
                    font: 10.5pt Arial;
                """
            )
            self.to_language_dropdown.setStyleSheet(
                f"""
                    background-color: {self.color_language_dropdown_background};
                    border-color: #000000;
                    color: #FFFFFF;
                    min-height: {20 * self.scaling_factor_height}px;
                    font: 10.5pt Arial;
                """
            )

            self.switch_control_label.setStyleSheet(
                f"""
                      QLabel {{
                          color: #000000;
                          font-size: 10pt;
                      }}
                      """
            )

            self.clear_button.setStyleSheet(
                f"""
                background-color: {self.color_copy_button_background};
                border-color: #127287;
                color: #FFFFFF;
                min-height: {27 * self.scaling_factor_height}px;
                font: 11.5pt Arial;
                """)

            self.group_box.setStyleSheet(
                f"""QGroupBox {{ 
                 background-color: #f8f8f8; 
                 border: {int(1 * self.scaling_factor_height)}px solid #b6b5b6; 
                 border-radius: {int(4 * self.scaling_factor_height)}px; 
                 font-size: 10pt;}}""")

            self.stylesheet_radio_buttons = f"""QRadioButton {{ 
                                                 font-size: 10.5pt;
                                                 color: #000000;}}

                                                 QRadioButton::indicator::unchecked {{
                                                 background-color: #96c5b8;
                                                 width: {int(6 * self.scaling_factor_height)}px;
                                                 height: {int(6 * self.scaling_factor_height)}px;
                                                 border-radius: {int(6 * self.scaling_factor_height)}px;
                                                 border: {int(3 * self.scaling_factor_height)}px solid #96c5b8;
                                                 }}

                                                 QRadioButton::indicator::checked {{
                                                 background-color: #272a29;
                                                 width: {int(6 * self.scaling_factor_height)}px;
                                                 height: {int(6 * self.scaling_factor_height)}px;
                                                 border-radius: {int(6 * self.scaling_factor_height)}px;
                                                 border: {int(3 * self.scaling_factor_height)}px solid #96c5b8;
                                                 }}"""
            self.radio_rephrase.setStyleSheet(self.stylesheet_radio_buttons)
            self.radio_debug_code.setStyleSheet(self.stylesheet_radio_buttons)
            self.radio_summarize.setStyleSheet(self.stylesheet_radio_buttons)
            self.radio_translate.setStyleSheet(self.stylesheet_radio_buttons)
            self.radio_email.setStyleSheet(self.stylesheet_radio_buttons)
            self.radio_explain.setStyleSheet(self.stylesheet_radio_buttons)
            self.radio_manual_prompts.setStyleSheet(self.stylesheet_radio_buttons)
            self.radio_grammer_checker.setStyleSheet(self.stylesheet_radio_buttons)

            self.proxy_checkbox_label.setStyleSheet(
                f"""
                        QLabel {{
                            color: #000000;
                            font-size: 10pt;
                        }}
                        """
            )


    def toggle_debug_code(self, state):
        self.keep_previous_checkbox.setHidden(state)
        self.keep_previous_checkbox.setChecked(False)

    def toggle_custom_prompt(self, state):
        self.keep_previous_checkbox.setHidden(state)
        self.keep_previous_checkbox.setChecked(False)

    def toggle_grammer_checker(self, state):
        self.keep_previous_checkbox.setHidden(state)
        self.keep_previous_checkbox.setChecked(False)

    def on_keep_previous_changed(self, state):
        self.keep_previous = state
    def update_rewrite_level(self, new_level):
        self.rewrite_level = new_level

    def toggle_email_details(self, state):
        self.checkbox_reply_to_email.setHidden(not state)
        if not self.radio_email.isChecked():
            self.editor_email_details.setHidden(not state)
            self.checkbox_reply_to_email.setChecked(False)

    def toggle_email_details_checkbox(self, state):
        self.editor_email_details.setHidden(not state)

    def sag(self):

        self.mode_detail.setText("Rewrites text in a reliable manner to maintain meaning")

    def set_rephrase_prompt(self, index):
        options = ["Standard", "Fluency", "Formal", "Academic", "Simple", "Creative", "Expand", "Shorten"]
        self.selected_option = options[index]

        if self.selected_option == "Standard":
            self.additionalPrompt = "Rewrite the following passage while preserving its original meaning and essence:\n\n"
            self.mode_detail.setText("  Rewrites text in a reliable manner to maintain meaning  ")

        elif self.selected_option == "Fluency":
            self.additionalPrompt = "Enhance the text for readability and ensure it flows smoothly, free from grammatical errors:\n\n"
            self.mode_detail.setText("  Ensures text is readable and free of error  ")

        elif self.selected_option == "Formal":
            self.additionalPrompt = "Transform the provided text into a more refined and professional version suitable for formal settings:\n\n"
            self.mode_detail.setText("  Presents text in a more sophisticated and professional way  ")

        elif self.selected_option == "Academic":
            self.additionalPrompt = "Paraphrase the academic content with a focus on scholarly language, maintaining accuracy and depth:\n\n"
            self.mode_detail.setText("  Rewrites academic text in a more scholarly way  ")

        elif self.selected_option == "Simple":
            self.additionalPrompt = "Simplify the given text to make it accessible to a broader audience while retaining its core meaning:\n\n"
            self.mode_detail.setText("  Presents text in a way most people can understand  ")

        elif self.selected_option == "Creative":
            self.additionalPrompt = ("Encourage a complete reimagination of the following passage creatively, allowing for creative expression that might alter the original meaning:\n\n")
            self.mode_detail.setText("  Expresses ideas in a completely new way that may change the meaning  ")

        elif self.selected_option == "Expand":
            self.additionalPrompt = "Elaborate on the provided text by adding depth and detail. Increase sentence length while ensuring coherence and retaining the original message:\n\n"
            self.mode_detail.setText("  Adds more detail and depth to increase sentence length  ")

        elif self.selected_option == "Shorten":
            self.additionalPrompt = "Concise the given text by eliminating unnecessary words, maintaining clarity and delivering a succinct yet comprehensive message:\n\n"
            self.mode_detail.setText("  Strips away extra words to provide a clear message  ")

    def toggle_language_dropdowns(self, checked):
        self.from_language_dropdown.setHidden(not checked)
        self.to_language_dropdown.setHidden(not checked)
        self.from_language_label.setHidden(not checked)
        self.to_language_label.setHidden(not checked)

    def toggle_radio_rephrase(self, checked):
        self.rephrase_options_dropdown.setHidden(not checked)
        self.mode_detail.setHidden(not checked)
        self.rewrite_level_selector_label.setHidden(not checked)
        self.rewrite_level_selector.setHidden(not checked)

    def toggle_manual_prompt_input(self, checked):
        self.input_text.setHidden(not checked)
        if checked:
            self.input_text.setFocus()

    def toggle_api_key(self):
        if not self.api_key_input.text() == "":
            if self.api_key_fixed is False:
                self.api_key_fixed = True
                self.api_key_input.setReadOnly(False)
                self.api_key_input.setText(self.api_key)
                self.api_key_input.setStyleSheet(f"""
                                            QLineEdit {{
                                                background-color: {self.color_API};
                                                color: {self.color_API_text};
                                                font-size: 10pt;
                                                min-height: {27 * self.scaling_factor_height}px;
                                                border: {1 * self.scaling_factor_width}px solid #565856;
                                                border-radius: {2 * self.scaling_factor_width}px;
                                            }}
                                        """)
                self.fix_api_key_button.setStyleSheet(
                    f"""
                    QPushButton {{
                        background-color: {self.color_API_button_background};
                        border-color: #0f1c19;
                        color: #000000;
                        min-height: {27 * self.scaling_factor_height}px;
                        font: 10.5pt Arial;
                    }}
                    """
                )
            else:
                self.api_key_fixed = False
                self.api_key_input.setReadOnly(True)
                self.api_key = self.api_key_input.text()
                self.api_key_input.setText(f"{'*' * len(self.api_key)}")
                self.api_key_input.setStyleSheet(f"""
                            QLineEdit {{
                                background-color: {self.color_API};
                                color: {self.color_API_text};
                                font-size: 10pt;
                                min-height: {27 * self.scaling_factor_height}px;
                                border: {1 * self.scaling_factor_width}px solid #565856;
                                border-radius: {2 * self.scaling_factor_width}px;
                            }}
                        """)

                self.fix_api_key_button.setStyleSheet(
                    f"""
                    QPushButton {{
                        background-color: {self.color_API_button_background_active};
                        border-color: #127287;
                        color: #FFFFFF;
                        min-height: {27 * self.scaling_factor_height}px;
                        font: 10.5pt Arial;
                    }}
                    """
                )


        else:
            self.api_key = self.api_key_input.text()
            self.fix_api_key_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.color_API_button_background};
                    border-color: #0f1c19;
                    color: #000000;
                    min-height: {27 * self.scaling_factor_height}px;
                    font: 10.5pt Arial;
                }}
                """
            )
        self.save_config_to_file()
    def toggle_proxy_settings(self, state):
        self.proxy_server_input.setHidden(not state)
        self.proxy_port_input.setHidden(not state)
        self.save_proxy_button.setHidden(not state)

    def load_config_from_file(self):
        if os.path.exists(self.CONFIG_FILE_PATH):
            with open(self.CONFIG_FILE_PATH, 'r') as config_file:
                config_data = json.load(config_file)
                saved_api_key = config_data.get('api_key')
                saved_proxy_enabled = config_data.get('proxy_enabled', False)
                self.saved_proxy_server = config_data.get('proxy_server', '')
                self.saved_proxy_port = config_data.get('proxy_port', '')
                from_language = config_data.get('from_language', '')
                to_language = config_data.get('to_language', '')

                from_index = self.from_language_dropdown.findText(from_language)
                to_index = self.to_language_dropdown.findText(to_language)

                if saved_api_key:
                    self.api_key = saved_api_key
                    self.api_key_input.setText(f"{'*' * len(self.api_key)}")
                    self.api_key_input.setStyleSheet(f"""
                                                QLineEdit {{
                                                    background-color: {self.color_API};
                                                    color: {self.color_API_text};
                                                    font-size: 10pt;
                                                    min-height: {27 * self.scaling_factor_height}px;
                                                    border: {1 * self.scaling_factor_width}px solid #565856;
                                                    border-radius: {2 * self.scaling_factor_width}px;
                                                }}
                                            """)
                if saved_proxy_enabled:
                    self.proxy_checkbox_isEnabled = True
                    self.proxy_server_input.setText(self.saved_proxy_server)
                    self.proxy_port_input.setText(self.saved_proxy_port)
                    self.toggle_proxy_settings(False)
                else:
                    self.proxy_checkbox_isEnabled = False

                if from_index != -1:
                    self.from_language_dropdown.setCurrentIndex(from_index)

                if to_index != -1:
                    self.to_language_dropdown.setCurrentIndex(to_index)

    def save_config_to_file(self):
        api_key_save = self.api_key
        proxy_enabled = self.proxy_checkbox_isEnabled
        proxy_server = self.proxy_server_input.text().strip()
        proxy_port = self.proxy_port_input.text().strip()

        from_language = self.from_language_dropdown.currentText()
        to_language = self.to_language_dropdown.currentText()

        config_data = {
            'api_key': api_key_save,
            'proxy_enabled': proxy_enabled,
            'proxy_server': proxy_server,
            'proxy_port': proxy_port,
            'from_language': from_language,
            'to_language': to_language
        }

        with open(self.CONFIG_FILE_PATH, 'w') as config_file:
            json.dump(config_data, config_file)

    def save_proxy_settings(self):
        proxy_server = self.proxy_server_input.text().strip()
        proxy_port = self.proxy_port_input.text().strip()
        if not proxy_server or not proxy_port:
            self.show_error_message("Please enter both proxy server and port.")
            self.proxy_checkbox.initial_toggle()
            return

        self.save_config_to_file()
        self.toggle_proxy_settings(False)

    def configure_proxy(self):

        if not self.proxy_checkbox_isEnabled:
            self.proxy_checkbox_isEnabled = True
            self.toggle_proxy_settings(True)
            proxy_server = self.proxy_server_input.text().strip()
            proxy_port = self.proxy_port_input.text().strip()

            try:
                self.client = openai.OpenAI(api_key=self.api_key,
                                            http_client=httpx.Client(proxies=f"http://{proxy_server}:{proxy_port}"))
            except:
                pass
        else:
            self.proxy_checkbox_isEnabled = False
            try:
                self.client = openai.OpenAI(api_key=self.api_key)
            except:
                pass
            self.toggle_proxy_settings(False)
        self.save_config_to_file()

    def run_regenerate_text(self):

        from_language = self.from_language_dropdown.currentText()
        to_language = self.to_language_dropdown.currentText()

        if self.api_key_fixed:
            self.set_api_key()

        if self.radio_rephrase.isChecked():
            if self.rewrite_level > 1:
                prompt = f" Generate {self.rewrite_level} different rewrites of the following text.\n"
            else:
                prompt = ""
            prompt += self.additionalPrompt
        elif self.radio_debug_code.isChecked():
            prompt = "Help me to solve the issue and write updated code. Also explain what the issue was and what you did to fix it.\n\nCode:\n"
        elif self.radio_explain.isChecked():
            prompt = "Could you offer a clear and concise explanation for the following text, making it both simple and comprehensive?:\n"
        elif self.radio_summarize.isChecked():
            prompt = ("Could you please provide a concise and comprehensive summary of the given text? "
                      "The summary should capture the main points and key details of the text while conveying the author's intended meaning accurately. "
                      "Please ensure that the summary is well-organized and easy to read, with clear headings and subheadings to guide the reader "
                      "through each section. The length of the summary should be appropriate to capture the main points and key details of the text, "
                      "without including unnecessary information or becoming overly long. The response should not exceed the length of the main given text:\n")
        elif self.radio_translate.isChecked():
            prompt = f"Can you help me translate the following word/phrase/sentence from {from_language} to {to_language}?:\n"
        elif self.radio_email.isChecked():
            prompt = "Write a professional and productive reply to this email. Keeping it concise:\n\n"
            if self.checkbox_reply_to_email.isChecked():
                prompt = prompt[:-4]
                prompt += ".\nTake into account the following aspects when formulating the reply:\n\n"
                prompt += self.editor_email_details.toPlainText()
                prompt += "\n\nEmail:\n"
        elif self.radio_manual_prompts.isChecked():
            prompt = self.input_text.toPlainText()
        elif self.radio_grammer_checker.isChecked():
            prompt = ("1. I want you to be a grammar checker similar to QuillBot and Grammarly.\n"
                      "2. You should be capable of identifying and correcting grammatical errors in a given text.)\n"
                      "3. You should provide suggestions for corrections, offering alternatives to improve sentence structure, word choice, and overall grammar.\n"
                      "4. The output should maintain the context and meaning of the original text while addressing the identified errors.\n"
                      "5. Produce only plain text output, without any additional elements.\n\n"
                      "text:\n")

        if self.checkbox_clipboard.isChecked():
            clipboard = QApplication.clipboard()
            prompt += clipboard.text()

        elif not self.checkbox_clipboard.isChecked():
            prompt += self.text_editor.toPlainText()

        elif not self.checkbox_clipboard.isChecked() and self.radio_manual_prompts.isChecked():
            prompt = self.input_text.toPlainText() + ":\n\n" + self.text_editor.toPlainText()

        self.save_config_to_file()

        self.run_regenerate_button.setEnabled(False)
        self.run_regenerate_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.color_generate_button_active};
                border-color: #4b4b4b;
                color: #a5a5a5;
                min-height: {27 * self.scaling_factor_height}px;
                font: 11.5pt Arial;
            }}
            """
        )
        self.run_regenerate_button.setText("Generating your response...")

        self.generate_response(prompt)

        self.run_regenerate_button.setEnabled(True)
        self.run_regenerate_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.color_generate_button};
                border-color: #127287;
                color: #FFFFFF;
                min-height: {27 * self.scaling_factor_height}px;
                font: 11.5pt Arial;
            }}
            """
        )
        self.run_regenerate_button.setText("Generate")

    def toggle_clipboard_text(self, state):
        self.text_editor.setHidden(state == Qt.Checked)

    def toggle_always_on_top(self):
        if self.windowFlags() & Qt.WindowStaysOnTopHint:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
            self.toggle_always_on_top_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.color_toggle_always_on_top_button_background};
                    border-color: #0f1c19;
                    color: #000000;
                    min-height: {27 * self.scaling_factor_height}px;
                    font: 10.5pt Arial;
                }}
                """
            )
        else:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            self.toggle_always_on_top_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.color_toggle_always_on_top_button_background_active};
                    border-color: #0f1c19;
                    color: #FFFFFF;
                    min-height: {27 * self.scaling_factor_height}px;
                    font: 10.5pt Arial;
                }}
                """
            )
        self.show()

    def set_api_key(self):
        self.api_key = self.api_key_input.text()
        if self.api_key:
            e = openai.OpenAIError
            self.show_api_key_error_alert(str(e))
        else:
            self.show_api_key_error_alert("Please enter a valid API key.")
        self.save_config_to_file()

    def get_num_hyphens(self):
        font_metrics = QFontMetrics(self.output_text.font())
        hyphen_width = font_metrics.width("-")
        editor_width = self.output_text.width()
        buffer = 50
        return (editor_width - buffer) // hyphen_width
    def generate_response(self, prompt):
        try:
            self.stop_generation = False

            block_color_status = 0

            if self.proxy_checkbox_isEnabled:
                self.proxy_checkbox_isEnabled = False
            else:
                self.proxy_checkbox_isEnabled = True

            self.configure_proxy()
            self.toggle_proxy_settings(False)

            if self.keep_previous_checkbox.isChecked() and self.output_text.toPlainText() != "":

                self.output_text.setHtml(self.output_text.toHtml() + f"<br><span style='color: #32779a;'>{'-' * self.get_num_hyphens()}</span><br><br>")
                full_text = self.output_text.toHtml()
            else:
                full_text = ""

            for part in self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="gpt-3.5-turbo",
                    stream=True,
            ):

                if self.stop_generation:
                    break
                new_content = part.choices[0].delta.content or ""
                if block_color_status == 0:
                    if new_content.startswith("```"):
                        block_color_status = 1
                        new_content = new_content.replace('\n', '<br>')
                        full_text += '<font color="green">' + new_content + '</font>'
                    elif new_content.startswith("`"):
                        new_content = new_content.replace('\n', '<br>')
                        full_text += '<font color="green">' + new_content + '</font>'
                    else:
                        new_content = new_content.replace('\n', '<br>')
                        full_text += f'<font color={self.color_output_text_responce_font}>' + new_content + '</font>'

                elif block_color_status == 1:
                    if new_content.startswith("``"):
                        block_color_status = 0
                    new_content = new_content.replace('\n', '<br>')
                    full_text += '<font color="green">' + new_content + '</font>'

                if new_content and not self.radio_grammer_checker.isChecked():
                    self.output_text.setHtml(full_text)

                self.output_text.verticalScrollBar().setValue(self.output_text.verticalScrollBar().maximum())

                QApplication.processEvents()

            code_blocks = re.findall(r"```([\s\S]+?)``", full_text)

            html_code = ''.join(code_blocks)

            soup = BeautifulSoup(html_code, 'html.parser')

            for br_tag in soup.find_all('br'):
                br_tag.replace_with('\n')

            plain_text = soup.get_text()

            language = ""
            for code_block in code_blocks:

                start_delimiter = '<font color="green">'
                end_delimiter = '</font>'

                start_index = code_block.find(start_delimiter)
                end_index = code_block.find(end_delimiter, start_index + len(start_delimiter))

                if start_index != -1 and end_index != -1:
                    language = code_block[start_index + len(start_delimiter):end_index]

                if language == "":
                    language = "python"

                lexer = get_lexer_by_name(language, stripall=True)
                if self.theme_flag == 1:
                    formatter = HtmlFormatter(style="native", noclasses=True, nobackground=True)
                else:
                    formatter = HtmlFormatter(style="default", noclasses=True, nobackground=True)


                highlighted_code = highlight(plain_text, lexer, formatter)
                try:
                    updated_html = full_text.replace('``</font><font color="green">`', '```')
                except:
                    updated_html = full_text

                full_text = updated_html.replace(f"```{code_block}```", highlighted_code)

            full_text = full_text.replace('<br><br></font><font color="green">',
                                          '<br></font><font color="green">')
            full_text = full_text.replace('</font><font color="green"><br><br>',
                                          '</font><font color="green">')

            if self.radio_grammer_checker.isChecked():
                soup = BeautifulSoup(full_text, 'html.parser')
                full_text_plain = soup.get_text()
                if self.checkbox_clipboard.isChecked():
                    original_text = QApplication.clipboard().text()
                else:
                    original_text = self.text_editor.toPlainText()
                if self.theme_flag == 1:
                    html_diff = generate_html_diff(original_text, full_text_plain, dark_mode=True)
                else:
                    html_diff = generate_html_diff(original_text, full_text_plain, dark_mode=False)
                self.output_text.setHtml(html_diff)
            else:
                self.output_text.setHtml(full_text)

            QApplication.processEvents()

        except openai.OpenAIError as err:
            self.output_text.setPlainText(str(err))

    def copy_text(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.output_text.toPlainText())

        self.copy_button.setEnabled(False)
        self.copy_button.setText("Copied!")
        self.copy_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.color_copy_button_background_active};
                border-color: #127287;
                color: #FFFFFF;
                min-height: {27 * self.scaling_factor_height}px;
                font: 11.5pt Arial;
            }}
            """
        )
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.clear_copied)
        self.timer.start(2000)

######################################################
    def clear_text(self):
        self.output_text.setPlainText("")
#######################################################

    def clear_copied(self):
        self.timer.stop()
        self.copy_button.setEnabled(True)
        self.copy_button.setText("Copy response")
        self.copy_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.color_copy_button_background};
                border-color: #127287;
                color: #FFFFFF;
                min-height: {27 * self.scaling_factor_height}px;
                font: 11.5pt Arial;
            }}
            """
        )
    def stop_generation_process(self):
        self.stop_generation = True

    def show_api_key_error_alert(self, error_message):
        alert = QMessageBox()
        alert.setWindowTitle("API Key Error")
        alert.setIcon(QMessageBox.Critical)
        alert.setText("Error setting API key!")
        alert.setStandardButtons(QMessageBox.Ok)
        alert.exec_()

    def show_error_message(self, message):
        error_message = QMessageBox()
        error_message.setWindowTitle("Error")
        error_message.setIcon(QMessageBox.Critical)
        error_message.setText(message)
        error_message.exec_()

    def set_widget_palette_color(self, widget, color):
        palette = widget.palette()
        palette.setColor(QPalette.WindowText, QColor(color))
        widget.setPalette(palette)

    def update_program(self):
        repo_owner = "AITwinMinds"
        repo_name = "ChatGPT-UI"

        # Construct the GitHub API URL
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"

        try:
            response = requests.get(api_url)
            data = response.json()

            latest_version = data['tag_name']
            # Compare the latest version with your current version
            current_version = "v3.0"  # Replace with your actual version
            if latest_version > current_version:
                QMessageBox.information(self, 'Update Available', f'New version {latest_version} is available!')
            else:
                QMessageBox.information(self, 'No Updates', 'You are using the latest version.')

        except Exception as e:
            print(f"Error checking for updates: {e}")

    def give_star(self):
        repo_owner = "AITwinMinds"
        repo_name = "ChatGPT-UI"

        github_url = f"https://github.com/{repo_owner}/{repo_name}"

        # Open the user's default web browser to the GitHub repository URL
        QDesktopServices.openUrl(QUrl(github_url))

    def about_app(self):
        about_dialog = AboutDialog()
        about_dialog.exec_()
def main():
    app = QApplication(sys.argv)

    app_icon = QIcon(QPixmap.fromImage(QtGui.QImage.fromData(ICON_DATA)))
    app.setWindowIcon(app_icon)

    app.setStyle('Fusion')

    ui = GPTUI()

    ui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
