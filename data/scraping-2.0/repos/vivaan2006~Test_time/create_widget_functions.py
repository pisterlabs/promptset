from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QFontMetrics
from PyQt5.QtWidgets import QSpacerItem, QGraphicsDropShadowEffect, QSizePolicy, QLabel, QListWidgetItem, QTabWidget, QTabBar, QStylePainter, QStyleOptionTab, QStyle, QComboBox, QVBoxLayout, QHBoxLayout, QScrollArea, QTextEdit, QLineEdit, QPushButton, QWidget, QListWidget

import openai

openai.api_key = "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" # put the key here remove before pushing

class create_QComboBox:
    def __init__(self, container, x_coordinate, y_coordinate, width, length):
        # Creates and associates QComboBox to specified container
        if container == "points_tab":
            self.QComboBox = QtWidgets.QComboBox(self.points_tab)

        # Geometry of QComboBox is specified by the passed function parameters
        self.QComboBox.setGeometry(QtCore.QRect(x_coordinate, y_coordinate, width, length))
        return self.QComboBox

class create_QCheckBox():
    def __init__(self, container, x_coordinate, y_coordinate, width, length):
        if container == "dashboard_tab":
            self.QCheckBox = QtWidgets.QCheckBox(self.dashboard_tab)
        elif container == "upcoming_events_tab":
            self.QCheckBox = QtWidgets.QCheckBox(self.upcoming_events_tab)
        elif container == "event":
            self.QCheckBox = QtWidgets.QCheckBox(self.event_object)
        self.QCheckBox.resize(width, length)
        self.QCheckBox.move(x_coordinate, y_coordinate)
        return self.QCheckBox

class create_QCalendar():
    def __init__(self, container, x_coordinate, y_coordinate, width, length):
        if container == "upcoming_events_tab":
            self.QCalender = QtWidgets.QCalendarWidget(self.upcoming_events_tab)
        elif container == "admin_events_tab":
            self.QCalender = QtWidgets.QCalendarWidget(self.admin_events_tab)
        self.QCalender.setGeometry(x_coordinate, y_coordinate, width, length)
        return self.QCalender

class create_QLabel():
    def __init__(self, container, object_name, text, x_coordinate, y_coordinate, width, length):
        # Creates and associates QLabel to specified container
        if container == "login_widget_container":
            self.QLabel = QtWidgets.QLabel(self.login_widget_container)
        elif container == "central_widget":
            self.QLabel = QtWidgets.QLabel(self.central_widget)
        elif container == "dashboard_tab":
            self.QLabel = QtWidgets.QLabel(self.dashboard_tab)
        elif container == "upcoming_events_tab":
            self.QLabel = QtWidgets.QLabel(self.upcoming_events_tab)
        elif container == "points_tab":
            self.QLabel = QtWidgets.QLabel(self.points_tab)
        elif container == "rewards_tab":
            self.QLabel = QtWidgets.QLabel(self.rewards_tab)
        elif container == "student_profile_tab":
            self.QLabel = QtWidgets.QLabel(self.student_profile_tab)
        elif container == "slideshow_description_groupbox":
            self.QLabel = QtWidgets.QLabel(self.slideshow_description_groupbox)
        elif container == "event":
            self.QLabel = QtWidgets.QLabel(self.event_object)
        elif container == "report_frame":
            self.QLabel = QtWidgets.QLabel(self.report_frame)
        elif container == "forgot_password_frame":
            self.QLabel = QtWidgets.QLabel(self.forgot_password_frame)
        elif container == "student_account_frame":
            self.QLabel = QtWidgets.QLabel(self.student_account_frame)

        # Administrator

        elif container == "admin_dashboard_tab":
            self.QLabel = QtWidgets.QLabel(self.admin_dashboard_tab)
        elif container == "admin_events_tab":
            self.QLabel = QtWidgets.QLabel(self.admin_events_tab)
        elif container == "maps_tab":
            self.QLabel = QtWidgets.QLabel(self.maps_tab)
        elif container == "admin_statistics_tab":
            self.QLabel = QtWidgets.QLabel(self.admin_statistics_tab)
        elif container == "admin_student_view_tab":
            self.QLabel = QtWidgets.QLabel(self.admin_student_view_tab)
        elif container == "admin_statistics_tab":
            self.QLabel = QtWidgets.QLabel(self.admin_statistics_tab)
        elif container == "rand":
            self.QLabel = QtWidgets.QLabel(self.rand_win_gb)
        elif container == "top":
            self.QLabel = QtWidgets.QLabel(self.top_win_gb)
        elif container == "admin_output_report_frame":
            self.QLabel = QtWidgets.QLabel(self.admin_output_report_frame)
        elif container == "admin_student_support_tab":
            self.QLabel = QtWidgets.QLabel(self.admin_student_support_tab)
        elif container == "create_rewards_frame":
            self.QLabel = QtWidgets.QLabel(self.create_rewards_frame)
        elif container == "admin_account_frame":
            self.QLabel = QtWidgets.QLabel(self.admin_account_frame)
        self.QLabel.setWordWrap(True)
        self.QLabel.setObjectName(object_name)
        self.QLabel.setText(text)
        # Geometry of QLabel is specified by the passed function parameters
        self.QLabel.setGeometry(QtCore.QRect(x_coordinate, y_coordinate, width, length))
        return self.QLabel

class create_QLineEdit():
    def __init__(self, container, object_name, read_only, x_coordinate, y_coordinate, width, length):
        # Creates and associates QLabel to specified container
        if container == "login_widget_container":
            self.QLineEdit = QtWidgets.QLineEdit(self.login_widget_container)
        elif container == "dashboard_tab":
            self.QLineEdit = QtWidgets.QLineEdit(self.dashboard_tab)
        elif container == "admin_dashboard_tab":
            self.QLineEdit = QtWidgets.QLineEdit(self.admin_dashboard_tab)
        elif container == "upcoming_events_tab":
            self.QLineEdit = QtWidgets.QLineEdit(self.upcoming_events_tab)
        elif container == "points_tab":
            self.QLineEdit = QtWidgets.QLineEdit(self.points_tab)
        elif container == "rewards_tab":
            self.QLineEdit = QtWidgets.QLineEdit(self.rewards_tab)
        elif container == "student_profile_tab":
            self.QLineEdit = QtWidgets.QLineEdit(self.student_profile_tab)

            # Administrator
        elif container == "admin_dashboard_tab":
            self.QLineEdit = QtWidgets.QLineEdit(self.admin_dashboard_tab)
        elif container == "admin_events_tab":
            self.QLineEdit = QtWidgets.QLineEdit(self.admin_events_tab)
        elif container == "maps_tab":
            self.QLineEdit = QtWidgets.QLineEdit(self.maps_tab)
        elif container == "admin_statistics_tab":
            self.QLineEdit = QtWidgets.QLineEdit(self.admin_statistics_tab)
        elif container == "admin_student_view_tab":
            self.QLineEdit = QtWidgets.QLineEdit(self.admin_student_view_tab)
        self.QLineEdit.setObjectName(object_name)
        # user cannot type in the boxes
        self.QLineEdit.setReadOnly(read_only)
        # Geometry of QLineEdit is specified by the passed function parameters
        self.QLineEdit.setFixedSize(width, length)
        self.QLineEdit.move(x_coordinate, y_coordinate)

        return self.QLineEdit

class create_QTextEdit():
    def __init__(self, container, object_name, read_only, x_coordinate, y_coordinate, width, length):
        # Creates and associates QLabel to specified container
        if container == "login_widget_container":
            self.QTextEdit = QtWidgets.QTextEdit(self.login_widget_container)
        elif container == "dashboard_tab":
            self.QTextEdit = QtWidgets.QTextEdit(self.dashboard_tab)
        elif container == "admin_dashboard_tab":
            self.QTextEdit = QtWidgets.QTextEdit(self.admin_dashboard_tab)
        elif container == "upcoming_events_tab":
            self.QTextEdit = QtWidgets.QTextEdit(self.upcoming_events_tab)
        elif container == "points_tab":
            self.QTextEdit = QtWidgets.QTextEdit(self.points_tab)
        elif container == "rewards_tab":
            self.QTextEdit = QtWidgets.QTextEdit(self.rewards_tab)
        elif container == "student_profile_tab":
            self.QTextEdit = QtWidgets.QTextEdit(self.student_profile_tab)

            # Administrator
        elif container == "admin_dashboard_tab":
            self.QTextEdit = QtWidgets.QTextEdit(self.admin_dashboard_tab)
        elif container == "admin_events_tab":
            self.QTextEdit = QtWidgets.QTextEdit(self.admin_events_tab)
        elif container == "maps_tab":
            self.QTextEdit = QtWidgets.QTextEdit(self.maps_tab)
        elif container == "admin_statistics_tab":
            self.QTextEdit = QtWidgets.QTextEdit(self.admin_statistics_tab)
        elif container == "admin_student_view_tab":
            self.QTextEdit = QtWidgets.QTextEdit(self.admin_student_view_tab)
        self.QTextEdit.setObjectName(object_name)
        # user cannot type in the boxes
        self.QTextEdit.setReadOnly(read_only)
        # Geometry of QLineEdit is specified by the passed function parameters
        self.QTextEdit.setFixedSize(width, length)
        self.QTextEdit.move(x_coordinate, y_coordinate)
        self.QTextEdit.setWordWrapMode(True)

        return self.QTextEdit

class create_QScrollArea():
    def __init__(self, container, object_name, layout, x_coordinate, y_coordinate, fixed_width, min_length):
        self.scrollArea_object_container = QtWidgets.QWidget()
        if container == "upcoming_events_tab":
            self.QScrollArea = QtWidgets.QScrollArea(self.upcoming_events_tab)
        elif container == "dashboard_tab":
            self.QScrollArea = QtWidgets.QScrollArea(self.dashboard_tab)
        elif container == "maps_tab":
            self.QScrollArea = QtWidgets.QScrollArea(self.maps_tab)
        elif container == "points_tab":
            self.QScrollArea = QtWidgets.QScrollArea(self.points_tab)
        elif container == "rewards_tab":
            self.QScrollArea = QtWidgets.QScrollArea(self.rewards_tab)
        elif container == "admin_statistics_tab":
            self.QScrollArea = QtWidgets.QScrollArea(self.admin_statistics_tab)
        elif container == "report_frame":
            self.QScrollArea = QtWidgets.QScrollArea(self.report_frame)
        self.QScrollArea.setFixedWidth(fixed_width)
        self.QScrollArea.setFixedHeight(min_length)
        self.QScrollArea.move(x_coordinate, y_coordinate)
        self.QScrollArea.setWidgetResizable(True)
        self.QScrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        if layout == "vertical_layout":
            self.scroll_vertical_layout = QtWidgets.QVBoxLayout(self.scrollArea_object_container)
            self.scrollArea_object_container.setLayout(self.scroll_vertical_layout)
            return [self.scrollArea_object_container, self.scroll_vertical_layout, self.QScrollArea]
        elif layout == "grid_layout":
            self.scroll_grid_layout = QtWidgets.QGridLayout(self.scrollArea_object_container)
            self.scrollArea_object_container.setLayout(self.scroll_grid_layout)
            self.QScrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            return [self.scrollArea_object_container, self.scroll_grid_layout, self.QScrollArea]

class create_QFrame():
    def __init__(self, container, object_name, orientation, x_coordinate, y_coordinate, width, length):
        if container == "login_widget_container":
            self.QFrame = QtWidgets.QFrame(self.login_widget_container)
        elif container == "dashboard_tab":
            self.QFrame = QtWidgets.QFrame(self.dashboard_tab)
        elif container == "admin_dashboard_tab":
            self.QFrame = QtWidgets.QFrame(self.admin_dashboard_tab)
        elif container == "upcoming_events_tab":
            self.QFrame = QtWidgets.QFrame(self.upcoming_events_tab)
        elif container == "points_tab":
            self.QFrame = QtWidgets.QFrame(self.points_tab)
        elif container == "rewards_tab":
            self.QFrame = QtWidgets.QFrame(self.rewards_tab)
        elif container == "student_profile_tab":
            self.QFrame = QtWidgets.QFrame(self.student_profile_tab)
        elif container == "report_frame":
            self.QFrame = QtWidgets.QFrame(self.report_frame)
        elif container == "forgot_password_frame":
            self.QFrame = QtWidgets.QFrame(self.forgot_password_frame)
        elif container == "student_account_frame":
            self.QFrame = QtWidgets.QFrame(self.student_account_frame)
            # Administrator
        elif container == "admin_dashboard_tab":
            self.QFrame = QtWidgets.QFrame(self.admin_dashboard_tab)
        elif container == "admin_events_tab":
            self.QFrame = QtWidgets.QFrame(self.admin_events_tab)
        elif container == "maps_tab":
            self.QFrame = QtWidgets.QFrame(self.maps_tab)
        elif container == "admin_statistics_tab":
            self.QFrame = QtWidgets.QFrame(self.admin_statistics_tab)
        elif container == "admin_student_view_tab":
            self.QFrame = QtWidgets.QFrame(self.admin_student_view_tab)
        elif container == "admin_output_report_frame":
            self.QFrame = QtWidgets.QFrame(self.admin_output_report_frame)
        elif container == "admin_student_support_tab":
            self.QFrame = QtWidgets.QFrame(self.admin_student_support_tab)
        elif container == "create_rewards_frame":
            self.QFrame = QtWidgets.QFrame(self.create_rewards_frame)
        elif container == "admin_account_frame":
            self.QFrame = QtWidgets.QFrame(self.admin_account_frame)
        self.QFrame.setObjectName(object_name)
        self.QFrame.setGeometry(QtCore.QRect(x_coordinate, y_coordinate, width, length))
        if orientation == "VLine":
            self.QFrame.setFrameShape(QtWidgets.QFrame.VLine)
        else:
            self.QFrame.setFrameShape(QtWidgets.QFrame.HLine)

class create_QPushButton():
    def __init__(self, container, object_name, text, icon, x_coordinate, y_coordinate, width, length):
        # Creates and associates QLabel to specified container
        if container == "login_widget_container":
            self.QPushButton = QtWidgets.QPushButton(self.login_widget_container)
        elif container == "central_widget":
            self.QPushButton = QtWidgets.QPushButton(self.central_widget)
        elif container == "student_profile_tab":
            self.QPushButton = QtWidgets.QPushButton(self.student_profile_tab)
        elif container == "rewards_tab":
            self.QPushButton = QtWidgets.QPushButton(self.rewards_tab)
        elif container == "admin_statistics_tab":
            self.QPushButton = QtWidgets.QPushButton(self.admin_statistics_tab)
        self.QPushButton.setObjectName(object_name)
        if text != "None":
            self.QPushButton.setText(text)
        if icon != "None":
            self.QPushButton.setIcon(QIcon(icon))
        # Geometry of QLineEdit is specified by the passed function parameters
        self.QPushButton.setFixedSize(width, length)
        self.QPushButton.move(x_coordinate, y_coordinate)

        return self.QPushButton

class create_horizontal_QSlider():
    def __init__(self, container, x_coordinate, y_coordinate, width, length):
        if container == "dashboard_tab":
            self.QSlider = QtWidgets.QSlider(Qt.Horizontal, self.dashboard_tab)
        self.QSlider.setGeometry(x_coordinate, y_coordinate, width, length)
        return self.QSlider

class TabBar(QTabBar):
    def tabSizeHint(self, index):
        self.setGeometry(0, 120, 180, 700)
        s = QTabBar.tabSizeHint(self, index)
        s.transpose()
        return s

    def paintEvent(self, event):
        painter = QStylePainter(self)
        opt = QStyleOptionTab()

        for i in range(self.count()):
            self.initStyleOption(opt, i)
            painter.drawControl(QStyle.CE_TabBarTabShape, opt)
            painter.save()

            s = opt.rect.size()
            s.transpose()
            r = QtCore.QRect(QtCore.QPoint(), s)
            r.moveCenter(opt.rect.center())
            opt.rect = r

            c = self.tabRect(i).center()
            painter.translate(c)
            painter.rotate(90)
            painter.translate(-c)
            painter.drawControl(QStyle.CE_TabBarTabLabel, opt)
            painter.restore()

class VerticalTabWidget(QTabWidget):
    def __init__(self, *args, **kwargs):
        QTabWidget.__init__(self, *args, **kwargs)
        self.setTabBar(TabBar())
        self.setTabPosition(QtWidgets.QTabWidget.West)
        self.setStyleSheet("""
            QTabBar::tab {
                height: 180px;
                width: 50px;
                background-color: #202020;
                color: white;
                font-size:10pt;
            }
            """
        )
        #self.setStyleSheet("QTabBar::tab {width: 50px;}")


class ChatGPTWindowWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setFixedHeight(570)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        shadow1 = QGraphicsDropShadowEffect()
        shadow1.setBlurRadius(20)

        self.list_widget = QListWidget()
        self.list_widget.setGraphicsEffect(shadow1)
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("Ask Me Anything...")

        self.line_edit.setFixedHeight(30)

        self.line_edit.setStyleSheet("border-radius:5px; font-size: 10pt; border: 1px solid gray")
        
        # Enable the clear button in the QLineEdit
        self.line_edit.setClearButtonEnabled(False)
        
        # Create an icon for the button
        icon = QIcon("ChatGPT Icons/send.svg")
        
        # Create an action with the icon
        action = self.line_edit.addAction(icon, QLineEdit.TrailingPosition)
        
        # Connect a slot to the triggered signal of the action
        action.triggered.connect(self.send_prompt)

         # creating a QGraphicsDropShadowEffect object
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)

        self.line_edit.returnPressed.connect(self.send_prompt)
        self.line_edit.setGraphicsEffect(shadow)

        layout.addWidget(self.list_widget)
        layout.addWidget(self.line_edit)

        self.setLayout(layout)

        #self.add_prompt_widget("Hello")

    def send_prompt(self):
        text = self.line_edit.text()
        
        # Fetch user prompt      
        self.add_prompt_widget(text)

        # Stop thread if already running
        try:
            self.request_thread.exit()
        except:
            pass
        
        # Create and start new thread
        self.request_thread = RequestThread()
        self.request_thread.prompt = text 
        self.request_thread.response_signal.connect(self.add_response_widget)
        self.request_thread.start()
        
        self.line_edit.clear()

    def add_prompt_widget(self, text):
        list_item = QListWidgetItem()

        # Create a QLabel widget with the item text
        prompt_widget = ChatGPTPromptWidget(text)

        # Set the label widget as the list item widget
        list_item.setSizeHint(prompt_widget.sizeHint())
        self.list_widget.addItem(list_item)
        self.list_widget.setItemWidget(list_item, prompt_widget)

    def add_response_widget(self, text):
        list_item = QListWidgetItem()

        # Create a QLabel widget with the item text
        prompt_widget = ChatGPTResponseWidget(text)

        # Set the label widget as the list item widget
        list_item.setSizeHint(prompt_widget.sizeHint())
        self.list_widget.addItem(list_item)
        self.list_widget.setItemWidget(list_item, prompt_widget)

class ChatGPTPromptWidget(QWidget):
    def __init__(self, text):
        super().__init__()
        self.text = text
        self.initUI()

    def initUI(self):
        # Create the main horizontal layout
        layout = QHBoxLayout(self)

        vbox_layout = QVBoxLayout()

        # Create an image label and add it to the layout
        image_label = QLabel(self)
        image_label.setStyleSheet("background-color:None")
        pixmap = QPixmap(r'ChatGPT Icons/user.png')  # Provide the path to your image file
        image_label.setPixmap(pixmap.scaledToHeight(50))
        image_label.setStyleSheet("padding-top:0px")

        spacer = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        
        vbox_layout.addWidget(image_label)
        vbox_layout.addItem(spacer)

        layout.addLayout(vbox_layout)


        self.text_widget = QLabel(self)
        self.text_widget.setAlignment(Qt.AlignLeft)
        self.text_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.text_widget.setWordWrap(True)
        self.text_widget.setStyleSheet("font-size: 11pt")

        layout.addWidget(self.text_widget)

        # Set the main layout for the widget
        self.setLayout(layout)

        self.text_widget.setText(self.text)

class ChatGPTResponseWidget(QWidget):
    def __init__(self, text):
        super().__init__()
        self.text = text
        self.initUI()


    def initUI(self):
        # Create the main horizontal layout
        layout = QHBoxLayout(self)

        vbox_layout = QVBoxLayout()

        # Create an image label and add it to the layout
        image_label = QLabel(self)
        pixmap = QPixmap(r'ChatGPT Icons/chatbot.png')  # Provide the path to your image file
        image_label.setPixmap(pixmap.scaledToHeight(50))
        image_label.setStyleSheet("padding-top:0px")

        spacer = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        
        vbox_layout.addWidget(image_label)
        vbox_layout.addItem(spacer)

        layout.addLayout(vbox_layout)


        self.text_widget = QLabel(self)
        self.text_widget.setAlignment(Qt.AlignLeft)
        self.text_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.text_widget.setWordWrap(True)
        self.text_widget.setStyleSheet("font-size: 11pt")

        layout.addWidget(self.text_widget)

        # Set the main layout for the widget
        self.setLayout(layout)

        self.text_widget.setText(self.text)

# Seperate thread to make requests to OpenAI and wait for responses
class RequestThread(QThread):
    prompt = ""
    response_signal = pyqtSignal(str)

    def run(self):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": "You are a chatbot"},
                    {"role": "user", "content": self.prompt},
                ]
        )

        result = ''
        for choice in response.choices:
            result += choice.message.content

        # Send results back to main thread
        self.response_signal.emit(result)