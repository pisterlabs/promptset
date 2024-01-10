from PySide6.QtCore import Slot, Signal, QSize, QPoint, QPointF, QEvent, QStringListModel, QRectF, QSizeF, Property, \
    QSequentialAnimationGroup, QPropertyAnimation, QRect, QParallelAnimationGroup, QEasingCurve, QTimer
from PySide6.QtGui import Qt, QFont, QMouseEvent, QPixmap, QKeyEvent, QPaintEvent, QPainter, QPen, QColor, QResizeEvent, \
    QEnterEvent, QPainterPath, QHoverEvent, QBrush
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QHBoxLayout, QPushButton, QSpacerItem, \
    QSizePolicy, QButtonGroup, QLineEdit, QPlainTextEdit, QLabel, QFrame, QMenu, QGraphicsDropShadowEffect, QFileDialog, \
    QGraphicsColorizeEffect, QComboBox, \
    QSlider, QApplication, QBoxLayout, QAbstractButton, QGraphicsRectItem, QGraphicsEllipseItem, QScrollBar
from qframelesswindow import FramelessMainWindow, TitleBar, TitleBarButton
from qframelesswindow.titlebar.title_bar_buttons import TitleBarButtonState

from AIChatEnum import TranslaterAPIType, SpeakerAPIType
from BaseClass import DataGUIInterface
from data import OpenAIConfigData, GPTParamsData, TranslaterConfigData, TranslaterConfigDataList, VITSConfigDataList, \
    VITSConfigData, Data
from event import *
from event_type import *


class QInWindowDialog(QWidget):
    """Setting dialog for the chatbot"""
    Resized = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._set_up_ui()
        self.close()

    def _set_up_ui(self):
        """Set up the UI"""
        self.setObjectName('chatbot_setting_dialog')
        # size as parent
        self.resize(self.parent().size())
        # when parent resized, the dialog resized too
        self.parent().Resized.connect(lambda: self.resize(self.parent().size()))
        # always on top, and always focus, and always in the center of the parent
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.WindowDoesNotAcceptFocus | Qt.FramelessWindowHint)
        # set the main layout
        self._main_layout = QVBoxLayout()
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)
        self._main_layout.setAlignment(Qt.AlignCenter)
        self.setLayout(self._main_layout)
        # create a placeholder for the main content
        self._main_content = QWidget()
        self._main_content.setObjectName('main_content')
        self._main_content.setFixedSize(500, 700)
        self._main_content.setContentsMargins(0, 0, 0, 0)
        self._main_content_layout = QVBoxLayout()
        self._main_content_layout.setContentsMargins(0, 0, 0, 0)
        self._main_content_layout.setSpacing(0)
        self._main_content.setLayout(self._main_content_layout)
        self._main_layout.addWidget(self._main_content)
        # create a title bar
        self._title_bar = QWidget()
        self._title_bar.setObjectName('title_bar')
        self._title_bar.setContentsMargins(0, 0, 0, 0)
        self._title_bar_layout = QHBoxLayout()
        self._title_bar_layout.setContentsMargins(0, 0, 0, 0)
        self._title_bar_layout.setSpacing(0)
        self._title_bar.setLayout(self._title_bar_layout)
        self._main_content_layout.addWidget(self._title_bar)
        # create a title label for the title bar
        self._title_label = QLabel('')
        self._title_label.setObjectName('title_label')
        self._title_label.setFixedHeight(30)
        self._title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._title_label.setAlignment(Qt.AlignCenter)
        self._title_bar_layout.addWidget(self._title_label, alignment=Qt.AlignLeft)
        # create a close button for the title bar
        # self._close_button = QPushButton('×')
        self._close_button = QTitlebarCloseButton(30, 30, 8)
        self._close_button.setObjectName('close_button')
        self._close_button.setFixedSize(30, 30)
        self._close_button.clicked.connect(self.safe_close)
        self._title_bar_layout.addWidget(self._close_button)
        # create a setting area for the main content, it is a scroll area
        self._setting_area = QScrollArea()
        self._setting_area.setFrameShape(QFrame.NoFrame)
        self._setting_area.setObjectName('setting_area')
        self._setting_area.setContentsMargins(0, 0, 0, 0)
        self._setting_area.setWidgetResizable(True)
        self._setting_area_layout = QVBoxLayout()
        self._setting_area_layout.setContentsMargins(0, 0, 0, 0)
        self._setting_area_layout.setSpacing(0)
        self._setting_area_layout.setAlignment(Qt.AlignCenter)
        self._setting_area_widget = QWidget()
        self._setting_area_widget.setObjectName('setting_area_widget')
        self._setting_area_widget.setContentsMargins(0, 0, 0, 0)
        self._setting_area_widget.setLayout(self._setting_area_layout)
        self._setting_area.setWidget(self._setting_area_widget)
        self._main_content_layout.addWidget(self._setting_area)

    @property
    def main_content(self):
        return self._main_content

    @property
    def setting_area(self):
        return self._setting_area

    @property
    def setting_area_layout(self):
        return self._setting_area_layout

    @property
    def setting_area_widget(self):
        return self._setting_area_widget

    @property
    def title_bar(self):
        return self._title_bar

    @property
    def title_bar_layout(self):
        return self._title_bar_layout

    @property
    def close_button(self):
        return self._close_button

    @property
    def main_content_layout(self):
        return self._main_content_layout

    @property
    def main_layout(self):
        return self._main_layout

    def set_title(self, title: str):
        self._title_label.setText(title)

    def set_size(self, width: int, height: int):
        self._main_content.setFixedSize(width, height)

    # rewrite the paintEvent to draw transparent background
    def paintEvent(self, e: QPaintEvent):
        background_color = QColor(0, 0, 0, 0)
        background_color.setAlpha(200)
        painter = QPainter(self)
        painter.fillRect(self.rect(), background_color)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            # if user click the space of the dialog, close the dialog
            if event.pos().x() < self._main_content.pos().x() or event.pos().x() > self._main_content.pos().x() + self._main_content.width() or event.pos().y() < self._main_content.pos().y() or event.pos().y() > self._main_content.pos().y() + self._main_content.height():
                self.area_safe_close()

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.Resized.emit()
        super().resizeEvent(event)

    def show(self) -> None:
        # todo: add an animation to the main content
        super().show()

    def close(self) -> bool:
        # todo: add an animation to the main content
        return super().close()

    def area_safe_close(self):
        """
        When user click the space of the dialog, close the dialog
        :return:
        """
        self.close()

    def safe_close(self):
        """
        When user click the close button, close the dialog
        :return:
        """
        self.close()


class QLabelInput(QWidget):
    """A container for a label and an input box"""

    def __init__(self, label_text, parent=None, label_width=110, label_height=30, password_mode=False, text=''):
        super().__init__(parent)
        self.setObjectName('label_input_box_container')
        # create a layout
        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(5)
        self.setLayout(self._layout)
        # create a label
        self._label = QLabel(label_text)
        self._label.setFixedSize(label_width, label_height)
        self._label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._label.setObjectName('label_with_input_box')
        self._layout.addWidget(self._label)
        # create a input box
        self._input_box = QLineEdit()
        self._input_box.setObjectName('input_box_with_label')
        self._input_box.setPlaceholderText(text)
        self._layout.addWidget(self._input_box)
        if password_mode:
            self._input_box.setEchoMode(QLineEdit.Password)

    def set_line_edit_text(self, text):
        self._input_box.setText(text)

    input_content = Property(str, fget=lambda self: self._input_box.text(),
                             fset=lambda self, v: self._input_box.setText(v))


class QLabelComboBox(QWidget):
    currentTextChanged = Signal(str)

    """A container for a label and a combobox"""

    def __init__(self, label_text, combobox_list, parent=None, label_width=110):
        super().__init__(parent)
        self.setObjectName('label_combobox_container')
        self._combobox_list = combobox_list
        # create a layout
        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(5)
        self.setLayout(self._layout)
        # create a label
        self._label = QLabel()
        self._label.setFixedSize(label_width, 30)
        self._label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._label.setObjectName('label_with_combobox')
        self._label.setText(label_text)
        self._layout.addWidget(self._label)
        # create a combobox
        self._combobox = QComboBox()
        self._combobox.setObjectName('combobox_with_label')
        self._combobox.setModel(QStringListModel(combobox_list))
        self._combobox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._combobox.currentTextChanged.connect(lambda text: self.currentTextChanged.emit(text))
        self._layout.addWidget(self._combobox)

    def setCurrentText(self, text):
        """
        Set the current text of the combobox.
        :param text: the text to be set.
        :return:
        """
        self._combobox.setCurrentText(text)

    def currentText(self):
        """
        Get the current text of the combobox.
        :return: text, str
        """
        return self._combobox.currentText()

    def type(self):
        """
        Get the index of the current text in the combobox list.
        :return: int, the index of the current text in the combobox list.
        """
        # match the current text with the combobox list
        return self._combobox_list.index(self._combobox.currentText())

    currentText = Property(str, fget=currentText, fset=setCurrentText)


class QLabelSliderInput(QWidget):
    """A container for a label, a horizontal slider and an input box
    :param label_text: the text of the label
    :param slider_range: the range of the slider
    :param multiply: whether the value of the slider should be multiplied by 100
    :param parent: the parent widget"""
    sliderValueChanged = Signal(int)

    def __init__(self, label_text, slider_range, multiply=True, parent=None, default_value=0):
        super().__init__(parent)
        self._multiply = multiply
        self._default_value = default_value * 10 if multiply else default_value
        self.setObjectName('label_slider_input_box_container')
        # create a layout
        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(5)
        self.setLayout(self._layout)
        # create a label
        self._label = QLabel()
        self._label.setFixedSize(110, 30)
        self._label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._label.setObjectName('label_with_slider_input_box')
        self._label.setText(label_text)
        self._layout.addWidget(self._label)
        # create a slider
        self._slider = QSlider()
        self._slider.setObjectName('slider_with_label')
        self._slider.setOrientation(Qt.Horizontal)
        self._slider.setRange(*slider_range)
        self._layout.addWidget(self._slider)
        # create a input box
        self._input_box = QLineEdit()
        self._input_box.setObjectName('input_box_with_slider')
        self._input_box.setFixedSize(50, 30)
        self._input_box.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._input_box)
        # when the slider value changed, the input box value will change
        self._slider.valueChanged.connect(self._slider_value_changed)
        self._slider.valueChanged.connect(lambda value: self.sliderValueChanged.emit(value))
        # when the input box value changed, the slider value will change
        self._input_box.textChanged.connect(self._input_box_value_changed)
        # if the input box value is invalid, the slider value will be the minimum value and the input box value will be empty
        self._input_box.editingFinished.connect(self._input_box_editing_finished)

    input_content = Property(str, fget=lambda self: self._input_box.text(),
                             fset=lambda self, v: self._input_box.setText(v))
    value = Property(int, fget=lambda self: self.get_value(), fset=lambda self, v: self.set_value(v))

    def get_value(self):
        return self._slider.value() / 10 if self._multiply else self._slider.value()

    def set_value(self, value):
        self._slider.setValue(value * 10 if self._multiply else value)
        self._input_box.setText(str(value))

    # when the slider value changed, the input box value will change
    def _slider_value_changed(self):
        try:
            self._input_box.setText(str(self._slider.value() / 10) if self._multiply else str(self._slider.value()))
        except:
            pass

    def _input_box_value_changed(self):
        try:
            self._slider.setValue(
                int(float(self._input_box.text()) * 10) if self._multiply else int(self._input_box.text()))
        except:
            pass

    def _input_box_editing_finished(self):
        try:
            self._input_box.setText(str(self._slider.value() / 10) if self._multiply else str(self._slider.value()))
        except:
            pass

    def set_to_default(self):
        self._slider.setValue(self._default_value)
        self._input_box.setText(str(self._default_value / 10) if self._multiply else str(self._default_value))


# todo: complete the chatbot button class
class QChatBotButton(QPushButton):
    """A button for the chatbot list"""
    editClicked = Signal(str)
    deleteClicked = Signal(str)
    checked = Signal(str)

    def __init__(self, name, id_, avatar_path='./resources/images/test_avatar_me.jpg', description='This is a ChatBot.',
                 parent=None):
        super().__init__('', parent)
        self.setObjectName('chatbot_button')
        self._name = name
        self._description = description
        self._id = id_
        self._avatar_path = avatar_path
        self._description = description
        self.setCheckable(True)
        self.setFixedSize(200, 80)
        self.clicked.connect(lambda: self.checked.emit(self._id))
        self.setChecked(True)
        # add a main container
        self._main_container = QWidget(self)
        self._main_container.setObjectName('chatbot_button_main_container')
        self._main_container.setFixedSize(200, 80)

        # add a horizontal layout
        self._layout = QSettableHLayout(content_margin=(15, 0, 15, 0), spacing=5,
                                        alignment=Qt.AlignLeft | Qt.AlignVCenter)
        self._main_container.setLayout(self._layout)
        # add a avatar
        self._avatar = QAvatarLabel(avatar_path, 50, shadow=False)
        self._layout.addWidget(self._avatar)
        # add a vertical layout
        self._v_layout = QSettableVLayout(content_margin=(5, 0, 0, 0), spacing=5,
                                          alignment=Qt.AlignLeft | Qt.AlignVCenter)
        self._layout.addLayout(self._v_layout)
        # add a name label
        self._name_label = QLabel(name)
        self._name_label.setObjectName('chatbot_name_label')
        self._v_layout.addWidget(self._name_label)
        # add a description label
        self._description_label = QLabel(description)
        self._description_label.setObjectName('chatbot_description_label')
        self._v_layout.addWidget(self._description_label)

        # add a floating button container
        self._floating_button_container = QWidget(self)
        self._floating_button_container.setObjectName('chatbot_button_floating_button_container')
        self._floating_button_container.setFixedSize(70, 80)
        self._floating_button_container.setHidden(True)
        # move the floating button container to the right center
        self._floating_button_container.move(130, 0)
        # add a floating button layout
        self._floating_button_layout = QSettableHLayout(content_margin=(10, 10, 10, 10), spacing=5,
                                                        alignment=Qt.AlignCenter)
        self._floating_button_container.setLayout(self._floating_button_layout)
        # add a floating edit button
        self._floating_edit_button = QOverflowEditButton(20, 20)
        self._floating_edit_button.setObjectName('chatbot_button_floating_edit_button')
        self._floating_edit_button.clicked.connect(lambda: self.editClicked.emit(self._id))
        self._floating_button_layout.addWidget(self._floating_edit_button)
        # add a floating delete button
        self._floating_delete_button = QOverflowDeleteButton(20, 20)
        self._floating_delete_button.setObjectName('chatbot_button_floating_delete_button')
        self._floating_delete_button.clicked.connect(lambda: self.deleteClicked.emit(self._id))
        self.deleteClicked.connect(lambda: self.deleteLater())
        self._floating_button_layout.addWidget(self._floating_delete_button)

    def on_chatbot_update(self, chatbot):
        if chatbot.chatbot_id != self._id:
            return
        self._name_label.setText(chatbot.character.name)
        self._description_label.setText(chatbot.character.description)
        self._avatar.set_image(chatbot.character.avatar_path)

    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.button() == Qt.RightButton:
            self._show_context_menu(e)
            super().mouseReleaseEvent(e)
        else:
            super().mouseReleaseEvent(e)

    def enterEvent(self, event: QEnterEvent) -> None:
        self._floating_button_container.show()
        super().enterEvent(event)

    def leaveEvent(self, event: QEnterEvent) -> None:
        self._floating_button_container.hide()
        super().leaveEvent(event)

    def _show_context_menu(self, e: QMouseEvent):
        menu = QMenu(self)
        # todo: add a edit function
        menu.addAction('Edit', lambda: self.editClicked.emit(self._id))
        menu.addAction('Delete', lambda: self.deleteClicked.emit(self._id))
        menu.exec_(e.globalPos())


class QTitleBar(TitleBar):
    """A title bar for the main window
    :param parent: the parent widget"""
    GlobalSettingClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # add a setting button
        self._setting_button = QTitleBarSettingButton(46, 32, 3, 3.8, parent=self)
        self._setting_button.clicked.connect(self.GlobalSettingClicked.emit)
        self.hBoxLayout.insertWidget(1, self._setting_button, alignment=Qt.AlignRight)


class QTitleBarSettingButton(TitleBarButton):
    """A setting button for the title bar
    :param width: the width of the button
    :param height: the height of the button
    :param circle_radius: the radius of the circle in the icon
    :param hexagon_side: the side length of the hexagon in the icon
    :param parent: the parent widget"""

    def __init__(self, width, height, circle_radius, hexagon_side, parent=None):
        self._width = width
        self._height = height
        self._circle_radius = circle_radius
        self._hexagon_side = hexagon_side
        super().__init__(parent)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color, bgColor = self._getColors()

        # draw background
        painter.setBrush(bgColor)
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())

        # draw icon
        painter.setBrush(Qt.NoBrush)
        pen = QPen(color, 1)
        pen.setCosmetic(True)
        painter.setPen(pen)

        r = self.devicePixelRatioF()
        painter.scale(1 / r, 1 / r)
        painter.drawEllipse(QPoint(int(self._width / 2 * r), int(self._height / 2 * r)), int(self._circle_radius * r),
                            int(self._circle_radius * r))
        d = self._hexagon_side * r
        point_a = QPointF(int(self._width / 2 * r) - 2 * d, int(self._height / 2 * r))
        point_b = QPointF(int(self._width / 2 * r) - 1 * d, int(self._height / 2 * r) - pow(3, 0.5) * d)
        point_c = QPointF(int(self._width / 2 * r) + 1 * d, int(self._height / 2 * r) - pow(3, 0.5) * d)
        point_d = QPointF(int(self._width / 2 * r) + 2 * d, int(self._height / 2 * r))
        point_e = QPointF(int(self._width / 2 * r) + 1 * d, int(self._height / 2 * r) + pow(3, 0.5) * d)
        point_f = QPointF(int(self._width / 2 * r) - 1 * d, int(self._height / 2 * r) + pow(3, 0.5) * d)
        path = QPainterPath(point_a)
        path.lineTo(point_b)
        path.lineTo(point_c)
        path.lineTo(point_d)
        path.lineTo(point_e)
        path.lineTo(point_f)
        path.lineTo(point_a)
        painter.drawPath(path)


class QCustomPushButton(QPushButton):
    """ Title bar button"""

    def __init__(self, size, parent=None):
        super().__init__(parent=parent)
        self.setCursor(Qt.ArrowCursor)
        self.setFixedSize(size)
        self._state = TitleBarButtonState.NORMAL

        # icon color
        self._normalColor = QColor(0, 0, 0)
        self._hoverColor = QColor(0, 0, 0)
        self._pressedColor = QColor(0, 0, 0)

        # background color
        self._normalBgColor = QColor(0, 0, 0, 0)
        self._hoverBgColor = QColor(0, 0, 0, 26)
        self._pressedBgColor = QColor(0, 0, 0, 51)

    def setState(self, state):
        """ set the state of button

        Parameters
        ----------
        state: TitleBarButtonState
            the state of button
        """
        self._state = state
        self.update()

    def isPressed(self):
        """ whether the button is pressed """
        return self._state == TitleBarButtonState.PRESSED

    def getNormalColor(self):
        """ get the icon color of the button in normal state """
        return self._normalColor

    def getHoverColor(self):
        """ get the icon color of the button in hover state """
        return self._hoverColor

    def getPressedColor(self):
        """ get the icon color of the button in pressed state """
        return self._pressedColor

    def getNormalBackgroundColor(self):
        """ get the background color of the button in normal state """
        return self._normalBgColor

    def getHoverBackgroundColor(self):
        """ get the background color of the button in hover state """
        return self._hoverBgColor

    def getPressedBackgroundColor(self):
        """ get the background color of the button in pressed state """
        return self._pressedBgColor

    def setNormalColor(self, color):
        """ set the icon color of the button in normal state

        Parameters
        ----------
        color: QColor
            icon color
        """
        self._normalColor = QColor(color)
        self.update()

    def setHoverColor(self, color):
        """ set the icon color of the button in hover state

        Parameters
        ----------
        color: QColor
            icon color
        """
        self._hoverColor = QColor(color)
        self.update()

    def setPressedColor(self, color):
        """ set the icon color of the button in pressed state

        Parameters
        ----------
        color: QColor
            icon color
        """
        self._pressedColor = QColor(color)
        self.update()

    def setNormalBackgroundColor(self, color):
        """ set the background color of the button in normal state

        Parameters
        ----------
        color: QColor
            background color
        """
        self._normalBgColor = QColor(color)
        self.update()

    def setHoverBackgroundColor(self, color):
        """ set the background color of the button in hover state

        Parameters
        ----------
        color: QColor
            background color
        """
        self._hoverBgColor = QColor(color)
        self.update()

    def setPressedBackgroundColor(self, color):
        """ set the background color of the button in pressed state

        Parameters
        ----------
        color: QColor
            background color
        """
        self._pressedBgColor = QColor(color)
        self.update()

    def setAllBackgroundColor(self, color):
        """ set the background color of the button in all state

        Parameters
        ----------
        color: QColor
            background color
        """
        self._normalBgColor = QColor(color)
        self._hoverBgColor = QColor(color)
        self._pressedBgColor = QColor(color)
        self.update()

    def setALlIconColor(self, color):
        """ set the icon color of the button in all state

        Parameters
        ----------
        color: QColor
            icon color
        """
        self._normalColor = QColor(color)
        self._hoverColor = QColor(color)
        self._pressedColor = QColor(color)
        self.update()

    def enterEvent(self, e):
        self.setState(TitleBarButtonState.HOVER)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.setState(TitleBarButtonState.NORMAL)
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return

        self.setState(TitleBarButtonState.PRESSED)
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton:
            return

        self.setState(TitleBarButtonState.HOVER)
        super().mouseReleaseEvent(e)

    def _getColors(self):
        """ get the icon color and background color """
        if self._state == TitleBarButtonState.NORMAL:
            return self._normalColor, self._normalBgColor
        elif self._state == TitleBarButtonState.HOVER:
            return self._hoverColor, self._hoverBgColor

        return self._pressedColor, self._pressedBgColor

    normalColor = property(getNormalColor, setNormalColor)
    hoverColor = property(getHoverColor, setHoverColor)
    pressedColor = property(getPressedColor, setPressedColor)
    normalBackgroundColor = property(getNormalBackgroundColor, setNormalBackgroundColor)
    hoverBackgroundColor = property(getHoverBackgroundColor, setHoverBackgroundColor)
    pressedBackgroundColor = property(getPressedBackgroundColor, setPressedBackgroundColor)


class QTitlebarCloseButton(QCustomPushButton):
    def __init__(self, width, height, icon_width, parent=None):
        size = QSize(width, height)
        super().__init__(size, parent)
        self._width = width
        self._height = height
        self._icon_width = icon_width
        self.setHoverColor(QColor('#FFFFFB'))

    def paintEvent(self, e: QPaintEvent) -> None:
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color, bgcolor = self._getColors()

        # draw icon
        painter.setBrush(Qt.NoBrush)
        pen = QPen(color, 1.3)
        pen.setCosmetic(True)
        painter.setPen(pen)

        r = self.devicePixelRatioF()
        painter.scale(1 / r, 1 / r)
        point_a = QPointF(int(self._width / 2 * r) - int(self._icon_width / 2 * r),
                          int(self._height / 2 * r) - int(self._icon_width / 2 * r))
        point_b = QPointF(int(self._width / 2 * r) + int(self._icon_width / 2 * r),
                          int(self._height / 2 * r) + int(self._icon_width / 2 * r))
        point_c = QPointF(int(self._width / 2 * r) + int(self._icon_width / 2 * r),
                          int(self._height / 2 * r) - int(self._icon_width / 2 * r))
        point_d = QPointF(int(self._width / 2 * r) - int(self._icon_width / 2 * r),
                          int(self._height / 2 * r) + int(self._icon_width / 2 * r))
        path_a_to_b = QPainterPath(point_a)
        path_a_to_b.lineTo(point_b)
        painter.drawPath(path_a_to_b)
        path_c_to_d = QPainterPath(point_c)
        path_c_to_d.lineTo(point_d)
        painter.drawPath(path_c_to_d)


class QLeftBarAddButton(QCustomPushButton):
    def __init__(self):
        super().__init__(QSize(200, 80))
        self.normalColor, self.hoverColor, self.pressedColor = QColor('#e2e2e2'), QColor('#BDC0BA'), QColor('#d2d2d2')
        self.setObjectName('left_bar_add_button')

    def paintEvent(self, e: QPaintEvent) -> None:
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color, bgcolor = self._getColors()

        width = self.width()
        height = self.height()

        # draw icon
        painter.setBrush(color)
        pen = QPen(Qt.NoPen)
        pen.setCosmetic(True)
        painter.setPen(pen)
        circle_radius = 25
        cross_width = circle_radius * 0.4
        # draw a circle in the center
        painter.drawEllipse(QPointF(width / 2, height / 2), circle_radius, circle_radius)
        painter.setPen(QPen(QColor('#ffffff'), cross_width * 0.3))
        # draw a cross in the center
        painter.drawLine(QPointF(width / 2 - cross_width, height / 2), QPointF(width / 2 + cross_width, height / 2))
        painter.drawLine(QPointF(width / 2, height / 2 - cross_width), QPointF(width / 2, height / 2 + cross_width))


class QMessageLabel(QLabel):
    LabelClicked = Signal()

    def __init__(self, message, is_me, max_width):
        super().__init__(message)
        self._message = message
        self._max_width = max_width
        self.setObjectName('message_label_me' if is_me else 'message_label_reception')
        self.setFont(QFont('HarmonyOS Sans SC', 14))
        self.setMaximumWidth(max_width)
        # set the message label's size
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)

    # click event
    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self.LabelClicked.emit()
            super().mousePressEvent(e)
        else:
            super().mousePressEvent(e)

    def sizeHint(self) -> QSize:
        """
        Rewrite the sizeHint function to make the label's size fit the text.
        :return:
        """
        max_width_without_padding = self._max_width - 20
        metrics = self.fontMetrics()
        width = metrics.horizontalAdvance(self._message)
        if width > max_width_without_padding:
            width = max_width_without_padding + 30
            height = metrics.boundingRect(0, 0, width, 0, Qt.TextWordWrap, self._message).height() + 40
        else:
            width = metrics.horizontalAdvance(self._message) + 30
            height = metrics.height() + 20
        return QSize(width, height)

    def set_max_width(self, width):
        self._max_width = width
        self.resize(self.sizeHint())
        self.setMaximumWidth(width)


class QMessagePlainTextEdit(QPlainTextEdit):
    SendMessage = Signal(str)

    def __init__(self, max_height, parent=None):
        super().__init__(parent)
        self.setObjectName('message_input_box')
        self.setFont(QFont('HarmonyOS Sans SC', 14))
        self.setFrameStyle(QFrame.NoFrame)
        self.setMaximumHeight(max_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.setPlaceholderText('Type your message here...')

    def keyPressEvent(self, e: QKeyEvent):
        # if user input a return key, then send the message
        if e.key() == Qt.Key_Return and not e.modifiers() & Qt.ShiftModifier:
            self.SendMessage.emit(self.toPlainText())
        # if user input a return key and shift key, then add a new line
        elif e.key() == Qt.Key_Return and Qt.ShiftModifier:
            self.insertPlainText('\n')
        else:
            super().keyPressEvent(e)


class QMessageOverflowButtonsContainer(QWidget):
    startPlay = Signal()
    stopPlay = Signal()
    resendClicked = Signal()
    deleteClicked = Signal()
    copyClicked = Signal()

    def __init__(self, size, move_point, is_user, parent=None, startPlay=None, stopPlay=None, resendClicked=None,
                 deleteClicked=None, copyClicked=None):
        super().__init__(parent)
        self._is_playing = False
        self.setObjectName('message_overflow_buttons_container')
        self.setFixedSize(size)
        self.move(move_point)
        # layout
        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(5)
        self._layout.setAlignment(Qt.AlignRight)
        self.setLayout(self._layout)
        button_side_length = 24
        # create a resend button
        self._resend_button = QOverflowResendButton(button_side_length, button_side_length)
        self._resend_button.clicked.connect(resendClicked)
        self._layout.addWidget(self._resend_button)
        # if not is_user, create a play sound button
        if not is_user:
            self._play_sound_button = QOverflowPlaySoundButton(button_side_length, button_side_length)
            self._play_sound_button.clicked.connect(self._play_sound)
            self._layout.addWidget(self._play_sound_button)
        # create a copy button
        self._copy_button = QOverflowCopyButton(button_side_length, button_side_length)
        self._copy_button.clicked.connect(copyClicked)
        self._layout.addWidget(self._copy_button)
        # create a delete button
        self._delete_button = QOverflowDeleteButton(button_side_length, button_side_length)
        self._delete_button.clicked.connect(deleteClicked)
        self._layout.addWidget(self._delete_button)

        self.startPlay.connect(startPlay)
        self.stopPlay.connect(stopPlay)
        self.resendClicked.connect(resendClicked)
        self.deleteClicked.connect(deleteClicked)
        self.copyClicked.connect(copyClicked)

    def set_resendable(self, resendable):
        if resendable:
            self._resend_button.setHidden(False)
        else:
            self._resend_button.setHidden(True)

    @property
    def is_playing(self):
        return self._is_playing

    @is_playing.setter
    def is_playing(self, value):
        self._play_sound_button.change_button_state(value)
        self._is_playing = value
        if value:
            self.show()
        if not value and not self.parent().is_hover:
            self.hide()

    def hide(self) -> None:
        if self._is_playing:
            return
        super().hide()

    def _play_sound(self):
        self._is_playing = not self._is_playing
        if self._is_playing:
            self.startPlay.emit()
        else:
            self.stopPlay.emit()
        self._play_sound_button.change_button_state(self._is_playing)


class QOverflowButton(QCustomPushButton):
    def __init__(self, width, height, parent=None, shadow=True):
        self._size = QSize(width, height)
        super().__init__(self._size, parent)
        self.setObjectName('message_overflow_button')
        # add shadow effect
        if shadow:
            self.setGraphicsEffect(
                QGraphicsDropShadowEffect(self, blurRadius=width / 4, offset=QPoint(0, 0), color=Qt.gray))
        # set the color
        self._set_default_colors()

    def _set_default_colors(self):
        self.normalColor = QColor(145, 152, 159)
        self.hoverColor = QColor(255, 255, 251)
        self.pressedColor = QColor(255, 255, 251)
        self.normalBackgroundColor = QColor('#FFFFFB')
        self.hoverBackgroundColor = QColor('#A5DEE4')
        self.pressedBackgroundColor = QColor('#81C7D4')

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        # if user press the button, then remove the shadow effect
        self.setGraphicsEffect(None)
        self.update()

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        # if user release the button, then add the shadow effect
        self.setGraphicsEffect(
            QGraphicsDropShadowEffect(self, blurRadius=self._size.width() / 4, offset=QPoint(0, 0), color=Qt.gray))
        self.update()

    def paintEvent(self, arg__1: QPaintEvent) -> None:
        color, background_color = self._getColors()
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        painter.setPen(Qt.NoPen)
        painter.setBrush(background_color)
        painter.drawEllipse(0, 0, self._size.width(), self._size.height())


class QOverflowCopyButton(QOverflowButton):
    def __init__(self, width, height, parent=None, shadow=True):
        super().__init__(width, height, parent, shadow)

    def paintEvent(self, arg__1: QPaintEvent) -> None:
        super().paintEvent(arg__1)
        color, background_color = self._getColors()
        # draw the copy icon, it is in a circle of 24 * 24, contains two rectangles without fill
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        painter.setPen(QPen(color, 1))
        painter.setBrush(Qt.NoBrush)
        rect_width = 8
        rect_height = 10
        rect_radius = 1.2
        button_width = self._size.width()
        offset_x = 3
        offset_y = 2
        # rectangle 1
        rect_1_start_x = (button_width - rect_width - offset_x) / 2
        rect_1_start_y = (button_width - rect_height + offset_y) / 2
        painter.drawRoundedRect(QRectF(rect_1_start_x, rect_1_start_y, rect_width, rect_height), rect_radius,
                                rect_radius)
        # rectangle 2
        """point like this:
           C============E====D====#
           #                      #
           B                      #
           #                      #
           A                      #
                                  #
                                  #
                                  #
                        G         #
                                  #
                                  F
                                  #
                  H===============#                 
        """
        point_a = QPointF((button_width - rect_width + offset_x) / 2, rect_1_start_y)
        point_b = QPointF(point_a.x(), point_a.y() - offset_y + rect_radius)
        point_c = QPointF((button_width - rect_width + offset_x) / 2, (button_width - rect_height - offset_y) / 2)
        point_d = QPointF(point_c.x() + rect_width - rect_radius, point_c.y())
        point_e = QPointF(point_d.x() - rect_radius, point_d.y())
        point_f = QPointF(point_c.x() + rect_width, point_c.y() + rect_height - rect_radius)
        point_g = QPointF(point_f.x() - rect_radius * 2, point_f.y() - rect_radius)
        point_h = QPointF(point_c.x() + rect_width - offset_x, point_c.y() + rect_height)
        size_of_arc_rect = QSizeF(rect_radius * 2, rect_radius * 2)
        path = QPainterPath()
        path.moveTo(point_a)
        path.lineTo(point_b)
        path.arcTo(QRectF(point_c, size_of_arc_rect), 180, - 90)
        path.lineTo(point_d)
        path.arcTo(QRectF(point_e, size_of_arc_rect), 90, -90)
        path.lineTo(point_f)
        path.arcTo(QRectF(point_g, size_of_arc_rect), 0, -90)
        path.lineTo(point_h)
        painter.drawPath(path)


class QOverflowResendButton(QOverflowButton):
    def __init__(self, width, height, parent=None, shadow=True):
        super().__init__(width, height, parent, shadow)

    def paintEvent(self, arg__1: QPaintEvent):
        super().paintEvent(arg__1)
        color, background_color = self._getColors()
        # draw the resend icon
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        painter.setPen(QPen(color, 1))
        painter.setBrush(Qt.NoBrush)
        button_width = self._size.width()
        angle = - 150
        line_length = 4.5
        circle_radius = 6
        circle_center = QPointF(button_width / 2, button_width / 2)
        circle_rect = QRectF(circle_center - QPointF(circle_radius, circle_radius),
                             QSizeF(circle_radius * 2, circle_radius * 2))
        start_point_left = circle_center - QPointF(circle_radius, 0)
        start_point_right = circle_center + QPointF(circle_radius, 0)
        # draw the down part
        path = QPainterPath()
        path.moveTo(start_point_left)
        path.arcTo(circle_rect, 180, angle)
        path.moveTo(start_point_left)
        path.lineTo(start_point_left - QPointF(0, line_length))
        # draw the up part
        path.moveTo(start_point_right)
        path.arcTo(circle_rect, 0, angle)
        path.moveTo(start_point_right)
        path.lineTo(start_point_right + QPointF(0, line_length))

        painter.drawPath(path)


class QOverflowDeleteButton(QOverflowButton):
    def __init__(self, width, height, parent=None, shadow=True):
        super().__init__(width, height, parent, shadow)
        self.setObjectName("message_overflow_delete_button")
        self.hoverBackgroundColor = QColor("#E83015")
        self.pressedBackgroundColor = QColor("#B54434")

    def paintEvent(self, arg__1: QPaintEvent):
        super().paintEvent(arg__1)
        color, background_color = self._getColors()
        # draw the delete icon
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        painter.setPen(QPen(color, 1))
        painter.setBrush(Qt.NoBrush)
        button_width = self._size.width()
        center = QPointF(button_width / 2, button_width / 2)
        trash_width = 10
        trash_height = 12
        up_line_length = trash_width * 0.4
        down_line_length = trash_width
        spacing = 2
        down_rect_width = trash_width * 0.8
        down_rect_height = trash_height - spacing * 2
        down_rect_radius = 2
        down_rect_radius_rect_size = QSizeF(down_rect_radius * 2, down_rect_radius * 2)
        # draw the up line
        path = QPainterPath()
        path.moveTo(center + QPointF(- up_line_length / 2, - trash_height / 2))
        path.lineTo(center + QPointF(up_line_length / 2, - trash_height / 2))
        # draw the down line
        path.moveTo(center + QPointF(- down_line_length / 2, - trash_height / 2 + spacing))
        path.lineTo(center + QPointF(down_line_length / 2, - trash_height / 2 + spacing))
        # draw the down rect
        point_a = center + QPointF(- down_rect_width / 2, - trash_height / 2 + spacing * 2)
        point_b = point_a + QPointF(0, down_rect_height - down_rect_radius)
        point_c = point_a + QPointF(0, down_rect_height - down_rect_radius * 2)
        point_d = point_a + QPointF(down_rect_width - down_rect_radius, down_rect_height)
        point_e = point_c + QPointF(down_rect_width - down_rect_radius * 2, 0)
        point_f = point_a + QPointF(down_rect_width, 0)
        path.moveTo(point_a)
        path.lineTo(point_b)
        path.arcTo(QRectF(point_c, down_rect_radius_rect_size), 180, 90)
        path.lineTo(point_d)
        path.arcTo(QRectF(point_e, down_rect_radius_rect_size), 270, 90)
        path.lineTo(point_f)
        # draw the down rect line
        path.moveTo(point_a + QPointF(down_rect_width / 3, 0))
        path.lineTo(point_a + QPointF(down_rect_width / 3, down_rect_height * 0.7))
        path.moveTo(point_a + QPointF(down_rect_width / 3 * 2, 0))
        path.lineTo(point_a + QPointF(down_rect_width / 3 * 2, down_rect_height * 0.7))

        painter.drawPath(path)


class QOverflowPlaySoundButton(QOverflowButton):
    def __init__(self, width, height, parent=None, shadow=True):
        super().__init__(width, height, parent, shadow)

    def change_button_state(self, is_playing):
        if is_playing:
            self.setAllBackgroundColor(QColor("#E83015"))
            self.setALlIconColor(QColor("#FFFFFB"))
        else:
            self._set_default_colors()

    def paintEvent(self, arg__1: QPaintEvent):
        super().paintEvent(arg__1)
        color, background_color = self._getColors()
        # draw the play sound icon
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        painter.setPen(QPen(color, 1))
        painter.setBrush(Qt.NoBrush)
        button_width = self._size.width()
        center = QPointF(button_width / 2, button_width / 2)
        # draw the ellipse
        first_ellipse_radius = 3
        spacing = 3.5
        offset = - 1.5
        ellipse_angle = 80
        ellipse_center = center + QPointF(- (spacing * 2 + first_ellipse_radius) / 2 + offset, 0)
        path = QPainterPath()
        # first ellipse
        first_ellipse_rect = QRectF(ellipse_center - QPointF(first_ellipse_radius, first_ellipse_radius),
                                    QSizeF(first_ellipse_radius * 2, first_ellipse_radius * 2))
        path.moveTo(ellipse_center + QPointF(first_ellipse_radius, 0))
        path.arcTo(first_ellipse_rect, 0, ellipse_angle / 2)
        path.moveTo(ellipse_center + QPointF(first_ellipse_radius, 0))
        path.arcTo(first_ellipse_rect, 0, - ellipse_angle / 2)
        # second ellipse
        second_ellipse_radius = first_ellipse_radius + spacing
        second_ellipse_rect = QRectF(ellipse_center - QPointF(second_ellipse_radius, second_ellipse_radius),
                                     QSizeF(second_ellipse_radius * 2, second_ellipse_radius * 2))
        path.moveTo(ellipse_center + QPointF(second_ellipse_radius, 0))
        path.arcTo(second_ellipse_rect, 0, ellipse_angle / 2)
        path.moveTo(ellipse_center + QPointF(second_ellipse_radius, 0))
        path.arcTo(second_ellipse_rect, 0, - ellipse_angle / 2)
        # third ellipse
        third_ellipse_radius = second_ellipse_radius + spacing
        third_ellipse_rect = QRectF(ellipse_center - QPointF(third_ellipse_radius, third_ellipse_radius),
                                    QSizeF(third_ellipse_radius * 2, third_ellipse_radius * 2))
        path.moveTo(ellipse_center + QPointF(third_ellipse_radius, 0))
        path.arcTo(third_ellipse_rect, 0, ellipse_angle / 2)
        path.moveTo(ellipse_center + QPointF(third_ellipse_radius, 0))
        path.arcTo(third_ellipse_rect, 0, - ellipse_angle / 2)

        painter.drawPath(path)


class QOverflowEditButton(QOverflowButton):
    def __init__(self, width, height, parent=None, shadow=True):
        super().__init__(width, height, parent, shadow)

    def paintEvent(self, e: QPaintEvent) -> None:
        super().paintEvent(e)
        color, background_color = self._getColors()
        # draw the edit icon
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        painter.setPen(QPen(color, 1))
        painter.setBrush(Qt.NoBrush)
        button_width = self._size.width()
        center = QPointF(button_width / 2, button_width / 2)
        icon_width = button_width * 0.5
        icon_radius = icon_width * 0.1

        path = QPainterPath()
        point_a = center + QPointF(icon_width / 2, - icon_width / 2)
        point_b = center + QPointF(- icon_width / 2, - icon_width / 2)
        point_c = center + QPointF(- icon_width / 2, icon_width / 2)
        point_d = center + QPointF(icon_width / 2, icon_width / 2)
        path.moveTo(point_a + QPointF(- icon_width * 0.4, 0))
        path.lineTo(point_b + QPointF(icon_radius, 0))
        path.arcTo(QRectF(point_b, QSizeF(icon_radius * 2, icon_radius * 2)), 90, 90)
        path.lineTo(point_c + QPointF(0, - icon_radius))
        path.arcTo(QRectF(point_c + QPointF(0, -icon_radius * 2), QSizeF(icon_radius * 2, icon_radius * 2)), 180, 90)
        path.lineTo(point_d + QPointF(- icon_radius, 0))
        path.arcTo(
            QRectF(point_d + QPointF(- icon_radius * 2, - icon_radius * 2), QSizeF(icon_radius * 2, icon_radius * 2)),
            270, 90)
        path.lineTo(point_a + QPointF(0, icon_width * 0.4))
        painter.drawPath(path)

        path = QPainterPath()
        path.moveTo(point_a)
        path.lineTo(center)
        painter.drawPath(path)


class QAvatarLabel(QLabel):
    clicked = Signal()

    def __init__(self, image_path, radius, parent=None, editable=False, shadow=True):
        super().__init__(parent)
        self._radius = radius
        self._avatar_path = image_path
        self.setObjectName('avatar_image_label')
        self.setFixedSize(radius, radius)
        self._image = QPixmap(self._avatar_path).scaled(radius, radius, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._editable = editable
        self.setScaledContents(True)
        self.setAlignment(Qt.AlignTop)
        if shadow:
            self.setGraphicsEffect(
                QGraphicsDropShadowEffect(self, offset=QPoint(0, 0), blurRadius=self._radius / 4, color=Qt.gray))

    # paint the image to a circle
    def paintEvent(self, e: QPaintEvent):
        # draw the image
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._image)
        painter.drawEllipse(0, 0, self._radius, self._radius)

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton and self._editable:
            self.clicked.emit()
        else:
            super().mousePressEvent(e)

    def enterEvent(self, event: QEnterEvent) -> None:
        # if the image is editable, then change the cursor to a hand, and darken the image
        if self._editable:
            self.setCursor(Qt.PointingHandCursor)
            self.setGraphicsEffect(QGraphicsColorizeEffect(self, color=QColor(88, 178, 220, 200)))
        super().enterEvent(event)

    def leaveEvent(self, event: QEnterEvent) -> None:
        # if the image is editable, then change the cursor to an arrow, and restore the image
        if self._editable:
            self.setCursor(Qt.ArrowCursor)
            self.setGraphicsEffect(
                QGraphicsDropShadowEffect(self, offset=QPoint(0, 0), blurRadius=self._radius / 4, color=Qt.gray))
        super().leaveEvent(event)

    def set_image(self, image):
        self._avatar_path = image
        self._image = QPixmap(image).scaled(self._radius, self._radius, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.update()

    avatar_path = Property(str, fget=lambda self: self._avatar_path, fset=set_image)


class SmoothScrollBar(QScrollBar):
    """ Smooth scroll bar """

    scrollFinished = Signal()

    def __init__(self, parent=None):
        QScrollBar.__init__(self, parent)
        self.rangeChanged.connect(lambda : self.scrollTo(self.maximum(), False))
        self.ani = QPropertyAnimation()
        self.ani.setTargetObject(self)
        self.ani.setPropertyName(b"value")
        self.ani.setEasingCurve(QEasingCurve.OutExpo)
        self.ani.setDuration(500)
        # self.bounce_ani = QPropertyAnimation()
        # self.bounce_ani.setTargetObject(self)
        # self.bounce_ani.setPropertyName(b"value")
        # self.bounce_ani.setEasingCurve(QEasingCurve.InCubic)
        # self.bounce_ani.setDuration(500)
        # self.virtual_max = self.maximum() + self.pageStep()
        # self.virtual_min = self.minimum() - self.pageStep()
        self.__value = self.value()
        self.ani.finished.connect(self.scrollFinished)

    def setValue(self, value: int, is_animate=True):
        if value == self.value():
            return

        if is_animate:
            # stop running animation
            self.ani.stop()
            self.scrollFinished.emit()
            self.ani.setStartValue(self.value())
            self.ani.setEndValue(value)
            self.ani.start()
        else:
            super().setValue(value)

    def scrollValue(self, value: int):
        """ scroll the specified distance """
        self.__value += value
        self.__value = max(self.minimum(), self.__value)
        self.__value = min(self.maximum(), self.__value)
        self.setValue(self.__value)

    def scrollTo(self, value: int, is_animate=True):
        """ scroll to the specified position """
        self.__value = value
        self.__value = max(self.minimum(), self.__value)
        self.__value = min(self.maximum(), self.__value)
        self.setValue(self.__value, is_animate)

    def resetValue(self, value):
        self.__value = value

    def mousePressEvent(self, e):
        self.ani.stop()
        super().mousePressEvent(e)
        self.__value = self.value()

    def mouseReleaseEvent(self, e):
        self.ani.stop()
        super().mouseReleaseEvent(e)
        self.__value = self.value()

    def mouseMoveEvent(self, e):
        self.ani.stop()
        super().mouseMoveEvent(e)
        self.__value = self.value()


class QNoBarScrollArea(QScrollArea):
    def __init__(self, widget, parent=None):
        super().__init__(parent)
        self.hScrollBar = SmoothScrollBar(self)
        self.hScrollBar.setOrientation(Qt.Horizontal)
        self.setHorizontalScrollBar(self.hScrollBar)
        self.vScrollBar = SmoothScrollBar(self)
        self.vScrollBar.setOrientation(Qt.Vertical)
        self.setVerticalScrollBar(self.vScrollBar)
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setFrameStyle(QFrame.NoFrame)
        self.setWidget(widget)

    def setScrollAnimation(self, orient, duration, easing=QEasingCurve.OutCubic):
        """ set scroll animation
        Parameters
      ----------
        orient: Orient
            scroll orientation
        duration: int
            scroll duration
        easing: QEasingCurve
            animation type
        """
        bar = self.vScrollBar if orient == Qt.Horizontal else self.hScrollBar
        bar.ani.setDuration(duration)
        bar.ani.setEasingCurve(easing)

    def wheelEvent(self, e):
        if e.modifiers() == Qt.NoModifier:
            self.vScrollBar.scrollValue(-e.angleDelta().y())


class QSettableVLayout(QVBoxLayout):
    def __init__(self, content_margin=(0, 0, 0, 0), spacing=0, alignment=None, parent=None):
        super().__init__(parent)
        self.setContentsMargins(content_margin[0], content_margin[1], content_margin[2], content_margin[3])
        self.setSpacing(spacing)
        if alignment:
            self.setAlignment(alignment)


class QSettableHLayout(QHBoxLayout):
    def __init__(self, content_margin=(0, 0, 0, 0), spacing=0, alignment=None, parent=None):
        super().__init__(parent)
        self.setContentsMargins(content_margin[0], content_margin[1], content_margin[2], content_margin[3])
        self.setSpacing(spacing)
        if alignment:
            self.setAlignment(alignment)


class TranslaterSettingGroup(QWidget, DataGUIInterface):
    def __init__(self, data: TranslaterConfigDataList, parent=None):
        super().__init__()
        self._api_type_list = data.api_type_list()
        # add a layout
        self._layout = QSettableVLayout(content_margin=(0, 0, 0, 0), spacing=15, alignment=None)
        self.setLayout(self._layout)
        # add a QLabelComboBox for the API type
        self._api_type_label_combobox = QLabelComboBox('API Type:', self._api_type_list, label_width=80)
        self._layout.addWidget(self._api_type_label_combobox)
        # add a widget for the baidu api
        self._baidu_api_widget = QWidget()
        self._baidu_api_widget_layout = QSettableHLayout(content_margin=(0, 0, 0, 0), spacing=0, alignment=None)
        self._baidu_api_widget.setLayout(self._baidu_api_widget_layout)
        self._baidu_api_app_id_lineedit = QLabelInput('App ID:', label_width=60)
        self._baidu_api_widget_layout.addWidget(self._baidu_api_app_id_lineedit)
        self._baidu_api_app_key_lineedit = QLabelInput('App Key:', label_width=70, password_mode=True)
        self._baidu_api_widget_layout.addWidget(self._baidu_api_app_key_lineedit)
        self._layout.addWidget(self._baidu_api_widget)
        # add a widget for the deepl api
        self._deepl_api_widget = QTranslaterAPIKeyContainer()
        # add a widget for the Google api
        self._google_api_widget = QTranslaterAPIKeyContainer()
        # add a widget for the youdao api
        self._youdao_api_widget = QWidget()
        self._youdao_api_widget_layout = QSettableHLayout(content_margin=(0, 0, 0, 0), spacing=0, alignment=None)
        self._youdao_api_widget.setLayout(self._youdao_api_widget_layout)
        self._youdao_api_app_id_lineedit = QLabelInput('App ID:', label_width=60)
        self._youdao_api_widget_layout.addWidget(self._youdao_api_app_id_lineedit)
        self._youdao_api_app_key_lineedit = QLabelInput('App Key:', label_width=70, password_mode=True)
        self._youdao_api_widget_layout.addWidget(self._youdao_api_app_key_lineedit)
        # add a widget for the openai api
        self._openai_api_widget = QWidget()
        self._openai_api_widget_layout = QSettableHLayout(content_margin=(0, 0, 0, 0), spacing=0, alignment=None)
        self._openai_api_widget.setLayout(self._openai_api_widget_layout)
        self._openai_api_app_key_combobox = QLabelComboBox('Model:', ['gpt-3.5-turbo', 'gpt-4'], label_width=70)
        self._openai_api_widget_layout.addWidget(self._openai_api_app_key_combobox)

        self._layout.addWidget(self._google_api_widget)
        self._layout.addWidget(self._youdao_api_widget)
        self._layout.addWidget(self._deepl_api_widget)
        self._layout.addWidget(self._openai_api_widget)
        # when the api type is changed, change the api widget
        self._api_type_label_combobox.currentTextChanged.connect(self._change_api_widget)
        self._load_data(data)
        self._change_api_widget(self._api_type_label_combobox.currentText)

    def _change_api_widget(self, api_type):
        if api_type == 'Baidu':
            self._baidu_api_widget.show()
            self._google_api_widget.hide()
            self._youdao_api_widget.hide()
            self._deepl_api_widget.hide()
            self._openai_api_widget.hide()
        elif api_type == 'Google':
            self._baidu_api_widget.hide()
            self._google_api_widget.show()
            self._youdao_api_widget.hide()
            self._deepl_api_widget.hide()
            self._openai_api_widget.hide()
        elif api_type == 'Youdao':
            self._baidu_api_widget.hide()
            self._google_api_widget.hide()
            self._youdao_api_widget.show()
            self._deepl_api_widget.hide()
            self._openai_api_widget.hide()
        elif api_type == 'DeepL':
            self._baidu_api_widget.hide()
            self._google_api_widget.hide()
            self._youdao_api_widget.hide()
            self._deepl_api_widget.show()
            self._openai_api_widget.hide()
        elif api_type == 'OpenAI':
            self._baidu_api_widget.hide()
            self._google_api_widget.hide()
            self._youdao_api_widget.hide()
            self._deepl_api_widget.hide()
            self._openai_api_widget.show()

    def _load_data(self, data: TranslaterConfigDataList):
        self._api_type_label_combobox.setCurrentText(
            AIChatEnum.TranslaterAPIType.from_value(
                data.get_active_translater_config().api_type).name if data.get_active_translater_config() else 'Baidu')
        for config in data:
            match config.api_type:
                case TranslaterAPIType.Baidu.value:
                    self._baidu_api_app_id_lineedit.input_content = config.app_id
                    self._baidu_api_app_key_lineedit.input_content = config.app_key
                case TranslaterAPIType.Google.value:
                    self._google_api_widget.api_key = config.api_key
                case TranslaterAPIType.Youdao.value:
                    self._youdao_api_app_id_lineedit.input_content = config.app_id
                    self._youdao_api_app_key_lineedit.input_content = config.app_key
                case TranslaterAPIType.DeepL.value:
                    self._deepl_api_widget.api_key = config.api_key
                case TranslaterAPIType.OpenAI.value:
                    self._openai_api_app_key_combobox.currentText = config.gpt_model

    def _data(self) -> TranslaterConfigDataList:
        data = TranslaterConfigDataList([
            {
                'api_type': TranslaterAPIType.Baidu.value,
                'app_id': self._baidu_api_app_id_lineedit.input_content,
                'app_key': self._baidu_api_app_key_lineedit.input_content,
                'active': 1 if self._api_type_label_combobox.currentText == 'Baidu' and self._baidu_api_app_id_lineedit.input_content and self._baidu_api_app_key_lineedit.input_content else 0
            },
            {
                'api_type': TranslaterAPIType.Google.value,
                'api_key': self._google_api_widget.api_key,
                'active': 1 if self._api_type_label_combobox.currentText == 'Google' and self._google_api_widget.api_key else 0
            },
            {
                'api_type': TranslaterAPIType.Youdao.value,
                'app_id': self._youdao_api_app_id_lineedit.input_content,
                'app_key': self._youdao_api_app_key_lineedit.input_content,
                'active': 1 if self._api_type_label_combobox.currentText == 'Youdao' and self._youdao_api_app_id_lineedit.input_content and self._youdao_api_app_key_lineedit.input_content else 0
            },
            {
                'api_type': TranslaterAPIType.DeepL.value,
                'api_key': self._deepl_api_widget.api_key,
                'active': 1 if self._api_type_label_combobox.currentText == 'DeepL' and self._deepl_api_widget.api_key else 0
            },
            {
                'api_type': TranslaterAPIType.OpenAI.value,
                'gpt_model': self._openai_api_app_key_combobox.currentText,
                'active': 1 if self._api_type_label_combobox.currentText == 'OpenAI' and self._openai_api_app_key_combobox.currentText else 0
            }
        ])
        return data

    def has_active_translater(self) -> bool:
        """
        check if there is an active translater
        :return:
        """
        return self._data().has_active_translater()

    data = Property(TranslaterConfigDataList, fget=_data, fset=_load_data)


class QTranslaterAPIKeyContainer(QWidget):
    def __init__(self, api_key_text=''):
        super().__init__()
        self._layout = QSettableHLayout(content_margin=(0, 0, 0, 0), spacing=15, alignment=None)
        self.setLayout(self._layout)
        self._api_key_lineedit = QLabelInput('API Key:', password_mode=True, label_width=80)
        self._layout.addWidget(self._api_key_lineedit)
        self._api_key_lineedit.input_content = api_key_text

    api_type = Property(str, fget=lambda self: self._api_type_label_combobox.currentText(),
                        fset=lambda self, value: self._api_type_label_combobox.setCurrentText(value))
    api_key = Property(str, fget=lambda self: self._api_key_lineedit.input_content,
                       fset=lambda self, value: self._api_key_lineedit.set_line_edit_text(value))


class QSelectableGroupBox(QWidget):
    """
    A group box with selectable buttons
    :param button_label_list: the label list of the buttons
    :param align_type: the alignment type of the buttons
    :param parent: the parent widget
    """
    selectionChanged = Signal(int)

    def __init__(self, button_label_list, align_type=AIChatEnum.AIGui.HorizontalAlignment, parent=None):
        super().__init__(parent)
        self._button_group = QButtonGroup()
        self._button_group.setExclusive(True)
        self._layout = QSettableHLayout(content_margin=(0, 0, 0, 0), spacing=0,
                                        alignment=Qt.AlignLeft) if align_type == AIChatEnum.AIGui.HorizontalAlignment else QSettableVLayout(
            content_margin=(0, 0, 0, 0), spacing=0, alignment=Qt.AlignTop)
        self.setLayout(self._layout)
        for button_label in button_label_list:
            button = QPushButton('    ' + button_label + '    ')
            button.setObjectName('selectable_group_box_button')
            button.setCheckable(True)
            self._button_group.addButton(button)
            self._button_group.setId(button, button_label_list.index(button_label))
            self._layout.addWidget(button)

        self._button_group.button(0).setChecked(True)
        # when the button is clicked, emit the signal
        self._button_group.buttonClicked.connect(self._button_clicked)

    def _button_clicked(self, button):
        self.selectionChanged.emit(self._button_group.id(button))


class OpenAISettingGroup(QWidget, DataGUIInterface):
    def __init__(self, data, parent=None, has_api_key=True):
        super().__init__(parent)
        self._has_api_key = has_api_key

        self.setObjectName('openai_setting_group')
        self._layout = QSettableVLayout(content_margin=(0, 0, 0, 0), spacing=15)
        self.setLayout(self._layout)

        if self._has_api_key:
            # in the first row, there is a label and an input box in a container for chatgpt api key
            self._chatgpt_api_key_input_box = QLabelInput('ChatGPT API Key: ', label_width=150, password_mode=True)
            self._layout.addWidget(self._chatgpt_api_key_input_box)
        # below the first row, there is a label and a combo box in a container for chatgpt model
        self._chatgpt_model_combo_box = QLabelComboBox('ChatGPT Model: ', ['gpt-3.5-turbo', 'gpt-4'],
                                                       label_width=150)
        self._layout.addWidget(self._chatgpt_model_combo_box)
        # in the second row, there is a label, a horizontal slider and an input box in a container
        self._temperature_input_box = QLabelSliderInput('Temperature: ', (0, 20), default_value=1)
        self._temperature_input_box.sliderValueChanged.connect(lambda: self._top_p_input_box.set_to_default())
        self._layout.addWidget(self._temperature_input_box)
        # in the third row, there is a label, a horizontal slider and an input box in a container
        self._top_p_input_box = QLabelSliderInput('Top P: ', (0, 10), default_value=1)
        self._top_p_input_box.sliderValueChanged.connect(lambda: self._temperature_input_box.set_to_default())
        self._layout.addWidget(self._top_p_input_box)
        # in the fourth row, there is a label, a horizontal slider and an input box in a container
        self._frequency_penalty_input_box = QLabelSliderInput('Frequency: ', (-20, 20), default_value=0)
        self._layout.addWidget(self._frequency_penalty_input_box)
        # in the fifth row, there is a label, a horizontal slider and an input box in a container
        self._presence_penalty_input_box = QLabelSliderInput('Presence: ', (-20, 20), default_value=0)
        self._layout.addWidget(self._presence_penalty_input_box)
        # in the sixth row, there is a label, a horizontal slider and an input box in a container
        self._max_tokens_input_box = QLabelSliderInput('Max Tokens: ', (0, 2048), False, default_value=512)
        self._layout.addWidget(self._max_tokens_input_box)
        self._load_data(data)

    def _load_data(self, data: OpenAIConfigData | GPTParamsData):
        if not data:
            return
        if self._has_api_key:
            self._chatgpt_api_key_input_box.input_content = data.openai_api_key
        self._chatgpt_model_combo_box.setCurrentText(data.model)
        self._temperature_input_box.value = data.temperature
        self._top_p_input_box.value = data.top_p
        self._frequency_penalty_input_box.value = data.frequency_penalty
        self._presence_penalty_input_box.value = data.presence_penalty
        self._max_tokens_input_box.value = data.max_tokens

    def _data(self) -> OpenAIConfigData | GPTParamsData:
        if self._has_api_key:
            data = OpenAIConfigData(openai_api_key=self._chatgpt_api_key_input_box.input_content,
                                    gpt_params={'model': self._chatgpt_model_combo_box.currentText,
                                                'temperature': self._temperature_input_box.value,
                                                'top_p': self._top_p_input_box.value,
                                                'frequency_penalty': self._frequency_penalty_input_box.value,
                                                'presence_penalty': self._presence_penalty_input_box.value,
                                                'max_tokens': self._max_tokens_input_box.value})
        else:
            data = GPTParamsData(model=self._chatgpt_model_combo_box.currentText,
                                 temperature=self._temperature_input_box.value,
                                 top_p=self._top_p_input_box.value,
                                 frequency_penalty=self._frequency_penalty_input_box.value,
                                 presence_penalty=self._presence_penalty_input_box.value,
                                 max_tokens=self._max_tokens_input_box.value)
        return data

    def is_set(self) -> bool:
        """
        Check if all the input boxes are set
        :return:
        """
        if self._has_api_key:
            return self._chatgpt_api_key_input_box.input_content and True
        else:
            return True

    data = Property(Data, _data, fset=_load_data)


class VITSSettingContainer(QWidget, DataGUIInterface):
    def __init__(self, data: VITSConfigData, parent=None):
        super().__init__()
        self._config_data = data
        self.setObjectName('vits_setting_container')
        self._layout = QSettableHLayout(content_margin=(0, 0, 0, 0), spacing=15)
        self.setLayout(self._layout)

        self._api_address_input_box = QLabelInput('API Address: ', text='e.g. 127.0.0.1')
        self._layout.addWidget(self._api_address_input_box)
        self._api_port_input_box = QLabelInput('API Port: ', label_width=70)
        self._layout.addWidget(self._api_port_input_box)

    @property
    def api_address(self):
        return self._api_address_input_box.input_content

    @api_address.setter
    def api_address(self, value):
        self._api_address_input_box.input_content = value

    @property
    def api_port(self):
        return self._api_port_input_box.input_content

    @api_port.setter
    def api_port(self, value):
        self._api_port_input_box.input_content = value


class VITSSettingGroup(QWidget, DataGUIInterface):
    def __init__(self, data: VITSConfigDataList, parent=None):
        super().__init__(parent)
        self._config_data = data
        self.setObjectName('vits_setting_group')
        self._layout = QSettableVLayout(content_margin=(0, 0, 0, 0), spacing=15)
        self.setLayout(self._layout)

        # in the ninth row, there is a label and a combo box in a container for vits api type
        self._vits_api_type_input_box = QLabelComboBox('API Type: ', data.api_type_list())
        self._vits_api_type_input_box.currentTextChanged.connect(self._change_vits_api_type)
        self._layout.addWidget(self._vits_api_type_input_box)
        self._vits_simple_setting_container = VITSSettingContainer(
            data.get_vits_config(SpeakerAPIType.VitsSimpleAPI.value))
        self._layout.addWidget(self._vits_simple_setting_container)
        self._nene_emotion_setting_container = VITSSettingContainer(
            data.get_vits_config(SpeakerAPIType.NeneEmotion.value))
        self._layout.addWidget(self._nene_emotion_setting_container)
        self._load_data(data)
        self._change_vits_api_type(self._vits_api_type_input_box.currentText)

    def _change_vits_api_type(self, api_type: str):
        if api_type == SpeakerAPIType.VitsSimpleAPI.name:
            self._vits_simple_setting_container.show()
            self._nene_emotion_setting_container.hide()
        elif api_type == SpeakerAPIType.NeneEmotion.name:
            self._vits_simple_setting_container.hide()
            self._nene_emotion_setting_container.show()

    def _load_data(self, data: VITSConfigDataList):
        self._vits_api_type_input_box.setCurrentText(SpeakerAPIType(
            data.get_active_vits_config().api_type).name if data.get_active_vits_config() else SpeakerAPIType.VitsSimpleAPI.name)
        for vits_config in data:
            if vits_config.api_type == SpeakerAPIType.VitsSimpleAPI.value:
                self._vits_simple_setting_container.api_address = vits_config.api_address
                self._vits_simple_setting_container.api_port = vits_config.api_port
            elif vits_config.api_type == SpeakerAPIType.NeneEmotion.value:
                self._nene_emotion_setting_container.api_address = vits_config.api_address
                self._nene_emotion_setting_container.api_port = vits_config.api_port

    def _data(self) -> VITSConfigDataList:
        return VITSConfigDataList([
            {
                'api_type': SpeakerAPIType.VitsSimpleAPI.value,
                'api_address': self._vits_simple_setting_container.api_address,
                'api_port': self._vits_simple_setting_container.api_port,
                'active': 1 if self._vits_api_type_input_box.currentText == 'VitsSimpleAPI' and self._vits_simple_setting_container.api_address and self._vits_simple_setting_container.api_port else 0
            },
            {
                'api_type': SpeakerAPIType.NeneEmotion.value,
                'api_address': self._nene_emotion_setting_container.api_address,
                'api_port': self._nene_emotion_setting_container.api_port,
                'active': 1 if self._vits_api_type_input_box.currentText == 'NeneEmotion' and self._nene_emotion_setting_container.api_address and self._nene_emotion_setting_container.api_port else 0
            }
        ])

    def has_active_vits(self):
        return self._data().has_active_vits()

    data = Property(VITSConfigDataList, _data, fset=_load_data)


class QRoundLabel(QLabel):
    def __init__(self, color, radius, parent):
        super().__init__(parent)
        self._color = color
        self._radius = radius

    def paintEvent(self, e: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self._color))
        painter.drawRoundedRect(self.rect(), self._radius, self._radius)
        super().paintEvent(e)


class LoadingAnimationButton(QAbstractButton):
    """
    A button with a loading animation.
    """
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hover = False
        self._click_animation = None
        self._animation = QSequentialAnimationGroup()
        self._hover_animation = QParallelAnimationGroup()
        self._animation_timer = QTimer()
        self._animation_timer.timeout.connect(self._loop_animation_start)
        self._width = 80
        self._height = 50
        center_x = self._width / 2
        center_y = self._height * 0.45
        eye_width = self._width * 0.1
        eye_height = self._height * 0.4
        spacing = self._width * 0.3
        self.setFixedSize(self._width, self._height)
        self.setStyleSheet('background: transparent;')
        self._right_eye = QRoundLabel(QColor('#33A6B8'), 5, self)
        self._left_eye = QRoundLabel(QColor('#33A6B8'), 5, self)
        self._init_animation(center_x, center_y, eye_width, eye_height, spacing)
        self._animation.setLoopCount(1)
        self._loop_animation_start()

    def _init_animation(self, center_x, center_y, eye_width, eye_height, spacing):
        eye_height_after = self._height * 0.01
        self._eye_blink = QParallelAnimationGroup()
        self._left_eye_blink = QPropertyAnimation(self._left_eye, b'geometry')
        self._left_eye_blink.setDuration(200)
        self._left_eye_blink.setStartValue(
            QRect(QPoint(int(center_x + spacing / 2 - eye_width / 2), int(center_y - eye_height / 2)),
                  QSize(int(eye_width), int(eye_height))))
        self._left_eye_blink.setKeyValueAt(0.5, QRect(
            QPoint(int(center_x + spacing / 2 - eye_width / 2), int(center_y - eye_height_after / 2)),
            QSize(int(eye_width), int(eye_height_after))))
        self._left_eye_blink.setEndValue(
            QRect(QPoint(int(center_x + spacing / 2 - eye_width / 2), int(center_y - eye_height / 2)),
                  QSize(int(eye_width), int(eye_height))))
        self._left_eye_blink.setEasingCurve(QEasingCurve.InCurve)
        self._right_eye_blink = QPropertyAnimation(self._right_eye, b'geometry')
        self._right_eye_blink.setDuration(200)
        self._right_eye_blink.setStartValue(
            QRect(QPoint(int(center_x - eye_width / 2 - spacing / 2), int(center_y - eye_height / 2)),
                  QSize(int(eye_width), int(eye_height))))
        self._right_eye_blink.setKeyValueAt(0.5, QRect(
            QPoint(int(center_x - eye_width / 2 - spacing / 2), int(center_y - eye_height_after / 2)),
            QSize(int(eye_width), int(eye_height_after))))
        self._right_eye_blink.setEndValue(
            QRect(QPoint(int(center_x - eye_width / 2 - spacing / 2), int(center_y - eye_height / 2)),
                  QSize(int(eye_width), int(eye_height))))
        self._right_eye_blink.setEasingCurve(QEasingCurve.InCurve)
        self._eye_blink.addAnimation(self._left_eye_blink)
        self._eye_blink.addAnimation(self._right_eye_blink)
        self._eye_blink.setLoopCount(2)
        self._animation.addAnimation(self._eye_blink)
        copied_left_eye_blink = self.copy_animation(self._left_eye_blink)
        copied_left_eye_blink.setKeyValueAt(0.5, QRect(
            QPoint(int(center_x + spacing / 2 - eye_width / 2), int(center_y - eye_height_after / 2)),
            QSize(int(eye_width), int(eye_height_after))))
        copied_right_eye_blink = self.copy_animation(self._right_eye_blink)
        copied_right_eye_blink.setKeyValueAt(0.5, QRect(
            QPoint(int(center_x - eye_width / 2 - spacing / 2), int(center_y - eye_height_after / 2)),
            QSize(int(eye_width), int(eye_height_after))))
        self._hover_animation.addAnimation(copied_left_eye_blink)
        self._hover_animation.addAnimation(copied_right_eye_blink)
        self._hover_animation.setLoopCount(2)
        self._eye_move = QParallelAnimationGroup()
        center_x_left = center_x - spacing * 0.3
        center_x_right = center_x + spacing * 0.5
        self._left_eye_move = QPropertyAnimation(self._left_eye, b'geometry')
        self._left_eye_move.setDuration(800)
        self._left_eye_move.setStartValue(
            QRect(QPoint(int(center_x - spacing / 2 - eye_width / 2), int(center_y - eye_height / 2)),
                  QSize(int(eye_width), int(eye_height))))
        self._left_eye_move.setKeyValueAt(0.3, QRect(
            QPoint(int(center_x_left - eye_width / 2 - spacing / 2), int(center_y - eye_height / 2)),
            QSize(int(eye_width), int(eye_height))))
        self._left_eye_move.setKeyValueAt(0.6, QRect(
            QPoint(int(center_x_right - eye_width / 2 - spacing / 2), int(center_y - eye_height / 2)),
            QSize(int(eye_width), int(eye_height))))
        self._left_eye_move.setEndValue(
            QRect(QPoint(int(center_x - spacing / 2 - eye_width / 2), int(center_y - eye_height / 2)),
                  QSize(int(eye_width), int(eye_height))))
        self._left_eye_move.setEasingCurve(QEasingCurve.InOutQuad)
        self._right_eye_move = QPropertyAnimation(self._right_eye, b'geometry')
        self._right_eye_move.setDuration(800)
        self._right_eye_move.setStartValue(
            QRect(QPoint(int(center_x - eye_width / 2 + spacing / 2), int(center_y - eye_height / 2)),
                  QSize(int(eye_width), int(eye_height))))
        self._right_eye_move.setKeyValueAt(0.3, QRect(
            QPoint(int(center_x_left - eye_width / 2 + spacing / 2), int(center_y - eye_height / 2)),
            QSize(int(eye_width), int(eye_height))))
        self._right_eye_move.setKeyValueAt(0.6, QRect(
            QPoint(int(center_x_right - eye_width / 2 + spacing / 2), int(center_y - eye_height / 2)),
            QSize(int(eye_width), int(eye_height))))
        self._right_eye_move.setEndValue(
            QRect(QPoint(int(center_x - eye_width / 2 + spacing / 2), int(center_y - eye_height / 2)),
                  QSize(int(eye_width), int(eye_height))))
        self._right_eye_move.setEasingCurve(QEasingCurve.InOutQuad)
        self._eye_move.addAnimation(self._left_eye_move)
        self._eye_move.addAnimation(self._right_eye_move)
        self._animation.addAnimation(self._eye_move)

    def _loop_animation_start(self):
        if not self._hover:
            self._hover_animation.stop()
            self._animation_timer.stop()
            self._animation_timer.setInterval(900)
            self._animation.start()
            self._animation_timer.start()
        else:
            self._animation.stop()
            self._animation_timer.stop()
            self._animation_timer.setInterval(2000)
            self._hover_animation.start()
            self._animation_timer.start()

    def enterEvent(self, event: QEnterEvent) -> None:
        self._hover = True
        self.setCursor(Qt.PointingHandCursor)
        self._loop_animation_start()

    def leaveEvent(self, event: QEnterEvent) -> None:
        self._hover = False
        self.setCursor(Qt.ArrowCursor)
        self._loop_animation_start()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        # if the mouse is not in the button, do nothing
        if not self.rect().contains(int(e.position().x()), int(e.position().y())):
            return
        else:
            self.clicked.emit()

    def paintEvent(self, e: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setPen(QPen(QBrush(QColor('#f2f2f2')), 3))
        painter.setBrush(QBrush(QColor('#ffffff')))
        # draw rounded rect
        painter.drawRoundedRect(self.rect(), 15, 15)

    @staticmethod
    def copy_animation(source_animation: QPropertyAnimation):
        target_object = source_animation.targetObject()
        property_name = source_animation.propertyName()
        copied_animation = QPropertyAnimation(target_object, property_name)
        copied_animation.setStartValue(source_animation.startValue())
        copied_animation.setEndValue(source_animation.endValue())
        copied_animation.setDuration(source_animation.duration())
        copied_animation.setEasingCurve(source_animation.easingCurve())
        return copied_animation
