"""
SettingsFrame Module
"""
import tkinter as tk
import tkinter.messagebox as messagebox
import inspect
from openaiclient.view.frame.baseframe import BaseFrame


class SettingsFrame(BaseFrame):
    """
    Class SettingsFrame:

    This class is for a frame that holds settings for
    a user to change. Created to work with the Completion
    and Edit requests to OpenAI.
    """

    def __init__(self, main, controller):
        super().__init__(main, controller)

        self._PADX = 10
        self._PADY = 5

        # variable to hold the settings we will be using
        # has a default value for testing.
        self.settings = {
            "testKey": "testVal",
            "testIntKey": 1,
            "testBoolKey": True,
            "testFloatKey": 1.0,
            "testNotInOptions": None
        }
        # hold the tk variables for returning inputs from the frame
        self.outputs = dict()

        # row count
        self.row = 0

    # creates all of the widgets for
    # the frame. This default method
    # can be overriden for different
    # types of input in other settings
    # classes.
    def create(self):
        """
        Creates the settings input frames by looping through the settings
        dictionary (self.settings) and creating a SettingsInputFrame object
        for each setting. It then attaches each frame in the next row on
        the frame with the grid() method.

        Finally, it creates the save and exit buttons and attaches them to
        the settings frame (self).
        """
        # key is the label, value is the value in the
        # dictionary of settings
        for key, value in self.settings.items():
            # get the name of the input type
            # and make the first letter lowercase
            typeOfValue = type(value).__name__
            typeOfValue = typeOfValue.replace(
                typeOfValue[0],
                typeOfValue[0].lower(),
                1
            )

            try:
                # get the correct setting function
                # based on value type
                func = getattr(self, typeOfValue+"Setting")
            except AttributeError:
                # skip the current attribute.
                # later, this will be logged into a logger
                print(f"No widget of type {typeOfValue} defined.")
                continue

            # get label and widget from function
            setting = func(key, value)

            # attach label
            setting.label.grid(
                column=0,
                row=self.row,
                padx=10,
                pady=5,
                sticky=tk.E
            )

            # attach widget
            setting.widget.grid(
                column=1,
                row=self.row,
                padx=self._PADX,
                pady=self._PADY,
                sticky=tk.W
            )

            # move down to the next row
            self.row += 1

        self.saveAndExitButtons()

        return self

    def saveSettings(self):
        """saves the settings inputted in the window"""
        for key in self.settings:
            try:
                val = self.outputs[key].get()
                self.setAttr(key, val)
            except KeyError:
                pass
            except tk.TclError as error:
                messagebox.showerror(
                    f"Incorrect input in {key}", f"Key \"{key}\" {str(error)}")
            except RuntimeError as error:
                messagebox.showerror(
                    f"Incorrect input in {key}", f"Key {str(error)}")
        
        self.exitSettings()

    def setAttr(self, key, val):
        """This should be overridden. Returns setter for saveSettings() method"""
        raise NotImplementedError

    def exitSettings(self):
        """exits the window without savings the changes"""
        self.master.destroy()

    def saveAndExitButtons(self):
        """Creates the save and exit button for the frame"""

        frame = tk.Frame(self)

        # button for savings settings
        tk.Button(
            master=frame,
            text="Save",
            command=self.saveSettings
        ).grid(
            column=0,
            row=0,
            padx=self._PADX,
            pady=self._PADY,
            ipadx=40
        )

        # button to close the window
        tk.Button(
            master=frame,
            text="Exit",
            command=self.exitSettings,
        ).grid(
            column=1,
            row=0,
            padx=self._PADX,
            pady=self._PADY,
            ipadx=40
        )

        frame.grid(
            column=0,
            row=self.row,
            columnspan=2,
            padx=self._PADX,
            pady=self._PADY
        )

    # string setting input
    def strSetting(self, key, value):
        """
        Creates a string setting input with the Entry object.
        """
        setting = self.Setting()

        setting.label = tk.Label(
            master=self,
            text=key
        )

        tkFunc = tk.Entry

        kwargs = self._kwargs(tkFunc, tk.StringVar, "textvariable", key)

        setting.widget = tkFunc(
            self,
            **kwargs
        )

        return setting

    # integer setting input
    def intSetting(self, key, value):
        """
        Creates an integer setting input with the Entry object.
        """
        setting = self.Setting()

        setting.label = tk.Label(
            master=self,
            text=key
        )

        tkFunc = tk.Entry

        kwargs = self._kwargs(tkFunc, tk.IntVar, "textvariable", key)

        setting.widget = tkFunc(
            self,
            **kwargs
        )

        return setting

    # float setting input
    def floatSetting(self, key, value):
        """
        Creates an integer setting input with the Entry object.
        """
        setting = self.Setting()

        setting.label = tk.Label(
            master=self,
            text=key
        )

        tkFunc = tk.Entry

        kwargs = self._kwargs(tkFunc, tk.DoubleVar, "textvariable", key)

        setting.widget = tkFunc(
            self,
            **kwargs
        )

        return setting

    # boolean setting input
    def boolSetting(self, key, value):
        """
        Creates a boolean setting input with on and off values as 1 and 0 and
        the input as a checkbutton.
        """
        setting = self.Setting()
        kwargs = dict()

        kwargs["onvalue"] = 1
        kwargs["offvalue"] = 0

        setting.label = tk.Label(
            master=self,
            text=key
        )

        tkFunc = tk.Checkbutton

        kwargs = self._kwargs(tkFunc, tk.IntVar, "variable", key, kwargs)

        setting.widget = tkFunc(
            self,
            **kwargs
        )

        return setting

    def _kwargs(self, tkFunc, tkVar, varKey, key, kwargs={}):
        """Sets up the kwargs for a setting input"""
        # add varKey: tkVar() to the kwargs
        kwargs = kwargs | {varKey: tkVar()}

        # point the output dictionary value key to
        # the tkVar() we just set up
        self.outputs[key] = kwargs[varKey]

        # set the value of the tkVar
        kwargs[varKey].set(self.settings[key])

        # return kwargs
        return kwargs

    class Setting:
        """Holds a setting with label and widget"""

        def __init__(self, label=None, widget=None):
            self._label = label
            self._widget = widget

        @property
        def label(self):
            """return label widget"""
            return self._label

        @label.setter
        def label(self, val):
            """set label widget"""
            self._label = val

        @property
        def widget(self):
            """return input widget"""
            return self._widget

        @widget.setter
        def widget(self, val):
            self._widget = val


if __name__ == "__main__":
    settingsWindow = tk.Tk()

    settingsFrame = SettingsFrame(
        main=settingsWindow,
        controller=None
    )

    settingsFrame.create()

    settingsFrame.pack()

    settingsWindow.mainloop()
