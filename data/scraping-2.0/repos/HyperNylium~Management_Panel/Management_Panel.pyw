###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
###
### Author/Creator: HyperNylium
###
### Website: http://www.hypernylium.com/
###
### GitHub: https://github.com/HyperNylium/
###
### CustomTkinter Version: 4.6.2 => 5.1.3 update
###
### License: Mozilla Public License Version 2.0
###
###
### TODO: Make the YT Downloader tab download audio files in a valid way rather than just downloading the video with only audio and converting it to audio
### TODO: Create a function like "configure_button_image(button, image_path, color, size)" to handle the repeated logic seen in lines 263, 266, 269.
###
### Done: Make a "Edit Mode" toggle both for "Games" and "Social Media" frames that will be on the "window" frame next to the X to close the menu.
###       It will allow you to edit the buttons in that frame. Will have functionality to add, remove, edit, and move button grid indexes (change order)
### DONE: Make a auto updater script that updates the main app instead of the user needing to go to github and download the new version
### DONE: Make a LaunchAtLogin setting in the settings.json file. This will create a shortcut in the startup folder (shell:startup) to launch the app on startup
### DONE: Make a Music tab that allows you to play music from a folder
### DONE: Fix window maximize issue on launch
### DONE: Fix assistant text boxes not being able to move up and down when the window height is changed
### DONE: make a check for updates function that checks for updates once clicked by a button instead of on launch
### DONE: when closing also save the width and height of the window for next launch in the settings.json
### DONE: make a dropdown menu in the settings tab for changing the default open tab on launch
### DONE: make all window.after() use schedule_create() instead
### DONE: finish making the app responsive
### DONE: re-make the check_for_updates() function.
### DISREGARDED: Instead of using tkinter.messagebox use CTkMessagebox (Didn't work out as i hoped it did. The library is not at fault, i just didn't like the way it worked)
###
###
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Imports
from os import system, startfile, execl, mkdir, rename, listdir, remove, getcwd, walk, makedirs
from os.path import exists, join, splitext, expanduser, relpath, abspath, splitdrive
from tkinter.messagebox import showerror, askyesno, showinfo
from subprocess import Popen, PIPE, CREATE_NO_WINDOW
from packaging.version import parse as parse_version
from tkinter import BooleanVar, DoubleVar, IntVar
from json import load as JSload, dump as JSdump
from datetime import datetime, date, timedelta
from tkinter.filedialog import askdirectory
from webbrowser import open as WBopen
from threading import Thread
from zipfile import ZipFile
from shutil import copy2
from time import sleep
import sys

try:
    from customtkinter import (
        CTk,
        CTkToplevel,
        CTkFrame, 
        CTkScrollableFrame, 
        CTkLabel, 
        CTkButton, 
        CTkImage, 
        CTkEntry, 
        CTkSwitch, 
        CTkOptionMenu, 
        CTkProgressBar, 
        CTkTextbox, 
        CTkSlider, 
        set_appearance_mode
    )
    from PIL.Image import open as PILopen, fromarray as PILfromarray
    from winshell import desktop, startup, CreateShortcut, shortcut
    from vlc import MediaPlayer, Media as vlcMedia
    from pytube import YouTube as PY_Youtube
    from requests.exceptions import Timeout
    from pyttsx3 import init as ttsinit
    from numpy import array as nparray
    from CTkListbox import CTkListbox
    from tinytag import TinyTag
    from requests import get
    import openai
except ImportError as importError:
    ModuleNotFound = str(importError).split("'")[1]
    usr_choice = askyesno(title="Import error", message=f"An error occurred while importing '{ModuleNotFound}'.\nWould you like to run the setup.bat script?")
    if usr_choice is True:
        system("setup.bat")
    sys.exit()

# Minimizes console window that launches with .py files if you want to use this app as a .py instead of a .pyw file
# ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 6)

# Sets the appearance mode of the window to dark 
# (in simpler terms, sets the window to dark mode).
# Don't want to burn them eyes now do we?
set_appearance_mode("dark") 

CurrentAppVersion = "4.3.2"
UpdateLink = "https://github.com/HyperNylium/Management_Panel"
DataTXTFileUrl = "http://www.hypernylium.com/projects/ManagementPanel/assets/data.txt"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

SETTINGSFILE = "settings.json"
model_prompt = "Hello, how can I help you today?"
UserDesktopDir = desktop() # shell:desktop
UserStartupDir = startup() # shell:startup
devices_per_row = 2  # Maximum number of devices per ro
DeviceFrames = []  # List to store references to DeviceFrame frames
devices = {} # dict to store device info (battey persetage, type)
after_events = {} # dict to store after events
all_buttons: list[CTkButton] = [] # list to store all buttons in the navigation bar
all_buttons_text: list[str] = [] # list to store all buttons text in the navigation bar
prev_x = 0 # variable to store previous x coordinate of the window
prev_y = 0 # variable to store previous y coordinate of the window
AppsLaucherGUISetup_max_buttons_per_row = 3 # Maximum number of buttons per row in the "Games" and "Social Media" frames
AppsLaucherGUISetup_row_num = 0 # Current row number in the "Games" and "Social Media" frames
AppsLaucherGUISetup_col_num = 0 # Current column number in the "Games" and "Social Media" frames


class TitleUpdater:
    def __init__(self, label: CTkLabel = None):
        if label is not None:
            self.label = label
        else:
            raise ValueError("label cannot be None")

    def start(self):
        """starts the time and date thread for infinite loop"""
        Thread(target=self.loop, daemon=True, name="TitleUpdater").start()

    def loop(self):
        while True:
            self.update()
            sleep(1)

    def update(self):
        """"Updates the title of the window to the current time and date"""
        if self.label is not None:
            if settings["AppSettings"]["NavigationState"] == "close":
                current_time = datetime.now().strftime('%I:%M %p')
                current_date = date.today().strftime('%d/%m/%Y')
            else:
                current_time = datetime.now().strftime('%I:%M:%S %p')
                current_date = date.today().strftime('%a, %b %d, %Y')
            self.label.configure(text=f"{current_date}\n{current_time}")
            del current_time, current_date
class MusicManager:
    def __init__(self):
        self.song_info = {} # Dictionary to store song info: {"song_name"(str): {"duration"(str): duration_in_seconds(int)}}
        self.song_list = []
        self.current_song_index = 0
        self.current_song_paused = False
        self.has_started_before = False
        self.music_dir_exists = None
        self.updating = False
        self.event_loop_running = False
        self.player = MediaPlayer()
        self.player.audio_set_volume(int(settings["MusicSettings"]["Volume"]))

    def cleanup(self):
        """Cleans up the music player"""
        self.event_loop_running = False
        try:
            self.player.stop()
            self.player.release()
        except:
            pass
        return

    def event_loop(self):
        """Event loop for the music player"""
        while self.event_loop_running:
            if self.is_playing() and not self.current_song_paused:
                current_pos_secs = self.player.get_time() / 1000 # Get current position in seconds
                total_duration = self.song_info[self.get_current_playing_song()]["duration"]
                remaining_time = total_duration - current_pos_secs

                formatted_remaining_time = str(timedelta(seconds=remaining_time)).split(".")[0]
                formatted_total_duration = str(timedelta(seconds=total_duration)).split(".")[0]

                time_left_label.configure(text=formatted_remaining_time)
                song_progressbar.set((current_pos_secs / total_duration))
                total_time_label.configure(text=formatted_total_duration)

                del current_pos_secs, total_duration, remaining_time, formatted_remaining_time, formatted_total_duration
                
            if not self.is_playing() and not self.current_song_paused and self.has_started_before:
                if settings["MusicSettings"]["LoopState"] == "all":
                    self.next()
                elif settings["MusicSettings"]["LoopState"] == "one":
                    self.player.stop()
                    self.play()
                elif settings["MusicSettings"]["LoopState"] == "off":
                    self.player.stop()
                    pre_song_btn.configure(state="disabled")
                    next_song_btn.configure(state="disabled")
                    play_pause_song_btn.configure(image=playimage, command=music_manager.play)
                    if self.updating is False:
                        SaveSettingsToJson("CurrentlyPlaying", False)
            sleep(1)
        return

    def start_event_loop(self):
        """Starts the event loop for the music player"""
        self.event_loop_running = True
        Thread(target=self.event_loop, daemon=True, name="MusicManager").start()

    def get_current_playing_song(self):
        """Returns the current playing song"""
        if self.event_loop_running:
            return self.song_list[self.current_song_index]

    def is_playing(self):
        """Returns True if a song is playing and False if not"""
        if self.event_loop_running:
            return bool(self.player.is_playing())

    def stop(self):
        """Stops the current song, resets the player and releases the media"""
        if self.is_playing():
            self.player.stop()

        if self.player.get_media():
            self.player.get_media().release()

        self.current_song_index = 0
        self.current_song_paused = False
        self.has_started_before = False

        stop_song_btn.configure(state="disabled")
        play_pause_song_btn.configure(image=playimage, command=self.play)
        all_music_frame.configure(label_text="Not playing anything")

        song_progressbar.set(0.0)
        time_left_label.configure(text="0:00:00")
        total_time_label.configure(text="0:00:00")

    def play(self):
        """Plays the current song or the first song in the list if no song is playing"""
        if len(self.song_list) > 0:
            if self.current_song_paused:
                self.player.play()
                self.current_song_paused = False
            else:
                song_path = join(settings["MusicSettings"]["MusicDir"], self.get_current_playing_song())
                self.player.set_media(vlcMedia(song_path))
                self.player.play()
                self.has_started_before = True
                self.current_song_paused = False
                if not self.updating:
                    all_music_frame.configure(label_text=f"Currently Playing: {splitext(self.get_current_playing_song())[0]}")
            stop_song_btn.configure(state="normal")
            pre_song_btn.configure(state="normal")
            next_song_btn.configure(state="normal")
            stop_song_btn.configure(state="normal")
            play_pause_song_btn.configure(image=pauseimage, command=music_manager.pause)
            SaveSettingsToJson("CurrentlyPlaying", True)
        return

    def pause(self):
        """Pauses the current playing song"""
        self.player.pause()
        self.current_song_paused = True
        stop_song_btn.configure(state="disabled")
        pre_song_btn.configure(state="disabled")
        next_song_btn.configure(state="disabled")
        play_pause_song_btn.configure(image=playimage, command=music_manager.play)
        if not self.updating:
            SaveSettingsToJson("CurrentlyPlaying", False)
        return

    def next(self):
        """Plays the next song in the list"""
        if len(self.song_list) > 0:
            self.current_song_index = (self.current_song_index + 1) % len(self.song_list)
            self.play()
        return

    def previous(self):
        """Plays the previous song in the list"""
        if len(self.song_list) > 0:
            self.current_song_index = (self.current_song_index - 1) % len(self.song_list)
            self.play()
        return

    def volume(self):
        """Sets the volume of the music player"""
        def savevolume():
            SaveSettingsToJson("Volume", musicVolumeVar.get())
        self.player.audio_set_volume(musicVolumeVar.get())
        volume_label.configure(text=f"{musicVolumeVar.get()}%")
        schedule_cancel(window, savevolume)
        schedule_create(window, 420, savevolume)
        return

    def loop(self):
        """Loops the current song or the whole playlist (keep in mind the playlist is the whole music directory)"""
        self.loopstate = settings["MusicSettings"]["LoopState"]
        if self.loopstate == "all":
            loop_playlist_btn.configure(image=CTkImage(change_image_clr(PILopen('assets/MusicPlayer/loop-1.png'), "#00ff00"), size=(25, 25)))
            SaveSettingsToJson("LoopState", "one")
        elif self.loopstate == "one":
            loop_playlist_btn.configure(image=CTkImage(change_image_clr(PILopen('assets/MusicPlayer/loop.png'), "#ff0000"), size=(25, 25)))
            SaveSettingsToJson("LoopState", "off")
        elif self.loopstate == "off":
            loop_playlist_btn.configure(image=CTkImage(change_image_clr(PILopen('assets/MusicPlayer/loop.png'), "#00ff00"), size=(25, 25)))
            SaveSettingsToJson("LoopState", "all")
        del self.loopstate
        return

    def changedir(self):
        """Changes the music directory"""
        if settings["MusicSettings"]["MusicDir"] != "" and exists(settings["MusicSettings"]["MusicDir"]):
            tmp_music_dir = askdirectory(title="Select Your Music Directory", initialdir=settings["MusicSettings"]["MusicDir"])
        else:
            tmp_music_dir = askdirectory(title="Select Your Music Directory", initialdir=expanduser("~"))
        if tmp_music_dir != "":
            SaveSettingsToJson("MusicDir", tmp_music_dir)
            self.stop()
            self.update()

        del tmp_music_dir
        return

    def update_all_music_frame(self):
        """Updates the all_music_frame frame"""
        delete_widgets = []
        for widget in all_music_frame.winfo_children():
            window.after(0, widget.grid_forget)
            delete_widgets.append(widget)

        for index, song_name in enumerate(self.song_list):
            CTkLabel(all_music_frame, text=f"{str(index+1)}. {splitext(song_name)[0]}", font=("sans-serif", 20)).grid(
                row=index, column=1, padx=(20, 0), pady=5, sticky="w")

        if len(delete_widgets) > 0:
            for widget in delete_widgets:
                window.after(0, widget.destroy)
                del widget

        all_music_frame.update()
        if self.music_dir_exists:
            all_music_frame.configure(label_text=f"Currently Playing: {splitext(self.get_current_playing_song())[0]}" if self.has_started_before and len(self.song_list) > 0 else "Not playing anything")

        self.updating = False

        del delete_widgets
        return

    def update(self):
        """Updates the music player"""
        def update_song_list():
            MusicDir = str(settings["MusicSettings"]["MusicDir"])
            if exists(MusicDir):
                self.music_dir_exists = True
                self.song_list = [file for file in listdir(MusicDir) if file.endswith(tuple(TinyTag.SUPPORTED_FILE_EXTENSIONS))]
                for song_name in self.song_list:
                    song_path = join(MusicDir, song_name)
                    tag = TinyTag.get(song_path)
                    self.song_info[song_name] = {"duration": tag.duration}
            else:
                self.music_dir_exists = False
                self.song_list = []
                all_music_frame.configure(label_text="Please choose a valid music directory by clicking the 'Change' button")
                SaveSettingsToJson("MusicDir", "")
                MusicDir = ""

            Thread(target=self.update_all_music_frame, daemon=True, name="update_all_music_frame").start()

            update_music_list.configure(state="normal")
            change_music_dir.configure(state="normal")
            pre_song_btn.configure(state="normal")
            play_pause_song_btn.configure(state="normal")
            next_song_btn.configure(state="normal")
            volume_slider.configure(state="normal")
            music_dir_label.configure(text=f"Music Directory: {shorten_path(MusicDir, 25)}" if MusicDir != "" else "Music Directory: None")

            if settings["MusicSettings"]["CurrentlyPlaying"] == True:
                self.play()
            return

        self.updating = True
        update_music_list.configure(state="disabled")
        change_music_dir.configure(state="disabled")
        stop_song_btn.configure(state="disabled")
        pre_song_btn.configure(state="disabled")
        play_pause_song_btn.configure(state="disabled")
        next_song_btn.configure(state="disabled")
        volume_slider.configure(state="disabled")
        all_music_frame.configure(label_text="Updating...")
        song_progressbar.set(0.0)
        time_left_label.configure(text="0:00:00")
        total_time_label.configure(text="0:00:00")

        if self.is_playing() and not self.current_song_paused:
            self.pause()

        Thread(target=update_song_list, daemon=True, name="MusicManager_updater").start()


def StartUp():
    """Main function that gets the app going. Should be called only once at the start of the app"""

    try:
        window.iconbitmap("assets/AppIcon/Management_Panel_Icon.ico")
    except Exception as e:
        showerror(title="Error loading window icon", message=f"An error occurred while loading the window icon\n{e}")
        sys.exit()

    global settings
    global UserPowerPlans, settingsSpeakResponceVar, settingsAlwayOnTopVar, settingslaunchwithwindowsvar
    global settingsCheckForUpdates, settingsAlphavar, musicVolumeVar, music_manager, EditModeVar

    default_settings = {
        "URLs": {
            "HyperNylium.com": "http://hypernylium.com/",
            "Github": "https://github.com/HyperNylium",
            "Discord": "https://discord.gg/4FHTjAgw95",
            "Instagram": "https://www.instagram.com/",
            "Youtube": "https://www.youtube.com/",
            "TikTok": "https://www.tiktok.com/",
            "Facebook": "https://www.facebook.com/",
            "Twitter": "https://twitter.com/"
        },
        "GameShortcutURLs": {
            "Game 1": "",
            "Game 2": "",
            "Game 3": ""
        },
        "OpenAISettings": {
            "VoiceType": 0,
            "OpenAI_API_Key": "",
            "OpenAI_model_engine": "text-davinci-003",
            "OpenAI_Max_Tokens": 1024,
            "OpenAI_Temperature": 0.5
        },
        "MusicSettings": {
            "MusicDir": "",
            "Volume": 0,
            "CurrentlyPlaying": False,
            "LoopState": "all"
        },
        "AppSettings": {
            "PreviouslyUpdated": False,
            "AlwaysOnTop": False,
            "LaunchAtLogin": False,
            "SpeakResponce": False,
            "CheckForUpdatesOnLaunch": True,
            "NavigationState": "open",
            "DownloadsFolderName": "YT_Downloads",
            "DefaultFrame": "Home",
            "Alpha": 1.0,
            "Window_State": "normal",
            "Window_Width": "",
            "Window_Height": "",
            "Window_X": "",
            "Window_Y": ""
        },
        "Devices": []
    }
    try:
        with open(SETTINGSFILE, 'r') as settings_file:
            settings = JSload(settings_file)

        if settings["AppSettings"]["PreviouslyUpdated"] == True:
            settings.update(default_settings)
            settings["AppSettings"]["PreviouslyUpdated"] = False
            with open(SETTINGSFILE, 'w') as settings_file:
                JSdump(settings, settings_file, indent=2)

    except FileNotFoundError:
        with open(SETTINGSFILE, 'w') as settings_file:
            JSdump(default_settings, settings_file, indent=2)
        settings = default_settings

    settingsSpeakResponceVar = BooleanVar()
    settingsAlwayOnTopVar = BooleanVar()
    settingslaunchwithwindowsvar = BooleanVar()
    settingsCheckForUpdates = BooleanVar()
    settingsAlphavar = DoubleVar()
    musicVolumeVar = IntVar()
    EditModeVar = BooleanVar()
    UserPowerPlans = None

    if settings["AppSettings"]["AlwaysOnTop"] == True:
        window.attributes('-topmost', True)
        settingsAlwayOnTopVar.set(True)

    if settings["AppSettings"]["LaunchAtLogin"] == True:
        shortcut_target_path = shortcut(join(UserStartupDir, "Management_Panel.lnk")).path
        if shortcut_target_path != file_path():
            reset_LaunchOnStartup_shortcut()
            SaveSettingsToJson("LaunchAtLogin", True)
        else:
            settingslaunchwithwindowsvar.set(True)
        del shortcut_target_path
    elif exists(join(UserStartupDir, "Management_Panel.lnk")):
        usr_res = askyesno(title="Startup shortcut found", message="Despite 'LaunchAtLogin' being turned off, we've discovered a startup shortcut for this app.\nWould you like the app to still lauch on startup?")
        if usr_res is True:
            reset_LaunchOnStartup_shortcut()
            SaveSettingsToJson("LaunchAtLogin", True)
        else:
            settingslaunchwithwindowsvar.set(False)
            LaunchOnStartupTrueFalse()
            SaveSettingsToJson("LaunchAtLogin", False)

    if settings["AppSettings"]["SpeakResponce"] == True:
        settingsSpeakResponceVar.set(True)

    if isinstance(settings["MusicSettings"]["Volume"], int):
        musicVolumeVar.set(settings["MusicSettings"]["Volume"])
    elif isinstance(settings["MusicSettings"]["Volume"], float):
        musicVolumeVar.set(int(settings["MusicSettings"]["Volume"]))

    check_for_updates_startup()

    music_manager = MusicManager()
def restart():
    """Restarts app"""
    python = sys.executable
    execl(python, python, *sys.argv)
def on_closing():
    """App termination function"""
    SaveSettingsToJson("CurrentlyPlaying", False)
    music_manager.cleanup()
    window.destroy()
    sys.exit()
def schedule_create(widget, ms, func, cancel_after_finished=False, *args, **kwargs):
    """Schedules a function to run after a given time in milliseconds and saves the event id in a dictionary with the function name as the key"""
    event_id = widget.after(ms, lambda: func(*args, **kwargs))
    after_events[func.__name__] = event_id
    if cancel_after_finished:
        widget.after(ms, lambda: schedule_cancel(widget, func))
def schedule_cancel(widget, func):
    """Cancels a scheduled function and deletes the event id from the dictionary using the function name as the parameter instead of the event id"""
    try:
        event_id = after_events.get(func.__name__)
        if event_id is not None:
            widget.after_cancel(event_id)
            del after_events[func.__name__]
    except: 
        pass
def NavbarAction(option: str):
    """Opens or closes the navigation bar and saves the state to settings.json"""
    SaveSettingsToJson("NavigationState", str(option))
    if option == "close":
        for button in all_buttons:
            button.configure(text="", anchor="center")
        navigation_frame_label.pack_configure(padx=7)
        navigation_frame_label.configure(font=("sans-serif", 15, "bold"))
        close_open_nav_button.configure(image=openimage, command=lambda: NavbarAction("open"))
        window.minsize(550, 420)
    elif option == "open":
        for button in all_buttons:
            button.configure(text=all_buttons_text[all_buttons.index(button)], anchor="w")
        navigation_frame_label.pack_configure(padx=20)
        navigation_frame_label.configure(font=("sans-serif", 18, "bold"))
        close_open_nav_button.configure(image=closeimage, command=lambda: NavbarAction("close"))
        window.minsize(650, 420)
    title_bar.update()

def get_data_content():
    """
    Gets the data from the data.txt file and returns it as a dictionary
    > Success: (Version, DevName, LastEditDate)
    > Error: (error, errorTitle, errorBody, info)
    """
    data = {}

    try:
        response = get(DataTXTFileUrl, timeout=3, headers=headers)
        lines = response.text.split('\n')
        delimiter = "="

        for line in lines:
            key_value = line.split(delimiter, 1)
            if len(key_value) == 2:
                key = key_value[0].strip()
                value = key_value[1].strip().replace(" ", "")
                data[key] = value

    except Timeout:
        data = {"error": "timeout", "errorTitle": "Request timed out", "errorBody": "Main data file request timed out\nThis can happen because:\n> You are offline\n> The webserver is not hosting the file at the moment\n> Your internet connection is slow\n\nThe app will now start in offline mode.", "info": "- timed out"}
    except Exception as e:
        data = {"error": "Unknown error", "errorTitle": "Launching in offline mode", "errorBody": "There was an error while retrieving the main data file\nThis can happen because:\n> You are offline\n> The webserver is not hosting the file at the moment\n\nThe app will now start in offline mode.", "info": "- offline mode"}

    return data
def check_for_updates_startup():
    """Checks for updates on startup if the user has the setting enabled"""
    global LiveAppVersion, Developer, LastEditDate, ShowUserInfo
    if settings["AppSettings"]["CheckForUpdatesOnLaunch"] == True:
        settingsCheckForUpdates.set(True)

        live_data = get_data_content()

        if "error" in live_data:
            LiveAppVersion = "N/A"
            Developer = "N/A"
            LastEditDate = "N/A"
            ShowUserInfo = live_data["info"]
            showerror(title=live_data["errorTitle"], message=live_data["errorBody"])
            return

        LiveAppVersion = live_data["Version"]
        Developer = live_data["DevName"]
        LastEditDate = live_data["LastEditDate"]

        live_version = parse_version(live_data["Version"])
        current_version = parse_version(CurrentAppVersion)
        if live_version < current_version:
            Developer = "Unknown"
            LastEditDate = "Unknown"
            ShowUserInfo = "- Unauthentic"
        elif live_version > current_version:
            ShowUserInfo = f"- Update available (v{live_version})"
        else:
            ShowUserInfo = "- Latest version"
    else:
        settingsCheckForUpdates.set(False)
        LiveAppVersion = "N/A"
        Developer = "N/A"
        LastEditDate = "N/A"
        ShowUserInfo = "- Check disabled"
    return
def check_for_updates_GUI():
    """Checks for updates (GUI only)"""
    global LiveAppVersion, Developer, LastEditDate, ShowUserInfo

    check_for_updates_button.configure(text="Checking for updates...", state="disabled")

    live_data = get_data_content()

    if "error" in live_data:
        LiveAppVersion = "N/A"
        Developer = "N/A"
        LastEditDate = "N/A"
        ShowUserInfo = live_data["info"]
        return

    LiveAppVersion = live_data["Version"]
    Developer = live_data["DevName"]
    LastEditDate = live_data["LastEditDate"]

    live_version = parse_version(live_data["Version"])
    current_version = parse_version(CurrentAppVersion)
    if live_version < current_version:
        Developer = "Unknown"
        LastEditDate = "Unknown"
        ShowUserInfo = "- Unauthentic"
    elif live_version > current_version:
        ShowUserInfo = f"- Update available (v{live_version})"
    else:
        ShowUserInfo = "- Latest version"

    home_frame_label_1.configure(text=f"Version: {CurrentAppVersion} {ShowUserInfo}")
    home_frame_label_2.configure(text=f"Creator/developer: {Developer}")
    home_frame_label_3.configure(text=f"Last updated: {LastEditDate}")
    check_for_updates_button.configure(text="Check complete", state="disabled")
    schedule_create(window, 3500, lambda: check_for_updates_button.configure(text="Check for updates", state="normal"), True)

    return
def check_for_updates_silent():
    """Checks for updates silently"""
    global LiveAppVersion, Developer, LastEditDate, ShowUserInfo

    live_data = get_data_content()

    if "error" in live_data:
        LiveAppVersion = "N/A"
        Developer = "N/A"
        LastEditDate = "N/A"
        ShowUserInfo = live_data["info"]
        return

    LiveAppVersion = live_data["Version"]
    Developer = live_data["DevName"]
    LastEditDate = live_data["LastEditDate"]

    live_version = parse_version(live_data["Version"])
    current_version = parse_version(CurrentAppVersion)
    if live_version < current_version:
        Developer = "Unknown"
        LastEditDate = "Unknown"
        ShowUserInfo = "- Unauthentic"
    elif live_version > current_version:
        ShowUserInfo = f"- Update available (v{live_version})"
    else:
        ShowUserInfo = "- Latest version"

    home_frame_label_1.configure(text=f"Version: {CurrentAppVersion} {ShowUserInfo}")
    home_frame_label_2.configure(text=f"Creator/developer: {Developer}")
    home_frame_label_3.configure(text=f"Last updated: {LastEditDate}")

    return

def on_drag_end(event):
    global prev_x, prev_y

    # Check if x and y values have changed
    if prev_x != event.x or prev_y != event.y:
        # Update the x and y values
        prev_x = event.x
        prev_y = event.y

        # The main point about the upcoming code is that when you start to
        # drag the window, it first cancels any existing function call to on_drag_stopped()
        # but then creates a new one right after that. This means that instead of the on_drag_stopped()
        # function being called every single pixel you move the window, it will
        # be called only after the window has stopped moving for 420ms.
        # this happens because the on_drag_end() gets called probably hundreds of times
        # you move the window. But with this code, it will only call the on_drag_stopped()
        # function once after you stop moving the window for 420ms because it keeps cancelling
        # the function call before it can even start. Cool, right? This is a more modefied version
        # so it works with this project (i am sure it would work with any project though)
        # you can find the full version that was built with tkinter and uses the .after() method
        # intead of my custom schedule_create() and schedule_cancel() functions here: https://github.com/HyperNylium/tkinter-window-drag-detection

        # Cancel any existing threshold check
        schedule_cancel(window, on_drag_stopped)

        # Schedule a new threshold check after 420ms
        schedule_create(window, 420, on_drag_stopped)
    return
def on_drag_stopped():
    if window.state() == "zoomed":
        SaveSettingsToJson("Window_State", "maximized")
    elif window.state() == "normal":
        SaveSettingsToJson("Window_State", "normal")
        SaveSettingsToJson("Window_Width", window.winfo_width())
        SaveSettingsToJson("Window_Height", window.winfo_height())
        SaveSettingsToJson("Window_X", window.winfo_x())
        SaveSettingsToJson("Window_Y", window.winfo_y())
    return

def systemsettings(setting: str):
    """Launches different settings within windows 10 and 11 (only tested on windows 11)"""
    if setting == "power":
        Popen(
            "cmd.exe /c control powercfg.cpl",
            stdout=PIPE,
            stderr=PIPE,
            stdin=PIPE,
            creationflags=CREATE_NO_WINDOW,
        )
    elif setting == "display":
        Popen(
            "cmd.exe /c control desk.cpl",
            stdout=PIPE,
            stderr=PIPE,
            stdin=PIPE,
            creationflags=CREATE_NO_WINDOW,
        )
    elif setting == "network":
        Popen(
            "cmd.exe /c %systemroot%\system32\control.exe /name Microsoft.NetworkAndSharingCenter",
            stdout=PIPE,
            stderr=PIPE,
            stdin=PIPE,
            creationflags=CREATE_NO_WINDOW,
        )
    elif setting == "sound":
        Popen(
            "cmd.exe /c control mmsys.cpl sounds",
            stdout=PIPE,
            stderr=PIPE,
            stdin=PIPE,
            creationflags=CREATE_NO_WINDOW,
        )
    elif setting == "apps":
        Popen(
            "cmd.exe /c start ms-settings:appsfeatures",
            stdout=PIPE,
            stderr=PIPE,
            stdin=PIPE,
            creationflags=CREATE_NO_WINDOW,
        )  # Put "appwiz.cpl" for control center version
    elif setting == "storage":
        Popen(
            "cmd.exe /c start ms-settings:storagesense",
            stdout=PIPE,
            stderr=PIPE,
            stdin=PIPE,
            creationflags=CREATE_NO_WINDOW,
        )
    elif setting == "windowsupdate":
        Popen(
            "cmd.exe /c %systemroot%\system32\control.exe /name Microsoft.WindowsUpdate",
            stdout=PIPE,
            stderr=PIPE,
            stdin=PIPE,
            creationflags=CREATE_NO_WINDOW,
        )
    elif setting == "taskmanager":
        Popen(
            "cmd.exe /c taskmgr",
            stdout=PIPE,
            stderr=PIPE,
            stdin=PIPE,
            creationflags=CREATE_NO_WINDOW,
        )
    elif setting == "vpn":
        Popen(
            "cmd.exe /c start ms-settings:network-vpn",
            stdout=PIPE,
            stderr=PIPE,
            stdin=PIPE,
            creationflags=CREATE_NO_WINDOW,
        )
    elif setting == "netdrive":
        system("cmd.exe /c netdrive -reset")
        showinfo(
            title="Network drives reset", message="All network drives have been reset"
        )

    del setting
    return
def LaunchGame(game_url: str = None, game_name: str = None, placed_frame: str = None) -> None:
    """Launches selected game"""

    if EditModeVar.get() is True:
        EditButton(game_name, game_url, placed_frame)
        return

    if game_url == None or game_url == "" or game_name == None or game_name == "":
        showerror(
            title="No game link found",
            message="Make sure you have configured a game shortcut link in you're settings and try restarting the app",
        )
    else:
        usr_input = askyesno(title="You are about to launch a game", message=f"Are you sure you want to launch '{game_name}'?\nClick 'Yes' to continue and 'No' to cancel.")
        if usr_input is True:
            WBopen(game_url)
        del usr_input
    del game_url, game_name
    return
def SocialMediaLoader(media_url: str = None, media_name: str = None, placed_frame: str = None) -> None:
    """Launches a website URL (either http or https)"""
    if EditModeVar.get() is True:
        EditButton(media_name, media_url, placed_frame)
        return
    WBopen(media_url)
    del media_url, media_name

def CenterWindowToDisplay(Screen: CTk, width: int, height: int, scale_factor: float = 1.0):
    """Centers the window to the main display/monitor"""
    screen_width = Screen.winfo_screenwidth()
    screen_height = Screen.winfo_screenheight()
    x = int(((screen_width/2) - (width/2)) * scale_factor)
    y = int(((screen_height/2) - (height/1.5)) * scale_factor)
    return f"{width}x{height}+{x}+{y}"
def CenterWindowToMain(Screen: CTkToplevel, width: int, height: int):
    """Centers the window to the main tkinter window"""
    main_screen_width = Screen.winfo_width()
    main_screen_height = Screen.winfo_height()
    main_screen_X = Screen.winfo_x()
    main_screen_Y = Screen.winfo_y()
    x = main_screen_X + (main_screen_width - width) // 2
    y = main_screen_Y + (main_screen_height - height) // 2
    return f"{width}x{height}+{int(x)}+{int(y)}"
def ResetWindowPos():
    """Resets window positions in settings.json"""
    SaveSettingsToJson("Window_State", "normal")
    SaveSettingsToJson("Window_Width", "")
    SaveSettingsToJson("Window_Height", "")
    SaveSettingsToJson("Window_X", "")
    SaveSettingsToJson("Window_Y", "")
    restart()

def AppsLaucherGUISetup(frame: str):
    global AppsLaucherGUISetup_row_num, AppsLaucherGUISetup_col_num

    if frame == "games_frame":
        master_frame = games_frame
        Property = "GameShortcutURLs"
        cmd = LaunchGame
    elif frame == "socialmedia_frame":
        master_frame = socialmedia_frame
        Property = "URLs"
        cmd = SocialMediaLoader
    else:
        return

    for button in master_frame.winfo_children():
        button.destroy()

    for url_name, url in settings[Property].items():
        CTkButton(master_frame, width=200, text=url_name, compound="top", fg_color=("gray75", "gray30"), font=("sans-serif", 22), corner_radius=10, command=lambda cmd=cmd, url=url, url_name=url_name, placed_frame=frame: cmd(url, url_name, placed_frame)).grid(row=AppsLaucherGUISetup_row_num, column=AppsLaucherGUISetup_col_num, padx=5, pady=10)
        AppsLaucherGUISetup_col_num += 1
        if AppsLaucherGUISetup_col_num >= AppsLaucherGUISetup_max_buttons_per_row:
            AppsLaucherGUISetup_col_num = 0
            AppsLaucherGUISetup_row_num += 1

    AppsLaucherGUISetup_row_num = 0
    AppsLaucherGUISetup_col_num = 0

    del frame, Property, cmd, master_frame
def EditModeInit():
    value = EditModeVar.get()
    if value is True:
        for button in games_frame.winfo_children():
            button.configure(fg_color="#1364cf")
            button.update()
        for button in socialmedia_frame.winfo_children():
            button.configure(fg_color="#1364cf")
            button.update()
    elif value is False:
        for button in games_frame.winfo_children():
            button.configure(fg_color=("gray75", "gray30"))
            button.update()
        for button in socialmedia_frame.winfo_children():
            button.configure(fg_color=("gray75", "gray30"))
            button.update()
    del value
    return
def EditButton(btn_title: str, btn_url: str, placed_frame: str):
    def remove_selected_btn():
        for button in master_frame.winfo_children():
            if button.cget('text') == btn_title:
                button.destroy()
                break
        del settings[Property][btn_title]
        with open(SETTINGSFILE, 'w') as settings_file:
            JSdump(settings, settings_file, indent=2)
        reload_func()
        EditModeInit()
        editmodewindow.destroy()
    def save_new_btn():
        new_title = button_title.get("0.0", "end-1c")
        new_url = button_url.get("0.0", "end-1c")
        for button in master_frame.winfo_children():
            if button.cget('text') == btn_title:
                button.configure(text=new_title, command=lambda cmd=cmd, url=new_url, url_name=new_title, placed_frame=placed_frame: cmd(url, url_name, placed_frame))
                button.update()
                break
        original_index = list(settings[Property].keys()).index(btn_title)
        del settings[Property][btn_title]
        new_url_dict = {}
        for idx, key in enumerate(settings[Property].keys()):
            if idx == original_index:
                new_url_dict[new_title] = new_url
            new_url_dict[key] = settings[Property][key]
        settings[Property] = new_url_dict
        with open(SETTINGSFILE, 'w') as settings_file:
            JSdump(settings, settings_file, indent=2)
        del new_title, new_url, original_index, new_url_dict
        editmodewindow.destroy()
    def preview_new_btn():
        editmodewindowpreview = CTkToplevel()
        editmodewindowpreview.title(f"Preview '{button_title.get('0.0', 'end-1c')}' button")
        editmodewindowpreview.attributes('-topmost', True)
        editmodewindowpreview.geometry(CenterWindowToMain(window, 400, 150))
        editmodewindowpreview.resizable(False, False)
        editmodewindowpreview.grab_set()

        new_btn = CTkButton(editmodewindowpreview, text=button_title.get("0.0", "end-1c"), font=("sans-serif", 22), fg_color=("gray75", "gray30"), corner_radius=10, command=lambda: WBopen(button_url.get("0.0", "end-1c")))
        new_btn.pack(pady=50)
    def change_button_position():
        def item_selected(listbox_selected_item):
            nonlocal selected_item
            selected_item = listbox_selected_item
        def move_selected_item_up():
            nonlocal selected_item
            if selected_item is None:
                return
            selected_item_index = listbox_items.index(selected_item)
            new_index = (selected_item_index - 1) % len(listbox_items)

            item_to_move_up = listbox_items[selected_item_index]
            listbox_items[selected_item_index] = listbox_items[new_index]
            listbox_items[new_index] = item_to_move_up

            listofbtns.delete(0, "end")
            for item in listbox_items:
                listofbtns.insert("end", item)
            listofbtns.activate(new_index)
            selected_item_index =- new_index
        def move_selected_item_down():
            nonlocal selected_item
            if selected_item is None:
                return
            selected_item_index = listbox_items.index(selected_item)
            new_index = (selected_item_index + 1) % len(listbox_items)

            item_to_move_up = listbox_items[selected_item_index]
            listbox_items[selected_item_index] = listbox_items[new_index]
            listbox_items[new_index] = item_to_move_up

            listofbtns.delete(0, "end")
            for item in listbox_items:
                listofbtns.insert("end", item)
            listofbtns.activate(new_index)
            selected_item_index =+ new_index
        def save_config():
            nonlocal selected_item
            if selected_item is None:
                return
            new_config = {}
            for item in listbox_items:
                new_config[item] = settings[Property][item]
            settings[Property] = new_config
            with open(SETTINGSFILE, 'w') as settings_file:
                JSdump(settings, settings_file, indent=2)
            reload_func()
            EditModeInit()
            changepositionwindow.destroy()
            editmodewindow.destroy()

        if placed_frame == "games_frame":
            frame_name = "Games"
            master_frame = games_frame
            Property = "GameShortcutURLs"
            reload_func = lambda: AppsLaucherGUISetup("games_frame")
        elif placed_frame == "socialmedia_frame":
            frame_name = "Social Media"
            master_frame = socialmedia_frame
            Property = "URLs"
            reload_func = lambda: AppsLaucherGUISetup("socialmedia_frame")

        changepositionwindow = CTkToplevel()
        changepositionwindow.title(f"Modify button positions for '{frame_name}' ")
        changepositionwindow.attributes('-topmost', True)
        changepositionwindow.geometry(CenterWindowToMain(window, 500, 450))
        changepositionwindow.resizable(False, False)
        changepositionwindow.grab_set()

        selected_item = None
        listbox_items = []
        listofbtns = CTkListbox(changepositionwindow, command=item_selected, font=("sans-serif", 22))
        listofbtns.pack(fill="both", expand=True, padx=10, pady=10)

        for button in master_frame.winfo_children():
            listofbtns.insert("END", button.cget('text'))
            listbox_items.append(button.cget('text'))

        currently_modifying_item_index = listbox_items.index(btn_title)
        listofbtns.activate(currently_modifying_item_index)

        move_up_btn = CTkButton(changepositionwindow, text="Move Up", font=("sans-serif", 22), fg_color=("gray75", "gray30"), corner_radius=10, command=move_selected_item_up)
        move_up_btn.pack(side="left", padx=5, pady=(20, 10), fill="x", expand=True, anchor="center")

        save_config_btn = CTkButton(changepositionwindow, text="Save", font=("sans-serif", 22), fg_color=("gray75", "gray30"), corner_radius=10, command=save_config)
        save_config_btn.pack(side="left", padx=5, pady=(20, 10), fill="x", expand=True, anchor="center")

        move_down_btn = CTkButton(changepositionwindow, text="Move Down", font=("sans-serif", 22), fg_color=("gray75", "gray30"), corner_radius=10, command=move_selected_item_down)
        move_down_btn.pack(side="left", padx=5, pady=(20, 10), fill="x", expand=True, anchor="center")

    if placed_frame == "games_frame":
        master_frame = games_frame
        Property = "GameShortcutURLs"
        cmd = LaunchGame
        reload_func = lambda: AppsLaucherGUISetup("games_frame")
    elif placed_frame == "socialmedia_frame":
        master_frame = socialmedia_frame
        Property = "URLs"
        cmd = SocialMediaLoader
        reload_func = lambda: AppsLaucherGUISetup("socialmedia_frame")

    editmodewindow = CTkToplevel()
    editmodewindow.title(f"Modify '{btn_title}' button")
    editmodewindow.attributes('-topmost', True)
    editmodewindow.geometry(CenterWindowToMain(window, 650, 450))
    editmodewindow.resizable(False, False)
    editmodewindow.grab_set()

    button_title_label = CTkLabel(editmodewindow, text="Button Title", font=("sans-serif", 25))
    button_title_label.pack(padx=10, pady=(20, 10), anchor="center")
    button_title = CTkTextbox(editmodewindow, width=30, height=5, border_width=0, corner_radius=10, font=("sans-serif", 22), activate_scrollbars=True, border_color="#242424")
    button_title.insert("0.0", btn_title)
    button_title.pack(fill="x", padx=10, pady=10)

    button_url_label = CTkLabel(editmodewindow, text="Button URL", font=("sans-serif", 25))
    button_url_label.pack(padx=10, pady=(20, 10), anchor="center")
    button_url = CTkTextbox(editmodewindow, width=30, height=100, border_width=0, corner_radius=10, font=("sans-serif", 22), activate_scrollbars=True, border_color="#242424")
    button_url.insert("0.0", btn_url)
    button_url.pack(fill="x", padx=10, pady=10)

    btn_level_1_frame = CTkFrame(editmodewindow, corner_radius=0, fg_color="transparent")
    btn_level_2_frame = CTkFrame(editmodewindow, corner_radius=0, fg_color="transparent")
    btn_level_1_frame.pack(fill="x", anchor="center")
    btn_level_2_frame.pack(fill="x", anchor="center")

    remove_btn = CTkButton(btn_level_1_frame, width=315, text="Remove", font=("sans-serif", 22), fg_color=("gray75", "gray30"), corner_radius=10, command=remove_selected_btn)
    remove_btn.grid(row=1, column=1, padx=5, pady=(20, 10), sticky="ew")

    save_btn = CTkButton(btn_level_1_frame, width=315, text="Save", font=("sans-serif", 22), fg_color=("gray75", "gray30"), corner_radius=10, command=save_new_btn)
    save_btn.grid(row=1, column=2, padx=5, pady=(20, 10), sticky="ew")

    preview_btn = CTkButton(btn_level_2_frame, width=315, text="Preview", font=("sans-serif", 22), fg_color=("gray75", "gray30"), corner_radius=10, command=preview_new_btn)
    preview_btn.grid(row=1, column=1, padx=5, pady=(20, 10), sticky="ew")

    change_btn_position = CTkButton(btn_level_2_frame, width=315, text="Edit Position", font=("sans-serif", 22), fg_color=("gray75", "gray30"), corner_radius=10, command=change_button_position)
    change_btn_position.grid(row=1, column=2, padx=5, pady=(20, 10), sticky="ew")
def AddButton(placed_frame: str):
    def preview_new_btn():
        addbtnwindowpreview = CTkToplevel()
        addbtnwindowpreview.title(f"Preview '{button_title.get('0.0', 'end-1c')}' button")
        addbtnwindowpreview.attributes('-topmost', True)
        addbtnwindowpreview.geometry(CenterWindowToMain(window, 400, 150))
        addbtnwindowpreview.resizable(False, False)
        addbtnwindowpreview.grab_set()

        new_btn = CTkButton(addbtnwindowpreview, text=button_title.get("0.0", "end-1c"), font=("sans-serif", 22), fg_color=("gray75", "gray30"), corner_radius=10, command=lambda: WBopen(button_url.get("0.0", "end-1c")))
        new_btn.pack(pady=50)
    def save_new_btn():
        new_title = button_title.get("0.0", "end-1c")
        new_url = button_url.get("0.0", "end-1c")
        settings[Property][new_title] = new_url
        with open(SETTINGSFILE, 'w') as settings_file:
            JSdump(settings, settings_file, indent=2)
        reload_func()
        EditModeInit()
        del new_title, new_url
        addbtnwindow.destroy()

    if placed_frame == "games_frame":
        frame_name = "Games"
        Property = "GameShortcutURLs"
        reload_func = lambda: AppsLaucherGUISetup("games_frame")
    elif placed_frame == "socialmedia_frame":
        frame_name = "Social Media"
        Property = "URLs"
        reload_func = lambda: AppsLaucherGUISetup("socialmedia_frame")

    addbtnwindow = CTkToplevel()
    addbtnwindow.title(f"Add button to '{frame_name}'")
    addbtnwindow.attributes('-topmost', True)
    addbtnwindow.geometry(CenterWindowToMain(window, 500, 370))
    addbtnwindow.resizable(False, False)
    addbtnwindow.grab_set()

    button_title_label = CTkLabel(addbtnwindow, text="Button Title", font=("sans-serif", 25))
    button_title_label.pack(padx=10, pady=(20, 10), anchor="center")
    button_title = CTkTextbox(addbtnwindow, width=30, height=5, border_width=0, corner_radius=10, font=("sans-serif", 22), activate_scrollbars=True, border_color="#242424")
    button_title.pack(fill="x", padx=10, pady=10)

    button_url_label = CTkLabel(addbtnwindow, text="Button URL", font=("sans-serif", 25))
    button_url_label.pack(padx=10, pady=(20, 10), anchor="center")
    button_url = CTkTextbox(addbtnwindow, width=30, height=100, border_width=0, corner_radius=10, font=("sans-serif", 22), activate_scrollbars=True, border_color="#242424")
    button_url.pack(fill="x", padx=10, pady=10)

    save_btn = CTkButton(addbtnwindow, text="Save", font=("sans-serif", 22), fg_color=("gray75", "gray30"), corner_radius=10, command=save_new_btn)
    save_btn.pack(side="left", padx=5, pady=(20, 10), fill="x", expand=True, anchor="center")

    preview_btn = CTkButton(addbtnwindow, text="Preview", font=("sans-serif", 22), fg_color=("gray75", "gray30"), corner_radius=10, command=preview_new_btn)
    preview_btn.pack(side="left", padx=5, pady=(20, 10), fill="x", expand=True, anchor="center")

def AlwaysOnTopTrueFalse():
    """Sets the window to always be on top or not and saves the state to settings.json"""
    value = settingsAlwayOnTopVar.get()
    window.attributes('-topmost', value)
    SaveSettingsToJson("AlwaysOnTop", value)
    del value
    return
def LaunchOnStartupTrueFalse():
    value = settingslaunchwithwindowsvar.get()
    if value is True:
        CreateShortcut(
            Path=join(UserStartupDir, "Management_Panel.lnk"),
            Target=file_path(),
            StartIn=getcwd(),
            Description="Shortcut for launching 'Management_Panel.pyw'",
            Icon=(join(getcwd(), "assets", "AppIcon", "Management_Panel_Icon.ico"), 0),
        )
    else:
        try:
            remove(join(UserStartupDir, "Management_Panel.lnk"))
        except FileNotFoundError:
            pass

    SaveSettingsToJson("LaunchAtLogin", value)
    del value
    return
def set_alpha(alpha_var: float):
    """Sets the window transparency and saves the state to settings.json"""
    def save_alpha_settings():
        SaveSettingsToJson("Alpha", alpha_var)
    window.attributes('-alpha', alpha_var)
    schedule_cancel(window, save_alpha_settings)
    schedule_create(window, 420, save_alpha_settings)
    del save_alpha_settings
    return

def YTVideoDownloaderContentType(vidtype: str):
    """Updates the video content type to either .mp4 or .mp3 according to whatever was selected in the dropdown"""
    global YTVideoContentType
    YTVideoContentType = vidtype
    del vidtype
def YTVideoDownloader(ContentType: str):
    """Downloads youtube videos and shows progress on GUI"""
    def DefaultStates(**kwargs):
        option = kwargs.get("option")
        if option == "YTReset":
            ytdownloader_progressbarpercentage.configure(text="0%")
            ytdownloader_progressbar.configure(progress_color="#1f6aa5")
            ytdownloader_progressbar.set(0)
            ytdownloader_progressbarpercentage.grid_forget()
            ytdownloader_progressbar.grid_forget()
            ytdownloader_OptionMenu.grid_configure(row=2, column=0, columnspan=2, padx=0, pady=0)
            ytdownloader_frame_button_1.grid_configure(row=3, rowspan=2, column=0, columnspan=2, padx=10, pady=20, sticky="ews")
            ytdownloader_entry.configure(state="normal")
            ytdownloader_OptionMenu.configure(state="normal")
            ytdownloader_frame_button_1.configure(text="Download", state="normal")
        elif option == "ErrorReset":
            ytdownloader_error_label.grid_forget()
            ytdownloader_OptionMenu.grid_configure(row=2, column=0, columnspan=2, padx=0, pady=0)
            ytdownloader_frame_button_1.grid_configure(row=3, rowspan=2, column=0, columnspan=2, padx=10, pady=20, sticky="ews")
            ytdownloader_entry.configure(state="normal")
            ytdownloader_OptionMenu.configure(state="normal")
            ytdownloader_frame_button_1.configure(text="Download", state="normal")
            window.update()
    def on_download_progress(stream, chunk, bytes_remaining):
        TotalSize = stream.filesize
        BytesDownloaded = TotalSize - bytes_remaining
        RawPersentage = BytesDownloaded / TotalSize * 100
        ConvertedPersentage = str(int(RawPersentage))
        ytdownloader_progressbarpercentage.configure(text=f"{ConvertedPersentage}%")
        ytdownloader_progressbar.set(float(ConvertedPersentage) / 100)
        ytdownloader_progressbarpercentage.update()
        ytdownloader_progressbar.update()
        ytdownloader_frame.update()
        if ConvertedPersentage == "100":
            ytdownloader_progressbar.configure(progress_color="green")
            ytdownloader_progressbar.update()
            schedule_create(window, 3500, DefaultStates, True, **{"option": "YTReset"})
    def YTDownloadThread():
        try:
            videourl = ytdownloader_entry.get().strip()
            if (videourl != "") and (videourl != None):
                ytdownloader_entry.configure(state="disabled")
                ytdownloader_OptionMenu.configure(state="disabled")
                ytdownloader_frame_button_1.configure(text="Downloading...", state="disabled")
                YTObject = PY_Youtube(videourl)
                CreatePath = join(UserDesktopDir, settings["AppSettings"]["DownloadsFolderName"])
                if not exists(CreatePath):
                    try:
                        mkdir(CreatePath)
                        mkdir(f"{CreatePath}\\Video")
                        mkdir(f"{CreatePath}\\Audio")
                    except OSError as error:
                        showerror(title="An error occurred", message=f"An error occurred while creating the downloads folder\nMore detailed error: {error}")
                        return
                if ContentType == "Video (.mp4)":
                    Video = YTObject.streams.get_by_resolution("720p")
                    VideoFilename = f"{CreatePath}\\Video\\{Video.default_filename}"
                    ytdownloader_OptionMenu.grid_configure(row=3, column=0, columnspan=2, padx=0, pady=0)
                    ytdownloader_frame_button_1.grid_configure(row=4, rowspan=2, column=0, columnspan=2, padx=10, pady=20, sticky="ews")
                    if exists(VideoFilename):
                        ytdownloader_error_label.configure(text=f"The video already exists", text_color="red")
                        ytdownloader_error_label.grid(row=2, column=0, columnspan=2, padx=0, pady=0)
                        schedule_create(window, 4000, DefaultStates, True, **{"option": "ErrorReset"})
                    else:
                        ytdownloader_progressbar.grid(row=2, column=0, columnspan=2, padx=20, pady=0, sticky="ew")
                        ytdownloader_progressbarpercentage.grid(row=2, column=0, columnspan=2, padx=0, pady=0)
                        DownloadThread = Thread(name="VidDownloadThread", daemon=True, target=lambda: Video.download(output_path=f"{CreatePath}\\Video"))
                        CallbackThread = Thread(name="VidCallbackThread", daemon=True, target=lambda: YTObject.register_on_progress_callback(on_download_progress))
                        DownloadThread.start()
                        CallbackThread.start()
                    ytdownloader_frame.update()
                elif ContentType == "Audio (.mp3)":
                    Audio = YTObject.streams.filter(only_audio=True).first()
                    BaseFilename, BaseFileext = splitext(Audio.default_filename)
                    YTVideoName = f"{CreatePath}\\Audio\\{Audio.default_filename}"
                    MP3Filename = f"{CreatePath}\\Audio\\{BaseFilename +'.mp3'}"
                    ytdownloader_OptionMenu.grid_configure(row=3, column=0, columnspan=2, padx=0, pady=0)
                    ytdownloader_frame_button_1.grid_configure(row=4, rowspan=2, column=0, columnspan=2, padx=10, pady=20, sticky="ews")
                    if (exists(YTVideoName)) or (exists(MP3Filename)):
                        ytdownloader_error_label.configure(text=f"The audio (.mp3/.mp4) already exists", text_color="red")
                        ytdownloader_error_label.grid(row=2, column=0, columnspan=2, padx=0, pady=0)
                        schedule_create(window, 4000, DefaultStates, True, **{"option": "ErrorReset"})
                    else:
                        def on_complete(stream, file_handle):
                            rename(file_handle, MP3Filename)
                        ytdownloader_progressbar.grid(row=2, column=0, columnspan=2, padx=20, pady=0, sticky="ew")
                        ytdownloader_progressbarpercentage.grid(row=2, column=0, columnspan=2, padx=0, pady=0)
                        CallbackThread = Thread(name="AudioCallbackThread", daemon=True, target=lambda: YTObject.register_on_progress_callback(on_download_progress))
                        DownloadThread = Thread(name="AudioDownloadThread", daemon=True, target=lambda: Audio.download(output_path=f"{CreatePath}\\Audio"))
                        CompleteCallbackThread = Thread(name="AudioCompleteCallbackThread", daemon=True, target=lambda: YTObject.register_on_complete_callback(on_complete))
                        CallbackThread.start()
                        DownloadThread.start()
                        CompleteCallbackThread.start()
                    ytdownloader_frame.update()
                else:
                    showerror(title="Unknown format", message=f"Format invalid. Only\nVideo: Video (.mp4)\nAudio = Audio (.mp3)")
                    return
        except Exception as viderror:
            viderror = str(viderror)
            if (viderror == "'streamingData'"):
                YTVideoDownloader(ContentType)
            elif (viderror == "regex_search: could not find match for (?:v=|\/)([0-9A-Za-z_-]{11}).*"):
                # showerror(title="URL error", message="The URL that you have inputed does not seem to be a vaild URL.\nPlease make sure you are inputing an actual URL from youtube")
                ytdownloader_OptionMenu.grid_configure(row=3, column=0, columnspan=2, padx=0, pady=0)
                ytdownloader_frame_button_1.grid_configure(row=4, rowspan=2, column=0, columnspan=2, padx=10, pady=20, sticky="ews")
                ytdownloader_error_label.configure(text=f"The link that you inputed is not a valid link", text_color="red")
                ytdownloader_error_label.grid(row=2, column=0, columnspan=2, padx=0, pady=0)
            else:
                showerror(title="Unknown error occurred", message=f"An unknown error occurred. Heres the log:\n{viderror}")
            schedule_create(window, 4000, DefaultStates, True, **{"option": "ErrorReset"})
    YTThread = Thread(name="YTDownloadThread", daemon=True, target=YTDownloadThread)
    YTThread.start()

def speak(audio):
    """Speaks any string that you give it. It runs on its own thread so it doesn't block the main thread by default.\n
    speak('hello world')
    """
    def engineSpeak(audio):
        engine = ttsinit('sapi5')
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[settings["OpenAISettings"]["VoiceType"]].id)
        engine.say(audio)
        engine.runAndWait()
        del engine
    SpeechThread = Thread(name="SpeechThread", daemon=True, target=lambda: engineSpeak(audio))
    SpeechThread.start()
def ChatGPT():
    """Sends requests to ChatGPT and puts Response in text box"""

    if (settings["OpenAISettings"]["OpenAI_API_Key"] == "") or (settings["OpenAISettings"]["OpenAI_API_Key"] == None):
        ans = askyesno(title="OpenAI API Key Error", message=f"It seems like you don't have an OpenAI API Key set.\nWould you like to set one now?")
        if ans:
            ChatGPTConfig()
        del ans
        return

    UserText = assistant_responce_box_1.get("0.0", "end").strip("\n")
    if UserText != "" and UserText != None:
        def generate_response(prompt):
            try:
                openai.api_key = settings["OpenAISettings"]["OpenAI_API_Key"]
                response = openai.Completion.create(
                    engine=settings["OpenAISettings"]["OpenAI_model_engine"],
                    prompt=prompt,
                    max_tokens=settings["OpenAISettings"]["OpenAI_Max_Tokens"],
                    temperature=settings["OpenAISettings"]["OpenAI_Temperature"]
                )
                message = response.choices[0].text.strip()
                assistant_responce_box_2.delete("0.0", "end")
                assistant_responce_box_2.insert("end", message)
                if settingsSpeakResponceVar.get() == True:
                    speak(message)
            except Exception as e:
                assistant_responce_box_2.delete("0.0", "end")
                showerror(title="OpenAI Error", message=e)
        assistant_responce_box_2.delete("0.0", "end")
        assistant_responce_box_2.insert("end", "Thinking...")
        prompt = f"User: {UserText}"
        AIThread = Thread(name="AIThread", daemon=True, target=lambda: generate_response(model_prompt + "\n" + prompt))
        AIThread.start()
def ChatGPTConfig():
    """Opens a window that allows you to modify the settings for ChatGPT"""

    def save_config():
        SaveSettingsToJson("OpenAI_API_Key", api_key_textbox.get("0.0", "end-1c").strip().replace("\n", "").replace(" ", ""))
        SaveSettingsToJson("OpenAI_Max_Tokens", int(max_tokens_var.get()))
        SaveSettingsToJson("OpenAI_Temperature", round(temperature_var.get(), 1))
        if voice_type_optionmenu.get() == "Male":
            SaveSettingsToJson("VoiceType", 0)
        else:
            SaveSettingsToJson("VoiceType", 1)
        chatgptconfigwindow.destroy()

    chatgptconfigwindow = CTkToplevel()
    chatgptconfigwindow.title("Modify ChatGPT settings")
    chatgptconfigwindow.attributes('-topmost', True)
    chatgptconfigwindow.geometry(CenterWindowToMain(window, 650, 450))
    chatgptconfigwindow.resizable(False, False)
    chatgptconfigwindow.grab_set()
    temperature_var = DoubleVar()
    max_tokens_var = IntVar()
    temperature_var.set(settings["OpenAISettings"]["OpenAI_Temperature"])
    max_tokens_var.set(settings["OpenAISettings"]["OpenAI_Max_Tokens"])

    model_engine_label = CTkLabel(chatgptconfigwindow, text=f"Model Engine: {settings['OpenAISettings']['OpenAI_model_engine']}", font=("sans-serif", 22))
    model_engine_label.pack(padx=10, pady=(20, 10), anchor="nw")

    api_key_label = CTkLabel(chatgptconfigwindow, text="OpenAI API Key", font=("sans-serif", 22))
    api_key_label.pack(padx=10, pady=(20, 10), anchor="center")
    api_key_textbox = CTkTextbox(chatgptconfigwindow, width=30, height=3, border_width=0, corner_radius=10, font=("sans-serif", 22), activate_scrollbars=True, border_color="#242424")
    api_key_textbox.insert("0.0", settings["OpenAISettings"]["OpenAI_API_Key"])
    api_key_textbox.pack(fill="x", padx=10, pady=10)

    max_tokens_frame = CTkFrame(chatgptconfigwindow, corner_radius=0, fg_color="transparent")
    max_tokens_frame.grid_columnconfigure(1, weight=1)
    max_tokens_frame.pack(fill="x", anchor="center")
    max_tokens_slider = CTkSlider(max_tokens_frame, width=300, height=20, from_=0, to=4000, command=lambda mtokens: max_tokens_label.configure(text=int(mtokens)), variable=max_tokens_var)
    max_tokens_slider.grid(row=1, column=1, padx=5, pady=(20, 10), sticky="ew")
    max_tokens_label = CTkLabel(max_tokens_frame, text=f"{settings['OpenAISettings']['OpenAI_Max_Tokens']}", font=("sans-serif", 22))
    max_tokens_label.grid(row=1, column=2, padx=5, pady=(20, 10), sticky="ew")

    temperature_frame = CTkFrame(chatgptconfigwindow, corner_radius=0, fg_color="transparent")
    temperature_frame.grid_columnconfigure(1, weight=1)
    temperature_frame.pack(fill="x", anchor="center")
    temperature_slider = CTkSlider(temperature_frame, width=300, height=20, from_=0.0, to=1.0, command=lambda temp: temperature_label.configure(text=round(temp, 1)), variable=temperature_var)
    temperature_slider.grid(row=1, column=1, padx=5, pady=(20, 10), sticky="ew")
    temperature_label = CTkLabel(temperature_frame, text=f"{settings['OpenAISettings']['OpenAI_Temperature']}", font=("sans-serif", 22))
    temperature_label.grid(row=1, column=2, padx=5, pady=(20, 10), sticky="ew")

    voice_type_frame = CTkFrame(chatgptconfigwindow, corner_radius=0, fg_color="transparent")
    voice_type_frame.grid_columnconfigure([1, 2], weight=1)
    voice_type_frame.pack(fill="x", anchor="center")
    voice_type_label = CTkLabel(voice_type_frame, text="Voice Type", font=("sans-serif", 22))
    voice_type_label.grid(row=1, column=1, padx=5, pady=(20, 10), sticky="e")
    voice_type_optionmenu = CTkOptionMenu(voice_type_frame, values=["Male", "Female"], command=None, fg_color="#343638", button_color="#4d4d4d", button_hover_color="#444", font=("sans-serif", 20), dropdown_font=("sans-serif", 17), anchor="center")
    voice_type_optionmenu.grid(row=1, column=2, padx=5, pady=(20, 10), sticky="w")
    if settings["OpenAISettings"]["VoiceType"] == 0:
        voice_type_optionmenu.set("Male")
    else:
        voice_type_optionmenu.set("Female")

    save_btn = CTkButton(chatgptconfigwindow, width=315, text="Save", font=("sans-serif", 22), fg_color=("gray75", "gray30"), corner_radius=10, command=save_config)
    save_btn.pack(padx=5, pady=(20, 10), fill="x", expand=True, anchor="center")

    api_key_textbox.focus_force()

def UpdatePowerPlans():
    """Gets all power plans that are listed at:\n
    control panel > hardware and sound > power options"""

    def update():
        global UserPowerPlans

        # Run a command to list the power plans without showing a window. I do nothing with the error output because this command does not return an error
        output, error = Popen(["powercfg", "/list"], stdout=PIPE, stderr=PIPE, stdin=PIPE, creationflags=CREATE_NO_WINDOW).communicate()
        output_text = output.decode("utf-8")

        # Extract power plan information from the output
        power_plans = {}
        for line in output_text.splitlines():
            if "GUID" in line:
                guid = line.split("(")[1].split(")")[0]
                name = line.split(":")[1].split("(")[0].strip()
                power_plans[guid] = name

        # Find the active power plan
        active_plan = None
        for line in output_text.splitlines():
            if "*" in line and "GUID" in line:
                active_plan = line.split("(")[1].split(")")[0]
                break

        # Store the active power plan in the dictionary
        power_plans["active"] = active_plan

        UserPowerPlans = power_plans

        # update the optionmenu with the power plans
        system_frame_power_optionmenu.configure(values=list(power_plans.keys())[:-1], command=lambda PlanName: ChangePowerPlan(PlanName), state="normal")
        system_frame_power_optionmenu.set(power_plans['active'])
        del output, error, output_text, power_plans, active_plan
        return

    system_frame_power_optionmenu.configure(values=["Loading..."], command=None, state="disabled")
    Thread(name="UpdatePowerPlansThread", daemon=True, target=update).start()
def ChangePowerPlan(PlanName: str):
    """Changes the selected power plan"""
    if PlanName not in UserPowerPlans:
        showerror(title="Power plan error", message=f"The power plan that you selected\n> {PlanName}\ndoesn't seem to exist.")
        return
    PowerPlanGUID = UserPowerPlans[PlanName]
    Popen(["powercfg", "/setactive", PowerPlanGUID], stdout=PIPE, stderr=PIPE, stdin=PIPE, creationflags=CREATE_NO_WINDOW)
    del PowerPlanGUID
    return

def GetDeviceInfo(device_name: str = None, connectivity_type: str = "Bluetooth", device_battery_data: str = "{104EA319-6EE2-4701-BD47-8DDBF425BBE5} 2", device_type_data: str = "DEVPKEY_DeviceContainer_Category"):
    """Gets Bluetooth device information based on the provided parameters"""
    if device_name is None:
        raise ValueError("device_name must be provided")

    try:
        powershell_script = f"""
            $devices = Get-PnpDevice -Class '{connectivity_type}' | Where-Object {{$_.FriendlyName -eq '{device_name}'}}
            $deviceProperties = $devices | Get-PnpDeviceProperty | Where-Object {{$_.KeyName -eq '{device_battery_data}' -or $_.KeyName -eq '{device_type_data}'}}
            $batteryData = $deviceProperties | Where-Object {{$_.KeyName -eq '{device_battery_data}'}}
            $typeData = $deviceProperties | Where-Object {{$_.KeyName -eq '{device_type_data}'}}
            $batteryData.Data, $typeData.Data
        """
        result = Popen(["powershell.exe", "-WindowStyle", "Hidden", "-Command", powershell_script], stdout=PIPE, stderr=PIPE, stdin=PIPE, creationflags=CREATE_NO_WINDOW)
        output, error = result.communicate()
        output = output.decode("utf-8")
        script_output = output.strip().splitlines()
        if len(script_output) == 2:
            battery_percentage, device_type = script_output
            return int(battery_percentage), str(device_type)
        else:
            return None
    except:
        return None
def AllDeviceDetails():
    """Gets all device details and puts the bluetooth devices on the GUI in different frames and uses different icons for each device type (mouse, keyboard, headphones)"""
    def UpdateDevices():
        devices_refresh_btn.grid_forget()
        refreshinglabel = CTkLabel(devices_frame, text="Refreshing...", font=("Arial", 25))
        refreshinglabel.place(relx=0.5, rely=0.4, anchor="center")

        for frame in DeviceFrames:
            frame.destroy()
        DeviceFrames.clear()
        devices.clear()

        for device in settings["Devices"]:
            DeviceDetails = GetDeviceInfo(device)
            devices[device] = DeviceDetails

        for index, (device_name, device_data) in enumerate(devices.items()):
            row = (index // devices_per_row) + 1  # Calculate the row number based on the index and skip the first row
            column = index % devices_per_row  # Calculate the column number based on the index

            deviceFrame = CTkFrame(devices_frame, bg_color="#242424")
            deviceFrame.grid(row=row, column=column, pady=10, padx=10, sticky="nesw")
            DeviceFrames.append(deviceFrame)  # Add the frame to the list

            if len(device_name) > 17:
                    device_name = device_name[:17] + "..."

            if device_data is not None:
                percentage, device_type = device_data

                # Set the header image based on the device type
                if device_type == 'Input.Mouse':
                    header_image = CTkImage(PILopen("assets/ExtraIcons/mouse.png"), size=(50, 50))
                elif device_type == 'Input.Keyboard': # Experimental value
                    header_image = CTkImage(PILopen("assets/ExtraIcons/keyboard.png"), size=(50, 50))
                elif (device_type == 'Audio.Headphone') or (device_type == 'Audio.Headset') or (device_type == 'Audio.Speaker') or (device_type == 'Communication.Headset.Bluetooth'):
                    header_image = CTkImage(PILopen("assets/ExtraIcons/headphones.png"), size=(50, 50))
                else:
                    header_image = CTkImage(PILopen("assets/ExtraIcons/unknown_device.png"), size=(50, 50))

                header_label = CTkLabel(deviceFrame, image=header_image, text="")
                header_label.grid(row=0, column=0, rowspan=2, padx=10)

                # Display the device name and percentage
                name_label = CTkLabel(deviceFrame, text=device_name, font=("Arial", 20, "bold"))
                name_label.grid(row=0, column=1, columnspan=2, sticky="w", padx=(0, 20), pady=(10, 0))

                percentage_label = CTkLabel(deviceFrame, text=f"Percentage: {str(percentage)}%", font=("Arial", 17))
                percentage_label.grid(row=1, column=1, columnspan=2, sticky="w", padx=(0, 20), pady=(0, 10))
            else:
                header_image = CTkImage(PILopen("assets/ExtraIcons/unknown_device.png"), size=(50, 50))
                header_label = CTkLabel(deviceFrame, image=header_image, text="")
                header_label.grid(row=0, column=0, rowspan=2, padx=10)

                name_label = CTkLabel(deviceFrame, text=device_name, font=("Arial", 20, "bold"))
                name_label.grid(row=0, column=1, sticky="w", padx=(0, 20), pady=(10, 0))

                percentage_label = CTkLabel(deviceFrame, text="No data", font=("Arial", 17))
                percentage_label.grid(row=1, column=1, sticky="w", padx=(0, 20), pady=(0, 10))
            try: 
                refreshinglabel.destroy()
            except: 
                pass
        devices_refresh_btn.configure(text="Refresh")
        devices_refresh_btn.grid(row=row + 1, column=0, columnspan=2, padx=10, pady=20, sticky="ew")
    if settings["Devices"] is None or len(settings["Devices"]) == 0 or settings["Devices"] == "":
        def defaultstates():
            refreshinglabel.destroy()
            devices_refresh_btn.configure(text="Load devices")
            devices_refresh_btn.grid(row=0, column=0, columnspan=2, padx=10, pady=20, sticky="ew")
        devices_refresh_btn.grid_forget()
        refreshinglabel = CTkLabel(devices_frame, text="No devices found", font=("Arial", 25))
        refreshinglabel.place(relx=0.5, rely=0.4, anchor="center")
        schedule_create(window, 4000, defaultstates, True)
    else:
        Thread(name="DeviceUpdateThread", daemon=True, target=lambda: UpdateDevices()).start()

def select_frame_by_name(frame_name: str):
    """Changes selected frame"""
    # set button color for selected button
    home_button.configure(fg_color=("gray75", "gray25") if frame_name == "Home" else "transparent")
    games_button.configure(fg_color=("gray75", "gray25") if frame_name == "Games" else "transparent")
    socialmedia_button.configure(fg_color=("gray75", "gray25") if frame_name == "Social Media" else "transparent")
    ytdownloader_button.configure(fg_color=("gray75", "gray25") if frame_name == "YT Downloader" else "transparent")
    assistant_button.configure(fg_color=("gray75", "gray25") if frame_name == "Assistant" else "transparent")
    music_button.configure(fg_color=("gray75", "gray25") if frame_name == "Music" else "transparent")
    devices_button.configure(fg_color=("gray75", "gray25") if frame_name == "Devices" else "transparent")
    system_button.configure(fg_color=("gray75", "gray25") if frame_name == "System" else "transparent")
    settings_button.configure(fg_color=("gray75", "gray25") if frame_name == "Settings" else "transparent")

    # show selected frame
    if frame_name == "Home":
        home_frame.pack(fill="both", expand=True)
    else:
        home_frame.pack_forget()
    if frame_name == "Games":
        games_frame.pack(anchor="center", fill="both", expand=True)
    else:
        games_frame.pack_forget()
    if frame_name == "Social Media":
        socialmedia_frame.pack(anchor="center", fill="both", expand=True)
    else:
        socialmedia_frame.pack_forget()
    if frame_name == "YT Downloader":
        ytdownloader_frame.pack(fill="both", expand=True, anchor="center")
        ytdownloader_entry.bind("<Return>", lambda event: YTVideoDownloader(YTVideoContentType))
    else:
        ytdownloader_frame.pack_forget()
        ytdownloader_entry.unbind("<Return>")
    if frame_name == "Assistant":
        assistant_frame.pack(fill="both", expand=True)
        assistant_responce_box_1.bind("<Shift-Return>", lambda event: ChatGPT())
    else:
        assistant_frame.pack_forget()
        assistant_responce_box_1.unbind("<Shift-Return>")
    if frame_name == "Music":
        music_frame.pack(fill="both", expand=True)
    else:
        music_frame.pack_forget()
    if frame_name == "Devices":
        devices_frame.pack(fill="both", expand=True)
        window.bind('<Control-r>', lambda event: AllDeviceDetails())
    else:
        devices_frame.pack_forget()
        window.unbind('<Control-r>')
    if frame_name == "System":
        system_frame.pack(anchor="center", pady=(0, 20), fill="x", expand=True)
        window.bind("<Control-r>", lambda event: UpdatePowerPlans())
    else:
        system_frame.pack_forget()
        window.unbind("<Control-r>")
    if frame_name == "Settings":
        settings_frame.pack(anchor="center", fill="both", expand=True)
    else:
        settings_frame.pack_forget()

    if frame_name == "Games" or frame_name == "Social Media":
        if frame_name == "Games":
            btn_origin_frame = "games_frame"
        elif frame_name == "Social Media":
            btn_origin_frame = "socialmedia_frame"
        toggle_edit_mode.pack(side="right", anchor="ne", padx=15, pady=(10, 0))
        add_new_btn.pack(side="right", anchor="ne", padx=5, pady=(5, 0))
        add_new_btn.configure(command=lambda: AddButton(btn_origin_frame))
    else:
        toggle_edit_mode.pack_forget()
        add_new_btn.pack_forget()

    if frame_name == "Assistant":
        assistant_settings_btn.pack(side="right", anchor="ne", padx=5, pady=(10, 0))
        assistant_submit_button.pack(side="right", anchor="ne", padx=5, pady=(10, 0))
    else:
        assistant_settings_btn.pack_forget()
        assistant_submit_button.pack_forget()

    SaveSettingsToJson("DefaultFrame", frame_name)
def SaveSettingsToJson(key: str, value):
    """Saves data to settings.json file"""
    for Property in ['URLs', 'GameShortcutURLs', 'OpenAISettings', 'MusicSettings', 'AppSettings']:
        if Property in settings and key in settings[Property]:
            settings[Property][key] = value
            break
    else:
        showerror(title="settings error", message=f"There was an error while writing to the settings file\nProperty: {Property}\nKey: {key}\nValue: {value}")
        return

    with open(SETTINGSFILE, 'w') as SettingsToWrite:
        JSdump(settings, SettingsToWrite, indent=2)
def responsive_grid(frame: CTkFrame, rows: int, columns: int):
    """Makes a grid responsive for a frame"""
    for row in range(rows+1):
        frame.grid_rowconfigure(row, weight=1)
    for column in range(columns+1):
        frame.grid_columnconfigure(column, weight=1)
def check_window_properties():
    """Checks if the window properties are set"""
    if (
        "AppSettings" in settings and
        all(key in settings["AppSettings"] for key in ["Window_Width", "Window_Height", "Window_X", "Window_Y", "Window_State"]) and
        all(settings["AppSettings"][key] != "" for key in ["Window_Width", "Window_Height", "Window_X", "Window_Y", "Window_State"])
    ):
        return True
    return False
def update_widget(widget, update=False, update_idletasks=False):
    """Updates a widget"""
    if update:
        widget.update()
    if update_idletasks:
        widget.update_idletasks()
def hextorgb(new_color_hex: str):
    new_color_hex = new_color_hex.lower().lstrip('#')
    new_color_rgb = tuple(int(new_color_hex[i:i+2], 16) for i in (0, 2, 4))
    return new_color_rgb
def change_image_clr(image, hex_color: str):
    target_rgb = hextorgb(hex_color)
    image = image.convert('RGBA')
    data = nparray(image)

    # red, green, blue, alpha = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
    alpha = data[..., 3]

    # Find areas with non-transparent pixels
    non_transparent_areas = alpha > 0

    # Replace the RGB values of non-transparent areas with the target RGB color
    data[..., 0][non_transparent_areas] = target_rgb[0]
    data[..., 1][non_transparent_areas] = target_rgb[1]
    data[..., 2][non_transparent_areas] = target_rgb[2]

    image_with_color = PILfromarray(data)
    return image_with_color
def shorten_path(text, max_length, replacement: str = "..."):
    if len(text) > max_length:
        return text[:max_length - 3] + replacement  # Replace the last three characters with "..."
    return text
def LaunchUpdater():
    check_for_updates_silent()
    cwd = getcwd()
    if getattr(sys, 'frozen', False):
        downurl = f"https://github.com/HyperNylium/Management_Panel/releases/download/v{LiveAppVersion}/Management_Panel-{LiveAppVersion}-windows.zip"
        local_path_zip = f"Management_Panel-{LiveAppVersion}-windows.zip"
        local_path = f"{cwd}\\Management_Panel"
    else:
        downurl = f"https://github.com/HyperNylium/Management_Panel/archive/refs/tags/v{LiveAppVersion}.zip"
        local_path_zip = f"Management_Panel-{LiveAppVersion}.zip"
        local_path = f"{cwd}\\Management_Panel-{LiveAppVersion}"

    if LiveAppVersion < CurrentAppVersion:
        showerror(title="Invalid version!", message=f"You have an invalid copy/version of this software.\n\nLive/Public version: {LiveAppVersion}\nYour version: {CurrentAppVersion}\n\nPlease go to:\nhttps://github.com/HyperNylium/Management_Panel\nto get the latest/authentic version of this app.")
        return

    elif LiveAppVersion != CurrentAppVersion or LiveAppVersion > CurrentAppVersion:
        usr_choice = askyesno(title='New Version!', message=f'New Version is v{LiveAppVersion}\nYour Version is v{CurrentAppVersion}\n\nNew Version of the app is now available to download/install\nClick "Yes" to update and "No" to cancel')
        if usr_choice is True:
            def updatewindow_on_closing():
                update_now_button.configure(state="normal")
                updatewindow.destroy()
            def launchupdater():
                def launch():
                    system(f"update.exe {LiveAppVersion} {SETTINGSFILE}")
                    sys.exit()
                Thread(name="LaunchUpdaterThread", daemon=True, target=launch).start()
                on_closing()
            def download_update():
                try:
                    updatewindow_label.configure(text=f"Downloading update v{LiveAppVersion}...")
                    downloadedoutof = CTkLabel(updatewindow, text=f"0 out of 0 bytes downloaded\n(0.0%)", font=("Arial", 20))
                    downloadedoutof.pack(fill="x", expand=True, padx=20, pady=0)
                    downloadprogress = CTkProgressBar(updatewindow, mode="determinate", height=15)
                    downloadprogress.set(0)
                    downloadprogress.pack(fill="x", expand=True, padx=20, pady=0)
                    response = get(downurl, stream=False, timeout=60, headers=headers, allow_redirects=True)
                    total_size_in_bytes = int(response.headers.get('content-length', 0))
                    bytes_downloaded = 0
                    block_size = 1024
                    with open(local_path_zip, 'wb') as updatefile:
                        for data in response.iter_content(block_size):
                            updatefile.write(data)
                            bytes_downloaded += len(data)
                            if total_size_in_bytes > 0:
                                progress = bytes_downloaded / total_size_in_bytes
                                progress_percent = (progress * 100)
                                downloadedoutof.configure(text=f"{bytes_downloaded} out of {total_size_in_bytes} bytes downloaded\n({progress_percent:.2f}%)")
                                downloadprogress.set(progress)
                                downloadedoutof.update()
                                downloadprogress.update()
                    downloadedoutof.destroy()
                    downloadprogress.destroy()

                    updatewindow_label.configure(text=f"Extracting update files...")
                    with ZipFile(local_path_zip, 'r') as zipObj:
                        zipObj.extractall()

                    remove(local_path_zip)

                    for root, dirs, files in walk(local_path):
                        relative_path = relpath(root, local_path)
                        dest_root = join(cwd, relative_path)

                        makedirs(dest_root, exist_ok=True)

                        for file in files:
                            if file.lower() == "update.exe":
                                src_file = join(root, file)
                                dest_file = join(dest_root, file)
                                copy2(src_file, dest_file)
                                break

                    updatewindow_label.configure(text=f"Launching update.exe to finish installing update...")
                    launchupdater()
                    return
                except Exception as e:
                    showerror(title="Update error", message=f"An error occurred while updating. Heres the full error:\n{e}")
                    update_now_button.configure(state="normal")
                    updatewindow.destroy()
                return

            updatewindow = CTkToplevel()
            updatewindow.title("Updater")
            updatewindow.attributes('-topmost', True)
            updatewindow.geometry(CenterWindowToMain(window, 500, 250))
            updatewindow.resizable(False, False)
            updatewindow.protocol("WM_DELETE_WINDOW", updatewindow_on_closing)
            update_now_button.configure(state="disabled")
            updatewindow_label = CTkLabel(updatewindow, text="Initializing...", font=("Arial", 20))
            updatewindow_label.pack(fill="x", expand=True, padx=20, pady=0)
            Thread(name="UpdateDownloadThread", daemon=True, target=download_update).start()
    else:
        showinfo(title="Update", message="You are on the latest version")
        return

    return
def file_path():
    """Gets the file path of the current file regardless of if its a .pyw file or a .exe file"""
    if getattr(sys, 'frozen', False):
        return sys.executable
    else:
        drive, rest_of_path = splitdrive(abspath(__file__))
        formatted_path = drive.upper() + rest_of_path
        return formatted_path
def reset_LaunchOnStartup_shortcut():
    settingslaunchwithwindowsvar.set(False)
    LaunchOnStartupTrueFalse()
    settingslaunchwithwindowsvar.set(True)
    LaunchOnStartupTrueFalse()
    return

window = CTk()
window.title("Management Panel")
window.protocol("WM_DELETE_WINDOW", on_closing)
screen_scale = window._get_window_scaling()
StartUp()

if check_window_properties():
    WINDOW_STATE = str(settings["AppSettings"]["Window_State"])
    WIDTH = int(settings["AppSettings"]["Window_Width"] / screen_scale)
    HEIGHT = int(settings["AppSettings"]["Window_Height"] / screen_scale)
    X = int(settings["AppSettings"]["Window_X"])
    Y = int(settings["AppSettings"]["Window_Y"])

    window.geometry(f"{WIDTH}x{HEIGHT}+{X}+{Y}")

    if WINDOW_STATE == "maximized":
        # Thank you Akascape for helping me out (https://github.com/TomSchimansky/CustomTkinter/discussions/1819)
        schedule_create(window, 50,  lambda: window.state('zoomed'), True)

    del WIDTH, HEIGHT, X, Y, WINDOW_STATE
else:
    window.geometry(CenterWindowToDisplay(window, 900, 420, screen_scale))

del screen_scale

# Bind keys Ctrl + Shift + Del to reset the windows positional values in settings.json
window.bind('<Control-Shift-Delete>', lambda event: ResetWindowPos())
window.bind('<Configure>', on_drag_end)

# Set alpha value of window from settings.json
window.attributes("-alpha", settings["AppSettings"]["Alpha"])

# Importing all icons and assigning them to there own variables to use later
try:
    homeimage = CTkImage(PILopen("assets/MenuIcons/about.png"), size=(25, 25))
    devicesimage = CTkImage(PILopen("assets/MenuIcons/devices.png"), size=(25, 25))
    gamesimage = CTkImage(PILopen("assets/MenuIcons/games.png"), size=(25, 25))
    ytdownloaderimage = CTkImage(PILopen("assets/MenuIcons/ytdownloader.png"), size=(25, 25))
    socialmediaimage = CTkImage(PILopen("assets/MenuIcons/socialmedia.png"), size=(25, 25))
    assistantimage = CTkImage(PILopen("assets/MenuIcons/assistant.png"), size=(25, 25))
    musicimage = CTkImage(PILopen("assets/MenuIcons/music.png"), size=(25, 25))
    systemimage = CTkImage(PILopen("assets/MenuIcons/system.png"), size=(25, 25))
    settingsimage = CTkImage(PILopen("assets/MenuIcons/settings.png"), size=(25, 25))
    addbtnimage = CTkImage(change_image_clr(PILopen('assets/ExtraIcons/add-btn.png'), "#ffffff"), size=(30, 30))
    assistantsettingsimage = CTkImage(change_image_clr(PILopen('assets/ExtraIcons/assistant-settings.png'), "#ffffff"), size=(35, 35))
    assistantsubmitimage = CTkImage(change_image_clr(PILopen('assets/ExtraIcons/assistant-submit.png'), "#ffffff"), size=(35, 35))
    modbtnpositionimage = CTkImage(change_image_clr(PILopen('assets/ExtraIcons/modify-btn-positions.png'), "#ffffff"), size=(30, 30))
    closeimage = CTkImage(PILopen("assets/MenuIcons/close.png"), size=(20, 20))
    openimage = CTkImage(PILopen("assets/MenuIcons/open.png"), size=(25, 25))
    previousimage = CTkImage(change_image_clr(PILopen('assets/MusicPlayer/previous.png'), "#ffffff"), size=(25, 25))
    pauseimage = CTkImage(change_image_clr(PILopen("assets/MusicPlayer/pause.png"), "#ffffff"), size=(25, 25))
    playimage = CTkImage(change_image_clr(PILopen('assets/MusicPlayer/play.png'), "#ffffff"), size=(25, 25))
    nextimage = CTkImage(change_image_clr(PILopen('assets/MusicPlayer/next.png'), "#ffffff"), size=(25, 25))
    stopimage = CTkImage(change_image_clr(PILopen('assets/MusicPlayer/stop.png'), "#ffffff"), size=(25, 25))
    if settings["MusicSettings"]["LoopState"] == "all":
        loopimage = CTkImage(change_image_clr(PILopen('assets/MusicPlayer/loop.png'), "#00ff00"), size=(25, 25))
    elif settings["MusicSettings"]["LoopState"] == "one":
        loopimage = CTkImage(change_image_clr(PILopen('assets/MusicPlayer/loop-1.png'), "#00ff00"), size=(25, 25))
    elif settings["MusicSettings"]["LoopState"] == "off":
        loopimage = CTkImage(change_image_clr(PILopen('assets/MusicPlayer/loop.png'), "#ff0000"), size=(25, 25))
    else:
        loopimage = CTkImage(change_image_clr(PILopen('assets/MusicPlayer/loop.png'), "#00ff00"), size=(25, 25))
        SaveSettingsToJson("LoopState", "all")
except Exception as e:
    showerror(title="Icon import error", message=f"Couldn't import an icon.\nDetails: {e}")
    on_closing()

# create navigation frame and 
navigation_frame = CTkFrame(window, corner_radius=0)
navigation_buttons_frame = CTkFrame(navigation_frame, corner_radius=0, fg_color="transparent")
navigation_frame.pack(side="left", fill="y")

# time&date label
navigation_frame_label = CTkLabel(navigation_frame, text="Loading...", font=("sans-serif", 18, "bold"))
navigation_frame_label.pack(side="top", padx=20, pady=20)

# navigation_buttons_frame pack. We need to pack it here so its under the navigation_frame_label (under the time and date)
navigation_buttons_frame.pack(fill="x", expand=True, anchor="center")

# X button and Edit Mode switch
navigation_bar_frame = CTkFrame(window, corner_radius=0, fg_color="transparent")
navigation_bar_frame.pack(side="top", fill="x", expand=False)

close_open_nav_button = CTkButton(navigation_bar_frame, width=25, height=25, text="", fg_color="transparent", image=closeimage, anchor="w", hover_color=("gray70", "gray30"), command=lambda: NavbarAction("close"))
close_open_nav_button.pack(side="left", anchor="nw", padx=0, pady=(5, 0))

add_new_btn = CTkButton(navigation_bar_frame, width=100, text="Add", fg_color=("gray75", "gray30"), image=addbtnimage, anchor="w", font=("sans-serif", 20), command=None)
toggle_edit_mode = CTkSwitch(navigation_bar_frame, text="Edit Mode", variable=EditModeVar, onvalue=True, offvalue=False, font=("sans-serif", 22), command=EditModeInit)


# menu btns
home_button = CTkButton(navigation_buttons_frame, corner_radius=10, width=10, height=40, text="Home", fg_color="transparent", image=homeimage, anchor="w", text_color=("gray10", "gray90"), font=("Arial", 22), hover_color=("gray70", "gray30"), command=lambda: select_frame_by_name("Home"))
home_button.grid(sticky="ew", pady=1)
games_button = CTkButton(navigation_buttons_frame, corner_radius=10, width=0, height=40, text="Games", fg_color="transparent", image=gamesimage, anchor="w", text_color=("gray10", "gray90"), font=("Arial", 22), hover_color=("gray70", "gray30"), command=lambda: select_frame_by_name("Games"))
games_button.grid(sticky="ew", pady=1)
socialmedia_button = CTkButton(navigation_buttons_frame, corner_radius=10, width=0, height=40, text="Social Media", fg_color="transparent", image=socialmediaimage, anchor="w", text_color=("gray10", "gray90"), font=("Arial", 22), hover_color=("gray70", "gray30"), command=lambda: select_frame_by_name("Social Media"))
socialmedia_button.grid(sticky="ew", pady=1)
ytdownloader_button = CTkButton(navigation_buttons_frame, corner_radius=10, width=0, height=40, text="YT Downloader", fg_color="transparent", image=ytdownloaderimage, anchor="w", text_color=("gray10", "gray90"), font=("Arial", 22), hover_color=("gray70", "gray30"), command=lambda: select_frame_by_name("YT Downloader"))
ytdownloader_button.grid(sticky="ew", pady=1)
assistant_button = CTkButton(navigation_buttons_frame, corner_radius=10, width=0, height=40, text="Assistant", fg_color="transparent", image=assistantimage, anchor="w", text_color=("gray10", "gray90"), font=("Arial", 22), hover_color=("gray70", "gray30"), command=lambda: select_frame_by_name("Assistant"))
assistant_button.grid(sticky="ew", pady=1)
music_button = CTkButton(navigation_buttons_frame, corner_radius=10, width=0, height=40, text="Music", fg_color="transparent", image=musicimage, anchor="w", text_color=("gray10", "gray90"), font=("Arial", 22), hover_color=("gray70", "gray30"), command=lambda: select_frame_by_name("Music"))
music_button.grid(sticky="ew", pady=1)
devices_button = CTkButton(navigation_buttons_frame, corner_radius=10, width=0, height=40, text="Devices", fg_color="transparent", image=devicesimage, anchor="w", text_color=("gray10", "gray90"), font=("Arial", 22), hover_color=("gray70", "gray30"), command=lambda: select_frame_by_name("Devices"))
devices_button.grid(sticky="ew", pady=1)
system_button = CTkButton(navigation_buttons_frame, corner_radius=10, width=0, height=40, text="System", fg_color="transparent", image=systemimage, anchor="w", text_color=("gray10", "gray90"), font=("Arial", 22), hover_color=("gray70", "gray30"), command=lambda: select_frame_by_name("System"))
system_button.grid(sticky="ew", pady=1)
settings_button = CTkButton(navigation_buttons_frame, corner_radius=10, width=0, height=40, text="Settings", fg_color="transparent", image=settingsimage, anchor="w", text_color=("gray10", "gray90"), font=("Arial", 22), hover_color=("gray70", "gray30"), command=lambda: select_frame_by_name("Settings"))
settings_button.grid(sticky="ew", pady=1)

del homeimage, devicesimage, gamesimage, ytdownloaderimage, socialmediaimage, assistantimage, musicimage, systemimage, settingsimage


# main frames
home_frame = CTkFrame(window, corner_radius=0, fg_color="transparent")
games_frame = CTkScrollableFrame(window, corner_radius=0, fg_color="transparent", border_width=3, border_color="#242424")
socialmedia_frame = CTkScrollableFrame(window, corner_radius=0, fg_color="transparent", border_width=3, border_color="#242424")
ytdownloader_frame = CTkFrame(window, corner_radius=0, fg_color="transparent")
assistant_frame = CTkFrame(window, corner_radius=0, fg_color="transparent")
music_frame = CTkFrame(window, corner_radius=0, fg_color="transparent")
devices_frame = CTkFrame(window, corner_radius=0, fg_color="transparent")
system_frame = CTkFrame(window, corner_radius=0, fg_color="transparent")
settings_frame = CTkScrollableFrame(window, corner_radius=0, fg_color="transparent", border_width=3, border_color="#242424")


# Create elements/widgets for frames
home_frame_label_1 = CTkLabel(home_frame, text=f"Version: {CurrentAppVersion} {ShowUserInfo}", font=("sans-serif", 28))
home_frame_label_1.pack(anchor="center", pady=(100, 0))
home_frame_label_2 = CTkLabel(home_frame, text=f"Creator/developer: {Developer}", font=("sans-serif", 28))
home_frame_label_2.pack(anchor="center", pady=10)
home_frame_label_3 = CTkLabel(home_frame, text=f"Last updated: {LastEditDate}", font=("sans-serif", 28))
home_frame_label_3.pack(anchor="center")
chkforupdatesframe = CTkFrame(home_frame, corner_radius=0, fg_color="transparent")
chkforupdatesframe.pack(anchor="s", fill="x", expand=True)
check_for_updates_button = CTkButton(chkforupdatesframe, text="Check for updates", fg_color=("gray75", "gray30"), font=("sans-serif", 22), corner_radius=10, command=check_for_updates_GUI)
update_now_button = CTkButton(chkforupdatesframe, text="Update now", fg_color=("gray75", "gray30"), font=("sans-serif", 22), corner_radius=10, command=LaunchUpdater)
check_for_updates_button.grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
update_now_button.grid(row=1, column=2, columnspan=2, padx=5, pady=10, sticky="ew")


YTVideoContentType = "Video (.mp4)"
ytdownloader_entry = CTkEntry(ytdownloader_frame, placeholder_text="Enter your video URL here", width=600, height=40, border_width=0, corner_radius=10, font=("sans-serif", 22), justify="center")
ytdownloader_entry.grid(row=1, column=0, columnspan=2, padx=10, pady=0)
ytdownloader_error_label = CTkLabel(ytdownloader_frame, text="", font=("sans-serif", 18))
ytdownloader_OptionMenu = CTkOptionMenu(ytdownloader_frame, values=["Video (.mp4)", "Audio (.mp3)"], command=lambda vidtype: YTVideoDownloaderContentType(vidtype), fg_color="#343638", button_color="#4d4d4d", button_hover_color="#444", font=("sans-serif", 17), dropdown_font=("sans-serif", 15), width=200, height=30, anchor="center")
ytdownloader_OptionMenu.grid(row=2, column=0, columnspan=2, padx=10, pady=0)
ytdownloader_OptionMenu.set("Video (.mp4)")
ytdownloader_frame_button_1 = CTkButton(ytdownloader_frame, text="Download", compound="top", fg_color=("gray75", "gray30"), font=("sans-serif", 22), corner_radius=10, command=lambda: YTVideoDownloader(YTVideoContentType))
ytdownloader_frame_button_1.grid(row=3, rowspan=2, column=0, columnspan=2, padx=10, pady=20, sticky="ews")
ytdownloader_progressbar= CTkProgressBar(ytdownloader_frame, mode="determinate", height=15)
ytdownloader_progressbar.set(0)
ytdownloader_progressbarpercentage= CTkLabel(ytdownloader_frame, text="0%", font=("sans-serif Bold", 18))


assistant_submit_button = CTkButton(navigation_bar_frame, text="", width=50, image=assistantsubmitimage, fg_color=("gray75", "gray30"), font=("sans-serif", 22), corner_radius=10, command=ChatGPT)
assistant_settings_btn = CTkButton(navigation_bar_frame, width=50, text="", fg_color=("gray75", "gray30"), image=assistantsettingsimage, anchor="w", font=("sans-serif", 20), command=ChatGPTConfig)
assistant_responce_box_frame = CTkFrame(assistant_frame, corner_radius=0, fg_color="transparent")
assistant_responce_box_frame.pack(fill="x", expand=True, anchor="center")
assistant_responce_box_1 = CTkTextbox(assistant_responce_box_frame, width=680, height=150, border_width=0, corner_radius=10, font=("sans-serif", 22), activate_scrollbars=True, border_color="#242424")
assistant_responce_box_1.grid(row=0, column=0, padx=10, pady=10)
assistant_responce_box_2 = CTkTextbox(assistant_responce_box_frame, width=680, height=150, border_width=0, corner_radius=10, font=("sans-serif", 22), activate_scrollbars=True, border_color="#242424")
assistant_responce_box_2.grid(row=1, column=0, padx=10, pady=10)


music_frame_container = CTkFrame(music_frame, corner_radius=0, fg_color="transparent")
all_music_frame = CTkScrollableFrame(music_frame, height=150, corner_radius=0, fg_color="transparent", border_width=2, border_color="#333", label_text="Updating...", label_font=("sans-serif", 18))
music_info_frame = CTkFrame(music_frame, corner_radius=0, fg_color="transparent")
music_controls_frame = CTkFrame(music_frame_container, corner_radius=0, fg_color="transparent")
music_volume_frame = CTkFrame(music_frame_container, corner_radius=0, fg_color="transparent")
music_progress_frame = CTkFrame(music_frame_container, corner_radius=0, fg_color="transparent")
all_music_frame.pack(fill="both", expand=True, anchor="s", pady=0)
music_info_frame.pack(fill="x", expand=True, anchor="s", pady=0)
music_frame_container.pack(fill="x", expand=True, anchor="s", pady=0)
music_controls_frame.pack(fill="x", expand=True, anchor="s", pady=0)
music_volume_frame.pack(fill="x", expand=True, anchor="s", pady=0)
music_progress_frame.pack(fill="x", expand=True, anchor="s", pady=0)
music_dir_label = CTkLabel(music_info_frame, text=f"Music Directory: {shorten_path(settings['MusicSettings']['MusicDir'], 45)}" if settings['MusicSettings']['MusicDir'] != "" else "Music Directory: None", font=("sans-serif", 18))
update_music_list = CTkButton(music_info_frame, width=80, text="Update", compound="top", fg_color=("gray75", "gray30"), font=("sans-serif", 18), corner_radius=10, command=music_manager.update)
change_music_dir = CTkButton(music_info_frame, width=80, text="Change", compound="top", fg_color=("gray75", "gray30"), font=("sans-serif", 18), corner_radius=10, command=music_manager.changedir)
music_dir_label.grid(row=2, column=1, padx=10, pady=5, sticky="w")
update_music_list.grid(row=2, column=2, padx=5, pady=5, sticky="e")
change_music_dir.grid(row=2, column=3, padx=5, pady=5, sticky="w")
music_info_frame.grid_columnconfigure([0, 3], weight=1)
stop_song_btn = CTkButton(music_controls_frame, width=40, height=40, text="", fg_color="transparent", image=stopimage, anchor="w", hover_color=("gray70", "gray30"), command=music_manager.stop)
pre_song_btn = CTkButton(music_controls_frame, width=40, height=40, text="", fg_color="transparent", image=previousimage, anchor="w", hover_color=("gray70", "gray30"), command=music_manager.previous)
play_pause_song_btn = CTkButton(music_controls_frame, width=40, height=40, text="", fg_color="transparent", image=playimage, anchor="w", hover_color=("gray70", "gray30"), command=music_manager.play)
next_song_btn = CTkButton(music_controls_frame, width=40, height=40, text="", fg_color="transparent", image=nextimage, anchor="w", hover_color=("gray70", "gray30"), command=music_manager.next)
loop_playlist_btn = CTkButton(music_controls_frame, width=40, height=40, text="", fg_color="transparent", image=loopimage, anchor="w", hover_color=("gray70", "gray30"), command=music_manager.loop)
stop_song_btn.grid(row=1, column=1, padx=5, pady=0, sticky="w")
pre_song_btn.grid(row=1, column=2, padx=10, pady=0, sticky="e")
play_pause_song_btn.grid(row=1, column=3, padx=10, pady=0, sticky="e")
next_song_btn.grid(row=1, column=4, padx=10, pady=0, sticky="e")
loop_playlist_btn.grid(row=1, column=5, padx=10, pady=0, sticky="w")
volume_slider = CTkSlider(music_volume_frame, width=250, from_=0, to=100, command=lambda volume: music_manager.volume(), variable=musicVolumeVar, button_color="#fff", button_hover_color="#ccc")
volume_label = CTkLabel(music_volume_frame, text=f"{musicVolumeVar.get()}%", font=("sans-serif", 18, "bold"), fg_color="transparent")
volume_label.grid(row=1, column=1, padx=0, pady=0, sticky="w")
volume_slider.grid(row=1, column=1, padx=40, pady=0, sticky="e")
music_volume_frame.grid_columnconfigure([0, 2], weight=1)
time_left_label = CTkLabel(music_progress_frame, text="0:00:00", font=("sans-serif", 18, "bold"), fg_color="transparent")
song_progressbar = CTkProgressBar(music_progress_frame, mode="determinate", height=15)
song_progressbar.set(0.0)
total_time_label = CTkLabel(music_progress_frame, text="0:00:00", font=("sans-serif", 18, "bold"), fg_color="transparent")
time_left_label.grid(row=1, column=0, padx=10, pady=0, sticky="w")
song_progressbar.grid(row=1, column=1, padx=10, pady=0, sticky="ew")
total_time_label.grid(row=1, column=2, padx=10, pady=0, sticky="e")
music_progress_frame.grid_columnconfigure(1, weight=1)


devices_spacing_label_1 = CTkLabel(devices_frame, width=340, height=0, text="").grid(row=0, column=0, padx=0, pady=0)
devices_spacing_label_2 = CTkLabel(devices_frame, width=340, height=0, text="").grid(row=0, column=1, padx=0, pady=0)
devices_refresh_btn = CTkButton(devices_frame, text="Load devices", compound="bottom", fg_color=("gray75", "gray30"), font=("sans-serif", 22), corner_radius=10, command=AllDeviceDetails)
devices_refresh_btn.grid(row=0, column=0, columnspan=2, padx=10, pady=20, sticky="ew")


system_frame_button_1 = CTkButton(system_frame, text="VPN settings", compound="top", width=200, fg_color=("gray75", "gray30"), font=("sans-serif", 22), corner_radius=10, command=lambda: systemsettings("vpn"))
system_frame_button_1.grid(row=0, column=1, padx=5, pady=10)
system_frame_button_2 = CTkButton(system_frame, text="Netdrive", compound="top", width=200, fg_color=("gray75", "gray30"), font=("sans-serif", 22), corner_radius=10, command=lambda: systemsettings("netdrive"))
system_frame_button_2.grid(row=0, column=2, padx=5, pady=10)
system_frame_button_3 = CTkButton(system_frame, text="Installed apps", compound="top", width=200, fg_color=("gray75", "gray30"), font=("sans-serif", 22), corner_radius=10, command=lambda: systemsettings("apps"))
system_frame_button_3.grid(row=0, column=3, padx=5, pady=10)
system_frame_button_4 = CTkButton(system_frame, text="Sound settings", compound="top", width=200, fg_color=("gray75", "gray30"), font=("sans-serif", 22), corner_radius=10, command=lambda: systemsettings("sound"))
system_frame_button_4.grid(row=1, column=1, padx=5, pady=10)
system_frame_button_5 = CTkButton(system_frame, text="Display settings", compound="top", width=200, fg_color=("gray75", "gray30"), font=("sans-serif", 22), corner_radius=10, command=lambda: systemsettings("display"))
system_frame_button_5.grid(row=1, column=2, padx=5, pady=10)
system_frame_button_6 = CTkButton(system_frame, text="Power settings", compound="top", width=200, fg_color=("gray75", "gray30"), font=("sans-serif", 22), corner_radius=10, command=lambda: systemsettings("power"))
system_frame_button_6.grid(row=1, column=3, padx=5, pady=10)
system_frame_button_7 = CTkButton(system_frame, text="Storage settings", compound="top", width=200, fg_color=("gray75", "gray30"), font=("sans-serif", 22), corner_radius=10, command=lambda: systemsettings("storage"))
system_frame_button_7.grid(row=2, column=1, padx=5, pady=10)
system_frame_button_8 = CTkButton(system_frame, text="Network setings", compound="top", width=200, fg_color=("gray75", "gray30"), font=("sans-serif", 22), corner_radius=10, command=lambda: systemsettings("network"))
system_frame_button_8.grid(row=2, column=2, padx=5, pady=10)
system_frame_button_9 = CTkButton(system_frame, text="Windows update", compound="top", width=200, fg_color=("gray75", "gray30"), font=("sans-serif", 22), corner_radius=10, command=lambda: systemsettings("windowsupdate"))
system_frame_button_9.grid(row=2, column=3, padx=5, pady=10)
system_frame_power_optionmenu = CTkOptionMenu(system_frame, values=["Loading..."], state="disabled", command=None, fg_color="#343638", button_color="#4d4d4d", button_hover_color="#444", font=("sans-serif", 17), dropdown_font=("sans-serif", 15), width=200, height=30, anchor="center")
system_frame_power_optionmenu.grid(row=4, column=2, padx=5, pady=10)


settingsgrid = CTkFrame(settings_frame, corner_radius=0, fg_color="transparent")
settingsgrid.pack(fill="x", expand=True, anchor="center")
settingsAlwayOnTopswitch = CTkSwitch(settingsgrid, text="", variable=settingsAlwayOnTopVar, onvalue=True, offvalue=False, font=("sans-serif", 22), command=AlwaysOnTopTrueFalse)
settingsAlwayOnToplabel = CTkLabel(settingsgrid, text="Always on top", font=("sans-serif", 22))
settingsAlwayOnTopswitch.grid(row=1, column=1, pady=5, sticky="e")
settingsAlwayOnToplabel.grid(row=1, column=2, pady=5, sticky="w")
settingslaunchwithwindowsswitch = CTkSwitch(settingsgrid, text="", variable=settingslaunchwithwindowsvar, onvalue=True, offvalue=False, font=("sans-serif", 22), command=LaunchOnStartupTrueFalse)
settingslaunchwithwindowslabel = CTkLabel(settingsgrid, text="Launch at login", font=("sans-serif", 22))
settingslaunchwithwindowsswitch.grid(row=2, column=1, pady=5, sticky="e")
settingslaunchwithwindowslabel.grid(row=2, column=2, pady=5, sticky="w")
settingsSpeakResponceswitch = CTkSwitch(settingsgrid, text="", variable=settingsSpeakResponceVar, onvalue=True, offvalue=False, font=("sans-serif", 22), command=lambda: SaveSettingsToJson("SpeakResponce", settingsSpeakResponceVar.get()))
settingsSpeakResponcelabel = CTkLabel(settingsgrid, text="Speak response from AI", font=("sans-serif", 22))
settingsSpeakResponceswitch.grid(row=3, column=1, pady=5, sticky="e")
settingsSpeakResponcelabel.grid(row=3, column=2, pady=5, sticky="w")
settingscheckupdatesswitch = CTkSwitch(settingsgrid, text="", variable=settingsCheckForUpdates, onvalue=True, offvalue=False, font=("sans-serif", 22), command=lambda: SaveSettingsToJson("CheckForUpdatesOnLaunch", settingsCheckForUpdates.get()))
settingscheckupdateslabel = CTkLabel(settingsgrid, text="Check for updates on launch", font=("sans-serif", 22))
settingscheckupdatesswitch.grid(row=4, column=1, pady=5, sticky="e")
settingscheckupdateslabel.grid(row=4, column=2, pady=5, sticky="w")
settingsresetwindowbtn = CTkButton(settings_frame, width=300, text="Reset window position", compound="top", fg_color=("gray75", "gray30"), font=("sans-serif", 20), corner_radius=10, command=ResetWindowPos)
settingsresetwindowbtn.pack(anchor="center", padx=10, pady=10)
settingsopensettingsbtn = CTkButton(settings_frame, width=300, text="Open settings.json", compound="top", fg_color=("gray75", "gray30"), font=("sans-serif", 20), corner_radius=10, command=lambda: startfile(SETTINGSFILE))
settingsopensettingsbtn.pack(anchor="center", padx=10, pady=10)
alpha_slider = CTkSlider(settings_frame, width=300, from_=0.5, to=1.0, command=set_alpha, variable=settingsAlphavar) # I set the _from param to 0.5 because anything lower than that is too transparent and you can't see the window let alone interact with it.
alpha_slider.pack(anchor="center", padx=10, pady=10)
settingsAlphavar.set(settings["AppSettings"]["Alpha"])


# select default frame in settings.json (can be changed in GUI from "settings_default_frame_optionmenu")
select_frame_by_name(settings["AppSettings"]["DefaultFrame"])

# Make frames .grid responsive
responsive_grid(navigation_buttons_frame, 10, 0) # 10 rows, 0 columns responsive
responsive_grid(games_frame, 2, 2) # 2 rows, 2 columns responsive
responsive_grid(socialmedia_frame, 2, 2) # 2 rows, 2 columns responsive
responsive_grid(ytdownloader_frame, 4, 1) # 4 rows, 1 column responsive
responsive_grid(assistant_frame, 0, 0) # 0 rows, 0 columns responsive
responsive_grid(assistant_responce_box_frame, 1, 0) # 1 rows, 0 columns responsive
responsive_grid(devices_frame, 3, 1) # 3 rows, 1 column responsive
responsive_grid(system_frame, 2, 3) # 2 rows, 3 columns responsive
responsive_grid(settings_frame, 2, 2) # 2 rows, 2 columns responsive
chkforupdatesframe.grid_columnconfigure([0, 3], weight=1)
music_controls_frame.grid_columnconfigure([0, 5], weight=1)
settingsgrid.grid_columnconfigure([0, 3], weight=1)

# add all buttons and their text to a list for later use
for widget in navigation_buttons_frame.winfo_children():
    if isinstance(widget, CTkButton):
        all_buttons.append(widget)
        all_buttons_text.append(widget.cget('text'))

# initialize and start the MusicManager
music_manager.update()
music_manager.start_event_loop()

# initialize and start the TitleUpdater
title_bar = TitleUpdater(navigation_frame_label)
title_bar.start()

# Start to update the powerplan optionmenu
UpdatePowerPlans()

# set the navigation state to the last known state in settings.json
if settings["AppSettings"]["NavigationState"] == "close":
    NavbarAction("close")
else:
    NavbarAction("open")

AppsLaucherGUISetup("games_frame") # create the games_frame content
AppsLaucherGUISetup("socialmedia_frame") # create the socialmedia_frame content

# if my personel netdrive script does not exist on the system, disable the button to launch it
if not exists(f"{UserDesktopDir}/Stuff/GitHub/Environment_Scripts/netdrive.bat"):
    system_frame_button_2.configure(state="disabled")

window.mainloop()