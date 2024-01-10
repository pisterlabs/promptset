import ctypes
import io
import os
import subprocess
from ctypes.wintypes import BOOL, HWND, LONG
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from tkinter.filedialog import askopenfilename

# The Pillow library is used to carry out image processing, such as resizing in our case. PIL is the Python Imaging Library which also allows Tkinter to display images.
try:
  from PIL import Image, ImageOps, ImageTk
except:
  print('Installing PIL.')
  subprocess.check_call(['pip', 'install', 'pillow'])
  print('Done.')
  from PIL import Image, ImageOps, ImageTk

# The tkcalendar library is a calendar widget for Tkinter. It is a drop-in replacement for the standard Tkinter calendar widget.
# This will be the additional feature for the system. 
try:
    from tkcalendar import Calendar as tkCalendar
    from tkcalendar import DateEntry
except:
    print('Installing tkcalendar.')
    subprocess.check_call(['pip', 'install', 'tkcalendar'])
    print('Done.')
    from tkcalendar import Calendar as tkCalendar
    from tkcalendar import DateEntry

try:
    import openai 
except:
    print('Installing openai.')
    subprocess.check_call(['pip', 'install', 'openai'])
    print('Done.')
    import openai
import urllib.request
import datetime
import math
import random
import sqlite3
from ctypes import windll

try:
    import pyglet 
except:
    print('Installing pyglet.')
    subprocess.check_call(['pip', 'install', 'pyglet'])
    print('Done.')
    import pyglet

# Pyglet library to add the fonts including Atkinson Hyperlegible and Avenir Next to the system.
pyglet.font.add_file('fonts\AtkinsonHyperlegible.ttf')
pyglet.font.add_file('fonts\AvenirNext-Bold.ttf')
pyglet.font.add_file('fonts\AvenirNext-Regular.ttf')
pyglet.font.add_file('fonts\AvenirNext-Medium.otf')

# Ctypes method that allows us to interact with windows and get the system resolution
# https://stackoverflow.com/a/3129524

user32 = windll.user32
  
# https://stackoverflow.com/a/68621773
# This bit of code allows us to remove the title bar from the window
# in case of a fullscreen application
# More information on the ctypes library can be found here:
# https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwindowlongptrw
# https://learn.microsoft.com/en-us/windows/win32/winmsg/window-styles
GetWindowLongPtrW = ctypes.windll.user32.GetWindowLongPtrW
SetWindowLongPtrW = ctypes.windll.user32.SetWindowLongPtrW

def get_handle(root) -> int:
    root.update_idletasks()
    # This gets the window's parent same as `ctypes.windll.user32.GetParent`
    return GetWindowLongPtrW(root.winfo_id(), GWLP_HWNDPARENT)

# Constants for the ctypes functions above 
GWL_STYLE = -16
GWLP_HWNDPARENT = -8
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
PINK = "#FFE3E1"
OTHERPINK = "#FA9494"
LIGHTYELLOW = "#FFF5E4"
ORANGE = "#FFAA22"
NICEPURPLE = "#B1B2FF"
NICEBLUE = "#AAC4FF"
LAVENDER = "#D2DAFF"
LIGHTPURPLE = "#EEF1FF"
DARKBLUE = "#3e477c"
NAVYBLUE = "#27364d"
WHITE = "#FFFFFF"
BLACK = "#000000"

LOGGEDINAS = "Viewer"
LOGINSTATE = False
LOGINID = "Viewer"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN WINDOW ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class Window(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        global dpi 
        global dpiError
        dpiError = False
        # This bit of code allows us to perform dpi awareness and allows us to
        # maintain the same size of the window on different resolutions and scalings 
        try:
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            print('ERROR. Could not set DPI awareness.')
            dpiError = True
        if dpiError:
            dpi = 96
        else:
            dpi = self.winfo_fpixels('1i')
        # The line below lets us get the primary display's resolution
        self.screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        if self.screensize == (1920, 1080):
            self.geometry(
                f'{math.ceil(1920 * dpi / 96)}x{math.ceil(1080 * dpi / 96)}')
        elif self.screensize > (1920, 1080):
            self.geometry(
                f'{math.ceil(1920 * dpi / 96)}x{math.ceil(1080 * dpi / 96)}')
        self.title("INTI Interactive System")
        self.resizable(False, False)
        self.configure(background=LAVENDER)
        for x in range(32):
            self.columnconfigure(x, weight=1, uniform='row')
            Label(self, width=1, bg=NICEPURPLE).grid(
                row=0, column=x, sticky=NSEW)
        for y in range(18):
            self.rowconfigure(y, weight=1, uniform='row')
            Label(self, width=1, bg=NICEPURPLE).grid(
                row=y, column=0, sticky=NSEW)

        FONTFORBUTTONS = "Bahnschrift Semibold"
        # print(LOGINID)
        # print(LOGGEDINAS)       
        #Frame that has everything stacked on top of it
        self.centercontainer = Frame(self, bg=LAVENDER)
        self.centercontainer.grid(row=2, column=2, rowspan=14,
                             columnspan=28, sticky=NSEW) 
        self.centercontainer.grid_propagate(False)
        
        for x in range(28):
            self.centercontainer.columnconfigure(x, weight=1, uniform='row')
            Label(self.centercontainer, width=1, bg=LAVENDER).grid(
                row=0, column=x, rowspan=1, columnspan=1, sticky=NSEW)
        for y in range(16):
            self.centercontainer.rowconfigure(y, weight=1, uniform='row')
            Label(self.centercontainer, width=1, bg=LAVENDER).grid(
                row=y, column=0, rowspan=1, columnspan=1, sticky=NSEW)
                        
        self.buttoncontainer = Frame(self, bg=DARKBLUE, highlightbackground=LIGHTYELLOW, highlightthickness=2)
        self.buttoncontainer.grid(row=0, column=0, rowspan=2,
                             columnspan=30, sticky=NSEW)
        self.buttoncontainer.grid_propagate(False)

        for x in range(30):
            self.buttoncontainer.columnconfigure(x, weight=1, uniform='row')
            Label(self.buttoncontainer, width=1, bg=DARKBLUE).grid(
                row=0, column=x, rowspan=1, columnspan=1, sticky=NSEW)
        for y in range(2):
            self.buttoncontainer.rowconfigure(y, weight=1, uniform='row')
            Label(self.buttoncontainer, width=1, bg=DARKBLUE).grid(
                row=y, column=0, rowspan=1, columnspan=1, sticky=NSEW)

        self.signupbutton = Button(self.buttoncontainer, text="Sign Up\n Page", bg=NICEBLUE,
                            fg="white",font=(FONTFORBUTTONS, 20),
                            borderwidth=2, relief="raised", height=1, width=1, highlightthickness=2,
                            command=lambda: [
                    self.show_frame(RegistrationPage),
                    self.togglebuttonrelief(self.signupbutton)
                    ])
        self.loginbutton = Button(self.buttoncontainer, text="Login\nPage", bg=NICEBLUE,
                            fg="white", font=(FONTFORBUTTONS, 20),
                            borderwidth=2, relief="raised", height=1, width=1, highlightthickness=2,
                            command=lambda: [
                    self.show_frame(LoginPage),
                    self.togglebuttonrelief(self.loginbutton)
                    ])

        self.signupbutton.grid(row=0, column=0, rowspan=2, columnspan=3, sticky=NSEW)
        self.loginbutton.grid(row=0, column=3, rowspan=2, columnspan=3, sticky=NSEW)

        self.mainpagebutton = Button(self.buttoncontainer, text="Main\nPage", bg=NICEBLUE, fg="white", font=(FONTFORBUTTONS, 20),
                                     borderwidth=2, relief="raised", height=1, width=1, highlightthickness=0,
                                     command=lambda: [
                    self.show_frame(MainPage),
                    self.togglebuttonrelief(self.mainpagebutton)
                    ])
        self.eventlistbutton = Button(self.buttoncontainer, text="Event\nList", bg=NICEBLUE, fg="white", font=(FONTFORBUTTONS, 20),
                                      borderwidth=2, relief="raised", height=1, width=1, highlightthickness=0,
                                      command=lambda: [
                    self.show_frame(EventView),
                    self.togglebuttonrelief(self.eventlistbutton)
                    ])
        self.eventregistrationbutton = Button(self.buttoncontainer, text="Event\nRegistration", bg=NICEBLUE, fg="white", font=(FONTFORBUTTONS, 20),
                                              borderwidth=2, relief="raised",
                                              height=1, width=1, highlightthickness=0,
                                              command=lambda: [
                    self.show_frame(EventRegistration),
                    self.togglebuttonrelief(self.eventregistrationbutton)
                    ])
        self.eventcreationbutton = Button(self.buttoncontainer, text="Event\nCreation\n(ADMIN)", bg=NICEBLUE, fg="white", font=(FONTFORBUTTONS, 20),
                                          borderwidth=2, relief="raised", height=1,width=1, highlightthickness=0,
                                          command=lambda: [
                    self.show_frame(EventCreation),
                    self.togglebuttonrelief(self.eventcreationbutton)
                    ])
        self.viewparticipantsbutton = Button(self.buttoncontainer, text="Management\nSuite\n(ADMIN)", bg=NICEBLUE, fg="white", font=(FONTFORBUTTONS, 20),
                                             borderwidth=2, relief="raised", height=1,width=1, highlightthickness=0,
                                             command=lambda: [
                    self.show_frame(ManagementSuite),
                    self.togglebuttonrelief(self.viewparticipantsbutton)
                    ])
        self.feedbackbutton = Button(self.buttoncontainer, text="Feedback\nForm", bg=NICEBLUE, fg="white", font=(FONTFORBUTTONS, 20),
                            borderwidth=2, relief="raised", height=1,width=1, highlightthickness=0,
                            command=lambda: [
                    self.show_frame(FeedbackForm),
                    self.togglebuttonrelief(self.feedbackbutton)
                    ])
        self.calendarbutton = Button(self.buttoncontainer, text="Calendar", bg=NICEBLUE, fg="white", font=(FONTFORBUTTONS, 20),
                                     borderwidth=2, relief="raised", height=1,width=1, highlightthickness=0,
                                     command=lambda: [
                    self.show_frame(CalendarPage),
                    self.togglebuttonrelief(self.calendarbutton)
                    ])

        # Sign out buttons + reminder frame

        self.bottomleftbuttons = Frame(self, bg=NAVYBLUE, width=1, height=1)
        self.bottomleftbuttons.grid(row=16, column=0, rowspan=2, columnspan=20, sticky=NSEW)
        self.bottomleftbuttons.grid_propagate(False)

        for x in range(20):
            self.bottomleftbuttons.columnconfigure(x, weight=1, uniform='row')
            Label(self.bottomleftbuttons, width=1, bg=NAVYBLUE).grid(
                row=0, column=x, rowspan=1, columnspan=1, sticky=NSEW)
        for y in range(2):
            self.bottomleftbuttons.rowconfigure(y, weight=1, uniform='row')
            Label(self.bottomleftbuttons, width=1, bg=NAVYBLUE).grid(
                row=y, column=0, rowspan=1, columnspan=1, sticky=NSEW)

        self.signoutbutton = Button(self.bottomleftbuttons,
                            text="Sign Out", bg=OTHERPINK, fg="white", font=(FONTFORBUTTONS, 20),
                            relief="solid", height=1, width=1, cursor="hand2",
                            command=lambda: [
                                self.show_frame(LoginPage),
                                self.togglebuttonrelief(self.loginbutton),
                                self.signout()])
                                
        self.studentbutton = Button(self.bottomleftbuttons,
                            text="Student\nButton", bg=NICEBLUE, fg="white", font=(FONTFORBUTTONS, 20),
                            relief="solid", height=1, width=1,
                            command=lambda: [
                                self.show_loggedin()])

        self.adminbutton = Button(self.bottomleftbuttons,
                            text="Admin\nButton", bg=NICEBLUE, fg="white", font=(FONTFORBUTTONS, 20),
                            relief="solid", height=1, width=1,
                            command=lambda: [
                                self.show_admin()])

        self.signoutbutton.grid(row=0, column=0, rowspan=2, columnspan=3, sticky=NSEW)
        # self.studentbutton.grid(row=0, column=3, rowspan=2, columnspan=3, sticky=NSEW)
        # self.adminbutton.grid(row=0, column=6, rowspan=2, columnspan=3, sticky=NSEW)

        self.remindercontainer = Frame(self.bottomleftbuttons, bg=LIGHTYELLOW, width=1, height=1)
        # self.remindercontainer.grid(row=0, column=9, rowspan=2, columnspan=11, sticky=NSEW)
        self.remindercontainer.grid_propagate(False)
        self.sidebarframe = Frame(self, bg=NAVYBLUE, width=1, height=1)
        self.sidebarframe.grid(row=2, column=0, rowspan=14, columnspan=2,
                             sticky=NSEW)
        self.sidebarframe.grid_propagate(False)

        for x in range(2):
                self.sidebarframe.columnconfigure(x, weight=1, uniform='row')
                Label(self.sidebarframe, width=1, bg=NAVYBLUE).grid(
                row=0, column=x, sticky=NSEW)
        for y in range(14):
                self.sidebarframe.rowconfigure(y, weight=1, uniform='row')
                Label(self.sidebarframe, width=1, bg=NAVYBLUE).grid(
                row=y, column=0, sticky=NSEW)
        

        self.calendarimage = Image.open(r"Assets\Main Assets\SideCalendar.png")
        self.calendarimage = ImageTk.PhotoImage(self.calendarimage.resize(
            (math.ceil(120 * dpi/96), math.ceil(120 * dpi/96)), Image.Resampling.LANCZOS))
        self.sidecalendar = Button(self.sidebarframe, image=self.calendarimage, bg=NAVYBLUE,cursor="hand2",
                                borderwidth=1, relief="flat", height=1, width=1,
                                command=lambda:[
                                    self.make_a_container()])


        
        #Clickable Calendar Frame
        #bind escape to close the window 
        self.bind("<Escape>", lambda e: self.destroy())
        self.welcomelabel("Stranger", "Viewer")
        self.createcalendarframe()
        self.createwindowmanagementframe()
        self.frames = {}
        # This class-based approach is used with heavy reference from Bryan Oakley's tutorial on StackOverflow
        # https://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter
        # https://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter/7557028#7557028
        # How this works is that the windows are assigned to a dictionary, and the show_frame function
        # is used to switch between the frames. This is done by bringing the frame to the front, and
        # then hiding the other frames. This is done by using the tkraise() function, which brings the
        # frame to the front, and then using the tkraise() function, which brings the frame to the front,
        # The Window() class is the main window, and the other classes are the frames that represent pages possess common functions 
        # and attributes when initialized as controller. The controller is the main window, and the pages are the frames.
        for F in (RegistrationPage, LoginPage, MainPage, 
                EventView, EventRegistration, EventCreation,
                ManagementSuite, CalendarPage, FeedbackForm):
            frame = F(parent=self.centercontainer, controller=self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, rowspan=16, columnspan=28, sticky=NSEW)

        #Shows the loading frame
        self.show_frame(LoginPage)
        self.togglebuttonrelief(self.loginbutton)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.grid()
        frame.tkraise()

    def get_page(self, page_class):
        return self.frames[page_class]

    def signout(self):
        self.mainpagebutton.grid_forget()
        self.eventlistbutton.grid_forget()
        self.eventregistrationbutton.grid_forget()
        self.eventcreationbutton.grid_forget()
        self.viewparticipantsbutton.grid_forget()
        self.calendarbutton.grid_forget()
        self.feedbackbutton.grid_forget()
        self.sidecalendar.grid_forget()
        self.welcomelabel("Stranger", "Viewer")
        self.welcomebuttonlabel.configure(cursor="arrow")
        global LOGGEDINAS
        global LOGINSTATE
        global LOGINID
        print(LOGINSTATE)
        if LOGINSTATE != False:
            messagebox.showinfo("Sign Out", "You have been signed out.")
        elif LOGINSTATE == False:
            messagebox.showerror("Sign Out", "You are already signed out.")
        LOGGEDINAS = "Viewer"
        LOGINSTATE = False
        LOGINID = "Viewer"

    def show_loggedin(self):
        self.mainpagebutton.grid(row=0, column=6, rowspan=2, columnspan=3,sticky=NSEW)
        self.eventlistbutton.grid(row=0, column=9, rowspan=2, columnspan=3, sticky=NSEW)
        self.eventregistrationbutton.grid(row=0, column=12, rowspan=2,columnspan=3,sticky=NSEW)
        self.calendarbutton.grid(row=0, column=15, rowspan=2,columnspan=3, sticky=NSEW)
        self.feedbackbutton.grid(row=0, column=18, rowspan=2,columnspan=3, sticky=NSEW)
        self.sidecalendar.grid(row=10, column=0, rowspan=2, columnspan=2, sticky=NSEW)

    def show_admin(self):
        self.mainpagebutton.grid(row=0, column=6, rowspan=2, columnspan=3,sticky=NSEW)
        self.eventlistbutton.grid(row=0, column=9, rowspan=2, columnspan=3, sticky=NSEW)
        self.eventregistrationbutton.grid(row=0, column=12, rowspan=2,columnspan=3,  sticky=NSEW)
        self.eventcreationbutton.grid(row=0, column=15, rowspan=2,columnspan=3, sticky=NSEW)
        self.viewparticipantsbutton.grid(row=0, column=18, rowspan=2,columnspan=3, sticky=NSEW)
        self.calendarbutton.grid(row=0, column=21, rowspan=2,columnspan=3, sticky=NSEW)
        self.feedbackbutton.grid(row=0, column=24, rowspan=2,columnspan=3, sticky=NSEW)
        self.sidecalendar.grid(row=10, column=0, rowspan=2, columnspan=2, sticky=NSEW)

    def welcomelabel(self, name, role):
        self.welcomeframe = Frame(self, bg=NICEBLUE, width=1, height=1)
        self.welcomeframe.grid(row=16, column=20, rowspan=2, columnspan=12, sticky=NSEW)
        self.welcomeframe.grid_propagate(False)

        for x in range(8):
            self.welcomeframe.columnconfigure(x, weight=1, uniform='row')
            Label(self.welcomeframe, width=1, bg=NICEBLUE).grid(
                row=0, column=x, sticky=NSEW)
        for y in range(2):
            self.welcomeframe.rowconfigure(y, weight=1, uniform='row')
            Label(self.welcomeframe, width=1, bg=NICEBLUE).grid(
                row=y, column=0, sticky=NSEW)
                
        self.welcomebuttonlabel = Button(self.welcomeframe, width=1, height=1,state=DISABLED,text="",font=("Atkinson Hyperlegible", 30), fg="white",  cursor="arrow",
                                    disabledforeground=WHITE, bg=DARKBLUE, command=lambda:self.make_a_container())
        self.welcomebuttonlabel.grid(row=0, column=0, rowspan=2, columnspan=8, sticky=NSEW)
        self.welcomebuttonlabel.configure(text=f"Welcome {name.capitalize()} as {role.capitalize()}!\nWe are glad to have you here!")
        self.welcomebuttonlabel.grid_propagate(False)

    def deletethewindowbar(self):
        hwnd:int = get_handle(self)
        style:int = GetWindowLongPtrW(hwnd, GWL_STYLE)
        style &= ~(WS_CAPTION | WS_THICKFRAME)
        SetWindowLongPtrW(hwnd, GWL_STYLE, style)
    
    def showthewindowbar(self):
        hwnd:int = get_handle(self)
        style:int = GetWindowLongPtrW(hwnd, GWL_STYLE)
        style |= (WS_CAPTION | WS_THICKFRAME)
        SetWindowLongPtrW(hwnd, GWL_STYLE, style)


    #Window management button frame
    def createwindowmanagementframe(self):
        self.windowmanagementframe = Frame(self, bg=NAVYBLUE, width=1, height=1,highlightthickness=1, highlightbackground=WHITE)
        self.windowmanagementframe.grid(row=0, column=30, rowspan=2, columnspan=2, 
                                    sticky=NSEW)
        self.windowmanagementframe.grid_propagate(False)

        for x in range(2):
            self.windowmanagementframe.columnconfigure(x, weight=1, uniform='row')
            Label(self.windowmanagementframe, width=1, bg=NAVYBLUE).grid(
                row=0, column=x, sticky=NSEW)
        for y in range(2):
            self.windowmanagementframe.rowconfigure(y, weight=1, uniform='row')
            Label(self.windowmanagementframe, width=1, bg=NAVYBLUE).grid(
                row=y, column=0, sticky=NSEW)

        self.minimizebutton = Button(self.windowmanagementframe, width=1, height=1,
                            text="Show", font=("Atkinson Hyperlegible", 12),
                            bg="#fdbc40", fg="WHITE", relief=RAISED,
                            command=lambda:[
                                self.state('normal'),
                                self.showthewindowbar()
                                ])
        self.minimizebutton.grid(row=0, column=0, sticky=NSEW)
        self.minimizebutton.grid_propagate(False)
        self.maximizebutton = Button(self.windowmanagementframe,
                            text="Hide", font=("Atkinson Hyperlegible", 12),
                            bg="#33c748", fg="WHITE", width=1, height=1, relief=RAISED,
                            command=lambda:[
                                self.deletethewindowbar(),
                                self.state("zoomed")
                                ])
        self.maximizebutton.grid(row=1, column=0, sticky=NSEW)
        self.maximizebutton.grid_propagate(False)
        self.closewindowbutton = Button(self.windowmanagementframe, text="Close", font=("Atkinson Hyperlegible", 12),
                                    bg="#fc5753", fg="WHITE", width=1, height=1, relief=RAISED,
                                    command=lambda:[
            self.destroy()
        ])
        self.closewindowbutton.grid(row=0, column=1, rowspan=2, columnspan=1, sticky=NSEW)
        self.closewindowbutton.grid_propagate(False)
    #this function toggles the relief to sunken every time the mouse clicks the button
    def togglebuttonrelief(self, button):
        self.buttonlist = [self.signupbutton, self.loginbutton, self.mainpagebutton,
        self.calendarbutton, self.eventlistbutton, self.eventregistrationbutton,
        self.eventcreationbutton, self.viewparticipantsbutton, self.feedbackbutton, self.calendarbutton]
        #sets every button to raised by default on click
        for b in self.buttonlist:
            b.configure(relief="raised")
        if button['relief'] == 'raised':
            button['relief'] = 'sunken'
        else:
            button['relief'] = 'raised'

    def enablethesebuttons(self):
        self.welcomebuttonlabel.configure(state="normal")
        self.signupbutton.configure(state="normal")

    def make_a_container(self):
        self.calendarframepopup.grid_remove()
        self.calendarframepopup.grid(row=10, column=20, rowspan=6, columnspan=12,
                            sticky=NSEW)
    def createcalendarframe(self):
        self.calendarframepopup = Frame(self, bg=OTHERPINK, width=1, height=1,
                                    borderwidth=1, relief="flat")
        self.calendarframepopup.grid_propagate(False)
        for x in range(12):
            self.calendarframepopup.columnconfigure(x, weight=1, uniform='row')
            Label(self.calendarframepopup, width=1, bg=OTHERPINK).grid(
                row=0, column=x, sticky=NSEW)
        for y in range(12):
            self.calendarframepopup.rowconfigure(y, weight=1, uniform='row')
            Label(self.calendarframepopup, width=1, bg=OTHERPINK).grid(
                row=y, column=0, rowspan=1, columnspan=1, sticky=NSEW,)
        self.introlabel = Label(self.calendarframepopup, text="What would you\nlike to do?",
                            font=("Atkinson Hyperlegible", 14), width=1, height=1,
                            bg=LAVENDER, fg="black")
        self.introlabel.grid(row=0, column=2, rowspan=2, columnspan=8, sticky=NSEW)
        self.introlabel.grid_propagate(False)
        self.viewbutton = Button(self.calendarframepopup,  cursor="hand2",
            text="View Calendar", font=("Atkinson Hyperlegible", 14),
            bg=DARKBLUE, fg="WHITE", width=1, height=1,
            command=lambda:[
                self.show_frame(CalendarPage),
                self.togglebuttonrelief(self.calendarbutton),
                self.calendarframepopup.grid_remove()
            ])
        self.loggedinaslabel = Label(self.calendarframepopup, 
            text="Logged in as:\n" + LOGINID, font=("Atkinson Hyperlegible", 14),
            bg=LAVENDER, fg="black", width=1, height=1)
        self.loggedinaslabel.grid(row=10, column=1, rowspan=2, columnspan=10, sticky=NSEW)
        self.loggedinaslabel.grid_propagate(False)
        self.viewbutton.grid(row=5, column=1, rowspan=2, columnspan=5, sticky=NSEW,padx=2)
        self.viewbutton.grid_propagate(False)
        self.editbutton = Button(self.calendarframepopup, cursor="hand2",
            text="Check My Registered Events", font=("Atkinson Hyperlegible", 14),
            bg=DARKBLUE, fg="WHITE", width=1, height=1,
            command=lambda:[self.getevents()])
        self.editbutton.grid(row=5, column=6, rowspan=2, columnspan=5, sticky=NSEW,padx=2)
        self.editbutton.grid_propagate(False)
        self.closebutton = Button(self.calendarframepopup,  cursor="hand2",
            text="Close", font=("Atkinson Hyperlegible", 14),
            bg=DARKBLUE, fg="WHITE", width=1, height=1,
            command=lambda:[
                self.calendarframepopup.grid_remove()
            ])
        self.closebutton.grid(row=8, column=1, rowspan=2, columnspan=10, sticky=NSEW,padx=2)
        self.closebutton.grid_propagate(False)
        self.signoutbutton.grid_propagate(False)
        self.studentbutton.grid_propagate(False)
        self.adminbutton.grid_propagate(False)
    def get_display_size(self):
        if self.screensize <= (1920, 1080):
            self.state('zoomed')
    def getevents(self):
        self.conn = sqlite3.connect('interactivesystem.db')
        self.c = self.conn.cursor()
        # Join query to get event date from tables eventregistration and eventcreation
        self.c.execute("SELECT eventregistration.event_registered, eventcreation.event_startdate FROM eventregistration INNER JOIN eventcreation ON eventregistration.eventkey_registered = eventcreation.eventkey_number WHERE email = ?", (LOGINID,))
        self.rows = self.c.fetchall() 
        if self.rows == []:
            messagebox.showinfo("Error", "You have not registered for any events yet!")
            return
        # presenting the information in a messagebox 
        # Given a return data ('Corgi Surfing', '2022-11-27')
        # The messagebox will show the following:
        # Corgi Surfing - 27/11/2022
        messagestoshow = []
        for row in self.rows:
            self.eventname = row[0]
            self.eventdate = row[1]
            self.message = self.eventname + " - " + self.eventdate
            messagestoshow.append(self.message)
        formattedmessage = "\n".join(messagestoshow)
        messagebox.showinfo("Registered Events", f"You have registered for:\n{formattedmessage}\nCheck out the calendar for more details!")

class RegistrationPage(Frame):
    def __init__(self, parent, controller):
        self.controller = controller
        Frame.__init__(self, parent, bg=LIGHTPURPLE)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        for x in range(42):
            self.columnconfigure(x, weight=1, uniform='x')
            Label(self, width=1, bg=LIGHTPURPLE, relief="flat").grid(
                row=0, column=x, rowspan=1, columnspan=1, sticky=NSEW)
        for y in range(21):
            self.rowconfigure(y, weight=1, uniform='x')
            Label(self, width=1, bg=LIGHTPURPLE, relief="flat").grid(
                row=y, column=0, rowspan=1, columnspan=1, sticky=NSEW)
        # constants
        self.bgimageregpageimg = Image.open(r"Assets\RegistrationPage\registrationpgbg.png")
        self.bgimageregpageimg = ImageTk.PhotoImage(self.bgimageregpageimg.resize(
            (math.ceil(1680 * dpi / 96), math.ceil(840 * dpi / 96)), Image.Resampling.LANCZOS))
        self.bgimageregpage = Label(self, image=self.bgimageregpageimg, width=1,height=1, bg=LIGHTPURPLE)
        self.bgimageregpage.grid(row=0, column=0, rowspan=21, columnspan=42, sticky=NSEW)
        FONTNAME = "Avenir Next Medium"
        FIRSTNAME = "First Name"
        LASTNAME = "Last Name"
        EMAILTEXT = "Please enter your student email."
        PASSWORDTEXT = "Please enter your password."
        CONFPASSTEXT = "Please confirm your password."
        # database functions
        conn = sqlite3.connect('interactivesystem.db')
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS registration(
            first_name text NOT NULL,
            last_name text NOT NULL,
            email text NOT NULL PRIMARY KEY,
            password text NOT NULL,
            role text NOT NULL
        )""")
        # c.execute ("DROP TABLE registration")
        # possibly, we could make two functions, one to validate input and another to actually send the data to the database, instead of checking validity itself in checkfields()

        def checkfields():
            # c.execute("DROP TABLE registration")
            firstnametext = firstnamefield.get()
            lastnametext = lastnamefield.get()
            emailtext = emailfield.get()
            passwordtext = passwordfield.get()
            confirmpasstext = confirmpasswordfield.get()
            #Raise an error if more than two @'s in the email
            if emailtext.count("@") > 1:
                messagebox.showerror("Invalid Email", "Please enter a valid email.")
                return
            if emailtext.count("@") == 0:
                messagebox.showerror("Invalid Email", "Please enter a valid email.")
                return
            try:
                emailending = emailfield.get().split("@")[1]
                namefield = emailfield.get().split("@")[0]
                if namefield == "" or " " in namefield:
                    messagebox.showerror("Invalid Email", "Please enter a valid email.\nThere should also be no spaces in the first half of the email.")
                    return
                if emailending == "student.newinti.edu.my":
                    role = "student"
                    validemail = True
                elif emailending == "newinti.edu.my":
                    role = "admin"
                    validemail = True
                else:
                    validemail = False
                    role = "invalid"
                if (FIRSTNAME in firstnametext) or (LASTNAME in lastnametext) or (EMAILTEXT in emailtext) or (PASSWORDTEXT in passwordtext) or (CONFPASSTEXT in confirmpasstext):
                    messagebox.showerror("Error", "Please fill in all fields.")
                elif passwordtext != confirmpasstext:
                    messagebox.showerror("Error", "Passwords do not match.")
                elif validemail == False:
                    messagebox.showerror(
                        "Error", "Please enter a valid email.")
                else:
                    with conn:
                        information = (firstnametext, lastnametext,
                                       emailtext, passwordtext, role)
                        c.execute(
                            """INSERT INTO registration VALUES(?, ?, ?, ?, ?)""", information)
                        messagebox.showinfo(
                            "Success", "You have successfully registered.")
                        controller.show_frame(LoginPage)
                        controller.togglebuttonrelief(controller.loginbutton)
                        cleareveryentry()
            except sqlite3.IntegrityError:
                messagebox.showerror("Error", "Email already in use.")
            except IndexError:
                emailwarning.configure(text="You have not entered an email")
                messagebox.showerror("Error", "Please enter a valid email.")

        def clearnamefields():
            if firstnamefield.get() == FIRSTNAME:
                firstnamefield.delete(0, END)
            if lastnamefield.get() == LASTNAME:
                lastnamefield.delete(0, END)

        def repopulatenamefields():
            if firstnamefield.get() == "":
                firstnamefield.insert(0, FIRSTNAME)
            if lastnamefield.get() == "":
                lastnamefield.insert(0, LASTNAME)
        emailwarning = Label(self, text="Please enter a valid email address.", font=(
            'Arial', 10), width=1, height=1, fg='#000000', bg='#FFF5E4')

        def clearemailfield():
            emailfield.configure(fg="black")
            if emailfield.get() == EMAILTEXT:
                emailfield.delete(0, END)
            try:
                emailending = emailfield.get().split("@")[1]
                if emailending in ["student.newinti.edu.my", "newinti.edu.my"]:
                    emailwarning.configure(fg="black")
                else:
                    emailwarning.configure(
                        text="Email entered is not with INTI or incomplete")
            except IndexError:
                emailwarning.configure(text="You have not entered an email")

        def showwarninglabelaboveentry():
            # configure emailwarning to show and become red when invalid email
            emailwarning.grid(row=5, column=33, columnspan=7, sticky=NSEW)
            emailwarning.configure(
                text="Please enter a valid email.", fg="red")

        def repopulateemailfield():
            try:
                emailending = emailfield.get().split("@")[1]
                if emailending not in ["student.newinti.edu.my", "newinti.edu.my"]:
                    if emailfield == "":
                        emailfield.insert(0, EMAILTEXT)
                    emailfield.configure(fg="red")
                    showwarninglabelaboveentry()
                else:
                    emailfield.configure(fg="black")
                    emailwarning.grid_forget()
            except IndexError:
                if emailfield.get() == EMAILTEXT or emailfield.get() == "":
                    emailfield.delete(0, END)
                    emailfield.insert(0, EMAILTEXT)
                emailfield.configure(fg="red")
                showwarninglabelaboveentry()

        def clearpasswordfield():
            passwordfield.configure(fg="black")
            passwordfield.configure(show="*")
            if passwordfield.get() == PASSWORDTEXT:
                passwordfield.delete(0, END)
            try:
                passwordcontents = passwordfield.get()
            except:
                pass

        def repopulatepasswordfield():
            if passwordfield.get() == "":
                passwordfield.insert(0, PASSWORDTEXT)
                passwordfield.configure(show="")
                passwordfield.configure(fg="red")
            else:
                passwordfield.configure(show="*")

        SCAMTEXT = "Please confirm your password "
        def clearconfpasswordfield():
            confirmpasswordfield.configure(fg="black")
            confirmpasswordfield.configure(show="*")
            if confirmpasswordfield.get() == CONFPASSTEXT or confirmpasswordfield.get() == SCAMTEXT:
                confirmpasswordfield.delete(0, END)
        def repopulateconfpasswordfield():    
            if confirmpasswordfield.get() != SCAMTEXT:    
                confirmpasswordfield.configure(show="*")
            if confirmpasswordfield.get() == "":
                confirmpasswordfield.insert(0, CONFPASSTEXT)
                confirmpasswordfield.configure(show="")
                confirmpasswordfield.configure(fg="red")

        def cleareveryentry():
            firstnamefield.delete(0, END)
            lastnamefield.delete(0, END)
            emailfield.delete(0, END)
            passwordfield.delete(0, END)
            confirmpasswordfield.delete(0, END)
            firstnamefield.insert(0, FIRSTNAME)
            lastnamefield.insert(0, LASTNAME)
            emailfield.insert(0, EMAILTEXT)
            passwordfield.insert(0, PASSWORDTEXT)
            confirmpasswordfield.insert(0, SCAMTEXT)
            passwordfield.configure(show="")
            confirmpasswordfield.configure(show="")
  
            

        # Labels
        # enterdetailslabel = Label(self, text="Please enter your details as shown in the entries.", font=(
        #     'Atkinson Hyperlegible', 16), width=1, height=1, fg='#000000', bg='#FFF5E4')
        # enterdetailslabel.grid(row=0, column=24,
        #                        rowspan=2, columnspan=17, sticky=NSEW)

        # Entries
        firstnamefield = Entry(self, width=1, bg='#FFFFFF',
                               font=(FONTNAME, 18), justify='center')
        firstnamefield.grid(row=3, column=23,
                            rowspan=2, columnspan=7, sticky=NSEW)
        firstnamefield.insert(0, FIRSTNAME)
        firstnamefield.grid_propagate(False)

        lastnamefield = Entry(self, width=1, bg='#FFFFFF',
                              font=(FONTNAME, 18), justify='center')
        lastnamefield.grid(row=3, column=33,
                           rowspan=2, columnspan=7, sticky=NSEW)
        lastnamefield.insert(0, LASTNAME)
        lastnamefield.grid_propagate(False)

        emailfield = Entry(self, width=1, bg='#FFFFFF',
                           font=(FONTNAME, 18), justify='center')
        emailfield.grid(row=6, column=23,
                        rowspan=2, columnspan=17, sticky=NSEW)
        emailfield.insert(0, EMAILTEXT)
        emailfield.grid_propagate(False)

        passwordfield = Entry(self, width=1, bg='#FFFFFF',
                              font=(FONTNAME, 18), justify='center')
        passwordfield.grid(row=9, column=23,
                           rowspan=2, columnspan=17, sticky=NSEW)
        passwordfield.insert(0, PASSWORDTEXT)
        passwordfield.grid_propagate(False)

        confirmpasswordfield = Entry(
            self, width=1, bg='#FFFFFF', font=(FONTNAME, 18), justify='center')
        confirmpasswordfield.grid(row=12, column=23,
                                  rowspan=2, columnspan=17, sticky=NSEW)
        confirmpasswordfield.insert(0, CONFPASSTEXT)
        confirmpasswordfield.grid_propagate(False)

        # Entry Binding
        firstnamefield.bind("<FocusIn>", lambda event: clearnamefields())
        lastnamefield.bind("<FocusIn>", lambda event: clearnamefields())
        firstnamefield.bind("<FocusOut>", lambda event: repopulatenamefields())
        lastnamefield.bind("<FocusOut>", lambda event: repopulatenamefields())
        emailfield.bind("<FocusIn>", lambda event: clearemailfield())
        emailfield.bind("<FocusOut>", lambda event: repopulateemailfield())
        passwordfield.bind("<FocusIn>", lambda event: clearpasswordfield())
        passwordfield.bind(
            "<FocusOut>", lambda event: repopulatepasswordfield())
        confirmpasswordfield.bind(
            "<FocusIn>", lambda event: clearconfpasswordfield())
        confirmpasswordfield.bind(
            "<FocusOut>", lambda event: repopulateconfpasswordfield())

        

        self.intibanner = Image.open(r"Assets\RegistrationPage\intibanner.png")
        self.intibanner = ImageTk.PhotoImage(self.intibanner.resize(
            (math.ceil(600 * dpi / 96), math.ceil(160 * dpi / 96)), Image.Resampling.LANCZOS))
        self.logolabel = Button(self, image=self.intibanner,
                           anchor=CENTER, width=1, height=1,
                           background= NICEBLUE, 
                           command = lambda:aboutINTIcontainer())
        self.logolabel.grid(row=1, column=3, columnspan=15,
                       rowspan=4, sticky=NSEW)
        # self.titleart = Image.open(r"assets\DR7j7r0.png")
        # self.titleart = ImageTk.PhotoImage(self.titleart.resize(
        #     (math.ceil(680 * dpi / 96), math.ceil(320 * dpi / 96)), Image.Resampling.LANCZOS))
        # titleartlabel = Button(self, image=self.titleart,
        #                        background= NICEBLUE, 
        #                        anchor=CENTER, width=1, height=1)
        # titleartlabel.grid(row=10, column=2, columnspan=17,
        #                    rowspan=8, sticky=NSEW)
        # titleartlabel.grid_propagate(False)
        # Buttons
        signupbutton = Button(self, text="SIGN UP", width=1, height=1, cursor="hand2", font=(
            'Atkinson Hyperlegible', 14), fg='#000000', command=lambda: checkfields(), bg=LIGHTYELLOW)
        signupbutton.grid(row=15, column=27, columnspan=9,
                          rowspan=2, sticky=NSEW)
        signupbutton.grid_propagate(False)

        loginbutton = Button(self, text="Click here to sign in.", cursor="hand2",
        font=('Atkinson Hyperlegible', 14), width=1, height=1,
        fg='#000000', command=lambda: [
        controller.show_frame(LoginPage),
        controller.togglebuttonrelief(controller.loginbutton),
        cleareveryentry()],
        bg=OTHERPINK)
        loginbutton.grid(row=18, column=27, columnspan=9,
                         rowspan=2, sticky=NSEW)

        def aboutINTIcontainer():
            calendarframepopup = Frame(controller, bg=NICEBLUE, width=1, height=1,
                                borderwidth=1, relief="flat")
            calendarframepopup.grid(row=6, column=4, rowspan=8, columnspan=10,
                             sticky=NSEW)
            calendarframepopup.grid_propagate(False)
            # self.calendarframepopup = calendarframepopup
            for x in range(10):
                calendarframepopup.columnconfigure(x, weight=1, uniform='row')
                Label(calendarframepopup, width=1, bg=NICEBLUE, borderwidth=0, relief="solid").grid(
                  row=0, column=x, sticky=NSEW)
            for y in range(8):
                calendarframepopup.rowconfigure(y, weight=1, uniform='row')
                Label(calendarframepopup, width=1, bg=NICEBLUE, borderwidth=0, relief="solid").grid(
                    row=y, column=0, rowspan=1, columnspan=1, sticky=NSEW,)
            randomlabel = Label(calendarframepopup, text="INTI Is Awesome!", font=("Comic Sans Ms", 18), width=1,height=1, fg="white",bg=DARKBLUE)
            randomlabel.grid(row=0, column=0, rowspan=1, columnspan=14, sticky=NSEW)
            randomlabel.grid_propagate(False)
            randombutton = Button(calendarframepopup, text="click me to close ", font=("Comic Sans Ms", 18), bg=DARKBLUE, fg="WHITE", command=lambda:[
            calendarframepopup.grid_forget()])
            randombutton.grid(row=9, column=0, rowspan=1, columnspan=14, sticky=NSEW)


class LoginPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=LIGHTPURPLE)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.controller = controller
        for x in range(42):
            self.columnconfigure(x, weight=1, uniform='x')
            Label(self, width=1, bg=LIGHTPURPLE).grid(
                row=0, column=x, rowspan=1, columnspan=1, sticky=NSEW)
        for y in range(21):
            self.rowconfigure(y, weight=1, uniform='x')
            Label(self, width=1, bg=LIGHTPURPLE).grid(
                row=y, column=0, rowspan=1, columnspan=1, sticky=NSEW)
        # Database Functions for Logging in and setting loginstate to student or teacher
        # Sqlite3 commands to fetch registered emails from database and assigning roles based on email ending.
        # If email is not found in database, it will return an error message.
        # If email is found in database, it will return a success message.
        global dpi 
        conn = sqlite3.connect('interactivesystem.db')
        c = conn.cursor()

        def checkcredentials():
            global LOGGEDINAS
            global LOGINSTATE
            global LOGINID
            if LOGGEDINAS != "Admin" and LOGGEDINAS != "Student" and LOGINSTATE != True:
                with conn:
                    c.execute("SELECT * FROM registration WHERE email = ? AND password = ?",
                              (emailfield.get(), passwordfield.get()))
                    for row in c.fetchall():
                        name = row[0]
                        email = row[2]
                        password = row[3]
                        role = row[4]
                        # print("Your name is: ", name)
                        # print("Email is: ", email)
                        # print("Password is :", password)
                        # print("Your role is : ", role)
                        # print(row)
                    try:
                        if role == "student":
                            messagebox.showinfo(
                                "Login Successful", f"Welcome {name}!")
                            LOGGEDINAS = "Student"
                            LOGINSTATE = True
                            LOGINID = email
                            controller.show_loggedin()
                            controller.welcomelabel(name, role)
                            controller.welcomebuttonlabel.configure(cursor="hand2")
                            controller.loggedinaslabel.configure(text=(f"Logged in as:\n{email}"))
                            controller.show_frame(MainPage)
                            controller.togglebuttonrelief(controller.mainpagebutton)
                            controller.enablethesebuttons()
                        elif role == "admin":
                            messagebox.showinfo(
                                "Login Successful", f"Welcome {name}!")
                            LOGGEDINAS = "Admin"
                            LOGINSTATE = True
                            LOGINID = email
                            controller.show_admin()
                            controller.welcomelabel(name, role)
                            controller.welcomebuttonlabel.configure(cursor="hand2")
                            controller.loggedinaslabel.configure(text=(f"Logged in as:\n{email}"))
                            controller.show_frame(MainPage)
                            controller.togglebuttonrelief(controller.mainpagebutton)
                            controller.enablethesebuttons()

                        else:
                            messagebox.showerror(
                                "Login Failed", "Invalid Email or Password")
                    except UnboundLocalError:
                        messagebox.showerror(
                            "Login Failed", "Invalid Email or Password")
            else:
                roles = LOGGEDINAS
                messagebox.showerror(
                    "Login Failed", f"You are already logged in as {roles}!")


        def signinbuttonpressed():
            checkcredentials()


        def clearemailfield():
            emailfield.configure(fg="black")
            if emailfield.get() == EMAILTEXT:
                emailfield.delete(0, END)
            try:
                emailending = emailfield.get().split("@")[1]
                if emailending in ["student.newinti.edu.my", "newinti.edu.my"]:
                    emailwarning.configure(fg="black")
                else:
                    emailwarning.configure(
                        text="Email entered is not with INTI or incomplete")
            except IndexError:
                emailwarning.configure(text="You have not entered an email")

        def showwarninglabelaboveentry():
            # configure emailwarning to show and become red when invalid email
            emailwarning.grid(row=6, column=31, columnspan=8,
                              rowspan=1, sticky=NSEW)
            emailwarning.configure(
                text="Please enter a valid email.", fg="red")

        def repopulateemailfield():
            try:
                emailending = emailfield.get().split("@")[1]
                if emailending not in ["student.newinti.edu.my", "newinti.edu.my"]:
                    if emailfield == "":
                        emailfield.insert(0, EMAILTEXT)
                    emailfield.configure(fg="red")
                    showwarninglabelaboveentry()
                else:
                    emailfield.configure(fg="black")
                    emailwarning.grid_forget()
            except IndexError:
                if emailfield.get() == EMAILTEXT or emailfield.get() == "":
                    emailfield.delete(0, END)
                    emailfield.insert(0, EMAILTEXT)
                emailfield.configure(fg="red")
                showwarninglabelaboveentry()

        def clearpasswordfield():
            passwordfield.configure(fg="black")
            passwordfield.configure(show="*")
            if passwordfield.get() == PASSWORDTEXT:
                passwordfield.delete(0, END)
            try:
                passwordcontents = passwordfield.get()
            except:
                pass

        def repopulatepasswordfield():
            if passwordfield.get() == "":
                passwordfield.insert(0, PASSWORDTEXT)
                passwordfield.configure(show="")
                passwordfield.configure(fg="red")
            else:
                passwordfield.configure(show="*")

        # Widgets
        # label = Label(self, text="This is the primary login page", font=(
        #     'Arial', 16), width=1, height=1, fg='#000000', bg='#FFF5E4')
        # label.grid(row=1, column=2, columnspan=18,
        #            rowspan=2, sticky=NSEW)
        EMAILTEXT = "Please enter your registered email address"
        PASSWORDTEXT = "Please enter your password"
        FONTNAME = "Avenir Next Medium"
        # Buttons
        # self.intibanner = Image.open(r"assets\Home-Banner-INTI.png")
        # self.intibanner = ImageTk.PhotoImage(self.intibanner.resize(
        #     (math.ceil(720 * dpi / 96), math.ceil(240 * dpi / 96)), Image.Resampling.LANCZOS))
        # logolabel = Button(self, image=self.intibanner,
        #                    anchor=CENTER, width=1, height=1)
        # logolabel.grid(row=1, column=24, columnspan=18,
        #                rowspan=5, sticky=NSEW)
        self.backgroundimageoriginal = Image.open(r"Assets\backgroundimage.png")
        self.backgroundimage = ImageTk.PhotoImage(self.backgroundimageoriginal.resize(
            (math.ceil(1680 * dpi / 96), math.ceil(840 * dpi / 96)), Image.Resampling.LANCZOS))
        
        self.backgroundimagelabel = Label(self, image=self.backgroundimage, width=1, height=1, bg=LIGHTPURPLE)
        self.backgroundimagelabel.grid(row=0, column=0, rowspan=21, columnspan=43, sticky=NSEW)
        self.backgroundimagelabel.grid_propagate(False)
        self.signinbuttonimage = Image.open(r"Assets\signinbutton.png")
        self.signinbuttonimage = ImageTk.PhotoImage(self.signinbuttonimage.resize(
            (math.ceil(440 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.signinbutton = Button(self, image=self.signinbuttonimage, width=1, height=1, cursor="hand2",
        bg=LIGHTPURPLE, relief="flat",command=lambda:signinbuttonpressed())
        self.signinbutton.grid(row=15, column=26, rowspan=2, columnspan=11, sticky=NSEW)
        self.signinbutton.grid_propagate(False)
        self.signupbuttonimage = Image.open(r"Assets\signupbutton.png")
        self.signupbuttonimage = ImageTk.PhotoImage(self.signupbuttonimage.resize(
            (math.ceil(600 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.signupbutton = Button(self, image=self.signupbuttonimage, width=1, height=1, cursor="hand2",
        bg=LIGHTPURPLE, borderwidth=1, relief="flat", command=lambda:[controller.show_frame(RegistrationPage),
        controller.togglebuttonrelief(controller.signupbutton)])
        self.signupbutton.grid(row=18, column=24, rowspan=2, columnspan=15,sticky=NSEW)
        self.signupbutton.grid_propagate(False)
        emailwarning = Label(self, text="Please enter a valid email address.", font=(
            'Arial', 10), width=1, height=1, fg='#000000', bg='#FFF5E4')
        emailfield = Entry(self, width=1, bg='#FFFFFF', highlightthickness=1,
                           font=(FONTNAME, 14), justify='center')
        emailfield.grid(row=7, column=25, columnspan=13,    
                        rowspan=2, sticky=NSEW)
        emailfield.insert(0, EMAILTEXT)
        emailfield.grid_propagate(False)
        passwordfield = Entry(self, width=1, bg='#FFFFFF', highlightthickness=1,
                              font=(FONTNAME, 14), justify='center')
        passwordfield.grid(row=12, column=25, columnspan=13,
                           rowspan=2, sticky=NSEW)
        passwordfield.insert(0, PASSWORDTEXT)
        passwordfield.grid_propagate(False)
        emailfield.bind("<FocusIn>", lambda a: clearemailfield())
        emailfield.bind("<FocusOut>", lambda a: repopulateemailfield())
        passwordfield.bind("<FocusIn>", lambda a: clearpasswordfield())
        passwordfield.bind("<FocusOut>", lambda a: repopulatepasswordfield()) 

        def resize():
            dimensions = [controller.winfo_width(), controller.winfo_height()]
            if controller.winfo_width() != dimensions[0] or controller.winfo_width != dimensions[1]:
                self.backgroundimageoriginal = Image.open(r"Assets\backgroundimage.png")
                self.backgroundimage = ImageTk.PhotoImage(self.backgroundimageoriginal.resize(
            (math.ceil(1680 * dpi / 96), math.ceil(840 * dpi / 96)), Image.Resampling.LANCZOS))
                self.backgroundimagelabel.config(image=self.backgroundimage)
        global eventID
        eventID = None
        controller.resizeDelay = 100
        def resizeEvent(event):
            global eventID
            if eventID:
                controller.after_cancel(eventID)
            if controller.state() == "zoomed":
                eventID = controller.after(controller.resizeDelay, resize)
        controller.bind('<Configure>', resizeEvent)
        if controller.screensize == (1920, 1080):
            self.removewindowbar()
    def removewindowbar(self):
        self.controller.deletethewindowbar()
        self.controller.state("zoomed")


        


class MainPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=LAVENDER)
        self.controller = controller
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        for x in range(42):
            self.columnconfigure(x, weight=1, uniform='x')
            Label(self, height=2, bg=LAVENDER).grid(
                row=0, column=x, rowspan=1, columnspan=1, sticky=N+S+E+W)
        for y in range(21):
            self.rowconfigure(y, weight=1, uniform='x')
            Label(self, width=5, bg=LAVENDER).grid(
                row=y, column=0, rowspan=1, columnspan=1, sticky=N+S+E+W)

        # Picture
        self.backgroundimageoriginal = Image.open(r"Assets\MainPage\MainPage.png")
        self.backgroundimage = ImageTk.PhotoImage(self.backgroundimageoriginal.resize(
            (math.ceil(1680 * dpi / 96), math.ceil(840 * dpi / 96)), Image.Resampling.LANCZOS))
        
        self.backgroundimagelabel = Label(self, image=self.backgroundimage, width=1, height=1, bg=LIGHTPURPLE)
        self.backgroundimagelabel.grid(row=0, column=0, rowspan=21, columnspan=43, sticky=N+S+E+W)
        self.backgroundimagelabel.grid_propagate(0)
        # Buttons
        self.feedbackimage = Image.open(r"Assets\MainPage\feedbackimage.png")
        self.feedbackimage = ImageTk.PhotoImage(self.feedbackimage.resize(
            (math.ceil(640 * dpi / 96), math.ceil(160 * dpi / 96)), Image.Resampling.LANCZOS)),
        feedbackbutton = Button(self, image=self.feedbackimage, width=1, height=1, relief="flat", fg='#000000', bg='#FFF5E4', cursor='hand2',
        command=lambda: [
        controller.show_frame(FeedbackForm),
        controller.togglebuttonrelief(controller.feedbackbutton) ])
        feedbackbutton.grid(row=8, column=2, columnspan=16,
                             rowspan=4, sticky=N+S+E+W)
        
        self.firsteventnamebutton = Button(self, text="Event 1:", font=(
         'Lucida Calligraphy', 14), width=1, height=1, relief="flat", fg='#000000', bg='#FFF5E4', cursor="hand2",
        command=lambda: [
        controller.show_frame(EventView),
        controller.togglebuttonrelief(controller.eventlistbutton),
        self.loadtheeventinquestion(self.lasteventindex)])

        self.firsteventnamebutton.grid(row=15, column=2, columnspan=16,
                             rowspan=2, sticky=N+S+E+W)
        self.secondeventsnamebutton = Button(self, text="Event 2:", font=(
         'Lucida Calligraphy', 14), width=1, height=1, relief="flat",fg='#000000', bg='#FFF5E4', cursor="hand2",
        command=lambda: [
        controller.show_frame(EventView),
        controller.togglebuttonrelief(controller.eventlistbutton),
        self.loadtheeventinquestion(self.lasteventindex-1)])
        self.secondeventsnamebutton.grid(row=18, column=2, columnspan=16,
                              rowspan=2, sticky=N+S+E+W)

        #Button
        eventlistbutton = Button(self, text="Event List", font=(
        'Lucida Calligraphy', 16), width=1, height=1, relief="flat",cursor="hand2", fg='#000000', bg='#FFF5E4',
        command=lambda: [
        controller.show_frame(EventView),controller.togglebuttonrelief(controller.eventlistbutton)]) 
        eventlistbutton.grid(row=16, column=21, columnspan=5,
                             rowspan=3, sticky=N+S+E+W)

        eventregistrationbutton = Button(self, text="Event\nRegistration", font=(
        'Lucida Calligraphy', 16), width=1, height=1, relief="flat", fg='#000000', bg='#FFF5E4',cursor="hand2",  command=lambda:
        [controller.show_frame(EventRegistration),
        controller.togglebuttonrelief(controller.eventregistrationbutton)])
        eventregistrationbutton.grid(row=16, column=28, columnspan=5,
                                     rowspan=3, sticky=N+S+E+W)

        calendarbutton = Button(self, text="Calendar", font=(
            'Lucida Calligraphy', 16), width=1, height=1,  relief="flat", fg='#000000', bg='#FFF5E4',cursor="hand2", command=
            lambda: [
            controller.show_frame(CalendarPage),
            controller.togglebuttonrelief(controller.calendarbutton)])
        calendarbutton.grid(row=16, column=35, columnspan=5,
                            rowspan=3, sticky=N+S+E+W)


        self.update_eventnames()
        self.refreshbutton = Button(self, text="Refresh", font=(
            'Lucida Calligraphy', 16), width=1, height=1, relief=RAISED, cursor="hand2",fg=BLACK, bg='#bcffff', command=
            lambda: [
            self.update_eventnames()])
        self.refreshbutton.grid(row=13, column=15, columnspan=3,
                            rowspan=1, sticky=N+S+E+W)
        self.conn = sqlite3.connect('interactivesystem.db')
        self.c = self.conn.cursor()
        with self.conn:
            self.c.execute("SELECT COUNT(event_name) FROM eventcreation")
        self.eventcount = self.c.fetchone()[0]
        self.lasteventindex = self.eventcount - 1

    def update_eventnames(self):
        self.conn = sqlite3.connect('interactivesystem.db')
        self.c = self.conn.cursor()
        with self.conn:
            self.c.execute("SELECT COUNT(event_name) FROM eventcreation")
        self.eventcount = self.c.fetchone()[0]
        self.lasteventindex = self.eventcount - 1
        self.c.execute("SELECT event_name FROM eventcreation ORDER BY eventkey_number DESC LIMIT 2")
        self.eventnames = self.c.fetchall()
        self.twoeventnames = [i[0] for i in self.eventnames] #first data for each tuple in list
        self.firsteventnamebutton.config(text=self.twoeventnames[0])
        self.secondeventsnamebutton.config(text=self.twoeventnames[1])
    def loadtheeventinquestion(self, index):
        eventviewrfrnce = self.controller.get_page(EventView)
        eventviewrfrnce.load_event(index)

class EventView(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=LAVENDER)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.controller = controller
        for x in range(42):
            self.columnconfigure(x, weight=1, uniform='x')
            Label(self, width=1, bg=LAVENDER).grid(
                row=0, column=x, rowspan=1, columnspan=1, sticky=NSEW)
        for y in range(21):
            self.rowconfigure(y, weight=1, uniform='x')
            Label(self, width=1, bg=LAVENDER).grid(
                row=y, column=0, rowspan=1, columnspan=1, sticky=NSEW)

        self.backgroundimageoriginal = Image.open(r"Assets\eventviewpage\backgroundimage.png")
        if controller.screensize == (1920, 1080):
            self.backgroundimage = ImageTk.PhotoImage(self.backgroundimageoriginal.resize(
                (math.ceil(1680 * dpi / 96), math.ceil(840 * dpi / 96)), Image.Resampling.LANCZOS))
        elif controller.screensize > (1920, 1080):
            self.backgroundimage = ImageTk.PhotoImage(self.backgroundimageoriginal.resize(
                (math.ceil(1680 * dpi / 96), math.ceil(840 * dpi / 96)), Image.Resampling.LANCZOS))
        
        self.backgroundimagelabel = Label(self, image=self.backgroundimage, width=1, height=1, bg=LIGHTPURPLE)
        self.backgroundimagelabel.grid(row=0, column=0, rowspan=21, columnspan=43, sticky=NSEW)
        self.backgroundimagelabel.grid_propagate(False)

        self.getregisteredbuttonimage = Image.open(r"Assets\eventviewpage\getregisteredbutton360x80.png")
        self.getregisteredbuttonimage = ImageTk.PhotoImage(self.getregisteredbuttonimage.resize(
            (math.ceil(360 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.getregisteredbutton = Button(self, image=self.getregisteredbuttonimage, width=1, height=1, bg=LIGHTPURPLE, cursor="hand2", command=lambda:
            [self.changetoeventregistration(self.imageindex)])
        self.getregisteredbutton.grid(row=13, column=5, columnspan=9,
                                    rowspan=2, sticky=NSEW) 
        def getregisteredbuttonhover(event):
            self.getregisteredbutton.config(borderwidth=2, relief=RAISED)
        def getregisteredbuttonleave(event):
            self.getregisteredbutton.config(borderwidth=0, relief=FLAT)
        self.getregisteredbutton.bind("<Enter>", getregisteredbuttonhover)
        self.getregisteredbutton.bind("<Leave>", getregisteredbuttonleave)
        # wait for everything to be initialized before calling this function


        self.showcaseimage = Label(self, image="", width=1, height=1, bg=LIGHTPURPLE)
        self.showcaseimage.grid(row=1, column=23, columnspan=17,
                     rowspan=17, sticky=NSEW)
        
        self.happeninglabelimage = Image.open(r"Assets\eventviewpage\whatshappening.png")
        self.happeninglabelimage = ImageTk.PhotoImage(self.happeninglabelimage.resize(
            (math.ceil(360 * dpi / 96), math.ceil(120 * dpi / 96)), Image.Resampling.LANCZOS))
        self.happeninglabel = Label(self, image=self.happeninglabelimage, width=1, height=1, bg=LIGHTPURPLE)
        self.happeninglabel.grid(row=0, column=21, columnspan=9,
                        rowspan=3, sticky=NSEW)

        self.eventdetailsimage = Image.open(r"Assets\eventviewpage\eventdetails.png")
        self.eventdetailsimage = ImageTk.PhotoImage(self.eventdetailsimage.resize(
            (math.ceil(480 * dpi / 96), math.ceil(360 * dpi / 96)), Image.Resampling.LANCZOS))
        self.eventdetails = Label(self, image=self.eventdetailsimage, width=1, height=1, relief="flat")
        self.eventdetails.grid(row=10, column=29, columnspan=12,
                        rowspan=9, sticky=NSEW)
        self.titleart = Image.open(r"Assets\eventviewpage\titleart.png")
        self.titleart = ImageTk.PhotoImage(self.titleart.resize(
            (math.ceil(320 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.titleartlabel = Label(self,
        text="", font=("Avenir Next Medium", 18), fg = "black",
        image=self.titleart, compound=CENTER, width=1, height=1, bg=LIGHTPURPLE, wraplength=300, justify=CENTER)
        self.titleartlabel.grid(row=11, column=31, columnspan=8, rowspan=2, sticky=NSEW)
        self.dateart = Image.open(r"Assets\eventviewpage\datepicture.png")
        self.dateart = ImageTk.PhotoImage(self.dateart.resize(
            (math.ceil(240 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.dateartlabel = Label(self,
        text="", font=("Avenir Next Medium", 18), fg = "black", relief="solid",
        image=self.dateart, compound=CENTER, width=1, height=1)
        self.dateartlabel.grid(row=14, column=33, columnspan=6, rowspan=2, sticky=NSEW)
        self.locationart = Image.open(r"Assets\eventviewpage\locationpicture.png")
        self.locationart = ImageTk.PhotoImage(self.locationart.resize(
            (math.ceil(240 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.locationartlabel = Label(self,
        text="", font=("Avenir Next Medium", 18), fg = "black", relief="solid",
        image=self.dateart, compound=CENTER, width=1, height=1)
        self.locationartlabel.grid(row=16, column=31, columnspan=6, rowspan=2, sticky=NSEW)

        self.leftarrowimage = Image.open(r"Assets\eventviewpage\Left Arrow.png")
        self.leftarrowimage = ImageTk.PhotoImage(self.leftarrowimage.resize(
            (math.ceil(120 * dpi / 96), math.ceil(120 * dpi / 96)), Image.Resampling.LANCZOS))
        self.leftarrowbutton = Button(self, image=self.leftarrowimage, width=1, height=1, relief="flat", cursor="hand2",
        command=lambda: self.previous_image())
        self.leftarrowbutton.grid(row=16, column=21, columnspan=3,
                                    rowspan=3, sticky= NSEW)
        self.rightarrowimage = Image.open(r"Assets\eventviewpage\Right Arrow.png")
        self.rightarrowimage = ImageTk.PhotoImage(self.rightarrowimage.resize(
            (math.ceil(120 * dpi / 96), math.ceil(120 * dpi / 96)), Image.Resampling.LANCZOS))
        self.rightarrowbutton = Button(self, image=self.rightarrowimage, width=1, height=1, relief="flat", cursor="hand2",
        command=lambda: self.next_image())
        self.rightarrowbutton.grid(row=16, column=25, columnspan=3,
                                rowspan=3, sticky=NSEW)
        
        self.eventsname = []
        self.after(100, self.updateevents)
        self.imageindex = 0

    def updateevents(self):
        # read all the events already created and store them in a list
        #clear the list
        self.eventsname.clear()
        self.conn = sqlite3.connect('interactivesystem.db')
        self.c = self.conn.cursor()
        #count events, if no events, then display no events
        self.c.execute("SELECT COUNT(event_name) FROM eventcreation")
        self.count = self.c.fetchone()[0]
        if self.count == 0:
            self.noeventsimage = Image.open(r"Assets\EventCreation\panelnoimage520x520.png")
            self.noeventsimage = ImageTk.PhotoImage(self.noeventsimage.resize(
                (math.ceil(680 * dpi / 96), math.ceil(680 * dpi / 96)), Image.Resampling.LANCZOS))
            self.showcaseimage.configure(image=self.noeventsimage)
            self.titleartlabel.configure(text="No Events")
            self.dateartlabel.configure(text="")
            self.locationartlabel.configure(text="")
            return
        with self.conn:
            self.c.execute("SELECT event_name FROM eventcreation")
            self.events = self.c.fetchall()
            for index, name in list(enumerate(self.events)):
                actualname = name[0]
                self.eventsname.append((index, actualname))
        # print(self.eventsname)
        self.read_blob(self.eventsname[0][1])
        self.titleartlabel.config(text=self.eventsname[0][1])
        self.update_location(self.eventsname[0][1])
        self.update_date(self.eventsname[0][1])
    def update_location(self, event):
        self.conn = sqlite3.connect('interactivesystem.db')
        self.c = self.conn.cursor()
        with self.conn:
            self.c.execute("SELECT venue_name FROM eventcreation WHERE event_name = ?", (event,))
            self.location = self.c.fetchone()
            self.locationartlabel.config(text=self.location[0])
    def update_date(self, event):
        self.conn = sqlite3.connect('interactivesystem.db')
        self.c = self.conn.cursor()
        with self.conn:
            self.c.execute("SELECT event_startdate FROM eventcreation WHERE event_name = ?", (event,))
            self.date = self.c.fetchone()
            self.dateartlabel.config(text=self.date[0])
    def read_blob(self, eventname):
        self.conn = sqlite3.connect('interactivesystem.db')
        self.c = self.conn.cursor()
        with self.conn:
            self.c.execute("SELECT event_image FROM eventcreation WHERE event_name = ?", (eventname,))
            self.blobData = io.BytesIO(self.c.fetchone()[0])
            self.img = Image.open(self.blobData)
            self.img = ImageTk.PhotoImage(self.img.resize(
                (math.ceil(680 * dpi / 96), math.ceil(680 * dpi / 96)), Image.Resampling.LANCZOS))
            self.showcaseimage.configure(image=self.img)

    def previous_image(self):
        # simultaneously add event if events are created not already in the list
        # clicking the left arrow button will show the last image at final index
        self.updateevents()
        if self.imageindex > 0:
            self.imageindex -= 1
        elif self.imageindex == 0:
            self.imageindex += len(self.eventsname) - 1
        if self.count != 0:
            self.read_blob(self.eventsname[self.imageindex][1])
            self.titleartlabel.config(text=self.eventsname[self.imageindex][1])
            self.update_location(self.eventsname[self.imageindex][1])
            self.update_date(self.eventsname[self.imageindex][1])
        # print(self.imageindex)

    def next_image(self):
        #this function is to change the image to the next image in the list
        # if imageindex already at the final index, it will jump to the first image by setting imageindex to 0
        self.updateevents()
        if self.imageindex < len(self.eventsname) - 1:
            self.imageindex += 1
        elif self.imageindex == len(self.eventsname) - 1:
            self.imageindex = 0
        if self.count != 0:
            self.read_blob(self.eventsname[self.imageindex][1])
            self.titleartlabel.config(text=self.eventsname[self.imageindex][1])
            self.update_location(self.eventsname[self.imageindex][1])
            self.update_date(self.eventsname[self.imageindex][1])

    def changetoeventregistration(self, index):
        # print(self.titleartlabel.cget("text"))
        if  self.titleartlabel.cget("text") != "No Events":
            self.controller.show_frame(EventRegistration)
            self.controller.togglebuttonrelief(self.controller.eventregistrationbutton)

            eventregistrationreference = self.controller.get_page(EventRegistration)
            eventregistrationreference.eventdropdown.event_generate("<<ComboboxSelected>>")

            eventregistrationreference.eventdropdown.event_generate("<FocusIn>")
            eventregistrationreference.eventdropdown.event_generate("<FocusOut>")
            try:
                eventregistrationreference.eventdropdown.current(index+1)
            except: #the index of the latest event has already been deleted, because of this, the index of the dropdown has to be -1 to not raise an error
                messagebox.showinfo("Error", "The event you were trying to register for has been deleted. Defaulting to the previous event.")
                eventregistrationreference.eventdropdown.current(index)
            eventregistrationreference.eventdropdown.event_generate("<<ComboboxSelected>>")
        else:
            messagebox.showinfo("Error", "You have selected an invalid event")
            print(index)
    #loading event from main page function #will receive either last index or second last index
    def load_event(self, index):
        self.imageindex = index
        self.updateevents()
        self.read_blob(self.eventsname[self.imageindex][1])
        self.titleartlabel.config(text=self.eventsname[self.imageindex][1])
        self.update_location(self.eventsname[self.imageindex][1])
        self.update_date(self.eventsname[self.imageindex][1])

    


            






class EventRegistration(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=PINK)
        FONTNAME = "Avenir Next"
        self.controller = controller
        for x in range(42):
            self.columnconfigure(x, weight=1, uniform='x')
            Label(self, width=1, bg=PINK).grid(
                row=0, column=x, rowspan=1, columnspan=1, sticky=NSEW)
        for y in range(21):
            self.rowconfigure(y, weight=1, uniform='x')
            Label(self, width=1, bg=PINK).grid(
                row=y, column=0, rowspan=1, columnspan=1, sticky=NSEW)
        self.bgwallpaper = Image.open(r"Assets\EventRegistration\wallpaperflare.jpg")
        self.bgwall = ImageTk.PhotoImage(self.bgwallpaper.resize(
             (math.ceil(1680 * dpi / 96), math.ceil(840 * dpi / 96)), Image.Resampling.LANCZOS))
        self.bgwalllabel = Label(self, image=self.bgwall, width=1, height=1, bg=LIGHTPURPLE)
        self.bgwalllabel.grid(row=0, column=0, rowspan=21, columnspan=43, sticky=N+S+E+W)
        self.bgwalllabel.grid_propagate(0)
        #enabling foreign keys in sqlite3
        self.conn = sqlite3.connect('interactivesystem.db')
        self.c = self.conn.cursor()
        with self.conn:
            self.c.execute("PRAGMA foreign_keys = ON")
        #checking if foreign keys are enabled
        self.c.execute("PRAGMA foreign_keys")
        self.foreignkey = self.c.fetchone()
        # Connect to database
        conn = sqlite3.connect('interactivesystem.db')
        # Create cursor
        #i've gone ahead and given u the liberty to create the eventcreation table here)
        with conn:
            self.c.execute("""CREATE TABLE IF NOT EXISTS eventcreation (
                eventkey_number TEXT PRIMARY KEY NOT NULL, 
                event_name TEXT NOT NULL,
                event_description TEXT NOT NULL,
                event_startdate TEXT NOT NULL,
                event_enddate TEXT NOT NULL,
                event_starttime TEXT NOT NULL,
                event_endtime TEXT NOT NULL,
                event_organizer TEXT NOT NULL,
                venue_name TEXT,
                host_name TEXT NOT NULL,
                event_image BLOB NULL
                )""")
        c = conn.cursor()
        # Create a table
        #drop table
        # c.execute("""DROP TABLE IF EXISTS eventregistration""")
        c.execute("""CREATE TABLE IF NOT EXISTS eventregistration (
            full_name text NOT NULL,
            icpass_number text NOT NULL, 
            phone_number text,
            email text NOT NULL,
            address text,
            event_registered text NOT NULL,
            eventkey_registered TEXT NOT NULL,
            FOREIGN KEY (eventkey_registered) REFERENCES eventcreation(eventkey_number) ON DELETE CASCADE
            )""")
        # Send entries to database
    
        def submit():
            full_nametext = fullnamefield.get()
            icpass_number = icnumberfield.get()
            phone_number = phonenumentry.get()
            email_registered = emailentry.get().strip()
            address = addressentry.get()
            event_registered = self.eventdropdown.get()
            #check if the event has already been registered by email 
            #why is this code allowing muplitple entries of the same email in 1 event
            # c.execute("SELECT email FROM eventregistration WHERE event_registered = ?", (event_registered,))
            # emailcheck = c.fetchall()
            # for email in emailcheck:
            #     if email == emailentry.get():
            #         messagebox.showerror("Error", "You have already registered for this event")
            c.execute("SELECT email, full_name, icpass_number, phone_number FROM eventregistration WHERE event_registered = ?", (event_registered,))
            emailcheck = c.fetchall()
            for emailnum in range(len(emailcheck)):
                if email_registered == emailcheck[emailnum][0]:
                    messagebox.showerror("Error", f"You have already registered for this event using the email {emailcheck[emailnum][0]}, name {emailcheck[emailnum][1]},\n ic/passport number {emailcheck[emailnum][2]} and phone number {emailcheck[emailnum][3]}")
                    return
            c.execute("SELECT eventkey_number FROM eventcreation WHERE event_name = ?", (self.eventdropdown.get(),))
            eventkey_registered = c.fetchone()[0]
            information = (full_nametext, icpass_number,
                           phone_number, email_registered, address, event_registered, eventkey_registered)
            try:
                if full_nametext == "" or icpass_number == "" or phone_number == "" or email_registered == "" or address == "":
                    messagebox.showerror(
                        "Error", "Please fill in all the fields")
                else:
                    with conn:
                        c.execute(
                            "INSERT INTO eventregistration VALUES (?,?,?,?,?,?,?)", information)
                        messagebox.showinfo(
                            "Success", "Registration Successful!")
                        fullnamefield.delete(0, END)
                        icnumberfield.delete(0, END)
                        phonenumentry.delete(0, END)
                        emailentry.delete(0, END)
                        addressentry.delete(0, END)
                        controller.show_frame(EventView)
                        controller.togglebuttonrelief(controller.eventlistbutton)
            except sqlite3.IntegrityError:
                messagebox.showerror("Error", "Email already registered")
        def defocus(event):
            if self.eventdropdown.get() == "Please select an event":
                confirmbutton.config(state=DISABLED, disabledforeground=OTHERPINK)
            else:
                confirmbutton.config(state=NORMAL)
            event.widget.master.focus_set()
            refresh()
        def focusout(event):
            event.widget.master.focus_set()
            #disable the confirm button if option selected is not an event, basically the first option
            if self.eventdropdown.get() == "Please select an event":
                confirmbutton.config(state=DISABLED, disabledforeground=OTHERPINK)
                try:
                    self.panel.config(image=self.panelnoimageimg)
                    for label in self.eventlabelist:
                        label.config(text="")
                    self.eventdescriptionlabel.config(text="Select an event from the menu!")
                except AttributeError:
                    pass
                return
            else:
                confirmbutton.config(state=NORMAL)
                self.read_blob(self.eventdropdown.get())
                self.gettheeventdetails(self.eventdropdown.get())
            refresh()
        # Widgets

        #dropdown for events
        conn = sqlite3.connect('interactivesystem.db')
        # Create cursor
        c = conn.cursor()
        self.event_list = ["Select an event"]
        self.current_eventkey = ""
        #refresh the self.event_list everytime the combobox is selected
        def refresh():
            c.execute("SELECT event_name, eventkey_number FROM eventcreation")
            self.event_list.clear()
            self.event_list.append("Please select an event")
            for row in c.fetchall():
                event_name = row[0]
                eventkey_number = row[1] 
                information = (event_name, eventkey_number)
                self.event_list.append(information[0])
                self.current_eventkey = information[1]
            


            self.eventdropdown['values'] = self.event_list
        with conn:
            c.execute("""SELECT event_name, eventkey_number FROM eventcreation""")
            self.event_list.clear()
            self.event_list.append("Please select an event")
            for eventname in c.fetchall():
                event_name = eventname[0]
                eventkey = eventname[1]
                information = (event_name, eventkey)
                self.event_list.append(information[0])
                
        self.eventdropdown = ttk.Combobox(
            self, values=self.event_list, width=1, state='readonly')
        self.eventdropdown.current(0)
        self.eventdropdown.grid(row=1, column=3, columnspan=16,
                           rowspan=2, sticky=NSEW)
        self.eventdropdown.bind('<FocusIn>', defocus)
        self.eventdropdown.bind('<<ComboboxSelected>>', focusout)
        self.eventdropdown.grid_propagate(False)
        
        separator = ttk.Separator(self, orient=HORIZONTAL)
        separator.grid(row=3, column=3, columnspan=16, pady=5, sticky=EW)
        icpasslabel = Label(self, text="IC No.",
                            font=(FONTNAME, 10), bg='#FFF5E4', width=1, height=1)
        icpasslabel.grid(row=7, column=3, columnspan=2,
                         rowspan=2, sticky=NSEW)
        icpasslabel.grid_propagate(False)
        phonenumberlabel = Label(
            self, text="Phone\nNo", font=(FONTNAME, 10), bg='#FFF5E4', width=1, height=1)
        phonenumberlabel.grid(
            row=10, column=3, columnspan=2, rowspan=2, sticky=NSEW)
        emaillabel = Label(self, text="Email", font=(
            FONTNAME, 14), bg='#FFF5E4', width=1, height=1)
        emaillabel.grid(row=13, column=3, columnspan=2,
                        rowspan=2, sticky=NSEW)
        emaillabel.grid_propagate(False)
        addresslabel = Label(self, text="Address",
                             font=(FONTNAME, 10), bg='#FFF5E4', width=1, height=1)
        addresslabel.grid(row=16, column=3, columnspan=2,
                          rowspan=2, sticky=NSEW)
        addresslabel.grid_propagate(False)

        # radio_1 = ttk.Radiobutton(self, text="Male  ", variable=var, value=0)
        # radio_1.grid(row=9, column=5,rowspan=2, columnspan=2, pady=(0, 10), sticky=NS)

        # radio_1 = ttk.Radiobutton(self, text="Female", variable=var, value=1, command=lambda:print(var.get()))
        # radio_1.grid(row=9, column=7,rowspan=2, columnspan=2, pady=(0, 10), sticky=NS)

        fullnamefield = Entry(self, width=1, bg='#FFFFFF',
                              font=(FONTNAME, 18), justify='center')
        fullnamefield.grid(row=4, column=3, columnspan=16,
                           rowspan=2, sticky=NSEW)
        fullnamefield.insert(0, "Full Name")
        #code to delete the default text when the user clicks on the entry
        def on_entry_click(event):
            if fullnamefield.get() == 'Full Name':
                fullnamefield.delete(0, "end")
                fullnamefield.insert(0, '')
                fullnamefield.config(fg='black')
        def on_focusout(event):
            if fullnamefield.get() == '':
                fullnamefield.insert(0, 'Full Name')
                fullnamefield.config(fg='grey')
        fullnamefield.bind('<FocusIn>', on_entry_click)
        fullnamefield.bind('<FocusOut>', on_focusout)
        fullnamefield.grid_propagate(False)


        icnumberfield = Entry(self, width=1, bg='#FFFFFF',
                              font=(FONTNAME, 18), justify='center')
        icnumberfield.grid(row=7, column=5, columnspan=14,
                           rowspan=2, sticky=NSEW)
        phonenumentry = Entry(self, width=1, bg='#FFFFFF',
                              font=(FONTNAME, 18), justify='center')
        phonenumentry.grid(row=10, column=5, columnspan=14,
                           rowspan=2, sticky=NSEW)
        emailentry = Entry(self, width=1, bg='#FFFFFF',
                           font=(FONTNAME, 18), justify='center')
        emailentry.grid(row=13, column=5, columnspan=14,
                        rowspan=2, sticky=NSEW)
        addressentry = Entry(self, width=1, bg='#FFFFFF',
                             font=(FONTNAME, 18), justify='center')
        addressentry.grid(row=16, column=5, columnspan=14,
                          rowspan=2, sticky=NSEW)
        # Buttons
        cancelbutton = Button(self, text="Cancel", font=(FONTNAME, 18), bg='White', command=lambda: [
        controller.show_frame(EventView),
        controller.togglebuttonrelief(controller.eventlistbutton)], width=1,height=1)
        cancelbutton.grid(row=18, column=3, columnspan=6,
                          rowspan=2, sticky=NSEW)
        confirmbutton = Button(self, text="Confirm", font=(
            FONTNAME, 18), bg='White', command=lambda: submit(), width=1,height=1, cursor='hand2', state=DISABLED,disabledforeground=OTHERPINK)
        confirmbutton.grid(row=18, column=13, columnspan=6,
                           rowspan=2, sticky=NSEW)
        self.panelnoimageimg = Image.open(r"Assets\EventCreation\panelnoimage520x520.png")
        self.panelnoimageimg = self.panelnoimageimg.resize((520, 520), Image.Resampling.LANCZOS)
        self.panelnoimageimg = ImageTk.PhotoImage(self.panelnoimageimg)
        self.panel = Label(self, image=self.panelnoimageimg,width=1,height=1, bg=ORANGE)
        self.panel.grid(row=1, column=28, columnspan=13,
                    rowspan=13, sticky=NSEW)
        self.panel.grid_propagate(False)

        # event details labels
        self.eventorganizerlabel = Label(self, text="", font=(FONTNAME, 14), bg=WHITE, fg=BLACK, width=1, height=1)
        self.eventorganizerlabel.grid(row=5, column=22, columnspan=5, rowspan=1, sticky=NSEW)
        self.datelabel = Label(self, text="", font=(FONTNAME, 12), bg=WHITE, fg=BLACK, width=1, height=1)
        self.datelabel.grid(row=8, column=22, columnspan=5, rowspan=1, sticky=NSEW)
        self.timelabel = Label(self, text="", font=(FONTNAME, 14), bg=WHITE, fg=BLACK, width=1, height=1)
        self.timelabel.grid(row=11, column=22, columnspan=5, rowspan=1, sticky=NSEW)
        self.venuelabel = Label(self, text="", font=(FONTNAME, 14), bg=WHITE, fg=BLACK, width=1, height=1)
        self.venuelabel.grid(row=14, column=22, columnspan=5, rowspan=1, sticky=NSEW)
        self.eventhostlabel = Label(self, text="", font=(FONTNAME, 14), bg=WHITE, fg=BLACK, width=1, height=1)
        self.eventhostlabel.grid(row=17, column=22, columnspan=5, rowspan=1, sticky=NSEW)
        self.eventdescriptionlabel = Label(self, text="Select an event from the menu!", font=(FONTNAME, 14), bg=WHITE, fg=BLACK, width=1, height=1)
        self.eventdescriptionlabel.grid(row=15, column=31, columnspan=9, rowspan=4, sticky=NSEW)
        #even label list for easy access
        self.eventlabelist = [self.eventorganizerlabel, self.datelabel, self.timelabel, self.venuelabel, self.eventhostlabel, self.eventdescriptionlabel]

    def gettheeventdetails(self, eventname): #eventname is eventdropdown.get()
        self.conn = sqlite3.connect('interactivesystem.db')
        self.c = self.conn.cursor()
        if self.c.execute("SELECT * FROM eventcreation WHERE event_name = ?", (eventname,)).fetchone() is None:
            self.eventorganizerlabel.config(text="Event Organizer")
            self.datelabel.config(text="", wraplength=100, justify=CENTER)
            self.timelabel.config(text="", wraplength=100, justify=CENTER)
            self.venuelabel.config(text="", wraplength=100, justify=CENTER)
            self.eventhostlabel.config(text="", wraplength=100, justify=CENTER)
            self.eventdescriptionlabel.config(text="", wraplength=100, justify=CENTER)
            self.conn.commit()
            self.conn.close()
        with self.conn:
            self.c.execute("SELECT * FROM eventcreation WHERE event_name = ?", (eventname,))
            self.eventdetails = self.c.fetchall()
            event_description = self.eventdetails[0][2]
            event_date = self.eventdetails[0][3] + " - " + self.eventdetails[0][4]
            event_time = self.eventdetails[0][5] + " - " + self.eventdetails[0][6]
            event_organizer = self.eventdetails[0][7]
            event_venue = self.eventdetails[0][8]
            event_host = self.eventdetails[0][9]
        #configuring the labels
        self.eventorganizerlabel.config(text=event_organizer)
        self.datelabel.config(text=event_date)
        self.timelabel.config(text=event_time)
        self.venuelabel.config(text=event_venue)
        self.eventhostlabel.config(text=event_host)
        self.eventdescriptionlabel.config(text=event_description)
        #setting the wraplength of the labels
        self.eventorganizerlabel.config(wraplength=300, justify=CENTER)
        self.datelabel.config(wraplength=300, justify=CENTER)
        self.timelabel.config(wraplength=300, justify=CENTER)
        self.venuelabel.config(wraplength=300, justify=CENTER)
        self.eventhostlabel.config(wraplength=300, justify=CENTER)
        self.eventdescriptionlabel.config(wraplength=300, justify=CENTER)

        self.conn.close()


    def read_blob(self, event_name):
        self.conn = sqlite3.connect('interactivesystem.db')
        self.c = self.conn.cursor()
        with self.conn:
            self.c.execute("SELECT event_image FROM eventcreation WHERE event_name=?", (event_name,))
            self.blobData = io.BytesIO(self.c.fetchone()[0])
            self.img = Image.open(self.blobData)
            self.img = ImageTk.PhotoImage(self.img.resize(
                 (math.ceil(520 * dpi / 96), math.ceil(520 * dpi / 96)), Image.Resampling.LANCZOS))
            self.panel.config(image=self.img)
            self.panel.grid_propagate(False)

    def refresh(self):
        self.conn = sqlite3.connect('interactivesystem.db')
        self.c = self.conn.cursor()
        self.c.execute("SELECT event_name, eventkey_number FROM eventcreation")
        self.event_list.clear()
        self.event_list.append("Please select an event")
        for row in self.c.fetchall():
            event_name = row[0]
            eventkey_number = row[1] 
            information = (event_name, eventkey_number)
            self.event_list.append(information[0])
            self.current_eventkey = information[1]
        self.eventdropdown['values'] = self.event_list

class EventCreation(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=PINK)
        FONTNAME = "Avenir Next Medium"
        self.controller = controller
        for x in range(42):
            self.columnconfigure(x, weight=1, uniform='x')
            Label(self, width=1, bg=PINK, relief=SOLID).grid(
                row=0, column=x, rowspan=1, columnspan=1, sticky=NSEW)
        for y in range(21):
            self.rowconfigure(y, weight=1, uniform='x')
            Label(self, width=1, bg=PINK, relief=SOLID).grid(
                row=y, column=0, rowspan=1, columnspan=1, sticky=NSEW)
        # Connect to database
        self.conn = sqlite3.connect('interactivesystem.db')
        # Create cursor
        self.c = self.conn.cursor()
        # Create a table
        # self.c.execute("""DROP TABLE eventcreation""")
        self.c.execute("""CREATE TABLE IF NOT EXISTS eventcreation (
            eventkey_number TEXT PRIMARY KEY NOT NULL, 
            event_name TEXT NOT NULL,
            event_description TEXT NOT NULL,
            event_startdate TEXT NOT NULL,
            event_enddate TEXT NOT NULL,
            event_starttime TEXT NOT NULL,
            event_endtime TEXT NOT NULL,
            event_organizer TEXT NOT NULL,
            venue_name TEXT,
            host_name TEXT NOT NULL,
            event_image BLOB NULL
            )""")

        self.bgimageoriginal = Image.open(r"Assets\EventCreation\eventcreationbg.png")
        self.bgimage = ImageTk.PhotoImage(self.bgimageoriginal.resize(
             (math.ceil(1680 * dpi / 96), math.ceil(840 * dpi / 96)), Image.Resampling.LANCZOS))
        self.bgimagelabel = Label(self, image=self.bgimage, width=1, height=1, bg=LIGHTPURPLE)
        self.bgimagelabel.grid(row=0, column=0, rowspan=21, columnspan=42, sticky=N+S+E+W)
        self.bgimagelabel.grid_propagate(False)

        self.eventnamefield = Entry(self, width=1, bg='#FFFFFF',
                              font=(FONTNAME, 18), justify='center')
        self.eventnamefield.grid(row=5, column=5, columnspan=11,
                           rowspan=2, sticky=NSEW)
        self.eventnamefield.insert(0, "Event Name")

        self.eventdescription = Text(self, width=1, height=1, bg='#FFFFFF',
                               font=(FONTNAME, 18), wrap=WORD)
        self.eventdescription.grid(row=8, column=3, columnspan=13,
                                rowspan=3, sticky=NSEW)
        self.eventdescription.insert("1.0", "Event Description", 'center')

        self.organizerfield = Entry(self, width=1, bg='#FFFFFF',
                                font=(FONTNAME, 18), justify='center')
        self.organizerfield.grid(row=14, column=7, columnspan=6,
                                    rowspan=1, sticky=NSEW)
        self.organizerfield.insert(0, "Organizing School")
        self.organizerfield.grid_propagate(False)
        self.hostnameentry = Entry(self, width=1, bg='#FFFFFF',
                           font=(FONTNAME, 18), justify='center')
        self.hostnameentry.grid(row=17, column=7, columnspan=6,
                        rowspan=1, sticky=NSEW)
        self.hostnameentry.insert(0, "Host Name")
        self.venuenameentry = Entry(self, width=1, bg='#FFFFFF',
                              font=(FONTNAME, 18), justify='center')
        self.venuenameentry.grid(row=5, column=28, columnspan=8,
                           rowspan=2, sticky=NSEW)
        self.venuenameentry.grid_propagate(False)
        self.venuenameentry.insert(0, "Venue Name")
        self.eventkeyfield = Entry(self, width=1, bg='#FFFFFF', fg= "red",
                              font=(FONTNAME, 18), justify='center')
        self.eventkeyfield.grid(row=19, column=20, columnspan=7,
                           rowspan=1, sticky=NSEW)
        self.eventkeyfield.grid_propagate(False)
        self.eventkeyfield.insert(0, "Event Key")

        self.filename = ""
 
        self.date_entrywidget = DateEntry(self, height=1, width=1, background=NAVYBLUE, 
        headersbackground = ORANGE,
        font=("Avenir Next Medium",16), justify='center',
        date_pattern='dd/mm/yyyy') 
        self.date_entrywidget.grid(row=10, column=20, columnspan=8,
                            rowspan=2, sticky=NSEW)

        #end date
        self.date_entrywidget2 = DateEntry(self, height=1, width=1, background=NAVYBLUE,
        headersbackground = ORANGE,
        font=("Avenir Next Medium",16), justify='center',
        date_pattern='dd/mm/yyyy')
        self.date_entrywidget2.grid(row=10, column=32, columnspan=8,
                            rowspan=2, sticky=NSEW)

        self.hourentry = Entry(self, width=1, bg='#FFFFFF',
                                font=(FONTNAME, 18), justify='center')
        self.hourentry.grid(row=14, column=20, columnspan=2,
                            rowspan=2, sticky=NSEW)
        self.hourentry.insert(0, "HH")
        self.minentry = Entry(self, width=1, bg='#FFFFFF',
                                font=(FONTNAME, 18), justify='center')
        self.minentry.grid(row=14, column=23, columnspan=2,
                            rowspan=2, sticky=NSEW)
        self.minentry.insert(0, "MM")
        #Am pm menu 
        self.ampmchoices = ["AM", "PM"]
        self.am_pmcombobox =  ttk.Combobox(self, width=1, font=(FONTNAME, 18), justify='center')
        self.am_pmcombobox['values'] = self.ampmchoices
        self.am_pmcombobox.grid(row=14, column=25, columnspan=3,
                            rowspan=2, sticky=NSEW)
        self.am_pmcombobox.current(0)
        
        self.endhourentry = Entry(self, width=1, bg='#FFFFFF',
                                font=(FONTNAME, 18), justify='center')
        self.endhourentry.grid(row=14, column=32, columnspan=2,
                            rowspan=2, sticky=NSEW)
        self.endhourentry.insert(0, "HH")
        self.endminentry = Entry(self, width=1, bg='#FFFFFF',
                                font=(FONTNAME, 18), justify='center')
        self.endminentry.grid(row=14, column=35, columnspan=2,
                            rowspan=2, sticky=NSEW)
        self.endminentry.insert(0, "MM")
        #Am pm combobox
        self.endampmchoices = ["AM", "PM"]
        self.endam_pmcombobox = ttk.Combobox(self, width=1, font=(FONTNAME, 18), justify='center')
        self.endam_pmcombobox['values'] = self.endampmchoices
        self.endam_pmcombobox.grid(row=14, column=37, columnspan=3,
                            rowspan=2, sticky=NSEW)
        self.endam_pmcombobox.current(0)
        self.cancelbutton = Button(self, text="Cancel", width=1,height=1,
        font=(FONTNAME, 18), bg='White', 
        command=lambda: [
        controller.show_frame(ManagementSuite),
        controller.togglebuttonrelief(controller.viewparticipantsbutton)])
        self.cancelbutton.grid(row=18, column=29, columnspan=5,
                          rowspan=2, sticky=NSEW)

        confirmbutton = Button(self, text="Continue\nto Insert Image", width=1,height=1,
        font=(FONTNAME, 18), bg='White', command=lambda: self.initializeuploadframe())
        confirmbutton.grid(row=18, column=36, columnspan=5,
                           rowspan=2, sticky=NSEW)

        self.date_entrywidget.set_date(datetime.date.today())
        self.date_entrywidget2.set_date(datetime.date.today())
        #~~~~~ Frame for Uploading Images~~~~~~
        self.uploadframe = Frame(self, width=1, height=1, bg=ORANGE)
        self.uploadframe.grid(row=0, column=0, columnspan=42, rowspan=21, sticky=NSEW)
        for i in range(42):
            self.uploadframe.grid_columnconfigure(i, weight=1)
            Label(self.uploadframe, width=1, bg=ORANGE).grid(row=0, column=i, sticky=NSEW)
        for j in range(21):
            self.uploadframe.grid_rowconfigure(j, weight=1)
            Label(self.uploadframe, width=1, bg=ORANGE).grid(row=j, column=0, sticky=NSEW)
        self.uploadframe.grid_propagate(False)
        self.uploadframe.grid_remove()

    def initializeuploadframe(self):
        self.uploadframe.grid()
        self.uploadframebg = Image.open(r"Assets\EventCreation\uploadimagebg.png")
        self.uploadbgimg = ImageTk.PhotoImage(self.uploadframebg.resize(
            (math.ceil(1680 * dpi / 96), math.ceil(840 * dpi / 96)), Image.Resampling.LANCZOS))
        self.uploadbgimglabel = Label(self.uploadframe, image=self.uploadbgimg, width=1, height=1, bg=ORANGE)
        self.uploadbgimglabel.grid(row=0, column=0, columnspan=42, rowspan=21, sticky=NSEW)
        self.uploadframe.tkraise()
        self.uploadimgbtnimg = Image.open(r"Assets\EventCreation\uploadimgbtn280x80.png")
        self.uploadimgbtnimg = ImageTk.PhotoImage(self.uploadimgbtnimg.resize(
            (math.ceil(280 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.uploadimgbtn = Button(self.uploadframe, image=self.uploadimgbtnimg, width=1, height=1,
        bg=ORANGE, command=lambda: self.upload_image())
        #~~~ Placeholder image~~~
        self.placeholderimg = Image.open(r"Assets\EventCreation\placeholderimage.png")

        self.uploadimgbtn.grid(row=6, column=18, columnspan=7, rowspan=2, sticky=NSEW)
        self.clearimgbtnimg = Image.open(r"Assets\EventCreation\clearimgbtn280x80.png")
        self.clearimgbtnimg = ImageTk.PhotoImage(self.clearimgbtnimg.resize(
            (math.ceil(280 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))   
        self.clearimgbtn = Button(self.uploadframe, image=self.clearimgbtnimg, width=1, height=1,
        bg=ORANGE, command=lambda: self.clear_image())
        self.clearimgbtn.grid(row=9, column=18, columnspan=7, rowspan=2, sticky=NSEW)
        self.prompttextwidget = Text(self.uploadframe, width=1, height=1, bg=LIGHTYELLOW, font=("Avenir Next", 14), wrap=WORD)
        self.prompttextwidget.grid(row=12, column=18, columnspan=7, rowspan=2, sticky=NSEW)
        self.prompttextwidget.insert(1.0,"Let's enter a prompt")
        self.generateanimagebtnimg = Image.open(r"Assets\EventCreation\generateanimage280x80.png")
        self.generateanimagebtnimg = ImageTk.PhotoImage(self.generateanimagebtnimg.resize(
            (math.ceil(280 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.generateanimagebtn = Button(self.uploadframe, image=self.generateanimagebtnimg, width=1, height=1,
        bg=LIGHTYELLOW, command=lambda: self.generate_image(self.prompttextwidget.get("1.0", END)))
        self.generateanimagebtn.grid(row=14, column=18, columnspan=7, rowspan=2, sticky=NSEW)
        self.panelimgnoimg = Image.open(r"Assets\EventCreation\panelnoimage520x520.png")
        self.panelimgnoimg = ImageTk.PhotoImage(self.panelimgnoimg.resize(
            (math.ceil(520 * dpi / 96), math.ceil(520 * dpi / 96)), Image.Resampling.LANCZOS))
        self.panel = Label(self.uploadframe, image=self.panelimgnoimg, width=1, height=1, bg=ORANGE)
        self.panel.grid(row=5, column=4, columnspan=13, rowspan=13, sticky=NSEW)
        self.canceluploadbtnimg = Image.open(r"Assets\EventCreation\returntoeventdetailsbtn280x80.png")
        self.canceluploadbtnimg = ImageTk.PhotoImage(self.canceluploadbtnimg.resize(
            (math.ceil(280 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.canceluploadbtn = Button(self.uploadframe, image=self.canceluploadbtnimg, width=1, height=1,
        bg=ORANGE, command=lambda: self.cancelupload())
        self.canceluploadbtn.grid(row=17, column=25, columnspan=7, rowspan=2, sticky=NSEW)
        self.confirmandsubmitbtnimg = Image.open(r"Assets\EventCreation\confdetailsandpostevent280x80.png")
        self.confirmandsubmitbtnimg = ImageTk.PhotoImage(self.confirmandsubmitbtnimg.resize(
            (math.ceil(280 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.confirmandsubmitbtn = Button(self.uploadframe, image=self.confirmandsubmitbtnimg, width=1, height=1,
        bg=ORANGE, command=lambda: self.insert_blob())
        self.confirmandsubmitbtn.grid(row=17, column=33, columnspan=7, rowspan=2, sticky=NSEW)

        # the function is right here bro
    def generate_image(self, inputprompt):
        try:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if inputprompt.strip() == "":
                messagebox.showerror("Error", "Please enter a prompt")
                return
            answer = messagebox.askyesno("Confirmation", f"Are you sure you want to generate an image using the prompt you have entered?\nPrompt: {inputprompt.strip()}\nThe system will become unresponsive for a few seconds.")
            # answer = messagebox.askyesno("Confirmation", f"Are you sure you want to generate an image using the prompt:\n{inputprompt}?\nThe system will become unresponsive for a few seconds.")
            # messagebox.showinfo("Generating Image", "Please wait while we generate your image\n The system will become unresponsive for a few seconds")
            if answer:
                image = (openai.Image.create(
                    prompt = inputprompt,
                    n = 1,
                    size = "512x512"
                    ))
                url = image["data"][0]["url"]
                #saves the file in this directory Assets\imagesgenerated by default        
                #saves as distinct file name
                # remove all spaces in the inputprompt string
                inputprompt = inputprompt.strip().replace(" ", "")
                actualfile = "Assets\imagesgenerated\imageof" + inputprompt + ".png"
                urllib.request.urlretrieve(url, actualfile)
                self.generatedimage = Image.open(actualfile)
                self.generatedimage = ImageTk.PhotoImage(self.generatedimage.resize(
                    (math.ceil(520 * dpi / 96), math.ceil(520 * dpi / 96)), Image.Resampling.LANCZOS))
                self.panel.configure(image=self.generatedimage)
                self.filename = actualfile
                messagebox.showinfo("Image Generated", "Your image has been generated, thanks for waiting!")
        except Exception as e:
            messagebox.showerror("Error", f"The system wasn't able to detect an OpenAPI key on your system.\nPlease ensure that you have set up your OpenAPI key as an environment variable. Error: {e}")


    def cancelupload(self):
        self.uploadframe.grid_remove()
        self.uploadframe.grid_propagate(False)
        self.controller.show_frame(EventCreation)
    
    def upload_image(self):
        try:
            self.filename = filedialog.askopenfilename(initialdir="Assets", 
                                                    title="Select A File", 
                                                    filetypes=(("all files", "*.*"), ("png files", "*.png"),("jpeg files", "*.jpg")))
            #This is the file we need to make as a blob
            self.img = Image.open(self.filename)
            self.img = ImageTk.PhotoImage(self.img.resize
            ((math.ceil(520 * dpi / 96), math.ceil(520 * dpi / 96)), Image.Resampling.LANCZOS))
            # Presents the images for future editing purposes or to just submit right away
            #store self.filename in the global name space
            self.panel.configure(image=self.img)
        except AttributeError:
            messagebox.showinfo("Error", "No image selected")
            self.panel.configure(image=self.panelimgnoimg)

    def convert_to_binary_data(self, image):
        # Convert digital data to binary format
        try:
            with open(image, 'rb') as file:
                blobData = file.read()
        except FileNotFoundError:
            with open(r"Assets\EventCreation\placeholderimage.png", 'rb') as file:
                blobData = file.read()
        return blobData
    def insert_blob(self):
            eventkey_number = self.eventkeyfield.get()
            event_nametext = self.eventnamefield.get()
            event_descriptiontext = self.eventdescription.get( "1.0", "end-1c" )
            event_startdate = self.date_entrywidget.get_date()
            event_enddate = self.date_entrywidget2.get_date()
            event_starttime = self.hourentry.get() + ":" + self.minentry.get() + " " + self.am_pmcombobox.get()
            event_endtime = self.endhourentry.get() + ":" + self.endminentry.get() + " " + self.endam_pmcombobox.get()
            event_organizer = self.organizerfield.get()
            venue_name = self.venuenameentry.get()
            hostname = self.hostnameentry.get()
            #checking if filename is empty
            if self.filename != self.panelimgnoimg:
                self.blobData = self.convert_to_binary_data(self.filename)
            elif self.filename == self.placeholderimg:
                self.blobData = self.convert_to_binary_data(self.placeholderimg)
            information = (eventkey_number, event_nametext, event_descriptiontext, event_startdate, event_enddate, event_starttime, event_endtime, event_organizer, venue_name, hostname, self.blobData)
            # Insert BLOB into table
            self.conn = sqlite3.connect('interactivesystem.db')
            self.c = self.conn.cursor()
            if "" not in information:
                try:
                    with self.conn:
                        #alter the table to add event description, event key, event date, event start time, 
                        # event end time, event organizer, venue name, host name
                        self.c.execute("""INSERT INTO eventcreation
                        (eventkey_number, event_name, event_description, event_startdate, event_enddate, event_starttime, event_endtime, event_organizer, venue_name, host_name, event_image) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (information))
                        messagebox.showinfo("Success", "Event Created")
                        eventregistrationreference = self.controller.get_page(EventRegistration)
                except sqlite3.IntegrityError:
                    messagebox.showerror("Error", "Event Key already exists")
            else:
                messagebox.showerror("Error", "Please fill in all fields")
                return
            #getting the index of the last event
            self.c.execute("SELECT COUNT(event_name) FROM eventcreation")
            self.eventindex = self.c.fetchone()[0]
            eventregistrationreference = self.controller.get_page(EventRegistration)
            eventregistrationreference.eventdropdown.event_generate("<FocusIn>")
            eventregistrationreference.eventdropdown.event_generate("<FocusOut>")
            eventregistrationreference.eventdropdown.current(self.eventindex)
            eventregistrationreference.eventdropdown.event_generate("<FocusIn>")
            eventregistrationreference.eventdropdown.event_generate("<FocusOut>")
            eventregistrationreference.eventdropdown.event_generate("<<ComboboxSelected>>")
            mainpagereference = self.controller.get_page(MainPage)
            mainpagereference.update_eventnames()

            self.conn.close()

    def clear_image(self):
        self.panel.configure(image=self.panelimgnoimg)
        self.filename = r"Assets\EventCreation\placeholderimage.png"

            


class ManagementSuite(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=PINK)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.controller = controller
        for x in range(42):
            self.columnconfigure(x, weight=1, uniform='x')
            Label(self, width=1, bg=PINK).grid(
                row=0, column=x, sticky=NSEW)
        for y in range(21):
            self.rowconfigure(y, weight=1, uniform='x')
            Label(self, width=1, bg=PINK).grid(
                row=y, column=0, sticky=NSEW)

        self.backgroundimageoriginal = Image.open(r"Assets\managementsuite\backgroundimage.png")
        self.backgroundimage = ImageTk.PhotoImage(self.backgroundimageoriginal.resize(
            (math.ceil(1680 * dpi / 96), math.ceil(840 * dpi / 96)), Image.Resampling.LANCZOS))
        self.backgroundimagelabel = Label(self, image=self.backgroundimage, width=1, height=1, bg=LIGHTPURPLE)
        self.backgroundimagelabel.grid(row=0, column=0, rowspan=21, columnspan=43, sticky=NSEW)
        self.backgroundimagelabel.grid_propagate(False)
        self.interfaceframe = Frame(self, bg=LIGHTPURPLE, width=1,height=1)
        self.studentcntimg = Image.open(r"Assets\managementsuite\manageeventswidgets\studentscountlabel200x80.png")
        self.studentcountimg = ImageTk.PhotoImage(self.studentcntimg.resize(
            (math.ceil(200 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.cancelimageorg = Image.open(r"Assets\managementsuite\manageeventswidgets\cancelbutton.png")
        self.cancelimage= ImageTk.PhotoImage(self.cancelimageorg.resize(
            (math.ceil(160 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))          
        self.cdeleteimageorg = Image.open(r"Assets\managementsuite\manageeventswidgets\confirmbutton.png")
        self.cdeleteimage = ImageTk.PhotoImage(self.cdeleteimageorg.resize(
            (math.ceil(160 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))

        self.createinterface()
        self.createlandingwidgets()

    def createinterface(self):
        for x in range(38): # 38
            self.interfaceframe.columnconfigure(x, weight=1, uniform='x')
            Label(self.interfaceframe, width=1, bg=LIGHTPURPLE).grid(
                row=0, column=x, sticky=NSEW)
        for y in range(17): # 17
            self.interfaceframe.rowconfigure(y, weight=1, uniform='x')
            Label(self.interfaceframe, width=1, bg=LIGHTPURPLE).grid(
                row=y, column=0, sticky=NSEW)
        self.interfaceframe.grid(row=3, column=2, rowspan=17, columnspan=38, sticky=NSEW)
        self.interfaceframe.grid_propagate(False)
        self.intframebackgroundoriginal = Image.open(r"Assets\managementsuite\blankinterface.png")
        self.intframebackground = ImageTk.PhotoImage(self.intframebackgroundoriginal.resize(
            (math.ceil(1520 * dpi / 96), math.ceil(680 * dpi / 96)), Image.Resampling.LANCZOS))
        self.intframebackgroundlabel = Label(self.interfaceframe, image=self.intframebackground, width=1, height=1, bg=LIGHTPURPLE)
        self.intframebackgroundlabel.grid(row=0, column=0, rowspan=17, columnspan=38, sticky=NSEW)
        self.intframebackgroundlabel.grid_propagate(False)


    def createlandingwidgets(self):
        for widgets in self.interfaceframe.winfo_children():
            widgets.destroy()
        self.createinterface()
        self.createeventsimage = Image.open(r"Assets\managementsuite\createevents.png")
        self.createeventsimage = ImageTk.PhotoImage(self.createeventsimage.resize(
            (math.ceil(560 * dpi / 96), math.ceil(120 * dpi / 96)), Image.Resampling.LANCZOS))
        self.createeventsbutton = Button(self.interfaceframe, image=self.createeventsimage, width=1, height=1, relief=RAISED, cursor="hand2",
        command=lambda: [self.controller.show_frame(EventCreation), self.controller.togglebuttonrelief(self.controller.eventcreationbutton)])
        self.createeventsbutton.grid(row=5, column=2, rowspan=3, columnspan=14, sticky=NSEW) 
        self.manageeventsimage = Image.open(r"Assets\managementsuite\manageevents.png")
        self.manageeventsimage = ImageTk.PhotoImage(self.manageeventsimage.resize(
            (math.ceil(560 * dpi / 96), math.ceil(120 * dpi / 96)), Image.Resampling.LANCZOS))
        self.manageeventsbutton = Button(self.interfaceframe, image=self.manageeventsimage, width=1, height=1, relief=RAISED, cursor="hand2",
        command=lambda: self.manageeventsframe())
        self.manageeventsbutton.grid(row=9, column=2, rowspan=3, columnspan=14, sticky=NSEW)
        self.viewparticipantsimage = Image.open(r"Assets\managementsuite\viewparticipants.png")
        self.viewparticipantsimage = ImageTk.PhotoImage(self.viewparticipantsimage.resize(
            (math.ceil(560 * dpi / 96), math.ceil(120 * dpi / 96)), Image.Resampling.LANCZOS))
        self.viewparticipantsbutton = Button(self.interfaceframe, image=self.viewparticipantsimage, width=1, height=1, relief=RAISED, cursor="hand2",
        command=lambda: self.view_participants())
        self.viewparticipantsbutton.grid(row=13, column=2, rowspan=3, columnspan=14, sticky=NSEW)
    def manageeventsframe(self):
        # for widgets in self.interfaceframe.winfo_children():
        #     widgets.destroy()
        self.createinterface()
        self.manageexistingeventsimage = Image.open(r"Assets\managementsuite\manageeventswidgets\manageexistingeventslabel.png")
        self.manageexistingeventsimage = ImageTk.PhotoImage(self.manageexistingeventsimage.resize(
            (math.ceil(200 * dpi / 96), math.ceil(120 * dpi / 96)), Image.Resampling.LANCZOS))
        self.manageexistingeventslabel = Label(self.interfaceframe, image=self.manageexistingeventsimage, width=1, height=1, bg=LIGHTPURPLE)
        self.manageexistingeventslabel.grid(row=1, column=1, rowspan=3, columnspan=5, sticky=NSEW)
        self.manageexistingeventslabel.grid_propagate(False)

        self.searchallimage = Image.open(r"Assets\managementsuite\manageeventswidgets\allsearchbutton.png")
        self.searchallimage = ImageTk.PhotoImage(self.searchallimage.resize(
            (math.ceil(200 * dpi / 96), math.ceil(40 * dpi / 96)), Image.Resampling.LANCZOS))
        self.searchallbutton = Button(self.interfaceframe, image=self.searchallimage, width=1, height=1, bg=LIGHTPURPLE, relief=FLAT, command=lambda:self.database_queries())
        self.searchallbutton.grid(row=13, column=1, rowspan=1, columnspan=5, sticky=NSEW)
        self.returnhomeimage = Image.open(r"Assets\managementsuite\manageeventswidgets\mainmenubutton.png")
        self.returnhomeimage = ImageTk.PhotoImage(self.returnhomeimage.resize(
            (math.ceil(200 * dpi / 96), math.ceil(40 * dpi / 96)), Image.Resampling.LANCZOS))
        self.returnhomebutton = Button(self.interfaceframe, image=self.returnhomeimage, width=1, height=1, bg=LIGHTPURPLE, relief=FLAT, command=lambda: self.createlandingwidgets())
        self.returnhomebutton.grid(row=15, column=1, rowspan=1, columnspan=5, sticky=NSEW)


        self.centerframe = Frame(self.interfaceframe, bg=LIGHTPURPLE, width=1, height=1)
        self.centerframe.grid(row=1, column=7, rowspan=15, columnspan=18, sticky=NSEW)
        self.centerframe.grid_propagate(False)
        for x in range(18):
            self.centerframe.columnconfigure(x, weight=1, uniform='x')
            Label(self.centerframe, width=1, bg=LIGHTPURPLE).grid(
                row=0, column=x, sticky=NSEW)
        for y in range(15):
            self.centerframe.rowconfigure(y, weight=1, uniform='y')
            Label(self.centerframe, width=1, bg=LIGHTPURPLE).grid(
                row=y, column=0, sticky=NSEW)
        self.centerframebackgroundimage = Image.open(r"Assets\managementsuite\manageeventswidgets\centerframebackground.png")
        self.centerframebackgroundimage = ImageTk.PhotoImage(self.centerframebackgroundimage.resize(
            (math.ceil(720 * dpi / 96), math.ceil(600 * dpi / 96)), Image.Resampling.LANCZOS))
        self.centerframebackgroundlabel = Label(self.centerframe, image=self.centerframebackgroundimage, width=1, height=1, bg=LIGHTPURPLE)
        self.centerframebackgroundlabel.grid(row=0, column=0, rowspan=15, columnspan=18, sticky=NSEW)
        self.rightframe = Frame(self.interfaceframe, bg=LIGHTPURPLE, width=1, height=1)
        self.rightframe.grid(row=1, column=26, rowspan=15, columnspan=11, sticky=NSEW)
        self.rightframe.grid_propagate(False)
        for x in range(11):
            self.rightframe.columnconfigure(x, weight=1, uniform='x')
            Label(self.rightframe, width=1, bg=LIGHTPURPLE).grid(
                row=0, column=x, sticky=NSEW)
        for y in range(15):
            self.rightframe.rowconfigure(y, weight=1, uniform='x')
            Label(self.rightframe, width=1, bg=LIGHTPURPLE).grid(
                row=y, column=0, sticky=NSEW)
        self.rightframebackgroundimage = Image.open(r"Assets\managementsuite\manageeventswidgets\rightframe.png")
        self.rightframebackgroundimage = ImageTk.PhotoImage(self.rightframebackgroundimage.resize(
            (math.ceil(440 * dpi / 96), math.ceil(600 * dpi / 96)), Image.Resampling.LANCZOS))
        self.rightframebackgroundlabel = Label(self.rightframe, image=self.rightframebackgroundimage, width=1, height=1, bg=LIGHTPURPLE)
        self.rightframebackgroundlabel.grid(row=0, column=0, rowspan=15, columnspan=11, sticky=NSEW)
        self.studentcountlabel = Label(self.rightframe, image=self.studentcountimg, width=1, height=1)



    def generate_widgets(self):
        self.labelbackground = Image.open(r"Assets\managementsuite\manageeventswidgets\bgfortitle.png")
        self.labelbackground = ImageTk.PhotoImage(self.labelbackground.resize(
            (math.ceil(480 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.editbuttonimage = Image.open(r"Assets\managementsuite\manageeventswidgets\editicon.png")
        self.editbuttonimage = ImageTk.PhotoImage(self.editbuttonimage.resize(
            (math.ceil(80 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.deletebuttonimage = Image.open(r"Assets\managementsuite\manageeventswidgets\deleteicon.png")
        self.deletebuttonimage = ImageTk.PhotoImage(self.deletebuttonimage.resize(
            (math.ceil(80 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))

    def database_queries(self):
        self.generate_widgets()
        self.conn = sqlite3.connect("interactivesystem.db")
        self.c = self.conn.cursor()
        self.c.execute("SELECT COUNT(event_name) FROM eventcreation")
        self.eventcount = self.c.fetchone()
        self.eventcount = self.eventcount[0]
        if self.eventcount == 0:
            messagebox.showinfo("No Events", "There are no events in the database")
            #if the last event was deleted already, clear the event list frame and right frame
            try:
                self.overallframes[0].destroy() 
            except:
                pass
            return
        self.c.execute("SELECT event_name, eventkey_number FROM eventcreation")
        self.events = self.c.fetchall()
        self.overallframes = {}
        self.framesneeded = math.ceil(self.eventcount / 4)
        #configure a label to display the page number
        self.page = 1
        self.pagecount = Label(self.centerframe, width=1,height=1, text="Page\n" + str(self.page) + " of " + str(self.framesneeded), bg=LIGHTPURPLE, fg="black", font=("Arial", 14))
        self.pagecount.grid(row=0, column=15, rowspan=2, columnspan=3, sticky=NSEW) 
        self.backgroundforframes = Image.open(r"Assets\managementsuite\manageeventswidgets\backgroundforpages.png")
        self.backgroundforframes = ImageTk.PhotoImage(self.backgroundforframes.resize(
                (math.ceil(640 * dpi / 96), math.ceil(440 * dpi / 96)), Image.Resampling.LANCZOS))
        #configure the next and previous buttons
        button = Button(self.centerframe, text="<", height=1,width=1, command=lambda:self.previous_page())
        button.grid(row=2, column=15, rowspan=1, columnspan=1, sticky=NSEW)
        button2 = Button(self.centerframe, text=">", height=1,width=1, command=lambda:self.next_page())
        button2.grid(row=2, column=17, rowspan=1, columnspan=1, sticky=NSEW)
        #configure the frames
        for x in range(self.framesneeded):
            self.overallframes[x] = Frame(self.centerframe, bg=LIGHTPURPLE, relief=FLAT, width=1, height=1)
            self.overallframes[x].grid(row=3, column=1, rowspan=11, columnspan=16, sticky=NSEW)
            self.overallframes[x].grid_propagate(False)
            for y in range(11):
                self.overallframes[x].rowconfigure(y, weight=1, uniform='y')
                Label(self.overallframes[x], width=1, bg=NAVYBLUE).grid(
                    row=y, column=0, sticky=NSEW)
            for z in range(16):
                self.overallframes[x].columnconfigure(z, weight=1, uniform='x')
                Label(self.overallframes[x], width=1, bg=NAVYBLUE).grid(
                    row=0, column=z, sticky=NSEW)
            Label(self.overallframes[x], image=self.backgroundforframes, width=1, height=1, bg=LIGHTPURPLE).grid(
                row=0, column=0, rowspan=11, columnspan=16, sticky=NSEW)
            self.overallframes[x].grid_remove()
        #print how many frames are needed
        #self.events = c.fetchall()
        #here's what we know, 
        # we have the index of the event details inside the tuple
        # we know the number of events
        #in order to know the number of frames needed, we need to know how many events there are
        # we can get this by doing len(self.events)
        # we can do this by using a for loop
        # we can use the index of the event to determine which frame to put it in
        # we can use the index of the event to determine which row and column to put it in
        # so the rowcount just needs to be index of event * 3, row=(indexofeventdetails-4*3) gives us 0, 3, 6, 9. Index of event details after 1st page is 4 - 4*3 = 0, 3, 6, 9
        # columns are always 0, 12, 14
        # to generate the number of frames we need, we can use math.ceil(self.eventcount / 4)
        for indexofeventdetails in range(self.eventcount):
            # print(indexofeventdetails)
            event_name = self.events[indexofeventdetails][0]
            event_key = self.events[indexofeventdetails][1]
            #x is the index comparison
            x = math.floor(indexofeventdetails / 4)
            if x == 0: 
                Label(self.overallframes[x], text=f"{event_name}", image=self.labelbackground, width=1, height=1, font=("Avenir Next Bold", 18),fg="white", bg=LIGHTPURPLE, compound=CENTER).grid(row=indexofeventdetails*3, column=0, rowspan=2, columnspan=12, sticky=NSEW)
                Button(self.overallframes[x], image=self.editbuttonimage, width=1, height=1, bg=LIGHTPURPLE, relief=FLAT, command=lambda x=event_name:self.edit_event(x)).grid(
                    row=indexofeventdetails*3, column=12, rowspan=2, columnspan=2, sticky=NSEW)
                Button(self.overallframes[x], image=self.deletebuttonimage, width=1, height=1, bg=LIGHTPURPLE, relief=FLAT, command=lambda x=event_name, y=event_key: self.confirm_delete(x,y)).grid(
                    row=indexofeventdetails*3, column=14, rowspan=2, columnspan=2, sticky=NSEW)
            else:
                Label(self.overallframes[x], text=f"{event_name}", image=self.labelbackground, width=1, height=1, font=("Avenir Next Bold", 18),fg="white", bg=LIGHTPURPLE, compound=CENTER).grid(row=(indexofeventdetails-4*x)*3, column=0, rowspan=2, columnspan=12, sticky=NSEW)
                Button(self.overallframes[x], image=self.editbuttonimage, width=1, height=1, bg=LIGHTPURPLE, relief=FLAT, command=lambda x=event_name:self.edit_event(x)).grid(
                    row=(indexofeventdetails-4*x)*3, column=12, rowspan=2, columnspan=2, sticky=NSEW)
                Button(self.overallframes[x], image=self.deletebuttonimage, width=1, height=1, bg=LIGHTPURPLE, relief=FLAT, command=lambda x=event_name, y=event_key: self.confirm_delete(x,y)).grid(
                    row=(indexofeventdetails-4*x)*3, column=14, rowspan=2, columnspan=2, sticky=NSEW)
        self.overallframes[0].grid()

    def edit_event(self, eventname):
        self.conn = sqlite3.connect("interactivesystem.db")
        self.c = self.conn.cursor()
        self.c.execute("SELECT * FROM eventcreation WHERE event_name=?", (eventname,))
        event_details = self.c.fetchone()
        if event_details == None:
            messagebox.showerror("Error", "Event may have already been deleted, please search again.")
            return
        self.tempframe = Frame(self.interfaceframe, bg=LIGHTPURPLE, relief=FLAT, width=1,height=1)
        self.tempframe.grid(row=1, column=7, rowspan=15, columnspan=18, sticky=NSEW)
        self.tempframe.grid_propagate(False)
        for y in range(15):
            self.tempframe.rowconfigure(y, weight=1, uniform='y')
            Label(self.tempframe, width=1, bg=NAVYBLUE).grid(
                row=y, column=0, sticky=NSEW)
        for z in range(18):
            self.tempframe.columnconfigure(z, weight=1, uniform='x')
            Label(self.tempframe, width=1, bg=NAVYBLUE).grid(
                row=0, column=z, sticky=NSEW)
        Label(self.tempframe, image=self.centerframebackgroundimage, width=1, height=1, bg=LIGHTPURPLE).grid(
            row=0, column=0, rowspan=15, columnspan=18, sticky=NSEW)
        #button to remove the frame
        Button(self.tempframe, text="X", width=1, height=1, bg=LIGHTPURPLE, relief=FLAT, command=lambda:self.tempframe.grid_remove()).grid(row=0, column=17, rowspan=1, columnspan=1, sticky=NSEW)
        event_key = event_details[0]
        event_name = event_details[1]
        event_description = event_details[2]
        event_startdate = event_details[3]
        event_enddate = event_details[4]
        event_starttime = event_details[5]
        event_endtime = event_details[6]
        event_organizer = event_details[7]
        venue_name = event_details[8]
        host_name = event_details[9]
        event_image = io.BytesIO(event_details[10])
        event_image = Image.open(event_image)
        self.event_image = ImageTk.PhotoImage(event_image.resize(
            (math.ceil(200 * dpi / 96), math.ceil(200 * dpi / 96)), Image.Resampling.LANCZOS))
        event_keybutton = Button(self.tempframe, text=f"Event Key: {event_key}", anchor=CENTER, width=1, height=1, font=("Avenir Next Medium", 16),fg="white", bg=ORANGE, compound=CENTER, state=DISABLED, disabledforeground="white")
        event_keybutton.grid(row=0, column=1, rowspan=1, columnspan=8, sticky=NSEW)
        event_namebutton = Button(self.tempframe, text=f"Event: {event_name}", anchor=W, width=1, height=1, font=("Avenir Next Medium", 16),fg="white", bg=NAVYBLUE, command=lambda:self.changing_details("normal",event_key, event_name, fieldchanged="event_name"))
        event_namebutton.grid(row=1, column=1, rowspan=1, columnspan=11, sticky=NSEW)
        event_descriptionbutton = Button(self.tempframe, text=f"Description:\n{event_description}", anchor=CENTER, width=1, height=1, font=("Avenir Next Medium", 16),fg="white", bg=NAVYBLUE, command=lambda:self.changing_details("normal",event_key, event_description, fieldchanged="event_description"), wraplength=350, justify=CENTER)
        event_descriptionbutton.grid(row=2, column=1, rowspan=3, columnspan=11, sticky=NSEW)
        event_datebutton = Button(self.tempframe, text=f"Date: {event_startdate} - {event_enddate}", anchor=W, width=1, height=1, font=("Avenir Next Medium", 16),fg="white", bg=NAVYBLUE,command=lambda:self.changing_details("date", event_key, (event_startdate, event_enddate)))
        event_datebutton.grid(row=5, column=1, rowspan=1, columnspan=11, sticky=NSEW)
        event_timebutton = Button(self.tempframe, text=f"Time: {event_starttime} - {event_endtime}", anchor=W, width=1, height=1, font=("Avenir Next Medium", 16),fg="white", bg=NAVYBLUE, command= lambda:self.changing_details("time", event_key,(event_starttime, event_endtime)))
        event_timebutton.grid(row=6, column=1, rowspan=1, columnspan=11, sticky=NSEW)
        event_organizerbutton = Button(self.tempframe, text=f"Organizer: {event_organizer}", anchor=W, width=1, height=1, font=("Avenir Next Medium", 16),fg="white", bg=NAVYBLUE, command=lambda:self.changing_details("normal", event_key,  event_organizer, fieldchanged="event_organizer"))
        event_organizerbutton.grid(row=7, column=1, rowspan=1, columnspan=11, sticky=NSEW)
        venue_namebutton = Button(self.tempframe, text=f"Venue: {venue_name}", anchor=W, width=1, height=1, font=("Avenir Next Medium", 16),fg="white", bg=NAVYBLUE, command=lambda:self.changing_details("normal", event_key, venue_name, fieldchanged="venue_name"))
        venue_namebutton.grid(row=8, column=1, rowspan=1, columnspan=11, sticky=NSEW)
        host_namebutton = Button(self.tempframe, text=f"Host: {host_name}", anchor=W, width=1, height=1, font=("Avenir Next Medium", 16),fg="white", bg=NAVYBLUE, command=lambda:self.changing_details("normal", event_key, host_name, fieldchanged="host_name"))
        host_namebutton.grid(row=9, column=1, rowspan=1, columnspan=11, sticky=NSEW)
        event_imagelabel = Label(self.tempframe, image=self.event_image, width=1, height=1, bg=LIGHTPURPLE)
        event_imagelabel.grid(row=1, column=13, rowspan=4, columnspan=4, sticky=NSEW) 
    def changing_details(self, entrytype, event_key, *args, **kwargs):
        self.conn = sqlite3.connect("interactivesystem.db")
        self.c = self.conn.cursor()
        field_changed = kwargs.get("fieldchanged")
        #refactor this function 
        def confirm_action(entrytype, *args, **kwargs):
            if entrytype == "normal":
                with self.conn:
                    self.c.execute(f"UPDATE eventcreation SET {field_changed} = ? WHERE eventkey_number = ?", (normalentry.get(), event_key))
                    messagebox.showinfo("Success", f"{field_changed} updated successfully where event key is {event_key}")
            elif entrytype == "date":
                with self.conn:
                    self.c.execute(f"UPDATE eventcreation SET event_startdate = ?, event_enddate = ? WHERE eventkey_number = ?", (start_dateentry.get(), enddateentry.get(), event_key))
                    messagebox.showinfo("Success",  f"event_startdate changed to {start_dateentry.get()} and\nevent_enddate changed to {enddateentry.get()}.\nUpdated successfully where event key is {event_key}.")
            elif entrytype == "time":
                with self.conn:
                    self.c.execute("UPDATE eventcreation SET event_starttime = ?, event_endtime = ? WHERE eventkey_number = ?", (starttimeentry.get(), endtimeentry.get(), event_key))
                    messagebox.showinfo("Success",  f"event_starttime changed to {starttimeentry.get()} and\nevent_endtime changed to {endtimeentry.get()}.\nUpdated successfully where event key is {event_key}.")
        for widget in self.tempframe.winfo_children():
            if widget.winfo_class() == "Entry":
                widget.destroy()
        if entrytype == "normal":
            normalentry = Entry(self.tempframe, width=1, font=("Avenir Next Bold", 18),fg="black", bg=LIGHTYELLOW, justify=CENTER)
            normalentry.grid(row=11, column=3, rowspan=2, columnspan=12, sticky=NSEW)
            normalentry.delete(0, END)
            normalentry.insert(0, args[0])
            normalentry.focus_set()
            confirmbutton = Button(self.tempframe, text=f"Confirm changes for {field_changed}", anchor=CENTER, width=1, height=1, font=("Avenir Next Bold", 18),fg="black", bg=OTHERPINK, command=lambda:confirm_action(entrytype, args[0]))
            confirmbutton.grid(row=13, column=3, rowspan=2, columnspan=12, sticky=NSEW)
        elif entrytype == "date":
            start_dateentry = Entry(self.tempframe, width=1, font=("Avenir Next Bold", 18),fg="black", bg=LIGHTYELLOW, justify=CENTER)
            start_dateentry.grid(row=11, column=2, rowspan=2, columnspan=6, sticky=NSEW)
            start_dateentry.delete(0, END)
            start_dateentry.insert(0, args[0][0])
            start_dateentry.focus_set()
            enddateentry = Entry(self.tempframe, width=1, font=("Avenir Next Bold", 18),fg="black", bg=LIGHTYELLOW, justify=CENTER)
            enddateentry.grid(row=11, column=10, rowspan=2, columnspan=6, sticky=NSEW)
            enddateentry.delete(0, END)
            enddateentry.insert(0, args[0][1])
            confirmbutton = Button(self.tempframe, text="Confirm changes for date", anchor=CENTER, width=1, height=1, font=("Avenir Next Bold", 18),fg="black", bg=ORANGE, command=lambda:confirm_action(entrytype))
            confirmbutton.grid(row=13, column=3, rowspan=2, columnspan=12, sticky=NSEW)
        elif entrytype == "time":
            starttimeentry = Entry(self.tempframe, width=1, font=("Avenir Next Bold", 18),fg="black", bg=LIGHTYELLOW, justify=CENTER)
            starttimeentry.grid(row=11, column=2, rowspan=2, columnspan=6, sticky=NSEW)
            starttimeentry.delete(0, END)
            starttimeentry.insert(0, args[0][0])
            starttimeentry.focus_set()
            endtimeentry = Entry(self.tempframe, width=1, font=("Avenir Next Bold", 18),fg="black", bg=LIGHTYELLOW, justify=CENTER)
            endtimeentry.grid(row=11, column=10, rowspan=2, columnspan=6, sticky=NSEW)
            endtimeentry.delete(0, END)
            endtimeentry.insert(0, args[0][1])
            confirmbutton = Button(self.tempframe, text="Confirm changes for time", anchor=CENTER, width=1, height=1, font=("Avenir Next Bold", 18),fg="black", bg=LIGHTYELLOW, command=lambda:confirm_action(entrytype))
            confirmbutton.grid(row=13, column=3, rowspan=2, columnspan=12, sticky=NSEW)
    def confirm_delete(self, event_name, event_key):
        for widget in self.rightframe.winfo_children():
            if widget.winfo_class() == "Button":
                widget.destroy()
        self.conn = sqlite3.connect("interactivesystem.db")
        self.c = self.conn.cursor()
        self.c.execute("SELECT * FROM eventcreation WHERE event_name=?", (event_name,))
        event_details = self.c.fetchone()
        if event_details == None:
            messagebox.showerror("Error", "Event may have already been deleted, please search again.")
            return
        self.numbofstudentlabel = Button(self.rightframe, state=DISABLED, text="", compound=CENTER, font=("Avenir Next Bold", 12),disabledforeground=BLACK,width=1, height=1, bg=LIGHTYELLOW, command=None, relief=FLAT)
        self.numbofstudentlabel.grid(row=8, column=7, rowspan=2, columnspan=2, sticky=NSEW)

        self.c.execute("SELECT full_name FROM eventregistration WHERE eventkey_registered = ?", (event_key,))
        self.numbofstudentlabel.config(text=f"{len(self.c.fetchall())}\nstudents")
        #checking if foreign key is enabled in sqlite3
        self.studentcountlabel.grid(row=8, column=2, rowspan=2, columnspan=5, sticky=NSEW)
        self.cancelimageorg = Image.open(r"Assets\managementsuite\manageeventswidgets\cancelbutton.png")
        self.cancelimage= ImageTk.PhotoImage(self.cancelimageorg.resize(
            (math.ceil(160 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))          
        self.cdeleteimageorg = Image.open(r"Assets\managementsuite\manageeventswidgets\confirmbutton.png")
        self.cdeleteimage = ImageTk.PhotoImage(self.cdeleteimageorg.resize(
            (math.ceil(160 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        canceldeletebutton = Button(self.rightframe, image=self.cancelimage, width=1, height=1, command=lambda:self.cancel_delete())
        canceldeletebutton.grid(row=11, column=1, rowspan=2, columnspan=4, sticky=NSEW)
        confirmdeletebutton = Button(self.rightframe, width=1, height=1, image=self.cdeleteimage, command=lambda:self.delete_event(event_name, event_key))
        confirmdeletebutton.grid(row=11, column=6, rowspan=2, columnspan=4, sticky=NSEW)
        confirmdeletelabel = Button(self.rightframe, state=DISABLED, text="Are you sure you want\nto delete this event?", anchor=CENTER, width=1, height=1, font=("Avenir Next Bold", 14),disabledforeground=BLACK, bg=LIGHTPURPLE)
        confirmdeletelabel.grid(row=1, column=1, rowspan=2, columnspan=9, sticky=NSEW)
        self.eventnamelabel = Button(self.rightframe,state=DISABLED, text=f"Event:{event_name}", anchor=CENTER, font=("Avenir Next", 12),disabledforeground=BLACK, bg=LIGHTPURPLE, width=1, height=1, wraplength=200, justify=CENTER)
        self.eventnamelabel.grid(row=3, column=3, rowspan=2, columnspan=5, sticky=NSEW)
        self.eventkeylabel = Button(self.rightframe,state=DISABLED, text=f"Key:{event_key}", anchor=CENTER, font=("Avenir Next", 12),disabledforeground=BLACK, bg=LIGHTPURPLE, width=1, height=1, wraplength=200, justify=CENTER)
        self.eventkeylabel.grid(row=5, column=3, rowspan=2, columnspan=5, sticky=NSEW)
    def cancel_delete(self):
        self.studentcountlabel.grid_remove()
        self.numbofstudentlabel.grid_remove()
        for widget in self.rightframe.winfo_children():
            if widget.winfo_class() == "Button":
                widget.destroy()
 
    def delete_event(self, event_name, event_key):
        self.conn = sqlite3.connect("interactivesystem.db")
        self.c = self.conn.cursor()
        #turning on foreign key
        with self.conn:
            self.c.execute("PRAGMA foreign_keys = ON")

        with self.conn:
            self.c.execute("DELETE FROM eventcreation WHERE event_name = ?", (event_name,))
            messagebox.showinfo("Success", f"Event with event name {event_name} and {event_key} deleted successfully")

        #clear the rightframe
        test = self.controller.get_page(MainPage)
        test.update_eventnames()
        resetdropdown = self.controller.get_page(EventRegistration)
        resetdropdown.eventdropdown.current(1)
        resetdropdown.eventdropdown.event_generate("<FocusIn>")
        resetdropdown.eventdropdown.event_generate("<FocusOut>")
        resetdropdown.eventdropdown.event_generate("<<ComboboxSelected>>")

        self.studentcountlabel.grid_remove()
        for widget in self.rightframe.winfo_children():
            if widget.winfo_class() == "Button":
                widget.destroy()


    def next_page(self):
        if self.page < self.framesneeded:
            self.page += 1
            self.pagecount.config(text="Page\n" + str(self.page) + " of " + str(self.framesneeded))
            self.overallframes[self.page-2].grid_remove()
            self.overallframes[self.page-1].grid()
    def previous_page(self):
        if self.page > 1:
            self.page -= 1
            self.pagecount.config(text="Page\n" + str(self.page) + " of " + str(self.framesneeded))
            self.overallframes[self.page].grid_remove()
            self.overallframes[self.page-1].grid()

    #View participants functions
    def view_participants(self):
        self.viewparticipants = Frame(self, width=1, height=1, bg=WHITE)
        self.viewparticipants.grid(row=3, column=2, rowspan=17, columnspan=38, sticky=NSEW)
        self.viewparticipants.grid_propagate(False)
        for x in range(38):
            self.viewparticipants.columnconfigure(x, weight=1, uniform="x")
            Label(self.viewparticipants, bg=WHITE).grid(row=0, column=x, sticky=NSEW)
        for y in range(17):
            self.viewparticipants.rowconfigure(y, weight=1, uniform="y")
            Label(self.viewparticipants, bg=WHITE).grid(row=y, column=0, sticky=NSEW)
        #background label
        self.searchbyeventsframe = Frame(self, width=1, height=1, bg=WHITE)
        self.searchbyeventsframe.grid(row=3, column=2, rowspan=17, columnspan=38, sticky=NSEW)
        self.searchbyeventsframe.grid_propagate(False)
        #~~~~~~~~~~~~~~ Search by events frame ~~~~~~~~~~~~~~~
        for x in range(38):
            self.searchbyeventsframe.columnconfigure(x, weight=1, uniform="x")
            Label(self.searchbyeventsframe, width=1,bg=WHITE).grid(row=0, column=x, sticky=NSEW)
        for y in range(17):
            self.searchbyeventsframe.rowconfigure(y, weight=1, uniform="x")
            Label(self.searchbyeventsframe, width=1,bg=WHITE).grid(row=y, column=0, sticky=NSEW)
        self.searchbyeventsframe.grid_remove()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~ IMAGES ~~~~~~~~~~~~~~
        self.vpbgorg = Image.open(r"Assets\managementsuite\viewparticipantswidgets\viewparticipantsbg.png")
        self.vpbg = ImageTk.PhotoImage(self.vpbgorg.resize(
            (math.ceil(1520 * dpi / 96), math.ceil(680 * dpi / 96)), Image.Resampling.LANCZOS))
        self.vpbglabel = Label(self.viewparticipants, image=self.vpbg,width=1,height=1,bg=WHITE)
        self.vpbglabel.grid(row=0, column=0, rowspan=17, columnspan=38, sticky=NSEW)
        self.searchregistrantsimg = Image.open(r"Assets\managementsuite\viewparticipantswidgets\searchregistrantsbutton280x80.png")
        self.searchregistrants = ImageTk.PhotoImage(self.searchregistrantsimg.resize(
            (math.ceil(280 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.searcheventsimg = Image.open(r"Assets\managementsuite\viewparticipantswidgets\searchbyevents280x80.png")
        self.searchevents = ImageTk.PhotoImage(self.searcheventsimg.resize(
            (math.ceil(280 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.backimg = Image.open(r"Assets\managementsuite\viewparticipantswidgets\vpbackbutton280x80.png")
        self.back = ImageTk.PhotoImage(self.backimg.resize(
            (math.ceil(280 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.searchregistrantsbutton = Button(self.viewparticipants, image=self.searchregistrants, width=1, height=1, relief=SOLID, bd=4, highlightthickness=1, highlightbackground=LIGHTPURPLE, command=lambda:self.viewparticipants.grid())
        self.searchregistrantsbutton.grid(row=2, column=2, rowspan=2, columnspan=7, sticky=NSEW)
        self.searcheventsbutton = Button(self.viewparticipants, image=self.searchevents, width=1, height=1, command=lambda:[self.searchbyeventsframe.grid(),self.show_searchevents()])
        self.searcheventsbutton.grid(row=6, column=2, rowspan=2, columnspan=7, sticky=NSEW)
        self.backbutton = Button(self.viewparticipants, image=self.back, width=1, height=1, command=lambda:[self.interfaceframe.tkraise(), self.searchbyeventsframe.grid_remove()])
        self.backbutton.grid(row=10, column=2, rowspan=2, columnspan=7, sticky=NSEW)
        self.searchframe = Frame(self.viewparticipants, width=1, height=1, bg=WHITE)
        self.searchframe.grid(row=0, column=10, rowspan=17, columnspan=11, sticky=NSEW)
        self.searchframe.grid_propagate(False)
        for x in range(11):
            self.searchframe.columnconfigure(x, weight=1, uniform="x")
            Label(self.searchframe, bg=ORANGE).grid(row=0, column=x, sticky=NSEW)
        for y in range(17):
            self.searchframe.rowconfigure(y, weight=1, uniform="x")
            Label(self.searchframe, bg=ORANGE).grid(row=y, column=0, sticky=NSEW)
        self.searchframebgorg = Image.open(r"Assets\managementsuite\viewparticipantswidgets\searchframebg.png")
        self.searchframebg = ImageTk.PhotoImage(self.searchframebgorg.resize(
            (math.ceil(440 * dpi / 96), math.ceil(680 * dpi / 96)), Image.Resampling.LANCZOS))
        self.searchframebglabel = Label(self.searchframe, image=self.searchframebg, width=1,height=1, bg=WHITE)
        self.searchframebglabel.grid(row=0, column=0, rowspan=17, columnspan=11, sticky=NSEW)
        self.searchbuttonimg = Image.open(r"Assets\managementsuite\viewparticipantswidgets\magnifyingbutton80x80.png")
        self.searchbutton = ImageTk.PhotoImage(self.searchbuttonimg.resize(
            (math.ceil(80 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.searchbuttonbutton = Button(self.searchframe, image=self.searchbutton, width=1, height=1, command=lambda:self.searchregistrants_function(searchentry.get()))
        self.searchbuttonbutton.grid(row=1, column=8, rowspan=2, columnspan=2, sticky=NSEW)
        searchentry = Entry(self.searchframe, width=1, bg=WHITE, fg="black", font=("Avenir Next", 14))
        searchentry.grid(row=2, column=1, rowspan=1, columnspan=6, sticky=NSEW)
    def searchregistrants_function(self, name): #name is searchentry.get()
        #database queries
        self.conn = sqlite3.connect("interactivesystem.db")
        self.c = self.conn.cursor() 
        if name == "":
            self.c.execute("SELECT DISTINCT full_name, icpass_number, email FROM eventregistration")
        else:
            self.c.execute("SELECT DISTINCT full_name, icpass_number, email FROM eventregistration WHERE full_name LIKE ?", ("%"+name+"%",))
        self.results = self.c.fetchall()
        self.count = len(self.results)
        if self.count == 0:
            messagebox.showerror("Error", "No students that have registered for events were found.")
            # destroy the searchresultsframe if it exists
            try:
                self.searchresultsframes[0].destroy()
                self.eventsregisteredforframe[0].destroy()
                self.searchquerylabel.destroy()
                self.pagecounter.destroy()
                self.previouspagebutton.destroy()
                self.nextpagebutton.destroy()
                self.frameforpagebuttons.destroy()
                return
            except:
                pass
            return
        self.vpframesneeded = math.ceil(self.count/5)
        self.pagenumber = 1
        self.pagecounter = Label(self.searchframe, width=1, height=1, text="Page "+str(self.pagenumber)+"/"+str(self.vpframesneeded), bg=NICEBLUE, fg="black", font=("Avenir Next", 12), justify=CENTER,anchor=CENTER)
        self.pagecounter.grid(row=5, column=5, rowspan=1, columnspan=3, sticky=NSEW)
        self.previouspagebutton = Button(self.searchframe, text="<", width=1, height=1, bg=NICEBLUE, fg="black", font=("Avenir Next", 12), justify=CENTER,anchor=CENTER, command=lambda:self.previouspagevp())
        self.previouspagebutton.grid(row=5, column=8, rowspan=1, columnspan=1, sticky=NSEW)
        self.searchquerylabel = Label(self.searchframe, width=1, height=1, text=f"\"{name}\"", bg=PINK, fg="black", font=("Avenir Next", 12), justify=CENTER,anchor=CENTER)
        self.searchquerylabel.grid(row=5, column=1, rowspan=1, columnspan=4, sticky=NSEW)
        if name == "":
            self.searchquerylabel.config(text="All Registrants")
        self.nextpagebutton = Button(self.searchframe, width=1, height=1, text=">", bg=NICEBLUE, fg="black", font=("Avenir Next", 12), justify=CENTER,anchor=CENTER, command=lambda:self.nextpagevp())
        self.nextpagebutton.grid(row=5, column=9, rowspan=1, columnspan=1, sticky=NSEW)
        self.studentprofileimg = Image.open(r"Assets\managementsuite\viewparticipantswidgets\studentprofile360x80.png")
        self.studentprofile = ImageTk.PhotoImage(self.studentprofileimg.resize(
            (math.ceil(360 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.searchresultsframes = {}
        # print(f"The return from searching up {name} are {self.results}")
        # print(f"The number of students found using this name is {self.count}")
        for f in range(self.vpframesneeded):
            self.searchresultsframes[f] = Frame(self.searchframe, width=1, height=1, bg=WHITE)
            self.searchresultsframes[f].grid(row=6, column=1, rowspan=10, columnspan=9, sticky=NSEW)
            self.searchresultsframes[f].grid_propagate(False)
            for x in range(9):
                self.searchresultsframes[f].columnconfigure(x, weight=1, uniform="x")
                Label(self.searchresultsframes[f], bg=WHITE).grid(row=0, column=x, sticky=NSEW)
            for y in range(10):
                self.searchresultsframes[f].rowconfigure(y, weight=1, uniform="x")
                Label(self.searchresultsframes[f], bg=WHITE).grid(row=y, column=0, sticky=NSEW)
            self.searchresultsframes[f].grid_remove()
        for indexofdetails in range(self.count):
            self.results[indexofdetails]
            fullname = self.results[indexofdetails][0]
            icpassnumber = self.results[indexofdetails][1]
            email = self.results[indexofdetails][2]
            i = math.floor(indexofdetails/5)
            if i == 0:   
                Button(self.searchresultsframes[i], image=self.studentprofile, width=1, height=1,
                text= f"Student name = {fullname}\nIC/Pass No. = {icpassnumber}\nEmail = {email}", compound=CENTER, font=("Avenir Next", 12), fg="black", bg=WHITE,
                command=lambda x=(fullname,icpassnumber,email) :self.generate_eventlist(x)).grid(row=indexofdetails*2, column=0, rowspan=2, columnspan=9, sticky=NSEW)
            else:
                Button(self.searchresultsframes[i], image=self.studentprofile, width=1, height=1,
                text= f"Student name = {fullname}\nIC/Pass No. = {icpassnumber}\nEmail = {email}", compound=CENTER, font=("Avenir Next", 12), fg="black", bg=WHITE,
                command=lambda x=(fullname,icpassnumber,email) :self.generate_eventlist(x)).grid(row=(indexofdetails-5)*2, column=0, rowspan=2, columnspan=9, sticky=NSEW)
        self.searchresultsframes[0].grid()
        # self.searchresultsframe = Frame(self.searchframe, width=1, height=1, bg=WHITE)
        # self.searchresultsframe.grid(row=6, column=1, rowspan=11, columnspan=9, sticky=NSEW)
        # for x in range(9):
        #     self.searchresultsframe.columnconfigure(x, weight=1, uniform="x")
        #     Label(self.searchresultsframe, width=1, bg=WHITE).grid(row=0, column=x, sticky=NSEW)
        # for y in range(11):
        #     self.searchresultsframe.rowconfigure(y, weight=1, uniform="x")
        #     Label(self.searchresultsframe, width=1, bg=WHITE).grid(row=y, column=0, sticky=NSEW)
        # self.searchresultsframe.grid_propagate(False)
        #display results
        # first in the database, we need to check if the person has the same name, will return 
        #searches for registrants in the eventregistration table using the name parameter
        #if the name parameter is empty, returns all registrants
        #if the name parameter is not empty, returns all registrants with the approx-same name entry in the name parameter
        #Generic student profile result picture button 
        # for loop to generate buttons for each result
        # each button will have the name of the student, and the ic/passport number
        # when the button is clicked, it will open a new window with the student's profile
        # the student's profile will have the student's name, ic/passport number, email, and the events that they have registered for
        # the student's profile will also have a button to remove the student from the event
        # the student's profile will also have a button to edit the student's profile
        # this is the code

        # self.eventsregisteredforframebgorg = Image.open(r"Assets\managementsuite\viewparticipantswidgets\eventsregisteredforframebg.png")
        # self.eventsregisteredforframebg = ImageTk.PhotoImage(self.eventsregisteredforframebgorg.resize(
        #     (math.ceil(680 * dpi / 96), math.ceil(680 * dpi / 96)), Image.Resampling.LANCZOS))
        # self.eventsregisteredforframebglabel = Label(self.eventsregisteredforframe, image=self.eventsregisteredforframebg, width=1,height=1, bg=WHITE)
        # self.eventsregisteredforframebglabel.grid(row=0, column=0, rowspan=17, columnspan=17, sticky=NSEW)
    #~~~~~ v PAGE FUNCTIONS FOR REGISTRANTS RESULTS v ~~~~~
    def nextpagevp(self):
        if self.pagenumber < self.vpframesneeded:
            self.pagenumber += 1
            self.pagecounter.config(text=f"Page {self.pagenumber}/{self.vpframesneeded}")
            self.searchresultsframes[self.pagenumber-2].grid_remove()
            self.searchresultsframes[self.pagenumber-1].grid()
    def previouspagevp(self):
        if self.pagenumber > 1:
            self.pagenumber -= 1
            self.pagecounter.config(text=f"Page {self.pagenumber}/{self.vpframesneeded}")
            self.searchresultsframes[self.pagenumber].grid_remove()
            self.searchresultsframes[self.pagenumber-1].grid()
    #~~~~~~~ ^ PAGE FUNCTIONS FOR REGISTRANTS RESULTS ^ ~~~~~~~
    def generate_eventlist(self, information:tuple): #searches the eventregistration list to find all instances of name, ic, email and returns the events
        #database queries
        #in case somebody clicks a name while looking at edit page
        #~~~~~ Pg functions for events results ~~~~~
        self.pageeventcounter = 1
        #~~~~~~~ ^ Pg functions for events results ^ ~~~~~~~
        try:
            self.frametoshowdetails.grid_remove()
        except:
            pass
        try:
            for widget in self.frametoshowdetails.winfo_children():
                widget.destroy()
        except:
            pass
        self.conn = sqlite3.connect("interactivesystem.db")
        self.c = self.conn.cursor()
        #unpacking the tuple 
        self.fullname = information[0]
        self.icpassnumber = information[1]
        self.email = information[2]
        self.c.execute("SELECT COUNT(event_registered) FROM eventregistration WHERE full_name = ? AND icpass_number = ? AND email = ?", (self.fullname, self.icpassnumber, self.email))
        self.countgenerateevents = self.c.fetchone()[0]
        if self.countgenerateevents == 0:
            self.frametoshowdetails = Frame(self.searchframe, width=1, height=1, bg=WHITE)
            self.frametoshowdetails.grid(row=6, column=1, rowspan=11, columnspan=9, sticky=NSEW)
            for x in range(9):
                self.frametoshowdetails.columnconfigure(x, weight=1, uniform="x")
                Label(self.frametoshowdetails, width=1, bg=WHITE).grid(row=0, column=x, sticky=NSEW)
            for y in range(11):
                self.frametoshowdetails.rowconfigure(y, weight=1, uniform="x")
                Label(self.frametoshowdetails, width=1, bg=WHITE).grid(row=y, column=0, sticky=NSEW)
            self.frametoshowdetails.grid_propagate(False)
            # if the person has no events registered for, display this
            self.noeventsregisteredforframelabel = Label(self.frametoshowdetails, text=f"No events registered\nfor {self.fullname}\n Please search again. Sorry!", font=("Avenir Next Medium", 18), bg=WHITE, width=1, height=1)
            self.noeventsregisteredforframelabel.grid(row=0, column=0, rowspan=11, columnspan=9, sticky=NSEW)
            buttontoexit = Button(self.frametoshowdetails, text="Exit", font=("Avenir Next", 20), bg=OTHERPINK, width=1, height=1, command=lambda: self.frametoshowdetails.destroy())
            buttontoexit.grid(row=10, column=8, rowspan=1, columnspan=1, sticky=NSEW)
            return
        self.c.execute("SELECT event_registered FROM eventregistration WHERE full_name = ? AND icpass_number = ? AND email = ?", (self.fullname, self.icpassnumber, self.email))
        self.results = self.c.fetchall()
        print(f"This student with name {self.fullname} has registered for {self.results}")
        print(f"The number of events this student has registered for is {self.countgenerateevents}")
        self.readstudentimg = Image.open(r"Assets\managementsuite\viewparticipantswidgets\readstudentdetails.png")
        self.readstudent = ImageTk.PhotoImage(self.readstudentimg.resize(
            (math.ceil(80 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.deletestudentimg = Image.open(r"Assets\managementsuite\viewparticipantswidgets\deletestudent80x80.png")
        self.deletestudent = ImageTk.PhotoImage(self.deletestudentimg.resize(
            (math.ceil(80* dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.eventsregframesneeded = math.ceil(self.countgenerateevents/5)
        self.frameforpagebuttons = Frame(self.viewparticipants, width=1, height=1, bg=NICEBLUE)
        self.frameforpagebuttons.grid(row=3, column=22, rowspan=1, columnspan=15, sticky=NSEW)
        for hehe in range(15):
            self.frameforpagebuttons.columnconfigure(hehe, weight=1, uniform="x")
            Label(self.frameforpagebuttons, bg=NICEBLUE, width=1, height=1).grid(row=0, column=hehe, sticky=NSEW)
        for w in range(1):
            self.frameforpagebuttons.rowconfigure(w, weight=1, uniform="x")
            Label(self.frameforpagebuttons, bg=NICEBLUE, width=1, height=1).grid(row=w, column=0, sticky=NSEW)
        self.frameforpagebuttons.grid_propagate(False)
        self.showingthedetailsofstudent = Label(self.frameforpagebuttons, text=f"Showing details of {self.fullname}", bg=PINK, fg=BLACK, font=("Avenir Next Medium", 12), width=1, height=1)
        self.showingthedetailsofstudent.grid(row=0, column=0, columnspan=8, sticky=NSEW)
        self.pgnumlabelregisteredevents = Label(self.frameforpagebuttons, text=f"Page {self.pageeventcounter}/{self.eventsregframesneeded}", bg=NICEBLUE, fg=BLACK, font=("Avenir Next Medium", 12), width=1, height=1)
        self.pgnumlabelregisteredevents.grid(row=0, column=8, columnspan=3, sticky=NSEW)
        self.previouspgbutton = Button(self.frameforpagebuttons, text="<", bg=NICEBLUE, fg=WHITE, font=("Avenir Next Bold", 12), width=1, height=1, command=lambda:self.previouspageevents())
        self.previouspgbutton.grid(row=0, column=11, columnspan=2, rowspan=1, sticky=NSEW)
        self.nextpgbutton =  Button(self.frameforpagebuttons, text=">", bg=NICEBLUE, fg=WHITE, font=("Avenir Next Bold", 12), width=1, height=1, command=lambda:self.nextpageevents())
        self.nextpgbutton.grid(row=0, column=13, columnspan=2, rowspan=1, sticky=NSEW)

        self.eventsregisteredforframe = {}
        self.frametoshowdetails = {}
        for z in range(self.eventsregframesneeded):
            self.eventsregisteredforframe[z] = Frame(self.viewparticipants, width=1, height=1, bg=WHITE)
            self.eventsregisteredforframe[z].grid(row=4, column=22, rowspan=11, columnspan=15, sticky=NSEW)
            for x in range(15):
                self.eventsregisteredforframe[z].columnconfigure(x, weight=1, uniform="row")
                Label(self.eventsregisteredforframe[z], bg=WHITE, width=1).grid(row=0, column=x, sticky=NSEW)
            for y in range(11):
                self.eventsregisteredforframe[z].rowconfigure(y, weight=1, uniform="row")
                Label(self.eventsregisteredforframe[z], bg=WHITE, width=1).grid(row=y, column=0, sticky=NSEW)
            self.eventsregisteredforframe[z].grid_propagate(False)
            self.frametoshowdetails[z] = Frame(self.eventsregisteredforframe[z], bg=NICEBLUE, height=1,width=1)
            self.frametoshowdetails[z].grid(row=0, column=0, rowspan=11, columnspan=15, sticky=NSEW)
            self.frametoshowdetails[z].grid_propagate(False)
            for x in range(15):
                self.frametoshowdetails[z].columnconfigure(x, weight=1, uniform="x")
                Label(self.frametoshowdetails[z], bg=NICEBLUE, width=1).grid(row=0, column=x, sticky=NSEW)
            for y in range(11):
                self.frametoshowdetails[z].rowconfigure(y, weight=1, uniform="x")
                Label(self.frametoshowdetails[z], bg=NICEBLUE, width=1).grid(row=y, column=0, sticky=NSEW)
            self.frametoshowdetails[z].grid_remove()
            self.eventsregisteredforframe[z].grid_remove()
        #display results
        # basically presenting the events that the student has registered for
        #initializing the read student and delete student images 
        #deleting the previous widgets in the eventsregisteredforframe
        # for widget in self.eventsregisteredforframe.winfo_children():
        #     if widget.winfo_class() == "Button":
        #         widget.destroy()
        for indexofdetails in range(self.countgenerateevents):
            self.results[indexofdetails]
            eventregistered = self.results[indexofdetails][0]
            j = math.floor(indexofdetails/5)
            #for fun random colors
            #reset colors
            textcolor = BLACK
            randomcolor = WHITE
            randomcolor = random.choice([NAVYBLUE, PINK, NICEBLUE, OTHERPINK, ORANGE, LAVENDER, LIGHTPURPLE, DARKBLUE])
            if randomcolor in [NAVYBLUE, DARKBLUE]:
                textcolor = WHITE
            elif randomcolor in [PINK, NICEBLUE, OTHERPINK, ORANGE, LIGHTPURPLE, LAVENDER]:
                textcolor = BLACK
            if j == 0:
                Button(self.eventsregisteredforframe[j], state=DISABLED, width=1 , height=1, text=f"Event: {eventregistered}", font=("Avenir Next Medium", 14), disabledforeground=textcolor, bg=randomcolor).grid(row=indexofdetails*2, column=0, rowspan=2, columnspan=11, sticky=NSEW)
                Button(self.eventsregisteredforframe[j], image=self.readstudent,
                        width=1, height=1,
                        text= f"EDIT", compound=CENTER, font=("Avenir Next Medium", 12), fg=WHITE, bg=WHITE,
                        command=lambda x=eventregistered, y=self.fullname:self.read_student_details(x,y,0)).grid(row=indexofdetails*2, column=11, rowspan=2, columnspan=2, sticky=NSEW)
                Button(self.eventsregisteredforframe[j], image=self.deletestudent,
                        width=1, height=1,
                        text= f"DELETE", compound=CENTER, font=("Avenir Next Medium", 12), fg=WHITE, bg=WHITE,
                        command=lambda x=eventregistered, y=self.fullname:self.delete_student(x,y,0)).grid(row=indexofdetails*2, column=13, rowspan=2, columnspan=2, sticky=NSEW)
            else:
                Button(self.eventsregisteredforframe[j], state=DISABLED, width=1 , height=1, text=f"Event: {eventregistered}", font=("Avenir Next Medium", 14), disabledforeground=textcolor, bg=randomcolor).grid(row=(indexofdetails-5*j)*2, column=0, rowspan=2, columnspan=11, sticky=NSEW)
                Button(self.eventsregisteredforframe[j], image=self.readstudent,
                        width=1, height=1,
                        text= f"EDIT", compound=CENTER, font=("Avenir Next Medium", 12), fg=WHITE, bg=WHITE,
                        command=lambda x=eventregistered, y=self.fullname:self.read_student_details(x,y,j)).grid(row=(indexofdetails-5*j)*2, column=11, rowspan=2, columnspan=2, sticky=NSEW)
                Button(self.eventsregisteredforframe[j], image=self.deletestudent,
                        width=1, height=1,
                        text= f"DELETE", compound=CENTER, font=("Avenir Next Medium", 12), fg=WHITE, bg=WHITE,
                        command=lambda x=eventregistered, y=self.fullname:self.delete_student(x,y,j)).grid(row=(indexofdetails-5*j)*2, column=13, rowspan=2, columnspan=2, sticky=NSEW)
        #displaying the first page of the events registered for
        # self.frametoshowdetails[0].grid()
        self.eventsregisteredforframe[0].grid()
        
    def nextpageevents(self):
        print(self.pageeventcounter)
        if self.pageeventcounter < self.eventsregframesneeded:
            self.pageeventcounter += 1
            self.pgnumlabelregisteredevents.config(text=f"Page {self.pageeventcounter}/{self.eventsregframesneeded}")
            self.eventsregisteredforframe[self.pageeventcounter-2].grid_remove()
            self.eventsregisteredforframe[self.pageeventcounter-1].grid()
    def previouspageevents(self):
        print(self.pageeventcounter)
        if self.pageeventcounter > 1:
            self.pageeventcounter -= 1
            self.pgnumlabelregisteredevents.config(text=f"Page {self.pageeventcounter}/{self.eventsregframesneeded}")
            self.eventsregisteredforframe[self.pageeventcounter].grid_remove()
            self.eventsregisteredforframe[self.pageeventcounter-1].grid()


    def read_student_details(self, eventname, studentname, index):
        #this takes in the page counter, which ends up always being 1 more than the index of the frame
        print(f"Read {studentname}'s details for {eventname}")
        print("Current page is", index)
        self.conn = sqlite3.connect("interactivesystem.db")
        self.c = self.conn.cursor()
        self.c.execute("SELECT * FROM eventregistration WHERE full_name = ? AND event_registered = ?", (studentname, eventname))
        self.results = self.c.fetchall()
        if len(self.results) == 0:
            messagebox.showerror("Error", "No such student found, you may have deleted the student's details already")
            return
        removetheframe = Button(self.frametoshowdetails[index], width=1, height=1, text="X", font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, command=lambda:[self.frametoshowdetails[index].grid_remove(), deleteentry()])
        removetheframe.grid(row=0, column=14, sticky=NSEW)
        self.fullname = self.results[0][0]
        self.icpassnumber = self.results[0][1]
        self.phonenumb = self.results[0][2]
        self.email = self.results[0][3]
        self.address = self.results[0][4]
        self.fullnameentry = Entry(self.frametoshowdetails[index], width=1, font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, justify=CENTER)
        self.fullnameentry.grid(row=7, column=2, rowspan=2, columnspan=11, sticky=NSEW)
        self.icpassnoentry = Entry(self.frametoshowdetails[index], width=1, font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, justify=CENTER)
        self.icpassnoentry.grid(row=7, column=2, rowspan=2, columnspan=11, sticky=NSEW)
        self.phonenumentry = Entry(self.frametoshowdetails[index], width=1, font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, justify=CENTER)
        self.phonenumentry.grid(row=7, column=2, rowspan=2, columnspan=11, sticky=NSEW)
        self.emailentry = Entry(self.frametoshowdetails[index], width=1, font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, justify=CENTER)
        self.emailentry.grid(row=7, column=2, rowspan=2, columnspan=11, sticky=NSEW)
        self.addressentry = Entry(self.frametoshowdetails[index], width=1, font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, justify=CENTER)
        self.addressentry.grid(row=7, column=2, rowspan=2, columnspan=11, sticky=NSEW)
        entrylist = [self.fullnameentry, self.icpassnoentry, self.phonenumentry, self.emailentry, self.addressentry]
        self.currentlyeditinglabel = Button(self.frametoshowdetails[index], state=DISABLED, text=f"Editing info of: {self.fullname}", font=("Avenir Next", 12), fg=BLACK, bg=WHITE, width=1, height=1)
        # self.currentlyeditinglabel.grid(row=0, column=0, columnspan=14, sticky=NSEW)
        self.confirmbutton = Button(self.frametoshowdetails[index], text="Confirm", font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, command=lambda:[self.confirm_editing_details(studentname, eventname, index)])
        # self.confirmbutton.grid(row=9, column=2, columnspan=11, sticky=NSEW)
        self.currentlyeditinglabel.grid_remove()
        self.confirmbutton.grid_remove()
        def deleteentry():        
            for entry in entrylist:
                entry.grid_remove()
            for widget in self.frametoshowdetails[index].winfo_children():
                if widget.winfo_class() == "Button":
                    widget.grid_remove()
        # i am losing my sanity here
        deleteentry()
        removetheframe = Button(self.frametoshowdetails[index], width=1, height=1, text="X", font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, command=lambda:[self.frametoshowdetails[index].grid_remove(), deleteentry()])
        removetheframe.grid(row=0, column=14, sticky=NSEW)
        self.frametoshowdetails[index].grid()
        self.frametoshowdetails[index].tkraise()
        self.frametoshowdetails[index].grid_propagate(False)
        #label to show current event 
        self.eventnamelabel = Label(self.frametoshowdetails[index], text=f"Details of {self.fullname} in the event: {eventname}", font=("Avenir Next Medium", 16), fg=BLACK, bg=WHITE, width=1, height=1)
        self.eventnamelabel.grid(row=0, column=0, columnspan=14, sticky=NSEW)
        fullnamebutton = Button(self.frametoshowdetails[index], text=f"Full name: {self.fullname}", font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, command=lambda:self.entryinitializer(self.fullnameentry, self.fullname, fieldchanged="full_name",  originaltext=self.fullname, eventregistered=eventname, currentindex=index) ).grid(row=1, column=1, columnspan=13, sticky=NSEW)
        icpassnumberbutton = Button(self.frametoshowdetails[index], text=f"IC/Passport number: {self.icpassnumber}", font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, command=lambda:self.entryinitializer(self.icpassnoentry, self.icpassnumber, fieldchanged="icpass_number", originaltext=self.icpassnumber, eventregistered=eventname, currentindex=index)).grid(row=2, column=1, columnspan=13, sticky=NSEW)
        phonenumbbutton = Button(self.frametoshowdetails[index], text=f"Phone number: {self.phonenumb}", font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, command=lambda:self.entryinitializer(self.phonenumentry, self.phonenumb, fieldchanged="phone_number", originaltext=self.phonenumb, eventregistered=eventname, currentindex=index)).grid(row=3, column=1, columnspan=13, sticky=NSEW)
        emailbutton = Button(self.frametoshowdetails[index], text=f"Email: {self.email}", font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, command=lambda:self.entryinitializer(self.emailentry, self.email, fieldchanged="email", originaltext=self.email, eventregistered=eventname, currentindex=index)).grid(row=4, column=1, columnspan=13, sticky=NSEW)
        addressbutton = Button(self.frametoshowdetails[index], text=f"Address: {self.address}", font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, command=lambda:self.entryinitializer(self.addressentry, self.address, fieldchanged="address", originaltext=self.address, eventregistered=eventname, currentindex=index)).grid(row=5, column=1, columnspan=13, sticky=NSEW)

    def entryinitializer(self, entrywanted, texttochange,  **kwargs):
        fieldchanged = kwargs.get("fieldchanged")
        originaltext = kwargs.get("originaltext")
        eventregistered = kwargs.get("eventregistered")
        currentindex = kwargs.get("currentindex")
        #these entries all occupy the same place, only grid() when called upon.
        #remove all entries when called upon
        entrylist = [self.fullnameentry, self.icpassnoentry, self.phonenumentry, self.emailentry, self.addressentry]
        for entry in entrylist:
            if entry != entrywanted:
                entry.grid_remove()
        #grid the entry wanted
        entrywanted.grid()
        entrywanted.delete(0, END)
        entrywanted.insert(0, texttochange)
        entrywanted.focus_set()
        #confirm button
        #currently editing label

        self.currentlyeditinglabel = Button(self.frametoshowdetails[currentindex],state=DISABLED, text=f"You are currently editing the {fieldchanged} of the student, {self.fullname}", font=("Avenir Next", 12), fg=BLACK, bg=WHITE, width=1, height=1)
        self.currentlyeditinglabel.grid(row=6, column=1, columnspan=13, sticky=NSEW)

        self.confirmbutton = Button(self.frametoshowdetails[currentindex], text=f"Confirm + {fieldchanged} ", font=("Avenir Next Bold", 16), fg=WHITE, bg=NAVYBLUE, command=lambda:self.confirmchanges(entrywanted, texttochange, fieldchanged, originaltext, eventregistered))
        self.confirmbutton.grid(row=9, column=2, columnspan=11, sticky=NSEW)
        # self.confirmbutton = Button(self.frametoshowdetails[currentindex], text="Confirm Edit", font=("Avenir Next Bold", 16), fg=WHITE, bg=NAVYBLUE, command=lambda:self.confirmchanges(entrywanted, texttochange, fieldchanged, originaltext, eventregistered)).grid(row=9, column=2, columnspan=11, sticky=NSEW)
    def confirmchanges(self, entrytogetinfo, texttochange, fieldchanged, originaltext, event_registered):
        #get the text from the entry
        textfromentry = entrytogetinfo.get()
        #update the database
        self.conn = sqlite3.connect("interactivesystem.db")
        self.c = self.conn.cursor()
        with self.conn:
            try:
                self.c.execute(f"UPDATE eventregistration SET {fieldchanged} = ? WHERE {fieldchanged} = ? AND event_registered = ?", (textfromentry, originaltext, event_registered))
                messagebox.showinfo("Success", f"Changes have been made, where {fieldchanged} = {originaltext} has been changed to {textfromentry} under event {event_registered}")
            except Exception as e:
                messagebox.showerror("Error", f"An error has occured: {e}")

    # def edit_student_details(self, details):
    #     def update_details():
    #         self.conn = sqlite3.connect("interactivesystem.db")
    #         self.c = self.conn.cursor()
    #         with self.conn:
    #             self.c.execute("UPDATE eventregistration SET full_name = ?, ic_passport_number = ?, phone_number = ?, email = ?, address = ? WHERE full_name = ?", (self.fullname, self.icpassnumber, self.phonenumb, self.email, self.address, details))
    
    def delete_student(self, eventname, studentname, pagenumber):
        print(f"Delete {studentname}'s details for {eventname}")
        print("Current page number", pagenumber)
        #Check if student still exists in the database and if not, return an error message
        self.conn = sqlite3.connect("interactivesystem.db")
        self.c = self.conn.cursor()
        with self.conn:
            self.c.execute("SELECT * FROM eventregistration WHERE full_name = ? AND event_registered = ?", (studentname, eventname))
            if self.c.fetchone() == None:
                messagebox.showerror("Error", "Student does not exist in the database, you may have already deleted the student")
                return
        self.frametodeletestudent = Frame(self.eventsregisteredforframe[pagenumber], height=1, width=1,bg=PINK, relief=SOLID)
        self.frametodeletestudent.grid(row=0, column=0, rowspan=11, columnspan=15, sticky=NSEW)
        self.frametodeletestudent.grid_propagate(False)
        for x in range(15):
            self.frametodeletestudent.columnconfigure(x, weight=1)
            Label(self.frametodeletestudent, width=1, bg=PINK).grid(row=0, column=x, sticky=NSEW)
        for y in range(11):
            self.frametodeletestudent.rowconfigure(y, weight=1)
            Label(self.frametodeletestudent, width=1, bg=PINK).grid(row=y, column=0, sticky=NSEW)
        Label(self.frametodeletestudent, text=f"Are you sure you want to delete {studentname}'s details for \n{eventname}?", font=("Avenir Next", 14), fg=BLACK, bg=PINK).grid(row=1, column=1, columnspan=13, sticky=NSEW)
        Label(self.frametodeletestudent, text="This action cannot be undone.", font=("Avenir Next", 14), fg=BLACK, bg=PINK).grid(row=2, column=1, columnspan=13, sticky=NSEW)
        Label(self.frametodeletestudent, text="Please enter the word DELETE to confirm.", font=("Avenir Next", 14), fg=BLACK, bg=PINK).grid(row=3, column=1, columnspan=13, sticky=NSEW)
        self.deleteentry = Entry(self.frametodeletestudent, font=("Avenir Next", 14), fg=BLACK, bg=WHITE, justify=CENTER)
        self.deleteentry.grid(row=4, column=1, columnspan=13, sticky=NSEW)
        self.deleteentry.focus_set()
        Button(self.frametodeletestudent, text="Confirm", font=("Avenir Next", 14), fg=WHITE, bg=NAVYBLUE, command=lambda:self.delete_student_confirmed(eventname, studentname)).grid(row=5, column=1, columnspan=13, sticky=NSEW)
        Button(self.frametodeletestudent, text="Cancel", font=("Avenir Next", 14), fg=WHITE, bg=NAVYBLUE, command=lambda:self.frametodeletestudent.grid_remove()).grid(row=6, column=1, columnspan=13, sticky=NSEW)
    def delete_student_confirmed(self, eventname, studentname):
        if self.deleteentry.get() == "DELETE":
            self.conn = sqlite3.connect("interactivesystem.db")
            self.c = self.conn.cursor()
            with self.conn:
                try:
                    self.c.execute("DELETE FROM eventregistration WHERE event_registered = ? AND full_name = ?", (eventname, studentname))
                    messagebox.showinfo("Success", f"{studentname}'s details for\n{eventname} has been deleted.")
                    self.frametodeletestudent.grid_remove()
                except Exception as e:
                    messagebox.showerror("Error", f"An error has occured: {e}")
        else:
            messagebox.showerror("Error", "The word DELETE was not entered.")

    def show_searchevents(self):
        self.searchbyeventsframe.tkraise()

        # ~~~~~ IMAGES ~~~~~
        self.searcheventsorg = Image.open(r"Assets\managementsuite\viewparticipantswidgets\vpsearcheventsbg.png")
        self.searcheventsbg = ImageTk.PhotoImage(self.searcheventsorg.resize(
            (math.ceil(1520 * dpi / 96), math.ceil(680 * dpi / 96)), Image.Resampling.LANCZOS))
        self.searcheventsbglabel = Label(self.searchbyeventsframe, image=self.searcheventsbg, width=1, height=1)
        self.searcheventsbglabel.grid(row=0, column=0, rowspan=17, columnspan=38, sticky=NSEW)
        self.searchregistrantsbutton = Button(self.searchbyeventsframe, image=self.searchregistrants, width=1, height=1, command=lambda:self.viewparticipants.tkraise())
        self.searchregistrantsbutton.grid(row=2, column=2, rowspan=2, columnspan=7, sticky=NSEW)
        self.searcheventsbutton = Button(self.searchbyeventsframe, image=self.searchevents, width=1, height=1,relief=SOLID, bd=4, highlightthickness=1, highlightbackground=LIGHTPURPLE, command=lambda:self.searchbyeventsframe.grid())
        self.searcheventsbutton.grid(row=6, column=2, rowspan=2, columnspan=7, sticky=NSEW)
        self.backbutton = Button(self.searchbyeventsframe, image=self.back, width=1, height=1, relief=SOLID, bd=4, highlightthickness=1, highlightbackground=LIGHTPURPLE, command=lambda:[self.interfaceframe.tkraise(), self.searchbyeventsframe.grid_remove()])
        self.backbutton.grid(row=10, column=2, rowspan=2, columnspan=7, sticky=NSEW)
        self.searcheventsframe = Frame(self.searchbyeventsframe, height=1, width=1, bg=PINK)
        self.searcheventsframe.grid(row=0, column=10, rowspan=17, columnspan=11, sticky=NSEW)
        for x in range(11):
            self.searcheventsframe.columnconfigure(x, weight=1,uniform="x")
            Label(self.searcheventsframe, width=1, bg=PINK).grid(row=0, column=x, sticky=NSEW)
        for y in range(17):
            self.searcheventsframe.rowconfigure(y, weight=1,uniform="x")
            Label(self.searcheventsframe, width=1, bg=PINK).grid(row=y, column=0, sticky=NSEW)
        self.searcheventsframe.grid_propagate(0)
        self.searcheventsframebgorg = Image.open(r"Assets\managementsuite\viewparticipantswidgets\searcheventsframebg.png")
        self.searcheventsframebg = ImageTk.PhotoImage(self.searcheventsframebgorg.resize(
            (math.ceil(440 * dpi / 96), math.ceil(680 * dpi / 96)), Image.Resampling.LANCZOS))
        self.searcheventsframebglabel = Label(self.searcheventsframe, image=self.searcheventsframebg, width=1, height=1)
        self.searcheventsframebglabel.grid(row=0, column=0, rowspan=17, columnspan=11, sticky=NSEW)
        self.searchbuttonimg1 = Image.open(r"Assets\managementsuite\viewparticipantswidgets\magnifyingbutton80x80.png")
        self.searchbutton1 = ImageTk.PhotoImage(self.searchbuttonimg.resize(
            (math.ceil(80 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.searcheventbutton = Button(self.searcheventsframe, image=self.searchbutton1, width=1, height=1, command=lambda:self.searchevents_function(eventsearchentry.get())) 
        self.searcheventbutton.grid(row=1, column=8, rowspan=2, columnspan=2, sticky=NSEW)
        self.searcheventbutton.grid_propagate(False)
        eventsearchentry =  Entry(self.searcheventsframe, width=1, font=("Avenir Next", 12), fg=BLACK, bg=WHITE)
        eventsearchentry.grid(row=2, column=1, rowspan=1, columnspan=6, sticky=NSEW)
        eventsearchentry.grid_propagate(False)
    def searchevents_function(self, eventname): #name is eventsearchentry.get()
        self.conn = sqlite3.connect("interactivesystem.db")
        self.c = self.conn.cursor()
        if eventname == "":
            self.c.execute("""SELECT DISTINCT event_name, eventkey_number FROM eventcreation""")
        else:
            self.c.execute("""SELECT DISTINCT event_name, eventkey_number FROM eventcreation WHERE event_name LIKE ?  """, ("%"+eventname+"%",))
        self.eresults = self.c.fetchall()
        self.ecount = len(self.eresults)
        if self.ecount == 0:
            messagebox.showerror("Error", "No events found.")
            return
        #in case somebody clicks an event while looking at edit page
        # later
        self.seventsframesneeded = math.ceil(self.ecount/5)
        self.seventspagenum = 1
        # Page Number Widgets For Search Events
        self.eventspagecounter = Label(self.searcheventsframe, text=f"{self.seventspagenum}/{self.seventsframesneeded}", font=("Avenir Next", 12), fg=BLACK, bg=PINK)
        self.eventspagecounter.grid(row=5, column=5, rowspan=1, columnspan=3, sticky=NSEW)
        self.prevpagebtn = Button(self.searcheventsframe, text="<", width=1, height=1, fg=BLACK, font=("Avenir Next", 12), justify=CENTER, anchor=CENTER, command=lambda:self.prevpageeventssearch())
        self.prevpagebtn.grid(row=5, column=8, rowspan=1, columnspan=1, sticky=NSEW)
        self.nextpagebtn = Button(self.searcheventsframe, text=">", width=1, height=1, fg=BLACK, font=("Avenir Next", 12), justify=CENTER, anchor=CENTER, command=lambda:self.nextpageeventssearch())
        self.nextpagebtn.grid(row=5, column=9, rowspan=1, columnspan=1, sticky=NSEW)
        self.searchqueryeventlabel = Label(self.searcheventsframe, text=f"\"{eventname}\"", font=("Avenir Next", 12),width=1, height=1,justify=CENTER, anchor=CENTER, fg=BLACK, bg=PINK)
        self.searchqueryeventlabel.grid(row=5, column=1, rowspan=1, columnspan=4, sticky=NSEW)
        self.eventprofileimg = Image.open(r"Assets\managementsuite\viewparticipantswidgets\eventprofile360x80.png")
        self.eventprofile = ImageTk.PhotoImage(self.eventprofileimg.resize(
            (math.ceil(360 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        if eventname == "":
            self.searchqueryeventlabel.config(text="All Events")
        self.searcheventsresultsframe = {}
        for f in range(self.seventsframesneeded):
            self.searcheventsresultsframe[f] = Frame(self.searcheventsframe, height=1, width=1, bg=PINK)
            self.searcheventsresultsframe[f].grid(row=6, column=1, rowspan=10, columnspan=9, sticky=NSEW)
            self.searcheventsresultsframe[f].grid_propagate(False)
            for x in range(9):
                self.searcheventsresultsframe[f].columnconfigure(x, weight=1, uniform="x")
                Label(self.searcheventsresultsframe[f], width=1, bg=PINK).grid(row=0, column=x, sticky=NSEW)
            for y in range(10):
                self.searcheventsresultsframe[f].rowconfigure(y, weight=1, uniform="x")
                Label(self.searcheventsresultsframe[f], width=1, bg=PINK).grid(row=y, column=0, sticky=NSEW)
            self.searcheventsresultsframe[f].grid_remove()
        for indexofeventdetails in range(self.ecount):
            self.eresults[indexofeventdetails]
            nameofevent = self.eresults[indexofeventdetails][0]
            eventkey = self.eresults[indexofeventdetails][1]
            k = math.floor(indexofeventdetails/5)
            if k == 0:
                Button(self.searcheventsresultsframe[k], image=self.eventprofile, width=1, height=1, 
                text= f"Event: {nameofevent}\nEvent key:{eventkey}", compound=CENTER, font=("Avenir Next", 12),
                fg=BLACK, bg=WHITE, wraplength=250, justify=CENTER,
                command=lambda x=(nameofevent, eventkey):self.generate_studentlist(x)).grid(row=indexofeventdetails*2, column=0, rowspan=2, columnspan=9, sticky=NSEW)
            else:
                Button(self.searcheventsresultsframe[k], image=self.eventprofile, width=1, height=1, 
                text= f"Event: {nameofevent}\nEvent key: {eventkey}", compound=CENTER, font=("Avenir Next", 12),
                fg=BLACK, bg=WHITE, wraplength=250, justify=CENTER,
                command=lambda x=(nameofevent, eventkey):self.generate_studentlist(x)).grid(row=(indexofeventdetails-5*k)*2, column=0, rowspan=2, columnspan=9, sticky=NSEW)
        self.searcheventsresultsframe[0].grid()
        #this generates the number of frames needed for the number of data in multiples of 5
    #~~~~~ v PAGE FUNCTIONS FOR EVENT RESULTS v ~~~~~
    def nextpageeventssearch(self):
        if self.seventspagenum < self.seventsframesneeded:
            self.seventspagenum += 1
            self.eventspagecounter.config(text=f"Page {self.seventspagenum}/{self.seventsframesneeded}")
            self.searcheventsresultsframe[self.seventspagenum-2].grid_remove()
            self.searcheventsresultsframe[self.seventspagenum-1].grid()
    def prevpageeventssearch(self):
        if self.seventspagenum > 1:
            self.seventspagenum -= 1
            self.eventspagecounter.config(text=f"Page {self.seventspagenum}/{self.seventsframesneeded}")
            self.searcheventsresultsframe[self.seventspagenum].grid_remove()
            self.searcheventsresultsframe[self.seventspagenum-1].grid()
    #~~~~~ ^ PAGE FUNCTIONS FOR EVENT RESULTS ^ ~~~~~
    def generate_studentlist(self, information:tuple):
        try:
            self.frametoshowdetailsevent.grid_remove()
        except:
            pass
        self.pagestudentcounter = 1
        try:
            self.frametoshowdetailsevent.grid_remove()
        except:
            pass
        try:
            for widget in self.frametoshowdetailsevent.winfo_children():
                widget.destroy()
        except:
            pass
        self.conn = sqlite3.connect("interactivesystem.db")
        self.c = self.conn.cursor()
        #unpacking the tuple
        self.nameofevent = information[0]
        self.eventkey = information[1]
        self.c.execute("""SELECT COUNT(full_name) FROM eventregistration where eventkey_registered=?""", (self.eventkey,))
        self.countofstudentsregistered = self.c.fetchone()[0]
        if self.countofstudentsregistered == 0:
            self.frametoshowdetailsevent = Frame(self.searcheventsframe, height=1, width=1, bg=PINK)
            self.frametoshowdetailsevent.grid(row=6, column=1, rowspan=10, columnspan=9, sticky=NSEW)
            self.frametoshowdetailsevent.grid_propagate(False)
            for x in range(9):
                self.frametoshowdetailsevent.columnconfigure(x, weight=1, uniform="x")
                Label(self.frametoshowdetailsevent, width=1, bg=PINK).grid(row=0, column=x, sticky=NSEW)
            for y in range(10):
                self.frametoshowdetailsevent.rowconfigure(y, weight=1, uniform="x")
                Label(self.frametoshowdetailsevent, width=1, bg=PINK).grid(row=y, column=0, sticky=NSEW)
            self.frametoshowdetailsevent.grid()
            self.nostudentsregisteredlabel = Label(self.frametoshowdetailsevent, text=f"No students registered\n for this event\n{self.nameofevent}", font=("Avenir Next", 18), width=1, height=1, justify=CENTER, anchor=CENTER, fg=BLACK, bg=OTHERPINK, wraplength=300)
            self.nostudentsregisteredlabel.grid(row=0, column=0, rowspan=9, columnspan=9, sticky=NSEW)
            exitbutton = Button(self.frametoshowdetailsevent, text="X", font=("Avenir Next Bold", 18), width=1, height=1, command=lambda:self.frametoshowdetailsevent.destroy(), bg=NAVYBLUE, fg=WHITE)
            exitbutton.grid(row=0, column=8, rowspan=1, columnspan=1, sticky=NSEW)
            return

        self.c.execute("""SELECT full_name, icpass_number, phone_number FROM eventregistration where eventkey_registered=?""", (self.eventkey,))
        self.results = self.c.fetchall()
        print(f"This event with name {self.nameofevent} has the registrants of {self.results}")
        print(f"Total number of students registered for this event is {self.countofstudentsregistered}")
        #deleting the previous widgets in the studentsregisteredframe
        self.readstudentimg = Image.open(r"Assets\managementsuite\viewparticipantswidgets\readstudentdetails.png")
        self.readstudent_ = ImageTk.PhotoImage(self.readstudentimg.resize(
            (math.ceil(80 * dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        self.deletestudentimg = Image.open(r"Assets\managementsuite\viewparticipantswidgets\deletestudent80x80.png")
        self.deletestudent_ = ImageTk.PhotoImage(self.deletestudentimg.resize(
            (math.ceil(80* dpi / 96), math.ceil(80 * dpi / 96)), Image.Resampling.LANCZOS))
        # for widget in self.studentsregisteredforframe.winfo_children():
        #     if widget.winfo_class() == "Button":
        #         widget.destroy()
        self.studentsregframesneeded = math.ceil(self.countofstudentsregistered/5)
        self.frameforpgbuttonsstudents = Frame(self.searchbyeventsframe, width=1,height=1, bg=WHITE)
        self.frameforpgbuttonsstudents.grid(row=3, column=22, rowspan=1, columnspan=15, sticky=NSEW)
        for _ in range(15):
            self.frameforpgbuttonsstudents.columnconfigure(_, weight=1, uniform="x")
            Label(self.frameforpgbuttonsstudents, width=1, height=1, bg=WHITE).grid(row=0, column=_, sticky=NSEW)
        for w in range(1):
            self.frameforpgbuttonsstudents.rowconfigure(w, weight=1, uniform="x")
            Label(self.frameforpgbuttonsstudents, width=1, height=1, bg=WHITE).grid(row=w, column=0, sticky=NSEW)
        self.frameforpgbuttonsstudents.grid_propagate(False)
        self.showingthedetailsofstudent_ = Label(self.frameforpgbuttonsstudents, text=f"Showing details of {self.nameofevent}", bg=PINK, fg=BLACK, font=("Avenir Next Medium", 14), width=1, height=1, wraplength= 300, justify=LEFT)
        self.showingthedetailsofstudent_.grid(row=0, column=0, columnspan=8, sticky=NSEW)
        self.pgnumlabelregisteredevents_ = Label(self.frameforpgbuttonsstudents, text=f"Page {self.pagestudentcounter}/{self.studentsregframesneeded}", bg=NICEBLUE, fg=BLACK, font=("Avenir Next Medium", 12), width=1, height=1)
        self.pgnumlabelregisteredevents_.grid(row=0, column=8, columnspan=3, sticky=NSEW)
        self.previouspgbutton_ = Button(self.frameforpgbuttonsstudents, text="<", bg=NICEBLUE, fg=WHITE, font=("Avenir Next Bold", 12), width=1, height=1, command=lambda:self.previouspagestudents())
        self.previouspgbutton_.grid(row=0, column=11, columnspan=2, rowspan=1, sticky=NSEW)
        self.nextpgbutton_ =  Button(self.frameforpgbuttonsstudents, text=">", bg=NICEBLUE, fg=WHITE, font=("Avenir Next Bold", 12), width=1, height=1, command=lambda:self.nextpagestudents())
        self.nextpgbutton_.grid(row=0, column=13, columnspan=2, rowspan=1, sticky=NSEW)

        self.studentsregisteredforframe = {}
        self.frametoshowdetailsevent = {}
        #destroying the previous frames in the studentsregisteredforframe
        # try:
        #     for frame in range(len(self.studentsregisteredforframe)):
        #         self.studentsregisteredforframe[frame].destroy()
        # except:
        #     pass
        # for frame in range(len(self.studentsregisteredforframe)):
        #         print(frame)
        for z in range(self.studentsregframesneeded):
            self.studentsregisteredforframe[z] = Frame(self.searchbyeventsframe, height=1, width=1, bg=PINK)
            self.studentsregisteredforframe[z].grid(row=4, column=22, rowspan=11, columnspan=15, sticky=NSEW)
            for x in range(15):
                self.studentsregisteredforframe[z].columnconfigure(x, weight=1, uniform="x")
                Label(self.studentsregisteredforframe[z], width=1, bg=PINK).grid(row=0, column=x, sticky=NSEW)
            for y in range(11):
                self.studentsregisteredforframe[z].rowconfigure(y, weight=1, uniform="x")
                Label(self.studentsregisteredforframe[z], width=1, bg=PINK).grid(row=y, column=0, sticky=NSEW)
            self.studentsregisteredforframe[z].grid_propagate(False)
            self.frametoshowdetailsevent[z] = Frame(self.studentsregisteredforframe[z], height=1, width=1, bg=PINK)
            self.frametoshowdetailsevent[z].grid(row=0,column=0,rowspan=11,columnspan=15,sticky=NSEW)
            self.frametoshowdetailsevent[z].grid_propagate(False)
            for x in range(15):
                self.frametoshowdetailsevent[z].columnconfigure(x, weight=1,uniform="x")
                Label(self.frametoshowdetailsevent[z], width=1, bg=PINK).grid(row=0, column=x, sticky=NSEW)
            for y in range(11):
                self.frametoshowdetailsevent[z].rowconfigure(y, weight=1,uniform="x")
                Label(self.frametoshowdetailsevent[z], width=1, bg=PINK).grid(row=y, column=0, sticky=NSEW)
            self.frametoshowdetailsevent[z].grid_remove()
            self.studentsregisteredforframe[z].grid_remove()


        for indexofstudentdetails in range(self.countofstudentsregistered):
            self.results[indexofstudentdetails]
            nameofstudent = self.results[indexofstudentdetails][0]
            icpass = self.results[indexofstudentdetails][1]
            phonenumber = self.results[indexofstudentdetails][2]
            j = math.floor(indexofstudentdetails/5)
            if j == 0:
                Button(self.studentsregisteredforframe[j], state=DISABLED, width=1,height=1,text= f"Name = {nameofstudent}\nIC/Passport = {icpass}\nPhone number = {phonenumber}", font=("Avenir Next", 14), fg=BLACK, bg=PINK).grid(row=indexofstudentdetails*2, column=0, rowspan=2, columnspan=11, sticky=NSEW)
                Button(self.studentsregisteredforframe[j], image=self.readstudent_, width=1, height=1,
                text="EDIT", compound=CENTER, font=("Avenir Next Medium", 12), fg=WHITE,bg=WHITE,
                command=lambda x = self.nameofevent, y=nameofstudent: self.read_student_dtlsevnt(x, y, 0)).grid(row=indexofstudentdetails*2, column=11, rowspan=2, columnspan=2, sticky=NSEW)
                Button(self.studentsregisteredforframe[j], image=self.deletestudent_, width=1, height=1,
                text="DELETE", compound=CENTER, font=("Avenir Next Medium", 12), fg=WHITE,bg=WHITE,
                command=lambda x = self.nameofevent, y=nameofstudent: self.delete_studentevent(x, y, 0)).grid(row=indexofstudentdetails*2, column=13, rowspan=2, columnspan=2, sticky=NSEW)
            else:
                Button(self.studentsregisteredforframe[j], state=DISABLED, width=1,height=1,text= f"Name = {nameofstudent}\nIC/Passport = {icpass}\nPhone number = {phonenumber}", font=("Avenir Next", 14), fg=BLACK, bg=PINK).grid(row=(indexofstudentdetails-5*j)*2, column=0, rowspan=2, columnspan=11, sticky=NSEW)
                Button(self.studentsregisteredforframe[j], image=self.readstudent_, width=1, height=1,
                text="EDIT", compound=CENTER, font=("Avenir Next Medium", 12), fg=WHITE,bg=WHITE,
                command=lambda x = self.nameofevent, y=nameofstudent: self.read_student_dtlsevnt(x, y, j)).grid(row=(indexofstudentdetails-5*j)*2, column=11, rowspan=2, columnspan=2, sticky=NSEW)
                Button(self.studentsregisteredforframe[j], image=self.deletestudent_, width=1, height=1,
                text="DELETE", compound=CENTER, font=("Avenir Next Medium", 12), fg=WHITE,bg=WHITE,
                command=lambda x = self.nameofevent, y=nameofstudent: self.delete_studentevent(x, y, j)).grid(row=(indexofstudentdetails-5*j)*2, column=13, rowspan=2, columnspan=2, sticky=NSEW)
        try:
            self.studentsregisteredforframe[0].grid()
        except KeyError:
            #the problem is that when no students are registered, the dictionary is empty and the program crashes
            pass
            messagebox.showinfo("No students registered for this event", "No students registered for this event")
            
            
    def previouspagestudents(self):
        if self.pagestudentcounter > 1:
            self.pagestudentcounter -= 1
            self.pgnumlabelregisteredevents_.config(text=f"Page {self.pagestudentcounter} of {self.studentsregframesneeded}")
            self.studentsregisteredforframe[self.pagestudentcounter].grid_remove()
            self.studentsregisteredforframe[self.pagestudentcounter-1].grid()
    def nextpagestudents(self):
        if self.pagestudentcounter < self.studentsregframesneeded:
            self.pagestudentcounter += 1
            self.pgnumlabelregisteredevents_.config(text=f"Page {self.pagestudentcounter} of {self.studentsregframesneeded}")
            self.studentsregisteredforframe[self.pagestudentcounter-2].grid_remove()
            self.studentsregisteredforframe[self.pagestudentcounter-1].grid()

    def read_student_dtlsevnt(self, nameofevent, nameofstudent, index):
        print(f"Name of event is {nameofevent} and name of student is {nameofstudent}")
        self.conn = sqlite3.connect("interactivesystem.db")
        self.c = self.conn.cursor()
        self.c.execute("""SELECT * FROM eventregistration where full_name=? AND event_registered =? """, (nameofstudent, nameofevent))
        self.results = self.c.fetchall()
        removeeventframe = Button(self.frametoshowdetailsevent[index], width=1, height=1, text="X", font=("Avenir Next Bold", 16),
        fg=BLACK, bg=PINK, command=lambda:[self.frametoshowdetailsevent[index].grid_remove(), deleteentry()])
        removeeventframe.grid(row=0, column=14, sticky=NSEW)
        self.fullname = self.results[0][0]
        self.icpassnumber = self.results[0][1]
        self.phonenumber = self.results[0][2]
        self.email = self.results[0][3]
        self.address = self.results[0][4]
        self.eventfullnameentry = Entry(self.frametoshowdetailsevent[index], width=1, font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, justify=CENTER)
        self.eventfullnameentry.grid(row=7, column=2, rowspan=2, columnspan=11, sticky=NSEW)
        self.eventicpassnoentry = Entry(self.frametoshowdetailsevent[index], width=1, font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, justify=CENTER)
        self.eventicpassnoentry.grid(row=7, column=2, rowspan=2, columnspan=11, sticky=NSEW)
        self.eventphonenumberentry = Entry(self.frametoshowdetailsevent[index], width=1, font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, justify=CENTER)
        self.eventphonenumberentry.grid(row=7, column=2, rowspan=2, columnspan=11, sticky=NSEW)
        self.eventemailentry = Entry(self.frametoshowdetailsevent[index], width=1, font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, justify=CENTER)
        self.eventemailentry.grid(row=7, column=2, rowspan=2, columnspan=11, sticky=NSEW)
        self.eventaddressentry = Entry(self.frametoshowdetailsevent[index], width=1, font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, justify=CENTER)
        self.eventaddressentry.grid(row=7, column=2, rowspan=2, columnspan=11, sticky=NSEW)
        entrylist = [self.eventfullnameentry, self.eventicpassnoentry, self.eventphonenumberentry, self.eventemailentry, self.eventaddressentry]
        for entry in entrylist:
            entry.grid_remove()
        self.currentlyeditinglabel_ = Button(self.frametoshowdetailsevent[index], state=DISABLED, text=f"You are currently editing the student, {self.fullname}", font=("Avenir Next", 14), fg=BLACK, bg=WHITE, width=1, height=1)
        self.confirmbutton_ = Button(self.frametoshowdetailsevent[index], text="Confirm", font=("Avenir Next Bold", 16), fg=BLACK, bg=WHITE, command=lambda:[self.confirm_editing_details(nameofstudent, nameofevent, index)])
        self.currentlyeditinglabel_.grid_remove()
        self.confirmbutton_.grid_remove()
        def deleteentry():
            for entry in entrylist:
                entry.grid_remove()
            for widget in self.frametoshowdetailsevent[index].winfo_children():
                if widget.winfo_class() == "Button":
                    widget.grid_remove()
        deleteentry()
        removeeventframe = Button(self.frametoshowdetailsevent[index], width=1, height=1, text="X", font=("Avenir Next Bold", 16),
        fg=BLACK, bg=PINK, command=lambda:[self.frametoshowdetailsevent[index].grid_remove(), deleteentry()])
        removeeventframe.grid(row=0, column=14, sticky=NSEW)
        self.frametoshowdetailsevent[index].grid()
        self.frametoshowdetailsevent[index].tkraise()

        self.studentnamelabel = Label(self.frametoshowdetailsevent[index], text=f"The details for student {self.fullname} in event: {nameofevent}", font=("Avenir Next Medium", 14), fg=BLACK, bg=WHITE, width=1, height=1)
        self.studentnamelabel.grid(row=0, column=0, rowspan=1, columnspan=14, sticky=NSEW)
        fullnamebutton = Button(self.frametoshowdetailsevent[index], text=f"Full name: {self.fullname}", font=("Avenir Next Medium", 16), fg=BLACK, bg=PINK, command=lambda:self.evententryinitializer(self.eventfullnameentry,self.fullname,fieldchanged="full_name", originaltext=self.fullname, eventregistered=nameofevent, currentindex=index)).grid(row=1, column=1, columnspan=13, sticky=NSEW)
        icpassnobutton = Button(self.frametoshowdetailsevent[index], text=f"IC/Passport number: {self.icpassnumber}", font=("Avenir Next Medium", 16), fg=BLACK, bg=PINK, command=lambda:self.evententryinitializer(self.eventicpassnoentry,self.icpassnumber,fieldchanged="icpass_number", originaltext=self.icpassnumber,eventregistered=nameofevent, currentindex=index)).grid(row=2, column=1, columnspan=13, sticky=NSEW)
        phonenumberbutton = Button(self.frametoshowdetailsevent[index], text=f"Phone number: {self.phonenumber}", font=("Avenir Next Medium", 16), fg=BLACK, bg=PINK, command=lambda:self.evententryinitializer(self.eventphonenumberentry,self.phonenumber,fieldchanged="phone_number", originaltext=self.phonenumber,eventregistered=nameofevent, currentindex=index)).grid(row=3, column=1, columnspan=13, sticky=NSEW)
        emailbutton = Button(self.frametoshowdetailsevent[index], text=f"Email: {self.email}", font=("Avenir Next Medium", 16), fg=BLACK, bg=PINK, command=lambda:self.evententryinitializer(self.eventemailentry,self.email,fieldchanged="email", originaltext=self.email,eventregistered=nameofevent, currentindex=index)).grid(row=4, column=1, columnspan=13, sticky=NSEW)
        addressbutton = Button(self.frametoshowdetailsevent[index], text=f"Address: {self.address}", font=("Avenir Next Medium", 16), fg=BLACK, bg=PINK, command=lambda:self.evententryinitializer(self.eventaddressentry,self.address,fieldchanged="address", originaltext=self.address,eventregistered=nameofevent, currentindex=index)).grid(row=5, column=1, columnspan=13, sticky=NSEW)
    def evententryinitializer(self, entrywanted, texttochange, **kwargs):
        fieldchanged = kwargs.get("fieldchanged")
        originaltext = kwargs.get("originaltext")
        eventregistered = kwargs.get("eventregistered")
        currentindex = kwargs.get("currentindex")
        entrylist = [self.eventfullnameentry, self.eventicpassnoentry, self.eventphonenumberentry, self.eventemailentry, self.eventaddressentry]
        for entry in entrylist:
            if entry != entrywanted:
                entry.grid_remove()
        entrywanted.grid()
        entrywanted.delete(0, END)
        entrywanted.insert(0, texttochange)
        entrywanted.focus_set()
        self.currentlyeditinglabel_ = Button(self.frametoshowdetailsevent[currentindex],state=DISABLED, text=f"You are currently editing the {fieldchanged} of the student, {self.fullname}", font=("Avenir Next", 12), fg=BLACK, bg=WHITE, width=1, height=1)
        self.currentlyeditinglabel_.grid(row=6, column=1, columnspan=13, sticky=NSEW)
        self.confirmbutton_ = Button(self.frametoshowdetailsevent[currentindex], text=f"Confirm + {fieldchanged} ", font=("Avenir Next Bold", 16), fg=WHITE, bg=NAVYBLUE, command=lambda:self.confirmchanges(entrywanted, texttochange, fieldchanged, originaltext, eventregistered))
        self.confirmbutton_.grid(row=9, column=2, columnspan=11, sticky=NSEW)


    def eventconfirmchanges(self, eventtogetinfo, texttochange, fieldchanged, originaltext, event_registered):
        textfromentry = eventtogetinfo.get()
        self.conn = sqlite3.connect("interactivesystem.db")
        self.c = self.conn.cursor()
        with self.conn:
            try:
                self.c.execute(f"""UPDATE eventregistratioon SET {fieldchanged} = ? WHERE {fieldchanged} = ? AND event_registered = ?""", (textfromentry, originaltext, event_registered))
                messagebox.showinfo("Success", f"Changes have been made where {fieldchanged} = {originaltext}  has been changed to {textfromentry} under event {event_registered}")
            except Exception as e:
                messagebox.showerror("Error", f"An error has occured: {e}")

    def delete_studentevent(self, eventname, studentname, pagenumber):
        print(f"Delete {studentname}'s details from {eventname}")
        #check if the student still exists in the database
        self.conn = sqlite3.connect("interactivesystem.db")
        self.c = self.conn.cursor()
        self.c.execute("SELECT * FROM eventregistration WHERE full_name = ? AND event_registered = ?", (studentname, eventname))
        self.results = self.c.fetchall()
        if len(self.results) == 0:
            messagebox.showerror("Error", f"Student {studentname} does not exist in the database, you may have already deleted them")
            return
        self.frametodeletestudentsevent = Frame(self.studentsregisteredforframe[pagenumber], height=1, width=1, bg=WHITE,relief=SOLID)
        self.frametodeletestudentsevent.grid(row=0, column=0,rowspan=11,columnspan=15, sticky=NSEW)
        self.frametodeletestudentsevent.grid_propagate(False)
        for x in range(15):
            self.frametodeletestudentsevent.columnconfigure(x, weight=1)
            Label(self.frametodeletestudentsevent, width=1, bg=WHITE).grid(row=0, column=x, sticky=NSEW)
        for y in range(11):
            self.frametodeletestudentsevent.rowconfigure(y, weight=1)
            Label(self.frametodeletestudentsevent, height=1, bg=WHITE).grid(row=y, column=0, sticky=NSEW)
        Label(self.frametodeletestudentsevent, text=f"Are you sure you\nwant to delete {studentname}'s details\n from {eventname}?", font=("Avenir Next", 14), fg=BLACK, bg=WHITE).grid(row=1, column=1, columnspan=13, sticky=NSEW)
        Label(self.frametodeletestudentsevent, text="This action cannot be undone", font=("Avenir Next", 14), fg=BLACK, bg=WHITE).grid(row=2, column=1, columnspan=13, sticky=NSEW)
        Label(self.frametodeletestudentsevent, text="Please type the word DELETE to confirm", font=("Avenir Next", 14), fg=BLACK, bg=WHITE).grid(row=3, column=1, columnspan=13, sticky=NSEW)
        self.deleteentryevent = Entry(self.frametodeletestudentsevent, font=("Avenir Next", 14), fg=BLACK, bg=WHITE)
        self.deleteentryevent.grid(row=4, column=1, columnspan=13, sticky=NSEW)
        self.deleteentryevent.focus_set()
        confirmdeleteevent = Button(self.frametodeletestudentsevent, text="Confirm Delete", font=("Avenir Next", 14), fg=BLACK, bg=PINK, command=lambda:self.delete_studenteventconfirm(eventname, studentname)).grid(row=5, column=1, columnspan=13, sticky=NSEW)
        cancelbuttonevent = Button(self.frametodeletestudentsevent, text="Cancel", font=("Avenir Next", 14), fg=BLACK, bg=PINK, command=lambda:self.frametodeletestudentsevent.grid_remove()).grid(row=6, column=1, columnspan=13, sticky=NSEW)
    def delete_studenteventconfirm(self, eventname, studentname):
        if self.deleteentryevent.get() == "DELETE":
            self.conn = sqlite3.connect("interactivesystem.db")
            self.c = self.conn.cursor()
            with self.conn:
                try:
                    self.c.execute("""DELETE FROM eventregistration WHERE event_registered = ? AND full_name = ?""", (eventname, studentname))
                    messagebox.showinfo("Success", f"{studentname}'s details have been deleted from {eventname}")
                    self.frametodeletestudentsevent.grid_remove()
                except Exception as e:
                    messagebox.showerror("Error", f"An error has occured: {e}")
        else:
            messagebox.showerror("Error", "The word DELETE was not entered correctly")

      
class FeedbackForm(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=PINK)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        for x in range(42):
            self.columnconfigure(x, weight=1, uniform='x')
            Label(self, height=2, bg=PINK, relief="solid").grid(
                row=0, column=x, rowspan=1, columnspan=1, sticky=N+S+E+W)
        for y in range(21):
            self.rowconfigure(y, weight=1, uniform='x')
            Label(self, width=5, bg=PINK, relief="solid").grid(
                row=y, column=0, rowspan=1, columnspan=1, sticky=N+S+E+W)
            
        # Picture
        self.backgroundimageoriginal = Image.open(r"Assets\Feedback Form\feedbackbackground.jpg")
        self.backgroundimage = ImageTk.PhotoImage(self.backgroundimageoriginal.resize(
            (math.ceil(1680 * dpi / 96), math.ceil(840 * dpi / 96)), Image.Resampling.LANCZOS))
        
        self.backgroundimagelabel = Label(self, image=self.backgroundimage, width=1, height=1, bg=LIGHTPURPLE)
        self.backgroundimagelabel.grid(row=0, column=0, rowspan=21, columnspan=43, sticky=N+S+E+W)
        self.backgroundimagelabel.grid_propagate(0)

        # Widgets
        label = Label(self, text="This is a feedback form to help us improve our app.\nPlease answer the questions below to the best of your ability.\nThank you for your time!", font=(
            'Segoe Ui Semibold', 14), width=1, height=1, fg='#000000', bg='#FFF5E4')
        label.grid(row=1, column=10, columnspan=22,
                   rowspan=2, sticky=N+S+E+W)
        conn = sqlite3.connect('interactivesystem.db')
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS feedback(
            satisfactionquestion1 text NOT NULL,
            satisfactionquestion2 text NOT NULL,
            likelihoodquestion text NOT NULL,
            easinessquestion text NOT NULL,
            yesnoquestion text NOT NULL,
            entryquestion text NOT NULL,
            emailofuser text NOT NULL
            )""")
        global LOGINID
        # Radiobutton example
        def ShowChoice():
            messagebox.showinfo(
                "The answers are:", f"First question answer: {question1answer.get()}\nSecond question answer: {question2answer.get()}\nLikelihood answer: {question3answer.get()}\nEasinessQ answer: {question4answer.get()}\nYes no answer: {yesnoquestionanswer.get()}\nText entry answer: {openendedentry.get()}")

        def dosomedatabasemagic():
            satisfactionans = question1answer.get()
            helpfulans = question2answer.get()
            likelihoodans = question3answer.get()
            easinessans = question4answer.get()
            ynans = yesnoquestionanswer.get()
            entryanswer = openendedentry.get()
            emailofuser = LOGINID
            information = (satisfactionans, helpfulans,
                           likelihoodans, easinessans, ynans, entryanswer, emailofuser)

            with conn:
                c.execute(
                    "INSERT INTO feedback VALUES (?,?,?,?,?,?,?)", information)
                messagebox.showinfo(
                    "Success", "Your answers have been recorded!")

        scaleofsatisfaction = [("1", "Very Unsatisfied"), ("2", "Unsatisfied"),
                               ("3", "Neutral"), ("4", "Satisfied"), ("5", "Very Satisfied")]  # Scale for satisfaction
        scaleofhelpful = [("1", "Very Unhelpful"), ("2", "Unhelpful"), ("3", "Plain"),
                          ("4", "Helpful"), ("5", "Very Helpful")]  # Scale for helpful
        scaleoflikelihood = [("1", "Very Unlikely"), ("2", "Unlikely"), ("3", "Neutral"), (
            "4", "Likely"), ("5", "Very Likely")]  # Example if want to create a scale for likelihood
        scaleofeasiness = [("1", "Very Difficult"), ("2", "Difficult"), ("3", "Neutral"), (
            "4", "Easy"), ("5", "Very Easy")]  # If want to create an easiness scale question
        # messagebox.showinfo("Welcome to the survey!", "This is a survey to help us improve our app. Please answer the questions below to the best of your ability. Thank you for your time!")
        yesnooptions = ["No", "Yes"]

        # Label
        question1label = Label(self, text="How would you rate our announcement system for your overall experience? ",
        font=("Helvetica", 14), width=1, height=1,
        bg=LIGHTYELLOW)
        question1label.grid(row=4, column=10, columnspan=22,
                            rowspan=1, sticky=NSEW)
        question1answer = StringVar()
        question1answer.set("Neutral")  # Satisfaction Q
        question2label = Label(self, text="Is this system very helpful to you that you are not miss or neglect any event? ", 
        font=("Helvetica", 14), width=1, height=1,
        bg=LIGHTYELLOW)
        question2label.grid(row=6, column=10, columnspan=22,
                            rowspan=1, sticky=NSEW)
        question2answer = StringVar()
        question2answer.set("Neutral")  # Satisfaction Q
        question3label = Label(self, text="How likely are you to recommend our app to your friends?",
        font=("Helvetica", 14), bg=LIGHTYELLOW)
        question3label.grid(row=8, column=10, columnspan=22,
                            rowspan=1, sticky=NSEW)
        question3answer = StringVar()
        question3answer.set("Neutral")  # Likelihood Q

        question4label = Label(self, text="How difficult / easy was it to find the event on this app?",
        font=("Helvetica", 14), width=1, height=1,
        bg=LIGHTYELLOW)
        question4label.grid(row=10, column=10, columnspan=22,
                            rowspan=1, sticky=NSEW)
        question4answer = StringVar()
        question4answer.set("Neutral")  # Easiness Q
        yesnoquestionlabel = Label(self, text="Were you able to find information you needed about the events?",
        font=("Helvetica", 14), width=1, height=1,
        bg=LIGHTYELLOW)
        yesnoquestionlabel.grid(
            row=12, column=10, columnspan=22, rowspan=1, sticky=NSEW)
        yesnoquestionanswer = StringVar()
        yesnoquestionanswer.set("Yes")

        count = 12  # starting on column 12
        count2 = 12
        count3 = 12  # Change this to customize the column layout if needed, in this case since different loops, it's easier to just create the count2 variable
        count4 = 12
        count5 = 14  # Changing for yes no to make it centered
        # Creating Satisfaction Scale(because using satisfaction options)
        for text, rating in scaleofsatisfaction:
            self.firstrow = Radiobutton(self,
                text=text,  # text of the radiobutton becomes 1, 2, 3, 4, 5
                # for a row(horizontal), each radiobutton needs to share same variable
                variable=question1answer,
                # value is going to be the rating in ("Number", "Rating") that will be stored as the value for the radiobutton
                value=rating,
                justify=CENTER,
                bg=ORANGE, font=("Helvetica", 18), height=1, width=1)
            # count becomes the column number, 12, 16, 20, 24, 28
            self.firstrow.grid(row=5, column=count, rowspan=1,
                               columnspan=2, sticky=N+S+E+W)
            self.firstrow.grid_propagate(0)
            count += 4

        # Creating Helpful Scale
        for text, rating in scaleofhelpful:
            self.secondrow = Radiobutton(self,
                    text=text,  # text of the radiobutton becomes 1, 2, 3, 4, 5
                    # for a row(horizontal), each radiobutton needs to share same variable
                    variable=question2answer,
                    # value is going to be the rating in ("Number", "Rating") that will be stored as the value for the radiobutton
                    value=rating,
                    justify=CENTER,
                    bg=ORANGE, font=("Helvetica", 18), width=1, height=1)
            # count becomes the column number, 12, 16, 20, 24, 28
            self.secondrow.grid(row=7, column=count2,
                                rowspan=1, columnspan=2, sticky=N+S+E+W)
            self.secondrow.grid_propagate(0)
            count2 += 4

        # Creating Likelihood Scale
        for text, rating in scaleoflikelihood:
            self.thirdrow = Radiobutton(self, text=text, variable=question3answer, value=rating, bg=ORANGE, font=("Helvetica", 18),
                                        justify=CENTER, width=1, height=1)
            self.thirdrow.grid(row=9, column=count3, rowspan=1,
                               columnspan=2, sticky=N+S+E+W)
            self.thirdrow.grid_propagate(0)
            count3 += 4  # have to set column=count2,because different type of answers
            # in options means different count needs to be used, basically, count is for satisfaction questions,
            # count2 is for likelihood questions, count3 for yes no questions
            # notice gap value is still 4

        # Creating Easiness Scale(because using easiness question)
        for text, rating in scaleofeasiness:
            self.fourthrow = Radiobutton(self, text=text, variable=question4answer, value=rating, bg=ORANGE, font=("Helvetica", 18),
                                         justify=CENTER, width=1, height=1)
            self.fourthrow.grid(row=11, column=count4,
                                rowspan=1, columnspan=2, sticky=N+S+E+W)
            self.fourthrow.grid_propagate(0)
            count4 += 4

        # Creating Yes No
        for text in yesnooptions:
            self.fifthrow = Radiobutton(self, text=text, variable=yesnoquestionanswer, value=text, bg=ORANGE, font=("Helvetica", 18),
                                        justify=CENTER, width=1, height=1)
            self.fifthrow.grid(row=13, column=count5,
                               rowspan=2, columnspan=3, sticky=N+S+E+W)
            self.fifthrow.grid_propagate(0)
            count5 += 11

        # Open Question
        openendquestionlabel = Label(self, text="Please leave any comments or suggestions below:", font=(
            "Helvetica", 14), width=1, height=1, bg=LIGHTYELLOW)
        openendquestionlabel.grid(
            row=15, column=10, columnspan=22, rowspan=1, sticky=NSEW)
        openendedentry = Entry(self, width=1, bg="white",
                               font=("Helvetica", 18), justify=CENTER)
        openendedentry.grid(row=16, column=10, columnspan=22,
                            rowspan=2, sticky=NSEW)

        # labels for scale
        satisfactionlabel = Label(self, text="More Unsatisfied", font=(
            "Helvetica", 8), bg=NICEPURPLE, justify="left", width=1, height=1)
        satisfactionlabel.grid(
            row=4, column=10, columnspan=2, rowspan=1, sticky=NSEW)
        dissatisfactionlabel = Label(self, text="More satisfied", font=(
            "Helvetica", 8), bg=NICEPURPLE, justify="right", width=1, height=1)
        dissatisfactionlabel.grid(
            row=4, column=30, columnspan=2, rowspan=1, sticky=NSEW)
        unhelpfullabel = Label(self, text="Very Unhelpful", font=(
            "Helvetica", 8), bg=NICEPURPLE, justify="left", width=1, height=1)
        unhelpfullabel.grid(row=6, column=10, columnspan=2,
                            rowspan=1, sticky=NSEW)
        helpfullabel = Label(self, text="Very Helpful", font=(
            "Helvetica", 8), bg=NICEPURPLE, justify="right", width=1, height=1)
        helpfullabel.grid(row=6, column=30, columnspan=2,
                          rowspan=1, sticky=NSEW)
        unlikelihoodlabel = Label(self, text="Less likelihood", font=(
            "Helvetica", 8), bg=NICEPURPLE, justify="left", width=1, height=1)
        unlikelihoodlabel.grid(
            row=8, column=10, columnspan=2, rowspan=1, sticky=NSEW)
        likelihoodlabel = Label(self, text="More likelihood", font=(
            "Helvetica", 8), bg=NICEPURPLE, justify="right", width=1, height=1)
        likelihoodlabel.grid(row=8, column=30, columnspan=2,
                             rowspan=1, sticky=NSEW)
        difficultlabel = Label(self, text="More difficult", font=(
            "Helvetica", 8), bg=NICEPURPLE, justify="left", width=1, height=1)
        difficultlabel.grid(row=10, column=10, columnspan=2,
                            rowspan=1, sticky=NSEW)
        easierlabel = Label(self, text="Easier", font=(
            "Helvetica", 8), bg=NICEPURPLE, justify="right", width=1, height=1)
        easierlabel.grid(row=10, column=30, columnspan=2,
                         rowspan=1, sticky=NSEW)

        # Button
        self.getanswers = Button(self, text="Cancel", command=lambda: [controller.show_frame(MainPage), 
        controller.togglebuttonrelief(controller.mainpagebutton)],
        bg=ORANGE, font=("Helvetica", 18), width=1, height=1)
        self.getanswers.grid(row=18, column=10, rowspan=2,
                             columnspan=6, sticky=N+S+E+W)
        self.getanswers.grid_propagate(0)

        self.getanswers = Button(self, text="Confirm", command=lambda: [
                                 ShowChoice(), dosomedatabasemagic()], bg=ORANGE, font=("Helvetica", 18), width=1, height=1)
        self.getanswers.grid(row=18, column=26, rowspan=2,
                             columnspan=6, sticky=N+S+E+W)
        self.getanswers.grid_propagate(0)


class CalendarPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=PINK)
        for x in range(42):
            self.columnconfigure(x, weight=1, uniform='x')
            Label(self, width=1, bg=NICEPURPLE).grid(
                row=0, column=x, rowspan=1, columnspan=1, sticky=NSEW)
        for y in range(21):
            self.rowconfigure(y, weight=1, uniform='x')
            Label(self, width=1, bg=NICEPURPLE).grid(
                row=y, column=0, rowspan=1, columnspan=1, sticky=NSEW)
        def hidetheentireframe():
            self.grid_remove()

        self.calendarpagebgimg = Image.open(r"Assets\CalendarPage\calendarpgbg.png")
        self.calendarpagebgimg = ImageTk.PhotoImage(self.calendarpagebgimg.resize(
            (math.ceil(1680 * dpi / 96), math.ceil(840 * dpi / 96)), Image.Resampling.LANCZOS))
        self.calendarpagebg = Label(self, image=self.calendarpagebgimg,
                                    anchor=CENTER, width=1, height=1)
        self.calendarpagebg.grid(row=0, column=0, columnspan=42,
                                    rowspan=21, sticky=NSEW)
        xbuttonlabel = Button(self, text="X", font=("Avenir Next Medium", 18), height=1,width=1,
        bg=DARKBLUE, fg="white",
        command= lambda:hidetheentireframe())
        xbuttonlabel.grid(row=0, column=40, rowspan=2, columnspan=2, sticky=NSEW)
        # Widgets
        label = Label(self, text="This is the Calendar", font=(
            'Segoe Ui Semibold', 14), width=1, height=1, fg='#000000', bg='#FFF5E4', justify="left")
        label.grid(row=0, column=2, columnspan=6,
                   rowspan=2, sticky=NSEW)
        self.cal = tkCalendar(self, width=1, height=1,
            background = DARKBLUE, foreground = 'white', 
            bordercolor = ORANGE, 
            headersbackground = NAVYBLUE, headersforeground = 'white', 
            selectbackground = NICEBLUE, selectforeground = 'black',
            showothermonthdays = False,
            selectmode="day",
            font=("Avenir Next Medium", 18),
            date_pattern="dd-mm-yyyy")
        self.cal.grid(row=2, column=2, columnspan=21, rowspan=17, sticky=NSEW)
        self.cal.bind("<<CalendarSelected>>", self.generate_buttons)

        #Go back to current date button
        self.gobackbutton = Button(self, text="Change view to current date", width=1, height=1,
        bg=ORANGE, font=("Atkinson Hyperlegible", 18), command=lambda: [self.go_to_today()])
        self.gobackbutton.grid(row=19, column=2, rowspan=2,
                             columnspan=8, sticky=NSEW)
        self.gobackbutton.grid_propagate(False)
        self.refreshbutton = Button(self, text="Refresh",
        bg=ORANGE, font=("Atkinson Hyperlegible", 18), width=1, height=1,
        command=lambda: [self.add_events()])
        self.refreshbutton.grid(row=19, column=15, rowspan=2,
                                columnspan=8, sticky=NSEW)
        self.refreshbutton.grid_propagate(False)
        self.buttonframe=Frame(self, bg = ORANGE, relief=RAISED, width=1, height=1,)
        self.buttonframe.grid(row=4, column=24, rowspan=15, columnspan=17, sticky=NSEW)
        self.buttonframe.grid_propagate(False)
        self.detailslabel = Label(self, text="Click on a date to view the events and their details.", width=1, height=1,
        font = ("Avenir Next Medium", 18), background=LIGHTYELLOW)
        self.detailslabel.grid(row=2, column=24, rowspan=2, columnspan=17, sticky=NSEW)
        self.add_events()
    def generate_buttons(self, event):
        detailslabel = self.detailslabel 
        for widgets in self.buttonframe.winfo_children():
            widgets.destroy()
        for x in range(15):
            self.buttonframe.columnconfigure(x, weight=1, uniform='x')
            Label(self.buttonframe, width=1, bg=ORANGE).grid(
                row=0, column=x, rowspan=1, columnspan=1, sticky=NSEW)
        for y in range(15):
            self.buttonframe.rowconfigure(y, weight=1, uniform='x')
            Label(self.buttonframe, width=1, bg=ORANGE).grid(
                row=y, column=0, rowspan=1, columnspan=1, sticky=NSEW)
        self.conn = sqlite3.connect('interactivesystem.db')
        self.c = self.conn.cursor()
        date = self.cal.selection_get()
        self.c.execute("""SELECT count(*) FROM eventcreation WHERE event_startdate = ?""", (date,))
        self.eventnumber = self.c.fetchall()
        for row in self.eventnumber:
            detailslabel.configure(text=f"Event details: There is/are {row[0]} event(s)\noccurring on {date}")
        self.c.execute("""SELECT event_name FROM eventcreation WHERE event_startdate = ?""", (date,))
        self.eventnames = self.c.fetchall()
        startingrowno = 0
        for index, name in list(enumerate(self.eventnames)):
            #Unpacking the tuple (name) to get the string
            name = name[0]
            Label(self.buttonframe, text=f"Event name: {name}", width=1, height=1, 
            bg = LIGHTPURPLE, fg = "black", relief="groove",
            font = ("Avenir Next Medium", 18), wraplength=400, justify=CENTER).grid(row=0+startingrowno, column=0, rowspan=2, columnspan=11, sticky=NSEW)
            Button(self.buttonframe, text="View details", width=1, height=1,
            bg = PINK, fg = "black", relief="groove",
            font = ("Avenir Next Medium", 18),
            # lambda command fix thanks to https://stackoverflow.com/questions/17677649/tkinter-assign-button-command-in-a-for-loop-with-lambda
            command=lambda x=name:self.createdetails(x)).grid(row=0+startingrowno, column=11, rowspan=2, columnspan=4, sticky=NSEW)
            startingrowno += 2

    def createdetails(self, name):
        #A frame to display the details of the event
        self.subframe = Frame(self.buttonframe, bg = NICEBLUE, relief=RAISED, height=1, width=1)
        self.subframe.grid(row=0, column=0, rowspan=15, columnspan=15, sticky=NSEW)
        for x in range(15):
            self.subframe.columnconfigure(x, weight=1, uniform='x')
            Label(self.subframe, width=1, bg=LIGHTYELLOW).grid(
                row=0, column=x, rowspan=1, columnspan=1, sticky=NSEW)
        for y in range(15):
            self.subframe.rowconfigure(y, weight=1, uniform='x')
            Label(self.subframe, width=1, bg=LIGHTYELLOW).grid(
                row=y, column=0, rowspan=1, columnspan=1, sticky=NSEW)
        self.subframe.grid_propagate(False)
        self.conn = sqlite3.connect('interactivesystem.db')
        self.c = self.conn.cursor()
        self.c.execute("""SELECT 
        event_name, event_startdate, event_enddate, event_starttime, event_endtime, event_organizer, venue_name, host_name FROM eventcreation WHERE event_name = ?""", (name,))
        self.eventdetails = self.c.fetchall()
        for row in self.eventdetails:
            Label(self.subframe, text="Event details", width=1,height=1,
            bg = LAVENDER, fg = "black", font = ("Avenir Next Medium", 18)).grid(
                row=0, column=0, rowspan=2, columnspan=15, sticky=NSEW)
            Label(self.subframe, text=f"Event: {row[0]}", width=1,height=1,
            bg = LAVENDER, fg = "black", font = ("Avenir Next Medium", 18)).grid(
                row=2, column=0, rowspan=2, columnspan=15, sticky=NSEW)
            Label(self.subframe, text=f"Event Date: From {row[1]} to {row[2]}", width=1,height=1,
            bg = LAVENDER, fg = "black", font = ("Avenir Next Medium", 18)).grid(
                row=4, column=0, rowspan=2, columnspan=15, sticky=NSEW)
            Label(self.subframe, text=f"Event time: {row[3]} - {row[4]}", width=1,height=1,
            bg = LAVENDER, fg = "black", font = ("Avenir Next Medium", 18)).grid(
                row=6, column=0, rowspan=2, columnspan=15, sticky=NSEW)
            Label(self.subframe, text=f"Event organizer: {row[5]}", width=1,height=1,
            bg = LAVENDER, fg = "black", font = ("Avenir Next Medium", 18)).grid(
                row=8, column=0, rowspan=2, columnspan=15, sticky=NSEW)
            Label(self.subframe, text=f"Host name: {row[7]}", width=1,height=1,
            bg = LAVENDER, fg = "black", font = ("Avenir Next Medium", 18)).grid(
                row=10, column=0, rowspan=2, columnspan=15, sticky=NSEW)
            Label(self.subframe, text=f"Venue: {row[6]}", width=1,height=1,
            bg = LAVENDER, fg = "black", font = ("Avenir Next Medium", 18)).grid(
                row=12, column=0, rowspan=2, columnspan=15, sticky=NSEW)
            Button(self.subframe, text="Back", width=1, height=1,
            bg = PINK, fg = "black", relief="groove",
            font = ("Avenir Next Medium", 18),
            command=lambda: self.subframe.grid_remove()).grid(row=14, column=0, rowspan=2, columnspan=15, sticky=NSEW)

    
    def go_to_today(self):
        self.cal.selection_set(datetime.date.today())
        self.cal.see(datetime.date.today())
    #read from the eventcreation table and add the events to the calendar
    def add_events(self):
        self.conn = sqlite3.connect('interactivesystem.db')
        self.c = self.conn.cursor()
        self.c.execute("SELECT event_startdate, event_name FROM eventcreation")
        self.rows = self.c.fetchall()
        for row in self.rows:
            #convert to datetime 
            self.date = datetime.datetime.strptime(row[0], r'%Y-%m-%d').date()
            self.cal.calevent_create(self.date, row[1], 'all')

            

    



def main():
    window = Window()
    window.mainloop()


if __name__ == "__main__":
    main()