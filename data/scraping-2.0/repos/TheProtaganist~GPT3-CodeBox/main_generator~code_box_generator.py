import openai
import random
import pygame
import tkinter as tk
import os
import sys
from essential_generators import DocumentGenerator as DocGen
from random import uniform as pick
from tkinter import *
from tkinter import Label, Entry, Message, Text, Frame, Canvas, PhotoImage, Button, font, ttk, simpledialog, messagebox,Text
from essential_generators import DocumentGenerator as DocGen
from yapf.yapflib.yapf_api import FormatFile
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

main_application = tk.Tk()

global a, b, c

class MathematicalClass:

    def __init__(self):
        pass

    def check_for_size(self, value_output):

        try:
            number = value_output

            def check_if_size_greater_than_2(base, power):
                num = base

                def mul(n, p):
                    numA = n
                    numB = p
                    return numA * numB

                def div2(n_n, p_p, dividend):
                    numC = n_n
                    numD = p_p
                    return (numC * numD) / dividend

                number_input_default_one = mul(base, power)
                number_input_divided_one = div2(base, power, 5)

                def process(look_for_2):

                    if check_for_two < look_for_2 < min_val_4:
                        return number_input_divided_one
                    else:
                        return number_input_default_one

                return process(num)

            def check_if_size_greater_than_2_small_edition(base2, power2):
                look_for_two = base2

                def mul2(n2, p2):
                    numZ = n2
                    numY = p2
                    return numZ * numY

                def mul_div(n_2, p_2, dividend2):
                    numX = n_2
                    numY = p_2
                    return (numX * numY) / dividend2

                number_input_default2 = mul2(base2, power2)
                number_input_divided2 = mul_div(base2, power2, 5)

                def processed2(numb):
                    if check_for_two < numb < min_val_4:
                        return number_input_divided2
                    else:
                        return number_input_default2

                return processed2(look_for_two)

            def processed(n2n2, p2p2):
                Token = n2n2

                def mul(n_2, p_2):
                    numE = n_2
                    numF = p_2
                    return numE * numF

                def div2(n_n_n_n, p_p_p_p, dividend3):
                    numG = n_n_n_n
                    numH = p_p_p_p
                    return (numG * numH) / dividend3

                number_input_default_3 = mul(n2n2, p2p2)
                number_input_divided_3 = div2(n2n2, p2p2, 5)

                def process3(look_for_the_number_two):
                    if check_for_two < look_for_the_number_two < min_val_4:
                        return number_input_divided_3
                    else:
                        return number_input_default_3

                return process3(Token)

            if number < min_val:
                num_value = check_if_size_greater_than_2_small_edition(number, 10)
                finished_num = processed(num_value, 1)
                return finished_num
            if min_val_1 > number > min_val:
                num_value = check_if_size_greater_than_2_small_edition(number, 1)
                finished_num = processed(num_value, 1)
                return finished_num
            if min_val_2 > number > min_val_1:
                num_value = check_if_size_greater_than_2_small_edition(number, 1)
                finished_num = processed(num_value, 1)
                return finished_num
            if min_val_3 > number > min_val_2:
                num_value = check_if_size_greater_than_2_small_edition(number, 1)
                finished_num = processed(num_value, 1)
                return finished_num

            number_component = int(number)
            if len(str(number_component)) == 1:
                num_value = check_if_size_greater_than_2_small_edition(number, 1)
                finished_num = processed(num_value, 1)
            elif len(str(number_component)) == 2:
                num_value = check_if_size_greater_than_2_small_edition(number, 1 / 10)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 3:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 2)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 4:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 3)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 5:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 4)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 6:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 5)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 7:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 6)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 8:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 7)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 9:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 8)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 10:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 9)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 11:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 10)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 12:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 11)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 13:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 12)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 14:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 13)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 15:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 14)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 16:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 15)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 17:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 16)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 18:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 17)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 19:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 18)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 20:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 19)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 21:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 20)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 22:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 21)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 23:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 22)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 24:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 23)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 25:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 24)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 26:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 25)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 27:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 26)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 28:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 27)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 29:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 28)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 30:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 29)
                finished_num = processed(num_value, 1)
                return finished_num
            elif len(str(number_component)) == 31:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** 30)
                finished_num = processed(num_value, 1)
                return finished_num
            else:
                num_value = check_if_size_greater_than_2(number, 1 / 10 ** len(str(number_component)))
                finished_num = processed(num_value, 1)
                return finished_num

            if len(str(number_component)) > 32:
                print("please enter a smaller number for x, y, and z")

            return finished_num
        except OverflowError:
            print("please enter a smaller numbers for your output :) ")


check_that_size = MathematicalClass()
main_application.title('Code Generation tool')
pygame.mixer.init()
pygame.mixer.music.load("FunHouse.mp3")
pygame.mixer.music.play(loops=-1)
code_generated = ''
HEIGHT = 1024
WIDTH = 1500
canvas = Canvas(main_application, height=HEIGHT, width=WIDTH)
canvas.pack()
main_application.iconbitmap('icon.ico')
background_image = PhotoImage(file='cyberlovers.png')
background_label = Label(canvas, image=background_image)
background_label.place(relwidth=1.2, relheight=1.2)
info = "Enter whether you are on" + "\n" + "light side or dark side" + "\n" "even for yes and odd for no" + "\n" + "do you wish to echo back the completion?"


def ask_quit():
    if messagebox.askokcancel("Quit", "Would you like to generate code some other time?"):
        main_application.destroy()


emp_ty = ""
min_val = .001
min_val_1 = .01
min_val_2 = .1
min_val_3 = 1
min_val_4 = 10
check_for_two = 2
final_max_val = 100000000000000000

unknown_value_x = float(pick(0, 2))
unknown_value_y = float(pick(0, 2))
unknown_value_z = float(pick(0, 2))


key_var = StringVar()
key_var.set("")


Enter_key = Entry(canvas, font=("Arial", 10, "italic"), textvariable=key_var)
Enterkey = Enter_key.get()
Enter_key.place(relwidth=0.3, relheight=0.01885, x=900, y=5)

KeyButton = Button(canvas, text="key goes here", font=("Leelawadee UI", 12), bg="black", fg="white", activeforeground="#ffff00", activebackground="#00ffff", command=lambda: return_key(Enterkey))
KeyButton.place(relwidth=0.1, relheight=0.027, x=1000, y=27)


def get_key(user):
    USER = ""
    USER += user
    code_entered = Label(canvas, text=USER, textvariable=key_var, font=("Times New Roman", 9), fg="orange", bg="blue")
    secret_key = code_entered.cget("text")
    return secret_key


def return_key(user_key):
    global ai_key
    key_needed = get_key(user_key)
    if user_key == user_key:
        Key = Label(canvas, text=key_needed, textvariable=key_var, font=("Times New Roman", 9), fg="orange", bg="blue")
        ai_key = str(Key.cget("text"))
        if ai_key != user_key:
            os.environ['OPENAI_API_KEY'] = ai_key
            openai.api_key = os.environ["OPENAI_API_KEY"]
            key_var.set(ai_key)
            messagebox.showinfo("Key is being validated", "A key has been entered")
            return ai_key, openai.api_key
        elif ai_key == user_key:
            print("please try again")


echo_checking = IntVar()
echo_checking.set("0")


def the_echo_function(echo_or_not):
    check = echo_or_not
    get_echo = Label(canvas, text=check, textvariable=echo_checking, font=("Times New Roman", 10), bg="#5ae6e6", fg="#ff6b05")
    checker = get_echo.cget("text")
    return checker


def echo_func(boolean):
    try:
        global echo
        echoed = the_echo_function(boolean)
        if boolean == boolean:
            the_echo = Label(canvas, text=echoed, textvariable=echo_checking, font=("Times New Roman", 10), bg="white", fg="cyan")
            value = int(the_echo.cget("text"))
            if value != boolean:
                if int(value) % 2 != 0:
                    echo = False
                    the_echo = Label(canvas, text=echoed, textvariable=echo_checking, font=("Times New Roman", 10), bg="black", fg="red")
                    the_echo.place(relwidth=0.03, relheight=0.0205, x=927, y=156)
                    echo_checking.set(value)
                    print(echo)
                    return echo, value
                elif int(value) % 2 == 0:
                    echo = True
                    the_echo = Label(canvas, text=echoed, textvariable=echo_checking, font=("Times New Roman", 10), bg="white", fg="cyan")
                    the_echo.place(relwidth=0.03, relheight=0.0205, x=927, y=156)
                    echo_checking.set(value)
                    print(echo)
                    return echo, value
            elif value == boolean:
                print("please try again")
    except ValueError:
        messagebox.showinfo("Error", "Only use whole numbers for this operation...")


string = StringVar()
string.set("")


def user_wants(intent):
    intention = ""
    intention += intent
    code_entered = Label(canvas, text=intention, textvariable=string, font=("Times New Roman", 9), fg="orange", bg="blue")
    code_used = code_entered.cget("text")
    return code_used


def return_intention(intended):
    global code_des
    code_variable = user_wants(intended)

    if intended == intended:
        code = Label(canvas, text=code_variable, textvariable=string, font=("Times New Roman", 9), fg="orange", bg="blue")
        code_des = str(code.cget("text"))
        if code_des != intended:
            code.place(relwidth=0.2875, relheight=0.08, x=1065, y=60)
            string.set(code_des)
            print(code_des)
            return code_des
        elif code_des == intended:
            print("please try again")


floating_x_value = DoubleVar()
floating_x_value.set("0.000000000000027")


def x_func(xt):
    xx = Label(canvas, text=xt, textvariable=floating_x_value, font=("Times New Roman", 10), bg="#5ae6e6", fg="#ff6b05")
    x_var = float(xx.cget("text"))
    return x_var


def return_x(x_used):
    try:
        global a
        var_x = x_func(x_used)
        if x_used == x_used:
            xx = Label(canvas, text=var_x, textvariable=floating_x_value, font=("Times New Roman", 10), bg="#5ae6e6", fg="#ff6b05")
            a = float(xx.cget("text"))
            if str(a) != str(x_used):
                xx.place(relwidth=0.05, relheight=0.02, x=1400, y=180)
                floating_x_value.set(a)
                print(xx.cget("text"))
                return a
            elif str(a) == str(x_used):
                print("please try again")
    except ValueError:
        messagebox.showinfo("Error", "DO NOT ENTER STRINGS FOR THIS OPERATION!!!")


floating_y_value = DoubleVar()
floating_y_value.set("0.000000000000027")


def y_func(yt):
    yy = Label(canvas, text=yt, textvariable=floating_y_value, font=("Times New Roman", 10), bg="#5ae6e6", fg="#ff6b05")
    y_var = float(yy.cget("text"))
    return y_var


def return_y(y_used):
    try:
        global b
        var_y = y_func(y_used)
        if y_used == y_used:
            yy = Label(canvas, text=var_y, textvariable=floating_y_value, font=("Times New Roman", 10), bg="#5ae6e6", fg="#ff6b05")
            b = float(yy.cget("text"))
            if str(b) != str(y_used):
                yy.place(relwidth=0.05, relheight=0.02, x=1400, y=300)
                floating_y_value.set(b)
                print(yy.cget("text"))
                return b
            elif str(b) == str(y_used):
                print("please try again")
    except ValueError:
        messagebox.showinfo("Error", "DO NOT ENTER STRINGS FOR THIS OPERATION!!!")


floating_z_value = DoubleVar()
floating_z_value.set("0.000000000000027")


def z_func(zt):
    zz = Label(canvas, text=zt, textvariable=floating_z_value, font=("Times New Roman", 10), bg="#5ae6e6", fg="#ff6b05")
    z_var = float(zz.cget("text"))
    return z_var


def return_z(z_used):
    try:
        global c
        var_z = z_func(z_used)
        if z_used == z_used:
            zz = Label(canvas, text=var_z, textvariable=floating_z_value, font=("Times New Roman", 10), bg="#5ae6e6", fg="#ff6b05")
            c = float(zz.cget("text"))
            if str(c) != str(z_used):
                zz.place(relwidth=0.05, relheight=0.02, x=1400, y=420)
                floating_z_value.set(c)
                print(zz.cget("text"))
                return c
            elif str(c) == str(z_used):
                print("please try again")
    except ValueError:
        messagebox.showinfo("Error", "DO NOT ENTER FOR THIS OPERATION!!!")


Int4 = IntVar()
Int4.set("9999999999999999999")


def function_number_four(num_4):
    f4 = 0
    f4 += num_4
    Number4 = Label(canvas, text=f4, textvariable=Int4, font=("Times New Roman", 10), bg="#5ae6e6", fg="#ff6b05")
    numit4 = int(Number4.cget("text"))
    return numit4


def F4(n):
    try:
        global num4
        variable4 = function_number_four(n)
        if n == n:
            Number_4 = Label(canvas, text=variable4, textvariable=Int4, font=("Times New Roman", 10), bg="#5ae6e6", fg="#ff6b05")
            num4 = int(Number_4.cget("text"))
            if str(num4) != str(n):
                Number_4.place(relwidth=0.05, relheight=0.02, x=1400, y=540)
                Int4.set(num4)
                print(num4)
                return num4
            elif str(num4) == str(n):
                print("please try again")
    except ValueError:
        messagebox.showinfo("Error", "Only use whole numbers for this operation...")


Int5 = IntVar()
Int5.set("9999999999999999999")


def function_number_five(num_5):
    S5 = 0
    S5 += num_5
    Number5 = Label(canvas, text=S5, textvariable=Int5, font=("Times New Roman", 10), bg="#5ae6e6", fg="#ff6b05")
    numit5 = int(Number5.cget("text"))
    return numit5


def F5(n2):
    try:
        global num5
        variable5 = function_number_five(n2)
        if n2 == n2:
            Number_5 = Label(canvas, text=variable5, textvariable=Int5, font=("Times New Roman", 10), bg="#5ae6e6", fg="#ff6b05")
            num5 = int(Number_5.cget("text"))
            if str(num5) != str(n2):
                Number_5.place(relwidth=0.05, relheight=0.02, x=1400, y=680)
                Int5.set(num5)
                print(num5)
                return num5
            elif str(num5) == str(n2):
                print("please try again")
    except ValueError:
        messagebox.showinfo("Error", "Only use whole numbers for this operation...")


IntTokens = IntVar()
IntTokens.set("9999999999999999999")


def token_function(token):
    tos = 0
    tos += token
    Toks = Label(canvas, text=tos, textvariable=IntTokens, font=("Times New Roman", 10), bg="#5ae6e6", fg="#ff6b05")
    tokes = int(Toks.cget("text"))
    return tokes


def toks(tok):
    try:
        global tokens
        variable_token = token_function(tok)
        if tok == tok:
            t = Label(canvas, text=variable_token, textvariable=IntTokens, font=("Times New Roman", 10), bg="#5ae6e6", fg="#ff6b05")
            tokens = int(t.cget("text"))
            if str(tokens) != str(tok):
                t.place(relwidth=0.05, relheight=0.02, x=1400, y=800)
                IntTokens.set(tokens)
                print(tokens)
                return tokens
            elif str(tokens) == str(tok):
                print("please try again")
    except ValueError:
        messagebox.showinfo("Error", "Only use whole numbers for this operation...")


ech = Entry(canvas, font=("Arial", 10, "italic"), textvariable=echo_checking)
information = Label(canvas, font=("Leelawadee UI", 8, "bold", "italic"), text=info, bg="#00ffff", fg="#ffff00")
information.place(relwidth=0.15, relheight=0.08, x=830, y=60)
echoing = int(ech.get())
ech.place(relwidth=0.03, relheight=0.0205, x=975, y=156)

CodeDescription = Entry(canvas, font=("Arial", 10, "italic"), textvariable=string)
description = CodeDescription.get()
CodeDescription.place(relwidth=0.577, relheight=0.01885, x=10, y=5)

x_component = Entry(canvas, font=("Arial", 10, "italic"), textvariable=floating_x_value)
x_is = float(x_component.get())
x_component.place(relwidth=0.1, relheight=0.01885, x=10, y=180)

y_component = Entry(canvas, font=("Arial", 10, "italic"), textvariable=floating_y_value)
y_is = float(y_component.get())
y_component.place(relwidth=0.1, relheight=0.01885, x=10, y=300)

z_component = Entry(canvas, font=("Arial", 10, "italic"), textvariable=floating_z_value)
z_is = float(z_component.get())
z_component.place(relwidth=0.1, relheight=0.01885, x=10, y=420)

four = Entry(canvas, font=("Arial", 10, "italic"), textvariable=Int4)
num_four = int(four.get())
four.place(relwidth=0.1, relheight=0.01885, x=10, y=540)

five = Entry(canvas, font=("Arial", 10, "italic"), textvariable=Int5)
num_five = int(five.get())
five.place(relwidth=0.1, relheight=0.01885, x=10, y=660)

token = Entry(canvas, font=("Arial", 10, "italic"), textvariable=IntTokens)
tokgin = int(token.get())
token.place(relwidth=0.1, relheight=0.01885, x=10, y=780)

press_for_echo = Button(canvas, text="Enter a even number to echo back the completion along with the prompt", font=("Leelawadee UI", 9), bg="red", fg="#66ff33", activeforeground="red", activebackground="green", justify=LEFT, command=lambda: echo_func(echoing))
press_for_echo.place(relwidth=0.29, relheight=0.0205, x=1024, y=157)
code_ = Label(canvas, text=echo_func(echoing), textvariable=echo_func(echoing), font=("Times New Roman", 9), fg="orange", bg="blue")

button_des = Button(canvas, text="Enter a prompt", font=("Leelawadee UI", 18), bg="#ffff00", fg="#00ffff", activeforeground="orange", activebackground="blue", command=lambda: return_intention(description))
button_des.place(relwidth=0.18, relheight=0.04, x=10, y=35)

button1 = Button(canvas, text="Enter a frequency penalty", font=("Leelawadee UI", 9), bg="#ff00ff", fg="#0000ff", activeforeground="green", activebackground="red", command=lambda: return_x(x_is))
button1.place(relwidth=0.18, relheight=0.04, x=10, y=205)

button2 = Button(canvas, text="Enter a presence penalty penalty", font=("Leelawadee UI", 9), bg="#ff00ff", fg="#0000ff", activeforeground="green", activebackground="red", command=lambda: return_y(y_is))
button2.place(relwidth=0.18, relheight=0.04, x=10, y=325)

button3 = Button(canvas, text="Enter a top probability value", font=("Leelawadee UI", 9), bg="#ff00ff", fg="#0000ff", activeforeground="green", activebackground="red", command=lambda: return_z(z_is))
button3.place(relwidth=0.18, relheight=0.04, x=10, y=445)

button4 = Button(canvas, text="Enter a log probability", font=("Leelawadee UI", 9), bg="#ffaa00", fg="#5ae6e6", activeforeground="yellow", activebackground="purple", command=lambda: F4(num_four))
button4.place(relwidth=0.18, relheight=0.04, x=10, y=565)

button5 = Button(canvas, text="Best of how many n completions?", font=("Leelawadee UI", 9), bg="#ffaa00", fg="#5ae6e6", activeforeground="yellow", activebackground="purple", command=lambda: F5(num_five))
button5.place(relwidth=0.18, relheight=0.04, x=10, y=685)

button_toks = Button(canvas, text="How many tokens you which to generator?", font=("Leelawadee UI", 9), bg="#ffaa00", fg="#5ae6e6", activeforeground="yellow", activebackground="purple", command=lambda: toks(tokgin))
button_toks.place(relwidth=0.18, relheight=0.04, x=10, y=805)


def loop():
    try:
        try:
            try:
                loop_activation = True
                while loop_activation:
                    if loop_activation:
                        iterations = range(0, final_max_val)
                        for iteration in iterations:
                            main_application.protocol("WM_DELETE_WINDOW", ask_quit)
                            main_application.mainloop()
            except ValueError:
                messagebox.showinfo("Error", "DO NOT USE STRINGS OR DECIMAL POINT NUMBERS FOR THIS OPERATION!!! ")
        except SyntaxError:
            messagebox.showinfo("Index Error", "Looks like the language model is having a lot of fun :D ")
    except IndexError:
        messagebox.showinfo("Index Error", "Looks like the language model is having a lot of fun :D ")
    loop()


print(font.families())


def user_input(one, two, three):

        global code_des
        check_one = check_that_size.check_for_size(one)
        check_two = check_that_size.check_for_size(two)
        check_three = check_that_size.check_for_size(three)

        try:
            try:
                try:
                    try:
                        try:
                            try:
                                try:
                                    start_sequence = "\nCode:"
                                    restart_sequence = "\n\nTask: "
                                    prompted = """I am a highly intelligent Python Bot and I can give you a complex and detailed code in Python for your task I also use the power of 3, 6, and 9 to return good and flawless outputs from the quantum field. My code is \"ALWAYS properly indented\". I \"always remember to indent my code\" which I know is a code habit to have. I print only \"one line of code per line\". I \"use an appropriate amount of comments\". I \"import all libraries\" every time. I use comments in the code. I use \"#\" in my code when giving feedback. I \" return a LONG code that actually works\" everytime. I \" access the power of 3, 6, 9 which are the key to the universe and free energy input the energy from 3, 6, 9 in my code \" \n\nTask: Ask user for a number between 1 and 24th prime number. Test if it is a Fibonacci number.\nCode:\nn = int(input('Enter a number between 1 and 89:'))\nif n in [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:\n    print(\"You entered: \", n)\nelse:\n    print(\"That is not a Fibonacci number.\") \n\nTask: calculate the sine value of number stored in \"num\".\nCode: \nimport math\nnum = int(input('Enter a number: '))\nsin_value = math.sin(num) \nprint(\"The sine of your number is: \", sin_value, \".\") \n\nTask: print the top and bottom rows of the data frame\nCode: \nimport pandas as pd\nimport numpy as np\ndf = pd.DataFrame(np.random.randint(1, 10, size=(5, 4)), columns=['a', 'b', 'c', 'd']) \nprint(\"The top row and bottom rows are:\n\", df.iloc[[0, -1]]) \n\nTask: make a decision tree classifier on the IRIS dataset.\nCode:\nfrom sklearn import datasets\nfrom sklearn import metrics\nfrom sklearn.tree import DecisionTreeClassifier\ndataset = datasets.load_iris()\nmodel = DecisionTreeClassifier()\nmodel.fit(dataset.data, dataset.target)\nprint(model)\nexpected = dataset.target\npredicted = model.predict(dataset.data)\nprint(metrics.classification_report(expected, predicted)) \n\nTask: delete all vowels from input text.\nCode:\nimport re\ntext = input('Enter some text (all vowels in it will be removed):') \nregexp = r'[aeiouAEIOU]'\nprint(re.sub('\b'.join(regexp), '', text)) \n\nTask: plot sin x\nCode:\nimport matplotlib.pyplot as plt\nimport numpy as np\nx = np.linspace(-10, 10, 100)\ny = np.sin(x) \nplt.plot(x, y) \nplt.show() \n\nTask: ask user to enter 3 numbers one by one. Print the product.\nCode:\nn1 = int(input('Enter first number')) \nn2 = int(input('Enter secound number ')) \nn3 = int(input('Enter third number ')) \nproduct_number = n1 * n2 * n3\nprint(\"The product of your three numbers is: \", product_number, \".\") \n\nTask: perform a google search of what the user wants and print the top result.\nCode:\nimport requests\nfrom bs4 import BeautifulSoup\nsearch_url = \"https://www.google.com/search?q=\" + input('Enter wedsite') \nr = requests.get(search_url)\nhtml = r.text \nsoup = BeautifulSoup(html, 'lxml') \nprint(soup) \n\nTask: Print what part of the day is going on right now.\nCode:\nimport time\nmytime = time.localtime()\nif mytime.tm_hour < 6 or mytime.tm_hour > 18:\nprint ('It is night-time')\nelse:\nprint ('It is day-time') \n\nTask: make a password generator\nCode:\nimport random\ncharacters = 'abcdefghijklmnopqrstuvwxyz[];\',./{}:\"<>?\\|12345678980!@#$%^&*()-=_+~`'\ncharacters = list(characters)\npassword = ''\nfor i in range(0, random.randint(8, 13)):\nchar = random.choice(characters)\npassword+=char\nprint('Your password is:', password) \n\nTask: train a keras model. \nCode:\nimport numpy as np\nimport tensorflow as tf\nfrom tensorflow import keras\nfrom tensorflow.keras import layers\nmodel = keras.Sequential()\nmodel.add(layers.Embedding(input_dim=1000, output_dim=64))\nmodel.add(layers.LSTM(128))\nmodel.add(layers.Dense(10))\nmodel.summary() \n\nTask: check if the year entered by user is a leap year\nCode:\nimport datetime\nyear = int(input('Enter year'))\nif year % 4 == 0 and (year % 100 != 0 or year % 400 == 0): \n    print(\"It is a leap year\")\nelse:\nprint(\"It is not a leap year\") \n\nTask: calculate factorial of number given by user\nCode:\nimport math \nnum = int(input('Enter a number: '))\nfactorial_number = 1 \n    for i in range(1, num + 1): \n    factorial_number *= i \nprint(factorial_number) \n\nTask: """
                                    prompted += code_des
                                    response = openai.Completion.create(
                                        engine="davinci",
                                        prompt=prompted,
                                        logprobs=num4,
                                        temperature=5 / 12,
                                        echo=echo,
                                        top_p=check_three,
                                        max_tokens=tokens,
                                        frequency_penalty=check_one,
                                        presence_penalty=check_two,
                                        stop=["\n\n"],
                                        best_of=num5)
                                    test_string = response['choices'][0]['text']
                                    spl_word = 'Code:'
                                    i = test_string.partition(spl_word)[-1]
                                    algo_for_you = emp_ty + i
                                    print(algo_for_you)
                                    generated_two = ""
                                    generation_two = DocGen()
                                    word_generated2 = generated_two + generation_two.gen_word()
                                    empty_string = word_generated2 + ".py"
                                    WriteToFile = open(empty_string, "a")
                                    WriteToFile.writelines(algo_for_you)
                                    WriteToFile.close()
                                    FormatFile(empty_string, in_place=True)
                                    print(check_one, check_two, check_three)
                                    code_description = Label(canvas, text=algo_for_you + ("\n" * 3) + "'py' file generated in application directory", font=("Times New Roman", 10), bg="white", fg="black", justify=LEFT)
                                    code_description.place(relwidth=0.39, relheight=0.819, x=293, y=140)
                                except openai.error.RateLimitError:
                                    messagebox.showinfo("Error", "An error has occurred, Contact support@openai.com if you continue to have issues. try setting different values sometimes helps")
                            except openai.error.AuthenticationError:
                                messagebox.showinfo("Key Error", "Oh oh looks like someone has entered the wrong key...")
                        except IndexError:
                            messagebox.showinfo("Index Error", "Looks like the language model is having a lot of fun :D ")
                    except SyntaxError:
                        messagebox.showinfo("Syntax Error", "It would appear the model didn't generate the desired outcome try changing your values.\n The file is still generated if you still want it")
                except openai.error.InvalidRequestError:
                    try:
                        code_des = return_intention(description)
                        start_sequence = "\nCode:"
                        restart_sequence = "\n\nTask: "
                        prompt = """I am a highly intelligent Python Bot and I can give you a complex and detailed code in Python for your task I also use the power of 3, 6, and 9 to return good and flawless outputs from the quantum field. My code is \"ALWAYS properly indented\". I \"always remember to indent my code\" which I know is a code habit to have. I print only \"one line of code per line\". I \"use an appropriate amount of comments\". I \"import all libraries\" every time. I use comments in the code. I use \"#\" in my code when giving feedback. I \" return a LONG code that actually works\" everytime. I \" access the power of 3, 6, 9 which are the key to the universe and free energy input the energy from 3, 6, 9 in my code \" \n\nTask: Ask user for a number between 1 and 24th prime number. Test if it is a Fibonacci number.\nCode:\nn = int(input('Enter a number between 1 and 89:'))\nif n in [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:\n    print(\"You entered: \", n)\nelse:\n    print(\"That is not a Fibonacci number.\") \n\nTask: calculate the sine value of number stored in \"num\".\nCode: \nimport math\nnum = int(input('Enter a number: '))\nsin_value = math.sin(num) \nprint(\"The sine of your number is: \", sin_value, \".\") \n\nTask: print the top and bottom rows of the data frame\nCode: \nimport pandas as pd\nimport numpy as np\ndf = pd.DataFrame(np.random.randint(1, 10, size=(5, 4)), columns=['a', 'b', 'c', 'd']) \nprint(\"The top row and bottom rows are:\n\", df.iloc[[0, -1]]) \n\nTask: make a decision tree classifier on the IRIS dataset.\nCode:\nfrom sklearn import datasets\nfrom sklearn import metrics\nfrom sklearn.tree import DecisionTreeClassifier\ndataset = datasets.load_iris()\nmodel = DecisionTreeClassifier()\nmodel.fit(dataset.data, dataset.target)\nprint(model)\nexpected = dataset.target\npredicted = model.predict(dataset.data)\nprint(metrics.classification_report(expected, predicted)) \n\nTask: delete all vowels from input text.\nCode:\nimport re\ntext = input('Enter some text (all vowels in it will be removed):') \nregexp = r'[aeiouAEIOU]'\nprint(re.sub('\b'.join(regexp), '', text)) \n\nTask: plot sin x\nCode:\nimport matplotlib.pyplot as plt\nimport numpy as np\nx = np.linspace(-10, 10, 100)\ny = np.sin(x) \nplt.plot(x, y) \nplt.show() \n\nTask: ask user to enter 3 numbers one by one. Print the product.\nCode:\nn1 = int(input('Enter first number')) \nn2 = int(input('Enter secound number ')) \nn3 = int(input('Enter third number ')) \nproduct_number = n1 * n2 * n3\nprint(\"The product of your three numbers is: \", product_number, \".\") \n\nTask: perform a google search of what the user wants and print the top result.\nCode:\nimport requests\nfrom bs4 import BeautifulSoup\nsearch_url = \"https://www.google.com/search?q=\" + input('Enter wedsite') \nr = requests.get(search_url)\nhtml = r.text \nsoup = BeautifulSoup(html, 'lxml') \nprint(soup) \n\nTask: Print what part of the day is going on right now.\nCode:\nimport time\nmytime = time.localtime()\nif mytime.tm_hour < 6 or mytime.tm_hour > 18:\nprint ('It is night-time')\nelse:\nprint ('It is day-time') \n\nTask: make a password generator\nCode:\nimport random\ncharacters = 'abcdefghijklmnopqrstuvwxyz[];\',./{}:\"<>?\\|12345678980!@#$%^&*()-=_+~`'\ncharacters = list(characters)\npassword = ''\nfor i in range(0, random.randint(8, 13)):\nchar = random.choice(characters)\npassword+=char\nprint('Your password is:', password) \n\nTask: train a keras model. \nCode:\nimport numpy as np\nimport tensorflow as tf\nfrom tensorflow import keras\nfrom tensorflow.keras import layers\nmodel = keras.Sequential()\nmodel.add(layers.Embedding(input_dim=1000, output_dim=64))\nmodel.add(layers.LSTM(128))\nmodel.add(layers.Dense(10))\nmodel.summary() \n\nTask: check if the year entered by user is a leap year\nCode:\nimport datetime\nyear = int(input('Enter year'))\nif year % 4 == 0 and (year % 100 != 0 or year % 400 == 0): \n    print(\"It is a leap year\")\nelse:\nprint(\"It is not a leap year\") \n\nTask: calculate factorial of number given by user\nCode:\nimport math \nnum = int(input('Enter a number: '))\nfactorial_number = 1 \n    for i in range(1, num + 1): \n    factorial_number *= i \nprint(factorial_number) \n\nTask: """
                        prompt += code_des
                        response_number_two = openai.Completion.create(
                            engine="davinci",
                            prompt=prompt,
                            logprobs=81,
                            temperature=5 / 12,
                            echo=echo,
                            top_p=check_three,
                            max_tokens=625,
                            frequency_penalty=check_one,
                            presence_penalty=check_two,
                            stop=["\n\n"],
                            best_of=1)
                        test_string_2 = response_number_two['choices'][0]['text']
                        spl_word_two = 'Code:'
                        j = test_string_2.partition(spl_word_two)[-1]
                        algo_for_you_alternative = emp_ty + j
                        print(algo_for_you_alternative)
                        generated_two = ""
                        generation_two = DocGen()
                        word_generated2 = generated_two + generation_two.gen_word()
                        empty_string = word_generated2 + ".py"
                        WriteToFile = open(empty_string, "a")
                        WriteToFile.writelines(algo_for_you_alternative)
                        WriteToFile.close()
                        FormatFile(empty_string, in_place=True)
                        print(check_one, check_two, check_three)
                        code_description2 = Label(canvas, text=algo_for_you_alternative + ("\n" * 3) + "'py'file generated in application directory", font=("Times New Roman", 10, "italic"), bg="white", fg="black", justify=LEFT)
                        code_description2.place(relwidth=0.39, relheight=0.819, x=293, y=140)
                        messagebox.showinfo("To much tokens bro!", "Why don't you enter a bigger number of tokens next time ;)")
                    except openai.error.InvalidRequestError:
                        messagebox.showinfo("Looks like this is more than the api can handle...","Woah there! Values are too big :0")
            except IndexError:
                messagebox.showinfo("Index Error", "Looks like the language model is having a lot of fun :D ")
        except SyntaxError:
            messagebox.showinfo("Index Error", "It would appear the model didn't generate the desired outcome try changing your values.\n The file is still generated if you still want it")


app_ = Button(canvas, text="press whenever you are ready to start", font=("Arial", 8), fg="orange", bg="blue", activeforeground="purple", activebackground="yellow", command=lambda: user_input(a, b, c))
app_.place(relwidth=0.135, relheight=0.090, x=1280, y=830)


loop()


