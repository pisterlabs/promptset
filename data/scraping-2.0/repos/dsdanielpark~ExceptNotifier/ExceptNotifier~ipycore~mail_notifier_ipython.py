# -*- coding: utf-8 -*-
# Copyright 2023 parkminwoo
import datetime
import smtplib
import datetime
from os import environ
from IPython.core.ultratb import AutoFormattedTB
from ExceptNotifier.sender.openai_receiver import receive_openai_advice
from ExceptNotifier.sender.mail_sender import send_gmail_msg
from ExceptNotifier.sender.bard_receiver import receive_bard_advice


DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def ExceptMailIpython(
    shell: object, etype: object, evalue: object, tb: object, tb_offset=1
) -> None:
    """ExceptNotifier function for overriding custom execute in ipython for sending Gmail.

    :param shell: Excecuted shell, ZMQInteractiveShell object.
    :type shell: object
    :param etype: Error type
    :type etype: object
    :param evalue: Error value
    :type evalue: object
    :param tb: TraceBack object of Ipython
    :type tb: object
    :param tb_offset: Offset of traceback, defaults to 1
    :type tb_offset: int, optional
    """
    itb = AutoFormattedTB(mode="Plain", tb_offset=1)
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
    stb = itb.structured_traceback(etype, evalue, tb)
    sstb = itb.stb2text(stb)
    start_time = datetime.datetime.now()
    exceptNotifier = {
        "TO": environ["_GAMIL_RECIPIENT_ADDR"],
        "FROM": environ["_GMAIL_SENDER_ADDR"],
        "SUBJECT": "[Except Notifier] Error! Python Code Exception Detected.",
        "BODY": f"IMPORTANT WARNING \nPython Exception Detected in Your Code. \n\nHi there, \nThis is an exception catch notifier. \n\n * Code Status: Fail.🛠 \n * Detail: Python Code Ran Exceptions. \n * Time: {start_time.strftime(DATE_FORMAT)} \n\n ** {sstb}",
    }
    send_gmail_msg(
        environ["_GMAIL_SENDER_ADDR"],
        environ["_GAMIL_RECIPIENT_ADDR"],
        environ["_GMAIL_APP_PASSWORD_OF_SENDER"],
        exceptNotifier["SUBJECT"],
        exceptNotifier["BODY"],
    )

    if environ.get("_OPEN_AI_API") is not None:
        try:
            error_message = f"error sheel=={shell}, error_type_document=={etype.__doc__}, error_value=={evalue}, error message in ipython cell=={sstb}"
            advice_msg = receive_openai_advice(
                environ["_OPEN_AI_MODEL"], environ["_OPEN_AI_API"], error_message,
            )  # NO-QA
            exceptNotifier_openai = {
                "SUBJECT": "[Except AI Debugging] Error! chatGPT Debugging guide.",
                "BODY": f"IMPORTANT WARNING: \nPython Exception Detected in Your Code. \n\nHi there, \nThis is advice from OpenAI ChatGPT \n\n {advice_msg}",
            }
            send_gmail_msg(
                environ["_GMAIL_SENDER_ADDR"],
                environ["_GAMIL_RECIPIENT_ADDR"],
                environ["_GMAIL_APP_PASSWORD_OF_SENDER"],
                exceptNotifier_openai["SUBJECT"],
                exceptNotifier_openai["BODY"],
            )

        except Exception as e:
            pass

    if environ.get("_BARD_API_KEY") is not None:
        try:
            error_message = f"error sheel=={shell}, error_type_document=={etype.__doc__}, error_value=={evalue}, error message in ipython cell=={sstb}"
            advice_msg = receive_bard_advice(
                environ["_BARD_API_KEY"], error_message,
            )  # NO-QA
            exceptNotifier_openai = {
                "SUBJECT": "[Except AI Debugging] Error! Google Bard Debugging guide.",
                "BODY": f"IMPORTANT WARNING: \nPython Exception Detected in Your Code. \n\nHi there, \nThis is advice from OpenAI ChatGPT \n\n {advice_msg}",
            }
            send_gmail_msg(
                environ["_GMAIL_SENDER_ADDR"],
                environ["_GAMIL_RECIPIENT_ADDR"],
                environ["_GMAIL_APP_PASSWORD_OF_SENDER"],
                exceptNotifier_openai["SUBJECT"],
                exceptNotifier_openai["BODY"],
            )

        except Exception as e:
            pass
