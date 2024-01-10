import re
import openai
import os
import time
from peewee import *
import datetime
from dotenv import load_dotenv
import tiktoken  # Cost 

request_amount = int()
enc = tiktoken.get_encoding("cl100k_base")  # Cost
db = SqliteDatabase('Tickets.db')
price = int(0)
user_problem = str()
answer = str()
state = False


# Database Classes
class Ticket(Model):
    user = CharField()
    CI = CharField()
    ticket_create_date = DateTimeField(default=datetime.datetime.now)
    user_input = TextField()
    final_chat_output = TextField()
    question_rounds = IntegerField()
    is_problem_fixed = BooleanField()
    cost = IntegerField()

    class Meta:
        database = db

# General function


class functions():

    def exit():
        # function to exit program
        exit()

    def create_ticket(username, client, user_problem, answer, request_amount, state, price):
        # function to create ticket
        Ticket.create(user=username, CI=client, user_input=user_problem, final_chat_output=answer,
                      question_rounds=request_amount, is_problem_fixed=state, cost=price)

    def keyword_check(user_question_split):
        # function to compare question to keyword list
        for question in user_question_split:
            for keyword in tech_support_keywords:
                if question == keyword:
                    return True
        return False


tech_support_keywords = [
    "error",
    "bug",
    "glitch",
    "crash",
    "freeze",
    "slow",
    "lag",
    "network",
    "connection",
    "wifi",
    "router",
    "modem",
    "internet",
    "software",
    "hardware",
    "driver",
    "printer",
    "scanner",
    "keyboard",
    "mouse",
    "monitor",
    "screen",
    "display",
    "resolution",
    "battery",
    "charging",
    "power",
    "charging",
    "virus",
    "malware",
    "spyware",
    "firewall",
    "security",
    "backup",
    "restore",
    "update",
    "upgrade",
    "install",
    "uninstall",
    "configure",
    "settings",
    "option",
    "menu",
    "browser",
    "chrome",
    "firefox",
    "safari",
    "edge",
    "operating system",
    "windows",
    "macos",
    "linux",
    "ios",
    "android",
    "application",
    "program",
    "file",
    "folder",
    "directory",
    "storage",
    "memory",
    "RAM",
    "CPU",
    "GPU",
    "motherboard",
    "BIOS",
    "boot",
    "startup",
    "shutdown",
    "login",
    "password",
    "username",
    "account",
    "email",
    "mail",
    "outlook",
    "gmail",
    "yahoo",
    "hotmail",
    "pop3",
    "imap",
    "smtp",
    "server",
    "host",
    "domain",
    "website",
    "webpage",
    "html",
    "css",
    "javascript",
    "coding",
    "programming",
    "debugging",
    "testing",
    "quality assurance",
    "QA",
    "support",
    "customer service",
    "help desk",
    "ticket",
    "chat",
    "phone",
    "email",
    "remote",
    "onsite",
    "diagnostics",
    "repair",
    "replacement",
    "warranty",
    "insurance",
    "technical",
    "IT",
    "information technology",
    "networking",
    "wireless",
    "ethernet",
    "cable",
    "fiber",
    "VPN",
    "remote access",
    "firewall",
    "router",
    "switch",
    "hub",
    "server",
    "workstation",
    "desktop",
    "laptop",
    "tablet",
    "smartphone",
    "cell phone",
    "mobile",
    "access point",
    "repeater",
    "extender",
    "NAS",
    "cloud",
    "backup",
    "data recovery",
    "data migration",
    "data transfer",
    "encryption",
    "password protection",
    "security",
    "antivirus",
    "antimalware",
    "firewall",
    "intrusion detection",
    "incident response",
    "log analysis",
    "patch management",
    "vulnerability assessment",
    "penetration testing",
    "auditing",
    "compliance",
    "HIPAA",
    "PCI DSS",
    "SOX",
    "GDPR",
    "CIS",
    "NIST",
    "ISO",
    "ITIL",
    "DevOps",
    "agile",
    "SCRUM",
    "files",
    "computer"
]
