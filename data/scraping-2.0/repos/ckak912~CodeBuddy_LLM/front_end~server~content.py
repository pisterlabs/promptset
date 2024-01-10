import atexit
import random
from datetime import datetime
import gzip
from helper import *
from imgcompare import *
import json
import math
import re
import sqlite3
import traceback
from yaml import load
#import zipfile

# IMPORTANT: When creating/modifying queries that include any user input,
#            please follow the recommendations on this page:
#            https://realpython.com/prevent-python-sql-injection/

BLANK_IMAGE = "/9j/4AAQSkZJRgABAQEAlgCWAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCALQA8ADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/KKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//Z"

class Content:
    def __init__(self, settings_dict):
        self.__settings_dict = settings_dict

        self.conn = open_db("CodeBuddy.db")
        self.conn.row_factory = sqlite3.Row

        self.execute("PRAGMA foreign_keys=OFF")
        self.execute("PRAGMA cache_size=1000000")
        self.execute("PRAGMA mmap_size=1000000")
        self.execute("PRAGMA temp_store=MEMORY")
        self.execute("PRAGMA journal_mode=OFF")

        atexit.register(self.close)

    def close(self):
        self.conn.close()

    def execute(self, sql, params=()):
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        lastrowid = cursor.lastrowid
        cursor.close()

        return lastrowid
    
    def execute_multiple(self, sql_statements, params_list, lastrowid_index=-1):
        if len(sql_statements) != len(params_list):
            raise Exception(f"The size of sql_statements ({len(sql_statements)}) must be identical to the size of param_tuples ({len(params_list)}).")

        cursor = self.conn.cursor()
        cursor.execute("BEGIN")

        lastrowid = -1
        for i, sql in enumerate(sql_statements):
            cursor.execute(sql, params_list[i])

            if i == lastrowid_index:
                lastrowid = cursor.lastrowid

        self.conn.commit()
        cursor.close()

        return lastrowid

    def fetchone(self, sql, params=()):
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        result = cursor.fetchone()
        cursor.close()

        return result

    def fetchall(self, sql, params=()):
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        result = cursor.fetchall()
        cursor.close()

        return result

    # This function creates tables as they were in version 5. Subsequent changes
    #   to the database are implemented as migration scripts.
    def create_database_tables(self):
        print("Creating the initial database schema...")
        self.execute('''CREATE TABLE IF NOT EXISTS metadata (version integer NOT NULL);''')
        self.execute('''INSERT INTO metadata (version) VALUES (5);''')

        self.execute('''CREATE TABLE IF NOT EXISTS users (
                          user_id text PRIMARY KEY,
                          name text,
                          given_name text,
                          family_name text,
                          picture text,
                          locale text,
                          ace_theme text NOT NULL DEFAULT "tomorrow");''')

        self.execute('''CREATE TABLE IF NOT EXISTS permissions (
                          user_id text NOT NULL,
                          role text NOT NULL,
                          course_id integer,
                        FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE ON UPDATE CASCADE
                     );''')

        self.execute('''CREATE TABLE IF NOT EXISTS course_registration (
                          user_id text NOT NULL,
                          course_id integer NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE ON UPDATE CASCADE,
                        FOREIGN KEY (course_id) REFERENCES courses (course_id) ON DELETE CASCADE ON UPDATE CASCADE
                     );''')

        self.execute('''CREATE TABLE IF NOT EXISTS courses (
                          course_id integer PRIMARY KEY AUTOINCREMENT,
                          title text NOT NULL UNIQUE,
                          introduction text,
                          visible integer NOT NULL,
                          passcode text,
                          date_created timestamp NOT NULL,
                          date_updated timestamp NOT NULL
                     );''')

        self.execute('''CREATE TABLE IF NOT EXISTS assignments (
                          course_id integer NOT NULL,
                          assignment_id integer PRIMARY KEY AUTOINCREMENT,
                          title text NOT NULL,
                          introduction text,
                          visible integer NOT NULL,
                          start_date timestamp,
                          due_date timestamp,
                          allow_late integer,
                          late_percent real,
                          view_answer_late integer,
                          has_timer int NOT NULL,
                          hour_timer int,
                          minute_timer int,
                          date_created timestamp NOT NULL,
                          date_updated timestamp NOT NULL,
                        FOREIGN KEY (course_id) REFERENCES courses (course_id) ON DELETE CASCADE ON UPDATE CASCADE
                     );''')

        self.execute('''CREATE TABLE IF NOT EXISTS problems (
                          course_id integer NOT NULL,
                          assignment_id integer NOT NULL,
                          problem_id integer PRIMARY KEY AUTOINCREMENT,
                          title text NOT NULL,
                          visible integer NOT NULL,
                          answer_code text NOT NULL,
                          answer_description text,
                          hint text,
                          max_submissions integer NOT NULL,
                          credit text,
                          data_url text,
                          data_file_name text,
                          data_contents text,
                          back_end text NOT NULL,
                          expected_text_output text NOT NULL,
                          expected_image_output text NOT NULL,
                          instructions text NOT NULL,
                          output_type text NOT NULL,
                          show_answer integer NOT NULL,
                          show_student_submissions integer NOT NULL,
                          show_expected integer NOT NULL,
                          show_test_code integer NOT NULL,
                          test_code text,
                          date_created timestamp NOT NULL,
                          date_updated timestamp NOT NULL,
                          FOREIGN KEY (course_id) REFERENCES courses (course_id) ON DELETE CASCADE ON UPDATE CASCADE,
                          FOREIGN KEY (assignment_id) REFERENCES assignments (assignment_id) ON DELETE CASCADE ON UPDATE CASCADE
                     );''')

        self.execute('''CREATE TABLE IF NOT EXISTS submissions (
                          course_id integer NOT NULL,
                          assignment_id integer NOT NULL,
                          problem_id integer NOT NULL,
                          user_id text NOT NULL,
                          submission_id integer NOT NULL,
                          code text NOT NULL,
                          text_output text NOT NULL,
                          image_output text NOT NULL,
                          passed integer NOT NULL,
                          date timestamp NOT NULL,
                        FOREIGN KEY (course_id) REFERENCES courses (course_id) ON DELETE CASCADE ON UPDATE CASCADE,
                        FOREIGN KEY (assignment_id) REFERENCES assignments (assignment_id) ON DELETE CASCADE ON UPDATE CASCADE,
                        FOREIGN KEY (problem_id) REFERENCES problems (problem_id) ON DELETE CASCADE ON UPDATE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE ON UPDATE CASCADE,
                        PRIMARY KEY (course_id, assignment_id, problem_id, user_id, submission_id)
                     );''')

        self.execute('''CREATE TABLE IF NOT EXISTS scores (
                          course_id integer NOT NULL,
                          assignment_id integer NOT NULL,
                          problem_id integer NOT NULL,
                          user_id text NOT NULL,
                          score real NOT NULL,
                        FOREIGN KEY (course_id) REFERENCES courses (course_id) ON DELETE CASCADE ON UPDATE CASCADE,
                        FOREIGN KEY (assignment_id) REFERENCES assignments (assignment_id) ON DELETE CASCADE ON UPDATE CASCADE,
                        FOREIGN KEY (problem_id) REFERENCES problems (problem_id) ON DELETE CASCADE ON UPDATE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE ON UPDATE CASCADE,
                        PRIMARY KEY (course_id, assignment_id, problem_id, user_id)
                     );''')

        self.execute('''CREATE TABLE IF NOT EXISTS user_assignment_start (
                          user_id text NOT NULL,
                          course_id text NOT NULL,
                          assignment_id text NOT NULL,
                          start_time timestamp NOT NULL,
                        FOREIGN KEY (course_id) REFERENCES courses (course_id) ON DELETE CASCADE ON UPDATE CASCADE,
                        FOREIGN KEY (assignment_id) REFERENCES assignments (assignment_id) ON DELETE CASCADE ON UPDATE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE ON UPDATE CASCADE
                     );''')

    def get_database_version(self):
        sql = "SELECT COUNT(*) AS count FROM sqlite_master"

        # This tells us whether we have created the initial database scheme.
        # If not, create it.
        num_tables = self.fetchone(sql)["count"]

        if num_tables == 0:
            self.create_database_tables()
            return 5

        sql = '''SELECT MAX(version) AS version
                 FROM metadata'''

        return self.fetchone(sql)["version"]

    def update_database_version(self, version):
        print(f"Updating database to version {version}")

        sql = '''DELETE FROM metadata'''
        self.execute(sql)

        sql = '''INSERT INTO metadata (version)
                 VALUES (?)'''
        self.execute(sql, (version,))

        print(f"Done updating database to version {version}")

    def find_when_content_updated(self):
        sql = '''SELECT scope, when_updated
                 FROM when_content_updated'''

        updated_dict = {}

        try:
            for row in self.fetchall(sql):
                updated_dict[row["scope"]] = str(row["when_updated"])
        except:
            print(traceback.format_exc())

        return updated_dict
    
    def update_when_content_updated(self, scope):
        sql = '''UPDATE when_content_updated
                 SET when_updated = datetime('now')
                 WHERE scope = ?'''

        try:
            self.execute(sql, (scope, ))
        except:
            print(traceback.format_exc())

    def delete_content_updated(self, scope):
        sql = '''DELETE FROM when_content_updated
                 WHERE scope = ?'''

        try:
            self.execute(sql, (scope, ))
        except:
            print(traceback.format_exc())

    def set_user_assignment_start_time(self, course_id, assignment_id, user_id, start_time):
        sql = '''INSERT INTO user_assignment_starts (course_id, assignment_id, user_id, start_time)
                 VALUES (?, ?, ?, ?)'''

        self.execute(sql, (course_id, assignment_id, user_id, start_time,))

    def get_user_assignment_start_time(self, course_id, assignment_id, user_id):
        sql = '''SELECT start_time
                 FROM user_assignment_starts
                 WHERE course_id = ?
                   AND assignment_id = ?
                   AND user_id = ?'''

        row = self.fetchone(sql, (course_id, assignment_id, user_id,))
        if row:
            return row["start_time"]

    def is_taking_restricted_assignment(self, user_id, assignment_id):
        sql = '''SELECT ((julianday(datetime('now')) - julianday(latest_start_time)) * 24 * 60) < minute_limit AS yes
                 FROM
                 (SELECT (hour_timer * 60 + minute_timer) AS minute_limit, MAX(start_time) AS latest_start_time
                  FROM user_assignment_starts uas
			            INNER JOIN assignments a
			               ON uas.course_id = a.course_id
				            AND uas.assignment_id = a.assignment_id
                  WHERE uas.user_id = ?
                    AND a.assignment_id != ?
			              AND a.has_timer = 1
		              )'''

        row = self.fetchone(sql, (user_id, assignment_id, ))
        if row:
            return bool(row["yes"])
        return False

    def get_all_user_assignment_expired(self, course_id, assignment_id):
        user_dict = {}

        sql = '''SELECT user_id, start_time
                 FROM user_assignment_starts
                 WHERE course_id = ?
                   AND assignment_id = ?'''

        for row in self.fetchall(sql, (course_id, assignment_id,)):
            #TODO: It is not efficient to hit the database each time through this loop.
            start_time = datetime.strftime(row["start_time"], "%a, %d %b %Y %H:%M:%S %Z")
            timer_ended = self.has_user_assignment_start_timer_ended(course_id, assignment_id, start_time)
            user_dict[row["user_id"]] = timer_ended

        return user_dict

    def has_user_assignment_start_timer_ended(self, course_id, assignment_id, start_time):
        if not start_time:
            return False

        curr_time = datetime.utcnow()
        start_time = datetime.strptime(start_time, "%a, %d %b %Y %H:%M:%S ")

        sql = '''SELECT hour_timer, minute_timer
                 FROM assignments
                 WHERE course_id = ?
                   AND assignment_id = ?'''

        row = self.fetchone(sql, (course_id, assignment_id,))

        if row:
            #TODO: This logic could be much simpler using datetime module.
            elapsed_time = curr_time - start_time
            seconds = elapsed_time.total_seconds()
            e_hours = math.floor(seconds/3600)
            e_minutes = math.floor((seconds/60) - (e_hours*60))
            e_seconds = (seconds - (e_minutes*60) - (e_hours*3600))

            if e_hours > int(row["hour_timer"]):
                return True
            elif e_hours == int(row["hour_timer"]) and e_minutes > int(row["minute_timer"]):
                return True
            elif e_hours == int(row["hour_timer"]) and e_minutes == int(row["minute_timer"]) and e_seconds > 0:
                return True

        return False

    def reset_user_assignment_start_timer(self, course_id, assignment_id, user_id):
        sql = '''DELETE FROM user_assignment_starts
                 WHERE course_id = ?
                   AND assignment_id = ?
                   AND user_id = ?'''

        self.execute(sql, (course_id, assignment_id, user_id))

    def user_exists(self, user_id):
        sql = '''SELECT user_id
                 FROM users
                 WHERE user_id = ?'''

        return self.fetchone(sql, (user_id,)) != None

    def administrator_exists(self):
        sql = '''SELECT COUNT(*) AS num_administrators
                 FROM permissions
                 WHERE role = "administrator"'''

        return self.fetchone(sql)["num_administrators"]

    def is_administrator(self, user_id):
        return self.user_has_role(user_id, 0, "administrator")

    def user_has_role(self, user_id, course_id, role):
        sql = '''SELECT COUNT(*) AS has_role
                 FROM permissions
                 WHERE role = ?
                   AND user_id = ?
                   AND course_id = ?'''

        return self.fetchone(sql, (role, user_id, course_id, ))["has_role"] > 0

    def get_users_from_role(self, course_id, role):
        sql = '''SELECT user_id
                 FROM permissions
                 WHERE role = ?
                   AND (course_id = ? OR course_id IS NULL)'''

        rows = self.fetchall(sql, (role, course_id,))
        return [row["user_id"] for row in rows]

    def get_course_id_from_role(self, user_id):
        sql = '''SELECT course_id
                 FROM permissions
                 WHERE user_id = ?'''

        row = self.fetchone(sql, (user_id,))

        if row:
            return row["course_id"]
        else:
            return -1 # The user is a student.

    def set_user_dict_defaults(self, user_dict):
        if "name" not in user_dict:
            user_dict["name"] = "[Unknown name]"
        if "given_name" not in user_dict:
            user_dict["given_name"] = "[Unknown given name]"
        if "family_name" not in user_dict:
            user_dict["family_name"] = "[Unknown family name]"
        if "locale" not in user_dict:
            user_dict["locale"] = ""

    def add_user(self, user_id, user_dict):
        self.set_user_dict_defaults(user_dict)

        sql = '''INSERT INTO users (user_id, name, given_name, family_name, locale, ace_theme, email_address)
                 VALUES (?, ?, ?, ?, ?, ?, ?)'''

        self.execute(sql, (user_id, user_dict["name"], user_dict["given_name"], user_dict["family_name"], user_dict["locale"], "tomorrow", user_dict["email_address"]))

        self.update_when_content_updated("user")

    def register_user_for_course(self, course_id, user_id):
        sql = '''INSERT INTO course_registrations (course_id, user_id)
                 VALUES (?, ?)'''

        self.execute(sql, (course_id, user_id,))

        self.update_when_content_updated("user")
        self.update_when_content_updated(str(course_id))

    def unregister_user_from_course(self, course_id, user_id):
        self.execute('''DELETE FROM course_registrations
                        WHERE course_id = ?
                          AND user_id = ?''', (course_id, user_id, ))

        self.execute('''DELETE FROM scores
                        WHERE course_id = ?
                          AND user_id = ?''', (course_id, user_id, ))

        self.execute('''DELETE FROM submissions
                        WHERE course_id = ?
                          AND user_id = ?''', (course_id, user_id, ))

        self.execute('''DELETE FROM user_assignment_starts
                        WHERE course_id = ?
                          AND user_id = ?''', (course_id, user_id, ))
        
        self.update_when_content_updated("user")
        self.update_when_content_updated(str(course_id))

    def is_user_registered(self, course_id, user_id):
        sql = '''SELECT 1
                 FROM course_registrations
                 WHERE course_id = ?
                   AND user_id = ?'''

        if self.fetchone(sql, (course_id, user_id,)):
            return True

        return False

    def get_user_info(self, user_id):
        null_user_info = {"user_id": None, "name": None, "given_name": None, "family_name": None, "locale": None, "email_address": None, "ace_theme": None, "use_auto_complete": True, "use_studio_mode": True, "enable_vim": False}

        sql = '''SELECT *
                 FROM users
                 WHERE user_id = ?'''

        user = self.fetchone(sql, (user_id,))

        if not user:
            return null_user_info

        return {"user_id": user_id, "name": user["name"], "given_name": user["given_name"], "family_name": user["family_name"], "locale": user["locale"], "email_address": user["email_address"], "ace_theme": user["ace_theme"], "use_auto_complete": user["use_auto_complete"], "use_studio_mode": user["use_studio_mode"], "enable_vim": user["enable_vim"]}

    def add_permissions(self, course_id, user_id, role):
        sql = '''SELECT role
                 FROM permissions
                 WHERE user_id = ?
                   AND (course_id = ? OR course_id IS NULL)'''

        # Admins are not assigned to a particular course.
        if not course_id:
            course_id = 0

        role_exists = self.fetchone(sql, (user_id, int(course_id),)) != None

        if role_exists:
            sql = '''UPDATE permissions
                     SET role = ?, course_id = ?
                     WHERE user_id = ?'''

            self.execute(sql, (role, course_id, user_id,))
        else:
            sql = '''INSERT INTO permissions (user_id, role, course_id)
                     VALUES (?, ?, ?)'''

            self.execute(sql, (user_id, role, course_id,))

        if course_id != 0:
            self.update_when_content_updated(str(course_id))

        self.update_when_content_updated("user")            

    def remove_permissions(self, course_id, user_id, role):
        sql = '''DELETE FROM permissions
                 WHERE user_id = ?
                   AND role = ?
                   AND (course_id = ? OR course_id IS NULL)'''

        # Admins are not assigned to a particular course.
        if not course_id:
            course_id = "0"

        self.execute(sql, (user_id, role, int(course_id),))

        if course_id != "0":
            self.update_when_content_updated(course_id)

        self.update_when_content_updated("user")

    def add_admin_permissions(self, user_id):
        self.add_permissions(None, user_id, "administrator")
        self.update_when_content_updated("user")

    def get_user_count(self):
        sql = '''SELECT COUNT(*) AS count
                 FROM users'''

        return self.fetchone(sql)["count"]

    def get_all_courses(self):
        courses = []

        sql = '''SELECT course_id, title, visible, introduction, passcode
                 FROM courses
                 WHERE visible = 1
                 ORDER BY title'''

        for course in self.fetchall(sql):
            if course["visible"] or show_hidden:
                course_basics = {"id": course["course_id"], "title": course["title"], "visible": course["visible"], "introduction": course["introduction"], "passcode": course["passcode"], "exists": True}
                courses.append([course["course_id"], course_basics])

        return courses

    def get_registered_courses(self, user_id):
        registered_courses = []

        sql = '''SELECT r.course_id, c.title, c.introduction, 'student' AS role
                 FROM course_registrations r
                 INNER JOIN courses c
                   ON r.course_id = c.course_id
                 WHERE r.user_id = ?
                   AND c.visible = 1

                 UNION

                 SELECT c.course_id, c.title, c.introduction, p.role AS role
                 FROM permissions p
                 INNER JOIN courses c
                   ON p.course_id = 0 OR p.course_id = c.course_id
                 WHERE p.user_id = ?
                 ORDER BY c.title'''

        unique_course_ids = set()
        
        for course in self.fetchall(sql, (user_id, user_id, )):
            course_id = course["course_id"]

            if course_id not in unique_course_ids:
                unique_course_ids.add(course_id)

                course_basics = {"id": course_id, "title": course["title"], "introduction": course["introduction"], "role": course["role"]}

                registered_courses.append([course["course_id"], course_basics])

        return registered_courses
    
    def retrieve_llm_feedback(self, exercise_id, course_id, assignment_id):

      # SQL query to retrieve feedback information from LLM_stuff table
      sql = '''SELECT LF.exercise_feedback, LF.LLM_assignment_id, LF.LLM_course_id, LF.LLM_exercise_id 
                  FROM LLM_stuff as LF
                  JOIN exercises as E on LF.LLM_exercise_id = E.exercise_id
                  WHERE LF.LLM_exercise_id = ? AND E.course_id = ? AND E.assignment_id = ?
                '''
      # Fetch feedback data from the database
      result = self.fetchall(sql, (exercise_id, course_id, assignment_id))
      
      if result and result[0]:
          return result[0]
      else:
          return None
      # return result[0]

    # This function contains the SQL logic for storing steps after you have retrieved them from OpenAI
    def store_llm_feedback(self, exercise_id, course_id, assignment_id, exercise_feedback_json_str):
      print(f"feedback beings stored: {exercise_feedback_json_str}")
      sql = '''INSERT OR REPLACE INTO LLM_STUFF (LLM_exercise_id, LLM_course_id, LLM_assignment_id, exercise_feedback)
                VALUES (?, ?, ?, ?)
              '''

      result = self.execute(sql, (exercise_id, course_id, assignment_id, exercise_feedback_json_str))
      if result:
          return result
      
      return None
    
    def store_hint_code(self, exercise_id, course_id, assignment_id, hint_code):
      sql = '''INSERT OR REPLACE INTO LLM_STUFF (LLM_exercise_id, LLM_course_id, LLM_assignment_id, hint_code)
                VALUES (?, ?, ?, ?)
              '''

      result = self.execute(sql, (exercise_id, course_id, assignment_id, hint_code))

      if result:
          return result
      
      return None

    # This function contains the SQL logic for storing pseudo code generated from the model
    def store_pseudo_code(self, exercise_id, course_id, assignment_id, pseudo_code):
      try:
        sql = '''INSERT OR REPLACE INTO LLM_STUFF (LLM_exercise_id, LLM_course_id, LLM_assignment_id, pseudo_code)
                  VALUES (?, ?, ?, ?)
                '''

        result = self.execute(sql, (exercise_id, course_id, assignment_id, pseudo_code))

        if result:
            return result

      except Exception as inst:
        print("Error in store_pseudo_code:", inst)

      return None

    def store_user_code(self, exercise_id, course_id, assignment_id, user_code):
      print("hello this is the store_user_code")
      print(f"this is the user code being stored: {user_code}")
      try:
        sql = '''INSERT OR REPLACE INTO LLM_user_code (exercise_id, course_id, assignment_id, code, date_created)
                  VALUES (?, ?, ?, ?, ?)
                '''

        result = self.execute(sql, (exercise_id, course_id, assignment_id, user_code, datetime.utcnow()) )

        if result:
            return result

      except Exception as inst:
        print("Error in store_user_code:", inst)

      return None

    def retrieve_pseudo_code(self, exercise_id, course_id, assignment_id):
      try:
        # SQL query to retrieve pseudo code from LLM_stuff table
        sql = '''SELECT LF.pseudo_code, LF.LLM_assignment_id, LF.LLM_course_id, LF.LLM_exercise_id 
                    FROM LLM_stuff as LF
                    JOIN exercises as E on LF.LLM_exercise_id = E.exercise_id
                    WHERE LF.LLM_exercise_id = ? AND E.course_id = ? AND E.assignment_id = ?
                  '''

        # Fetch feedback data from the database
        result = self.fetchall(sql, (exercise_id, course_id, assignment_id))
        
        if result and result[0]:
            return result[0]
        else:
            return None

      except Exception as inst:
        print("Error in retrieve_pseudo_code:", inst)

      return None
      
    def retrieve_hint_code(self, exercise_id, course_id, assignment_id):
      # SQL query to retrieve hint code from LLM_stuff table
      sql = '''SELECT LF.hint_code, LF.LLM_assignment_id, LF.LLM_course_id, LF.LLM_exercise_id 
                  FROM LLM_stuff as LF
                  JOIN exercises as E on LF.LLM_exercise_id = E.exercise_id
                  WHERE LF.LLM_exercise_id = ? AND E.course_id = ? AND E.assignment_id = ?
                '''

      # Fetch feedback data from the database
      result = self.fetchall(sql, (exercise_id, course_id, assignment_id))
      
      if result and result[0]:
          return result[0]
      else:
          return None
      

    # TODO: This function could probably be removed. get_exercise_statuses could be used in its place.
    def get_exercises(self, course_basics, assignment_basics, show_hidden=True):
        sql = '''SELECT exercise_id, title, visible, enable_pair_programming
                 FROM exercises
                 WHERE course_id = ?
                   AND assignment_id = ?
                 ORDER BY title'''

        exercises = []
        for exercise in self.fetchall(sql, (course_basics["id"], assignment_basics["id"],)):
            if exercise["visible"] or show_hidden:
                exercises.append(dict(exercise))

        exercises = sort_list_of_dicts_nicely(exercises, ["title", "exercise_id"])

        exercises2 = []
        for exercise in exercises:
            exercise_basics = {"enable_pair_programming": exercise["enable_pair_programming"], "id": exercise["exercise_id"], "title": exercise["title"], "visible": exercise["visible"], "exists": True, "assignment": assignment_basics}

            exercises2.append([exercise["exercise_id"], exercise_basics, course_basics['id'], assignment_basics['id']])

        return exercises2

    def get_partner_info(self, course_id):
        # Gets list of users.
        users = [x[1] for x in self.get_registered_students(course_id)]

        # Adds users to dict to find duplicate names.
        user_duplicates_dict = {}
        for user in users:
            if user["name"] in user_duplicates_dict.keys():
                user_duplicates_dict[user["name"]].append({'id': user["id"], 'email': user['email']})
            else:
                user_duplicates_dict[user["name"]] = [{'id': user["id"], 'email': user['email']}]

        # Adds all users to a dictionary with name (and obscured email if applicable) as key and id as value.
        user_dict = {}
        for user in user_duplicates_dict:
            if len(user_duplicates_dict[user]) > 1:
                for user_info in user_duplicates_dict[user]:
                    user_dict[user + " — " + self.obscure_email(user_info['email'], list(map(lambda x: x['email'], user_duplicates_dict[user])))] = user_info['id']
            else:
                user_dict[user] = user_duplicates_dict[user][0]['id']

        return user_dict

    def obscure_email(self, full_email, all_emails):
        email = full_email.split("@")[0] if "@" in full_email else full_email
        email_end = full_email.split("@")[1] if "@" in full_email else full_email

        temp_email = email[0]
        for other_email in all_emails:
            other_email = other_email.split("@")[0] if "@" in other_email else other_email
            if other_email == email:
                pass
            else:
                for i in range(len(temp_email), len(min(email, other_email))):
                    if  temp_email == other_email[:i]:
                        temp_email = temp_email + email[i]
                    else:
                        break

        # Obscures all but essential characters of email.
        return temp_email + (("*")*(len(email)-len(temp_email))) + "@" + email_end

    def get_registered_students(self, course_id):
        registered_students = []

        sql = '''SELECT r.user_id, u.name, u.email_address
                 FROM course_registrations r
                 INNER JOIN users u
                   ON r.user_id = u.user_id
                 WHERE r.course_id = ?
                 ORDER BY u.name'''

        for student in self.fetchall(sql, (course_id,)):
            student_info = {"id": student["user_id"], "name": student["name"], 'email': student['email_address']}
            registered_students.append([student["user_id"], student_info])

        return registered_students

    # Indicates whether or not a student has passed each assignment in the course.
    async def get_assignment_statuses(self, course_id, user_id, show_hidden):
        course_basics = self.get_course_basics(course_id)

        sql = '''SELECT assignment_id,
                        title,
                        visible,
                        start_date,
                        due_date,
                        SUM(passed) AS num_passed,
                        COUNT(assignment_id) AS num_exercises,
                        SUM(passed) = COUNT(assignment_id) AS passed,
                        (SUM(passed) > 0 OR num_submissions > 0) AND SUM(passed) < COUNT(assignment_id) AS in_progress,
                        minutes_since_start,
                        has_timer,
                        hour_timer,
                        minute_timer,
                        restrict_other_assignments
                 FROM (
                   SELECT a.assignment_id,
                          a.title,
                          a.visible,
                          a.start_date,
                          a.due_date,
                          IFNULL(MAX(s.passed), 0) AS passed,
                          COUNT(s.submission_id) AS num_submissions,
                          a.has_timer,
                          a.hour_timer,
                          a.minute_timer,
                          a.restrict_other_assignments,
													(JulianDay(DATETIME('now')) - JulianDay(uas.start_time)) * 24 * 60 AS minutes_since_start
                   FROM exercises e
                   LEFT JOIN submissions s
                     ON e.course_id = s.course_id
                     AND e.assignment_id = s.assignment_id
                     AND e.exercise_id = s.exercise_id
                     AND (s.user_id = ? OR s.user_id IS NULL)
                   INNER JOIN assignments a
                     ON e.course_id = a.course_id
                     AND e.assignment_id = a.assignment_id
                   LEFT JOIN user_assignment_starts uas
                     ON a.course_id = uas.course_id
										 AND a.assignment_id = uas.assignment_id
                     AND a.has_timer = 1
										 AND (uas.user_id = ? OR uas.user_id IS NULL)
                   WHERE e.course_id = ?
                     AND e.visible = 1
                   GROUP BY e.assignment_id, e.exercise_id
                 )
                 GROUP BY assignment_id, title
                 ORDER BY title'''

        statuses = []
        for row in self.fetchall(sql, (user_id, user_id, course_id,)):
            assignment = dict(row)

            if assignment["visible"] or show_hidden:
                statuses.append(assignment)

        statuses2 = []

        # We have to check for this because otherwise the instructor has to make a submission before students will see the assignments.
        if len(statuses) == 0:
            for assignment_basics in self.get_assignments(course_basics, show_hidden):
                assignment_basics[1]["num_passed"] = 0
                assignment_basics[1]["num_exercises"] = 0
                assignment_basics[1]["passed"] = 0
                assignment_basics[1]["in_progress"] = 0
                assignment_basics[1]["time_has_expired"] = False
                statuses2.append([assignment_basics[0], assignment_basics[1]])
        else:
            for status in sort_list_of_dicts_nicely(statuses, ["title", "assignment_id"]):
                assignment_dict = {"id": status["assignment_id"], "title": status["title"], "visible": status["visible"], "start_date": status["start_date"], "due_date": status["due_date"], "passed": status["passed"], "in_progress": status["in_progress"], "num_passed": status["num_passed"], "num_exercises": status["num_exercises"], "has_timer": status["has_timer"], "time_has_expired": status["minutes_since_start"] > status["hour_timer"] * 60 + status["minute_timer"] if status["minutes_since_start"] else False, "restrict_other_assignments": status["restrict_other_assignments"]}

                if assignment_dict["start_date"]:
                    assignment_dict["start_date"] = assignment_dict["start_date"].strftime('%Y-%m-%dT%H:%M:%SZ')
                if assignment_dict["due_date"]:
                    assignment_dict["due_date"] = assignment_dict["due_date"].strftime('%Y-%m-%dT%H:%M:%SZ')

                statuses2.append([status["assignment_id"], assignment_dict])

        return statuses2

    # Gets the number of submissions a student has made for each exercise
    # in an assignment and whether or not they have passed the exercise.
    # TODO: Pass basics info into this function?
    def get_exercise_statuses(self, course_id, assignment_id, user_id, current_exercise_id=None, show_hidden=True, nice_sort=True):
        # This happens when you are creating a new assignment.
        if not assignment_id:
            return []

        sql = '''SELECT e.exercise_id as id,
                        e.title,
                        e.enable_pair_programming,
                        IFNULL(MAX(s.passed), 0) AS passed,
                        COUNT(s.submission_id) AS num_submissions,
                        COUNT(s.submission_id) > 0 AND IFNULL(MAX(s.passed), 0) = 0 AS in_progress,
                        IFNULL(sc.score, 0) as score,
                        e.weight,
                        e.visible
                 FROM exercises e
                 LEFT JOIN submissions s
                   ON e.course_id = s.course_id
                   AND e.assignment_id = s.assignment_id
                   AND e.exercise_id = s.exercise_id
                   AND s.user_id = ?
                 LEFT JOIN scores sc
                   ON e.course_id = sc.course_id
                   AND e.assignment_id = sc.assignment_id
                   AND e.exercise_id = sc.exercise_id
                   AND (sc.user_id = ? OR sc.user_id IS NULL)
                 WHERE e.course_id = ?
                   AND e.assignment_id = ?
                 GROUP BY e.assignment_id, e.exercise_id
                 ORDER BY e.title'''

        statuses = []
        for row in self.fetchall(sql, (user_id, user_id, int(course_id), int(assignment_id),)):
            if row["visible"] or show_hidden:
                statuses.append(dict(row))

        if nice_sort:
            statuses = sort_list_of_dicts_nicely(statuses, ["title", "id"])

        statuses2 = []

        for status in statuses:
            if current_exercise_id and status["id"] == current_exercise_id:
                statuses2.append([status["id"], status])

        for status in statuses:
            if current_exercise_id and status["id"] == current_exercise_id:
                next

            statuses2.append([status["id"], status])

        return statuses2

    ## Calculates the average score across all students for each assignment in a course, as well as the number of students who have completed each assignment.
    def get_course_summary_scores(self, course_id, assignments):
        sql = '''SELECT COUNT(*) AS num_students
                 FROM course_registrations
                 WHERE course_id = ?'''

        num_students = self.fetchone(sql, (course_id, ))["num_students"]

        sql = '''
          WITH
            student_info AS (
              SELECT user_id
              FROM course_registrations
              WHERE course_id = ?
            ),

            assignment_info AS (
              SELECT a.assignment_id, SUM(e.weight) * 100.0 AS max_score
              FROM assignments a
              INNER JOIN exercises e
                 ON a.course_id = e.course_id
                AND a.assignment_id = e.assignment_id
              WHERE a.course_id = ?
                AND a.visible = 1
                AND e.visible = 1
              GROUP BY a.assignment_id
            ),

            assignment_score_info AS (
              SELECT s.assignment_id, s.user_id, SUM(s.score * e.weight) AS score
              FROM scores s
              INNER JOIN assignments a
                 ON s.course_id = a.course_id
                AND s.assignment_id = a.assignment_id
              INNER JOIN exercises e
                 ON s.course_id = e.course_id
                AND s.assignment_id = e.assignment_id
                AND s.exercise_id = e.exercise_id
              INNER JOIN course_registrations cr
                 ON s.course_id = cr.course_id
                AND s.user_id = cr.user_id
              WHERE s.course_id = ?
                AND a.visible = 1
                AND e.visible = 1
              GROUP BY s.assignment_id, s.user_id
            ),

            student_scores AS (
              SELECT ai.assignment_id,
                     IFNULL(asi.score, 0) * 100.0 / ai.max_score AS score,
                     IFNULL(asi.score, 0) = ai.max_score AS completed
              FROM assignment_info ai
              INNER JOIN student_info si
              LEFT JOIN assignment_score_info asi
                 ON ai.assignment_id = asi.assignment_id
                AND si.user_id = asi.user_id
            )

            SELECT assignment_id, sum(completed) AS num_students_completed, ROUND(avg(score), 1) AS avg_score
            FROM student_scores
            GROUP BY assignment_id
          '''

        course_scores = {}

        for row in self.fetchall(sql, (course_id, course_id, course_id, )):
            assignment_dict = {"assignment_id": row["assignment_id"],
                "num_students": num_students,
                "num_students_completed": row["num_students_completed"],
                "avg_score": row["avg_score"]}

            course_scores[row["assignment_id"]] = assignment_dict

        for assignment in assignments:
            assignment_id = assignment[0]

            if assignment_id not in course_scores:
                course_scores[assignment_id] = {"num_students_completed": 0,
                    "num_students": num_students,
                    "avg_score": "0.0"}

        return course_scores

    ## Calculates the average score across all students for each exercise in an assignment,
    ## as well as the number of students who have completed each exercise.
    def get_assignment_summary_scores(self, course_basics, assignment_basics):
        sql = '''
          WITH
            student_info AS (
              SELECT user_id
              FROM course_registrations
              WHERE course_id = ?
          ),

            exercise_score_info AS (
              SELECT s.exercise_id, s.user_id, SUM(s.score) AS score
              FROM scores s
              INNER JOIN assignments a
                 ON s.course_id = a.course_id
                AND s.assignment_id = a.assignment_id
              INNER JOIN exercises e
                 ON s.course_id = e.course_id
                AND s.assignment_id = e.assignment_id
                AND s.exercise_id = e.exercise_id
              INNER JOIN course_registrations cr
                 ON s.course_id = cr.course_id
                AND s.user_id = cr.user_id
              WHERE s.course_id = ?
                AND s.assignment_id = ?
                AND a.visible = 1
                AND e.visible = 1
              GROUP BY s.exercise_id, s.user_id
          )

          SELECT e.exercise_id,
                 ROUND(avg(IFNULL(esi.score, 0)), 1) AS avg_score
          FROM exercises e
          INNER JOIN student_info si
          LEFT JOIN exercise_score_info esi
             ON e.exercise_id = esi.exercise_id
            AND si.user_id = esi.user_id
          WHERE e.course_id = ?
            AND e.assignment_id = ?
            AND e.visible = 1
          GROUP BY e.exercise_id
          '''

        assignment_scores = {}

        for row in self.fetchall(sql, (course_basics["id"], course_basics["id"], assignment_basics["id"], course_basics["id"], assignment_basics["id"], )):
            assignment_scores[row["exercise_id"]] = row["avg_score"]

        if len(assignment_scores) == 0:
            for exercise in self.get_exercises(course_basics, assignment_basics, show_hidden=False):
                assignment_scores[exercise[0]] = 0.0

        return assignment_scores

    # Gets all users who have submitted on a particular assignment and creates a list of their average scores for the assignment.
    def get_assignment_scores(self, course_basics, assignment_basics):
        scores = []

        sql = '''WITH
                   assignment_scores AS (
                     SELECT u.name,
                            s.user_id,
                            SUM(s.score * e.weight) / b.total_weight AS percent_passed,
                            sub.last_submission_time
                     FROM scores s
                     INNER JOIN users u
                       ON s.user_id = u.user_id
                     INNER JOIN (
                       SELECT SUM(weight) AS total_weight
                       FROM exercises
                       WHERE course_id = ?
                         AND assignment_id = ?
                         AND visible = 1
                     ) b
                     INNER JOIN exercises e
                       ON s.exercise_id = e.exercise_id
                     INNER JOIN (
                       SELECT user_id, strftime('%Y-%m-%d %H:%M:%S', MAX(date)) AS last_submission_time
                       FROM submissions
                       WHERE course_id = ?
                         AND assignment_id = ?
                       GROUP BY user_id
                     ) sub
                       ON s.user_id = sub.user_id
                     WHERE s.course_id = ?
                       AND s.assignment_id = ?
                       AND s.user_id NOT IN
                       (
                        SELECT user_id
                        FROM permissions
                      WHERE course_id = 0 OR course_id = ?
                       )
                       AND s.exercise_id NOT IN
                       (
                         SELECT exercise_id
                         FROM exercises
                         WHERE course_id = ?
                           AND assignment_id = ?
                           AND visible = 0
                       )
                     GROUP BY s.course_id, s.assignment_id, s.user_id
                     HAVING SUM(s.score) > 0
                   ),

                   exercise_pass_dates AS (
                     SELECT exercise_id, user_id, min(date) AS earliest_pass_date
                     FROM submissions s
                     WHERE course_id = ?
                       AND assignment_id = ?
                       AND passed = 1
                       AND user_id NOT IN (
                          SELECT user_id
                          FROM permissions
                          WHERE course_id = 0 OR course_id = ?
                       )
                     GROUP BY exercise_id, user_id
                   ),

                   when_users_passed AS (
                     SELECT user_id, max(earliest_pass_date) AS when_passed
                     FROM exercise_pass_dates
                     GROUP BY user_id
                     HAVING COUNT(*) >= (
                        SELECT COUNT(*) as num
                        FROM exercises
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND visible = 1
                        )

                     UNION

                     SELECT user_id, ''
                     FROM exercise_pass_dates
                     GROUP BY user_id
                     HAVING COUNT(*) < (
                       SELECT COUNT(*) as num
                       FROM exercises
                       WHERE course_id = ?
                         AND assignment_id = ?
                         AND visible = 1
                     )
                     ORDER BY user_id
                 )

                 SELECT assignment_scores.*, when_users_passed.when_passed
                 FROM assignment_scores
                 INNER JOIN when_users_passed
                   ON assignment_scores.user_id = when_users_passed.user_id

                 UNION

                 SELECT name, user_id, 0, '', ''
                 FROM users
                 WHERE user_id IN (SELECT user_id FROM course_registrations WHERE course_id = ?)
                   AND user_id NOT IN (SELECT user_id FROM assignment_scores)
                 '''

        course_id = course_basics["id"]
        assignment_id = assignment_basics["id"]

        for user in self.fetchall(sql, (course_id, assignment_id, course_id, assignment_id, course_id, assignment_id, course_id, course_id, assignment_id, course_id, assignment_id, course_id, course_id, assignment_id, course_id, assignment_id, course_id, )):
            scores_dict = {"name": user["name"], "user_id": user["user_id"], "percent_passed": user["percent_passed"], "when_passed": user["when_passed"], "last_submission_time": user["last_submission_time"]}
            scores.append([user["user_id"], scores_dict])

        return scores

    # Get score for each assignment for a particular student.
    def get_student_assignment_scores(self, course_id, user_id):
        scores = []

        sql = '''SELECT assignment_totals.assignment_id, assignment_totals.title, round(assignment_totals.total_score / max_possible.sum_weights, 2) AS score
                 FROM (
                   SELECT a.assignment_id, a.title, SUM(s.score * e.weight) AS total_score
                   FROM exercises e
                   INNER JOIN scores s
                     ON e.course_id = s.course_id
                    AND e.assignment_id = s.assignment_id
                    AND e.exercise_id = s.exercise_id
                   INNER JOIN assignments a
                     ON e.course_id = a.course_id
                    AND e.assignment_id = a.assignment_id
                   WHERE e.course_id = ?
                    AND s.user_id = ?
                    AND e.visible = 1
                    AND a.visible = 1
                   GROUP BY e.assignment_id
                 ) assignment_totals
                 INNER JOIN (
                   SELECT assignment_id, SUM(weight) AS sum_weights
                   FROM exercises
                   WHERE course_id = ?
                   GROUP BY assignment_id
                 ) max_possible
                   ON assignment_totals.assignment_id = max_possible.assignment_id

                 UNION

                 SELECT assignment_id, title, 0.0 as score
                 FROM assignments
                 WHERE course_id = ?
                   AND visible = 1
                   AND assignment_id NOT IN (
                     SELECT DISTINCT assignment_id
                     FROM scores
                     WHERE course_id = ?
                       AND user_id = ?
                     )
                 ORDER BY assignment_totals.title
                 '''

        for row in self.fetchall(sql, (course_id, user_id, course_id, course_id, course_id, user_id, )):
            scores.append([row["assignment_id"], row["title"], row["score"]])

        return scores

    def get_exercise_scores(self, course_id, assignment_id, exercise_id):
        scores = []

        sql = '''WITH exercise_scores AS (
                   SELECT u.name, s.user_id, sc.score, COUNT(s.submission_id) AS num_submissions
                   FROM submissions s
                   INNER JOIN users u
                     ON u.user_id = s.user_id
                   INNER JOIN scores sc
                   ON sc.course_id = s.course_id
                     AND sc.assignment_id = s.assignment_id
                     AND sc.exercise_id = s.exercise_id
                     AND sc.user_id = s.user_id
                   WHERE s.course_id = ?
                     AND s.assignment_id = ?
                     AND s.exercise_id = ?
                   GROUP BY s.user_id
                 )

                 SELECT *
                 FROM exercise_scores

                 UNION

                 SELECT name, user_id, 0, 0
                 FROM users
                 WHERE user_id IN (SELECT user_id FROM course_registrations WHERE course_id = ?)
                   AND user_id NOT IN (SELECT user_id FROM exercise_scores)
              '''

        for user in self.fetchall(sql, (int(course_id), int(assignment_id), int(exercise_id), int(course_id), )):
            scores_dict = {"name": user["name"], "user_id": user["user_id"], "num_submissions": user["num_submissions"], "score": user["score"]}
            scores.append([user["user_id"], scores_dict])

        return scores

    def save_exercise_score(self, course_id, assignment_id, exercise_id, user_id, score):
        # We only update the score if it's higher than what was there previously. We also account for the scenario where it is their first submission.
        sql = '''WITH user_scores AS (
                   SELECT score
		               FROM scores
		               WHERE course_id = ?
		                 AND assignment_id = ?
				             AND exercise_id = ?
				             AND user_id = ?

		               UNION
		
		               SELECT 0
                 )

                 INSERT OR REPLACE INTO scores (course_id, assignment_id, exercise_id, user_id, score)
                 SELECT ?, ?, ?, ?, ?
                 WHERE ? > (SELECT MAX(score) FROM user_scores)'''

        self.execute(sql, (course_id, assignment_id, exercise_id, user_id, course_id, assignment_id, exercise_id, user_id, score, score))

    def get_submissions(self, course_id, assignment_id, exercise_id, user_id, exercise_details):
        sql = '''SELECT o.submission_id, t.title, o.txt_output, o.jpg_output
                 FROM test_outputs o
                 INNER JOIN tests t
                   ON o.test_id = t.test_id
                 INNER JOIN submissions s
                   ON o.submission_id = s.submission_id
                 LEFT JOIN users u
                   ON s.partner_id = u.user_id
                 WHERE t.course_id = ?
                   AND t.assignment_id = ?
                   AND t.exercise_id = ?
                   AND s.user_id = ?'''

        test_outputs = {}
        for row in self.fetchall(sql, (int(course_id), int(assignment_id), int(exercise_id), user_id,)):
            submission_id = row["submission_id"]
            test_title = row["title"]

            if not submission_id in test_outputs:
                test_outputs[submission_id] = {}

            if not test_title in test_outputs[submission_id]:
                test_outputs[submission_id][test_title] = {}

            test_outputs[submission_id][test_title]["txt_output"] = row["txt_output"]
            test_outputs[submission_id][test_title]["jpg_output"] = row["jpg_output"]
            test_outputs[submission_id][test_title]["txt_output_formatted"] = format_output_as_html(row["txt_output"])

        sql = '''SELECT s.submission_id, s.code, s.passed, s.date, u.name AS partner_name
                 FROM submissions s
                 LEFT JOIN users u
                   ON s.partner_id = u.user_id
                 WHERE s.course_id = ?
                   AND s.assignment_id = ?
                   AND s.exercise_id = ?
                   AND s.user_id = ?
                 
                 UNION

                 SELECT -1, code, FALSE, NULL, NULL
                 FROM presubmissions
                 WHERE course_id = ?
                   AND assignment_id = ?
                   AND exercise_id = ?
                   AND user_id = ?

                 ORDER BY s.submission_id'''

        presubmission = None
        submissions = []

        for row in self.fetchall(sql, (course_id, assignment_id, exercise_id, user_id, course_id, assignment_id, exercise_id, user_id,)):
            submission_test_outputs = {}

            if row["submission_id"] == -1:
                  presubmission = row["code"]
            else:
              if row["submission_id"] in test_outputs:
                  submission_test_outputs = test_outputs[row["submission_id"]]
                  check_test_outputs(exercise_details, submission_test_outputs)
                  sanitize_test_outputs(exercise_details, submission_test_outputs)

              submissions.append({"id": row["submission_id"], "code": row["code"], "passed": row["passed"], "date": row["date"].strftime("%a, %d %b %Y %H:%M:%S UTC"), "partner_name": row["partner_name"], "test_outputs": submission_test_outputs})

        return presubmission, submissions

    #TODO: Is there some way to do this without going to the database?
    async def get_num_submissions(self, course_id, assignment_id, exercise_id, user_id):
        sql = '''SELECT COUNT(submission_id) AS num
                 FROM submissions
                 WHERE course_id = ?
                   AND assignment_id = ?
                   AND exercise_id = ?
                   AND user_id = ?'''

        return self.fetchone(sql, (course_id, assignment_id, exercise_id, user_id, ))["num"]

    def get_most_recent_submission_code(self, course_id, assignment_id, exercise_id, user_id):
        sql = '''SELECT code
                 FROM submissions
                 WHERE course_id = ?
                   AND assignment_id = ?
                   AND exercise_id = ?
                   AND user_id = ?
                   AND passed = 1
                 ORDER BY date DESC
                 LIMIT 1'''

        result = self.fetchone(sql, (course_id, assignment_id, exercise_id, user_id,))

        if result:
            return result["code"]
        else:
            return ""

    def get_peer_code(self, course_id, assignment_id, exercise_id, user_id):
        sql = '''SELECT user_id, code, MAX(date)
                 FROM submissions
                 WHERE course_id = ?
                   AND assignment_id = ?
                   AND exercise_id = ?
                   AND passed = 1
                   AND user_id != ?
                   AND (partner_id != ? OR partner_id IS NULL)
                 GROUP BY user_id
                 ORDER BY user_id'''

        peer_code_dict = {}
        for row in self.fetchall(sql, (course_id, assignment_id, exercise_id, user_id, user_id)):
            peer_code_dict[row["user_id"]] = row["code"]

        if len(peer_code_dict) == 0:
            return ""

        peer_ids = list(peer_code_dict.keys())
        random.shuffle(peer_ids)

        return peer_code_dict[peer_ids[0]]

    def get_help_requests(self, course_id):
        help_requests = []

        sql = '''SELECT r.course_id, a.assignment_id, e.exercise_id, c.title as course_title, a.title as assignment_title, e.title as exercise_title, r.user_id, u.name, r.code, r.text_output, r.image_output, r.student_comment, r.suggestion, r.approved, r.suggester_id, r.approver_id, r.date, r.more_info_needed
                 FROM help_requests r
                 INNER JOIN users u
                   ON r.user_id = u.user_id
                 INNER JOIN courses c
                   ON r.course_id = c.course_id
                 INNER JOIN assignments a
                   ON r.assignment_id = a.assignment_id
                 INNER JOIN exercises e
                   ON r.exercise_id = e.exercise_id
                 WHERE r.course_id = ?
                 ORDER BY r.date DESC'''

        for request in self.fetchall(sql, (course_id,)):
            help_requests.append({"course_id": request["course_id"], "assignment_id": request["assignment_id"], "exercise_id": request["exercise_id"], "course_title": request["course_title"], "assignment_title": request["assignment_title"], "exercise_title": request["exercise_title"], "user_id": request["user_id"], "name": request["name"], "code": request["code"], "text_output": request["text_output"], "image_output": request["image_output"], "student_comment": request["student_comment"], "suggestion": request["suggestion"], "approved": request["approved"], "suggester_id": request["suggester_id"], "approver_id": request["approver_id"], "date": request["date"], "more_info_needed": request["more_info_needed"]})

        return help_requests

    def get_student_help_requests(self, user_id):
        help_requests = []

        sql = '''SELECT r.course_id, a.assignment_id, e.exercise_id, c.title as course_title, a.title as assignment_title, e.title as exercise_title, r.user_id, u.name, r.code, r.text_output, r.image_output, r.student_comment, r.suggestion, r.approved, r.suggester_id, r.approver_id, r.more_info_needed
                 FROM help_requests r
                 INNER JOIN users u
                   ON r.user_id = u.user_id
                 INNER JOIN courses c
                   ON r.course_id = c.course_id
                 INNER JOIN assignments a
                   ON r.assignment_id = a.assignment_id
                 INNER JOIN exercises e
                   ON r.exercise_id = e.exercise_id
                 WHERE r.user_id = ?
                 ORDER BY r.date DESC'''

        for request in self.fetchall(sql, (user_id,)):
            help_requests.append({"course_id": request["course_id"], "assignment_id": request["assignment_id"], "exercise_id": request["exercise_id"], "course_title": request["course_title"], "assignment_title": request["assignment_title"], "exercise_title": request["exercise_title"], "user_id": request["user_id"], "name": request["name"], "code": request["code"], "text_output": request["text_output"], "image_output": request["text_output"], "image_output": request["image_output"], "student_comment": request["student_comment"], "suggestion": request["suggestion"], "approved": request["approved"], "suggester_id": request["suggester_id"], "approver_id": request["approver_id"], "more_info_needed": request["more_info_needed"]})

        return help_requests

    def get_exercise_help_requests(self, course_id, assignment_id, exercise_id, user_id):
        sql = '''SELECT text_output
                 FROM help_requests
                 WHERE course_id = ?
                 AND assignment_id = ?
                 AND exercise_id = ?
                 AND user_id = ?'''
        row = self.fetchone(sql, (course_id, assignment_id, exercise_id, user_id,))
        orig_output = re.sub("#.*", "", row["text_output"])

        sql = '''SELECT r.course_id, r.assignment_id, r.exercise_id, r.user_id, u.name, r.code, r.text_output, r.image_output, r.student_comment, r.suggestion, r.approved, r.suggester_id, r.approver_id, r.more_info_needed
                 FROM help_requests r
                 INNER JOIN users u
                   ON r.user_id = u.user_id
                 WHERE r.course_id = ?
                   AND r.assignment_id = ?
                   AND r.exercise_id = ?
                   AND NOT r.user_id = ?
                 ORDER BY r.date DESC'''

        requests = self.fetchall(sql, (course_id, assignment_id, exercise_id, user_id,))

        nlp = spacy.load('en_core_web_sm')
        orig = nlp(orig_output)
        help_requests = []

        for request in requests:
            curr = nlp(re.sub("#.*", "", request["text_output"]))
            psim = curr.similarity(orig)
            request_info = {"psim": psim, "course_id": request["course_id"], "assignment_id": request["assignment_id"], "exercise_id": request["exercise_id"], "user_id": request["user_id"], "name": request["name"], "code": request["code"], "text_output": request["text_output"], "image_output": request["text_output"], "image_output": request["image_output"], "student_comment": request["student_comment"], "suggestion": request["suggestion"], "approved": request["approved"], "suggester_id": request["suggester_id"], "approver_id": request["approver_id"], "more_info_needed": request["more_info_needed"]}
            help_requests.append(request_info)

        return sorted(help_requests, key=lambda x: x["psim"], reverse=True)

    def get_help_request(self, course_id, assignment_id, exercise_id, user_id):
        sql = '''SELECT r.user_id, u.name, r.code, r.text_output, r.image_output, r.student_comment, r.suggestion, r.approved, r.suggester_id, r.approver_id, r.more_info_needed
                 FROM help_requests r
                 INNER JOIN users u
                   ON r.user_id = u.user_id
                 WHERE r.course_id = ?
                   AND r.assignment_id = ?
                   AND r.exercise_id = ?
                   AND r.user_id = ?'''

        request = self.fetchone(sql, (course_id, assignment_id, exercise_id, user_id,))
        if request:
            help_request = {"course_id": course_id, "assignment_id": assignment_id, "exercise_id": exercise_id, "user_id": request["user_id"], "name": request["name"], "code": request["code"], "text_output": request["text_output"], "image_output": request["image_output"], "student_comment": request["student_comment"], "approved": request["approved"], "suggester_id": request["suggester_id"], "approver_id": request["approver_id"], "more_info_needed": request["more_info_needed"]}
            if request["suggestion"]:
                help_request["suggestion"] = request["suggestion"]
            else:
                help_request["suggestion"] = None

            return help_request

    def compare_help_requests(self, course_id, assignment_id, exercise_id, user_id):
        # Get the original help request, including its output type
        sql = '''SELECT r.text_output, e.expected_text_output, r.image_output, e.expected_image_output, e.output_type
                 FROM help_requests r
                 INNER JOIN exercises e
                   ON e.course_id = r.course_id
                   AND e.assignment_id = r.assignment_id
                   AND e.exercise_id = r.exercise_id
                 WHERE r.course_id = ?
                   AND r.assignment_id = ?
                   AND r.exercise_id = ?
                   AND r.user_id = ?'''

        row = self.fetchone(sql, (course_id, assignment_id, exercise_id, user_id,))

        #the original output type will be either txt or jpg depending on the output type of the exercise
        orig_output = None

        if row["output_type"] == "jpg":
            if row["image_output"] != row["expected_image_output"]:
                orig_output = row["image_output"]

        else:
            if row["text_output"] != row["expected_text_output"]:
                orig_output = row["text_output"]

        #get all other help requests in the course that have the same output type
        if orig_output:
                sql = '''SELECT r.course_id, r.assignment_id, r.exercise_id, r.user_id, u.name, r.code, r.text_output, r.image_output, r.student_comment, r.suggestion
                        FROM help_requests r
                        INNER JOIN users u
                          ON r.user_id = u.user_id
                        INNER JOIN exercises e
                          ON e.course_id = r.course_id
                          AND e.assignment_id = r.assignment_id
                          AND e.exercise_id = r.exercise_id
                        WHERE r.course_id = ?
                          AND NOT r.user_id = ?
                          AND e.output_type = ?'''
                requests = self.fetchall(sql, (course_id, user_id, row["output_type"]))
                sim_dict = []

                #jpg output uses the diff_jpg function in helper.py, txt output uses .similarity() from the Spacy module
                if row["output_type"] == "jpg":
                    for request in requests:
                        diff_image, diff_percent = diff_jpg(orig_output, request["image_output"])
                        if diff_percent < .10:
                            request_info = {"psim": 1 - diff_percent, "course_id": request["course_id"], "assignment_id": request["assignment_id"], "exercise_id": request["exercise_id"], "user_id": request["user_id"], "name": request["name"], "student_comment": request["student_comment"], "code": request["code"], "text_output": request["text_output"], "suggestion": request["suggestion"]}
                            sim_dict.append(request_info)

                else:
                    nlp = spacy.load('en_core_web_sm')
                    orig = nlp(orig_output)

                    for request in requests:
                        curr = nlp(request["text_output"])
                        psim = curr.similarity(orig)
                        sim = False

                        #these thresholds can be changed in the future
                        if len(orig) < 10 and len(curr) < 10:
                            if psim > .30:
                                sim = True
                        elif len(orig) < 100 and len(curr) < 100:
                            if psim > .50:
                                sim = True
                        elif len(orig) < 200 and len(curr) < 200:
                            if psim > .70:
                                sim = True
                        else:
                            if psim > .90:
                                sim = True

                        if sim:
                            request_info = {"psim": psim, "course_id": request["course_id"], "assignment_id": request["assignment_id"], "exercise_id": request["exercise_id"], "user_id": request["user_id"], "name": request["name"], "student_comment": request["student_comment"],  "code": request["code"], "text_output": request["text_output"], "suggestion": request["suggestion"]}
                            sim_dict.append(request_info)

                return sim_dict

    def get_same_suggestion(self, help_request):
        sql = '''SELECT r.suggestion, e.output_type, r.text_output, r.image_output
                 FROM help_requests r
                 INNER JOIN exercises e
                   ON e.exercise_id = r.exercise_id
                 WHERE r.course_id = ?
                   AND r.suggestion NOT NULL
                   AND e.output_type = (
                       SELECT output_type
                       FROM exercises
                       WHERE exercise_id = ?
                   )'''

        matches = self.fetchall(sql, (help_request["course_id"], help_request["exercise_id"]))
        for match in matches:
            if match["output_type"] == "jpg":
                if match["image_output"] == help_request["image_output"]:
                    return match["suggestion"]
            else:
                if match["text_output"] == help_request["text_output"]:
                    return match["suggestion"]

    # FYI: This is different from the get_submissions() function
    def get_exercise_submissions(self, course_id, assignment_id, exercise_id):
        exercise_submissions = []

        sql = '''WITH exercise_submissions AS (
                   SELECT MAX(s.date), s.code, u.user_id, u.name, sc.score, s.passed, p.name AS partner_name
                   FROM submissions s
                   LEFT JOIN users p
                     ON s.partner_id = p.user_id
                   INNER JOIN users u
                     ON s.user_id = u.user_id
                   INNER JOIN scores sc
                     ON s.user_id = sc.user_id
                     AND s.course_id = sc.course_id
                     AND s.assignment_id = sc.assignment_id
                     AND s.exercise_id = sc.exercise_id
                   WHERE s.course_id = ?
                     AND s.assignment_id = ?
                     AND s.exercise_id = ?
                     AND s.user_id IN
                     (
                        SELECT user_id
                        FROM course_registrations
                        WHERE course_id = ?
                     )
                   GROUP BY s.user_id
                 )

                 SELECT *
                 FROM exercise_submissions

                 UNION

                 SELECT NULL, NULL, user_id, name, 0, 0, NULL
                 FROM users
                 WHERE user_id IN (SELECT user_id FROM course_registrations WHERE course_id = ?)
                   AND user_id NOT IN (SELECT user_id FROM exercise_submissions)
                 ORDER BY name'''

        for submission in self.fetchall(sql, (course_id, assignment_id, exercise_id, course_id, course_id, )):
            submission_info = {"user_id": submission["user_id"], "name": submission["name"], "code": submission["code"], "score": submission["score"], "passed": submission["passed"], "partner_name": submission["partner_name"]}
            exercise_submissions.append([submission["user_id"], submission_info])

        return exercise_submissions

    def specify_course_basics(self, course_basics, title, visible):
        course_basics["title"] = title
        course_basics["visible"] = visible

    def specify_course_details(self, course_details, introduction, passcode, allow_students_download_submissions, date_created, date_updated):
        course_details["introduction"] = introduction
        course_details["passcode"] = passcode
        course_details["allow_students_download_submissions"] = allow_students_download_submissions

        course_details["date_updated"] = date_updated

        if course_details["date_created"]:
            course_details["date_created"] = date_created
        else:
            course_details["date_created"] = date_updated

    def specify_assignment_basics(self, assignment_basics, title, visible):
        assignment_basics["title"] = title
        assignment_basics["visible"] = visible

    def specify_assignment_details(self, assignment_details, introduction, date_created, date_updated, start_date, due_date, allow_late, late_percent, view_answer_late, enable_help_requests, has_timer, hour_timer, minute_timer, restrict_other_assignments, allowed_ip_addresses, allowed_external_urls):
        assignment_details["introduction"] = introduction
        assignment_details["date_updated"] = date_updated
        assignment_details["start_date"] = start_date
        assignment_details["due_date"] = due_date
        assignment_details["allow_late"] = allow_late
        assignment_details["late_percent"] = late_percent
        assignment_details["view_answer_late"] = view_answer_late
        assignment_details["enable_help_requests"] = enable_help_requests
        assignment_details["has_timer"] = has_timer
        assignment_details["hour_timer"] = hour_timer
        assignment_details["minute_timer"] = minute_timer
        assignment_details["restrict_other_assignments"] = restrict_other_assignments
        assignment_details["allowed_ip_addresses"] = allowed_ip_addresses
        assignment_details["allowed_external_urls"] = allowed_external_urls

        if assignment_details["date_created"]:
            assignment_details["date_created"] = date_created
        else:
            assignment_details["date_created"] = date_updated

    def get_course_basics(self, course_id):
        null_course = {"id": "", "title": "", "visible": True, "exists": False}

        if not course_id:
            return null_course

        sql = '''SELECT course_id, title, visible
                 FROM courses
                 WHERE course_id = ?'''

        row = self.fetchone(sql, (int(course_id),))

        if not row:
            return null_course

        return {"id": row["course_id"], "title": row["title"], "visible": bool(row["visible"]), "exists": True}
    
    def get_assignments(self, course_basics, show_hidden=True):
        sql = '''SELECT assignment_id as id, title, visible
                 FROM assignments a
                 WHERE course_id = ?'''

        # We initially structure it this way to make sorting easier.
        assignments = []
        for row in self.fetchall(sql, (course_basics["id"],)):
            if row["visible"] or show_hidden:
                assignments.append(dict(row))

        assignments = sort_list_of_dicts_nicely(assignments, ["title", "id"])

        # We restructure it to be consistent with courses and exercises
        assignments2 = []
        for assignment in assignments:
            assignments2.append([assignment["id"], assignment])

        return assignments2
    
    def get_assignment_basics(self, course_basics, assignment_id):
        if not assignment_id:
            return {"id": "", "title": "", "visible": True, "exists": False, "course": course_basics}

        sql = '''SELECT assignment_id, title, visible
                 FROM assignments
                 WHERE course_id = ?
                   AND assignment_id = ?'''

        row = self.fetchone(sql, (int(course_basics['id']), assignment_id,))
        if row is None:
            return {"id": "", "title": "", "visible": True, "exists": False, "course": course_basics}
        else:
            return {"id": row["assignment_id"], "title": row["title"], "visible": bool(row["visible"]), "exists": True, "course": course_basics}

    def get_exercise_basics(self, course_basics, assignment_basics, exercise_id):
        if not exercise_id:
            return {"id": "", "title": "", "visible": True, "exists": False, "assignment": assignment_basics}

        sql = '''SELECT exercise_id, title, visible, enable_pair_programming
                 FROM exercises
                 WHERE course_id = ?
                   AND assignment_id = ?
                   AND exercise_id = ?'''

        row = self.fetchone(sql, (course_basics['id'], assignment_basics['id'], exercise_id,))
        if row is None:
            return {"enable_pair_programming": False, "id": "", "title": "", "visible": True, "exists": False, "assignment": assignment_basics}
        else:
            return {"enable_pair_programming": row["enable_pair_programming"], "id": row["exercise_id"], "title": row["title"], "visible": bool(row["visible"]), "exists": True, "assignment": assignment_basics}

    def get_next_prev_exercises(self, course_id, assignment_id, exercise, exercises):
        prev_exercise = None
        next_exercise = None

        if len(exercises) > 0 and exercise:
            this_exercise = [i for i in range(len(exercises)) if exercises[i][0] == int(exercise)]
            if len(this_exercise) > 0:
                this_exercise_index = [i for i in range(len(exercises)) if exercises[i][0] == int(exercise)][0]

                if len(exercises) >= 2 and this_exercise_index != 0:
                    prev_exercise = exercises[this_exercise_index - 1][1]

                if len(exercises) >= 2 and this_exercise_index != (len(exercises) - 1):
                    next_exercise = exercises[this_exercise_index + 1][1]

        return {"previous": prev_exercise, "next": next_exercise}

    def delete_old_presubmissions(self):
        sql = '''DELETE
                 FROM presubmissions
                 WHERE date_updated < datetime('now', '-12 months')'''

        self.execute(sql)

    def get_presubmission(self, course_id, assignment_id, exercise_id, user_id):
        sql = '''SELECT code
                 FROM presubmissions
                 WHERE course_id = ?
                   AND assignment_id = ?
                   AND exercise_id = ?
                   AND user_id = ?'''

        row = self.fetchone(sql, (int(course_id), int(assignment_id), int(exercise_id), user_id))

        return row["code"] if row else None

    def get_course_details(self, course_id):
        null_course = {"introduction": "", "passcode": None, "date_created": None, "date_updated": None, "allow_students_download_submissions": False}

        if not course_id:
            return null_course

        sql = '''SELECT introduction, passcode, date_created, date_updated, allow_students_download_submissions
                 FROM courses
                 WHERE course_id = ?'''

        row = self.fetchone(sql, (course_id,))

        if not row:
            return null_course

        course_details = {"introduction": row["introduction"], "passcode": row["passcode"], "date_created": row["date_created"], "date_updated": row["date_updated"], "allow_students_download_submissions": row["allow_students_download_submissions"]}

        sql = '''SELECT COUNT(*) > 0 AS yes
                 FROM assignments
                 WHERE course_id = ?
                   AND restrict_other_assignments = 1'''

        course_details["check_for_restrict_other_assignments"] = bool(self.fetchone(sql, (course_id, ))["yes"])

        return course_details

    def get_assignment_details(self, course_basics, assignment_id):
        null_assignment = {"introduction": "", "date_created": None, "date_updated": None, "start_date": None, "due_date": None, "allow_late": False, "late_percent": None, "view_answer_late": False, "enable_help_requests": 1, "has_timer": 0, "hour_timer": None, "minute_timer": None, "restrict_other_assignments": False, "allowed_ip_addresses": None, "allowed_external_urls": None, "due_date_passed": None}

        if not assignment_id:
            return null_assignment

        sql = '''SELECT introduction, date_created, date_updated, start_date, due_date, allow_late, late_percent, view_answer_late, enable_help_requests, allowed_ip_addresses, allowed_external_urls, has_timer, hour_timer, minute_timer, restrict_other_assignments
                 FROM assignments
                 WHERE course_id = ?
                   AND assignment_id = ?'''

        row = self.fetchone(sql, (course_basics['id'], assignment_id,))

        if not row:
            return null_assignment

        assignment_dict = {"introduction": row["introduction"], "date_created": row["date_created"], "date_updated": row["date_updated"], "start_date": row["start_date"], "due_date": row["due_date"], "allow_late": row["allow_late"], "late_percent": row["late_percent"], "view_answer_late": row["view_answer_late"], "allowed_ip_addresses": row["allowed_ip_addresses"], "allowed_external_urls": row["allowed_external_urls"], "enable_help_requests": row["enable_help_requests"], "has_timer": row["has_timer"], "hour_timer": row["hour_timer"], "minute_timer": row["minute_timer"], "restrict_other_assignments": row["restrict_other_assignments"], "due_date_passed": None}

        curr_datetime = datetime.utcnow()
        if assignment_dict["due_date"]:
            assignment_dict["due_date_passed"] = curr_datetime > assignment_dict["due_date"]

        if assignment_dict["allowed_ip_addresses"]:
            assignment_dict["allowed_ip_addresses_list"] = assignment_dict["allowed_ip_addresses"].split("\n")

        if assignment_dict["allowed_external_urls"] != "":
            assignment_dict["allowed_external_urls_dict"] = {}
            for url in assignment_dict["allowed_external_urls"].split("\n"):
                url = url.strip()
                assignment_dict["allowed_external_urls_dict"][url] = urllib.parse.quote(url)

        return assignment_dict

    def get_exercise_details(self, course_basics, assignment_basics, exercise_id):
        null_exercise = {"instructions": "", "back_end": "python", "output_type": "txt", "allow_any_response": False, "solution_code": "", "solution_description": "", "hint": "", "max_submissions": 0, "starter_code": "", "credit": "", "data_files": [], "what_students_see_after_success": 1, "date_created": None, "date_updated": None, "enable_pair_programming": False, "verification_code": "", "weight": 1.0, "tests": {}}

        if not exercise_id:
            return null_exercise

        sql = '''SELECT instructions, back_end, output_type, allow_any_response, solution_code, solution_description, hint, max_submissions, starter_code, credit, data_files, what_students_see_after_success, date_created, date_updated, enable_pair_programming, verification_code, weight
                 FROM exercises
                 WHERE course_id = ?
                   AND assignment_id = ?
                   AND exercise_id = ?'''

        row = self.fetchone(sql, (course_basics['id'], assignment_basics['id'], exercise_id,))

        if not row:
            return null_exercise

        exercise_dict = {"instructions": row["instructions"], "back_end": row["back_end"], "output_type": row["output_type"], "allow_any_response": row["allow_any_response"], "solution_code": row["solution_code"], "solution_description": row["solution_description"], "hint": row["hint"], "max_submissions": row["max_submissions"], "starter_code": row["starter_code"], "credit": row["credit"], "data_files": json.loads(row["data_files"]), "what_students_see_after_success": row["what_students_see_after_success"], "date_created": row["date_created"], "date_updated": row["date_updated"], "enable_pair_programming": row["enable_pair_programming"], "verification_code": row["verification_code"], "weight": row["weight"], "tests": {}}

        sql = '''SELECT test_id,
                   title,
                   before_code,
                   after_code,
                   instructions,
                   can_see_test_code,
                   can_see_expected_output,
                   can_see_code_output,
                   txt_output,
                   jpg_output
                 FROM tests
                 WHERE course_id = ?
                   AND assignment_id = ?
                   AND exercise_id = ?'''

        tests = self.fetchall(sql, (course_basics['id'], assignment_basics['id'], exercise_id,))

        for test in tests:
            exercise_dict["tests"][test["title"]] = {"test_id": test["test_id"], "before_code": test["before_code"], "after_code": test["after_code"], "instructions": test["instructions"], "can_see_test_code": test["can_see_test_code"], "can_see_expected_output": test["can_see_expected_output"], "can_see_code_output": test["can_see_code_output"], "txt_output": test["txt_output"], "jpg_output": test["jpg_output"]}

        return exercise_dict

    def get_log_table_contents(self, file_path, year="No filter", month="No filter", day="No filter"):
        new_dict = {}
        line_num = 1
        with gzip.open(file_path) as read_file:
            header = read_file.readline()
            for line in read_file:
                line_items = line.decode().rstrip("\n").split("\t")

                #Get ids to create links to each course, assignment, and exercise in the table
                course_id = line_items[1]
                assignment_id = line_items[2]
                exercise_id = line_items[3]

                line_items[6] = f"<a href='/course/{course_id}'>{line_items[6]}</a>"
                line_items[7] = f"<a href='/assignment/{course_id}/{assignment_id}'>{line_items[7]}</a>"
                line_items[8] = f"<a href='/exercise/{course_id}/{assignment_id}/{exercise_id}'>{line_items[8]}</a>"

                line_items = [line_items[0][:2], line_items[0][2:4], line_items[0][4:6], line_items[0][6:]] + line_items[4:]

                new_dict[line_num] = line_items
                line_num += 1

        # Filter by date.
        year_dict = {}
        month_dict = {}
        day_dict = {}

        for key, line in new_dict.items():
            if year == "No filter" or line[0] == year:
                year_dict[key] = line
        for key, line in year_dict.items():
            if month == "No filter" or line[1] == month:
                month_dict[key] = line
        for key, line in month_dict.items():
            if day == "No filter" or line[2] == day:
                day_dict[key] = line

        return day_dict

    def get_root_dirs_to_log(self):
        root_dirs_to_log = set(["home", "course", "assignment", "exercise", "check_exercise", "edit_course", "edit_assignment", "edit_exercise", "delete_course", "delete_assignment", "delete_exercise", "view_answer", "import_course", "export_course"])
        return root_dirs_to_log

    def sort_nested_list(self, nested_list, key="title"):
        l_dict = {}
        for row in nested_list:
            l_dict[row[1][key]] = row

        return [l_dict[key] for key in sort_nicely(l_dict)]

    def has_duplicate_title(self, entries, this_entry, proposed_title):
        for entry in entries:
            if entry[0] != this_entry and entry[1]["title"] == proposed_title:
                return True

        return False

    def save_course(self, course_basics, course_details):
        if course_basics["exists"]:
            sql = '''UPDATE courses
                     SET title = ?, visible = ?, introduction = ?, passcode = ?, allow_students_download_submissions = ?, date_updated = ?
                     WHERE course_id = ?'''

            self.execute(sql, [course_basics["title"], course_basics["visible"], course_details["introduction"], course_details["passcode"], course_details["allow_students_download_submissions"], course_details["date_updated"], course_basics["id"]])

            self.update_when_content_updated(course_basics["id"])
        else:
            sql = '''INSERT INTO courses (title, visible, introduction, passcode, allow_students_download_submissions, date_created, date_updated)
                     VALUES (?, ?, ?, ?, ?, ?, ?)'''

            course_basics["id"] = self.execute(sql, (course_basics["title"], course_basics["visible"], course_details["introduction"], course_details["passcode"], course_details["allow_students_download_submissions"], course_details["date_created"], course_details["date_updated"], ))
            course_basics["exists"] = True

            self.update_when_content_updated("user")

        return course_basics["id"]

    def save_assignment(self, assignment_basics, assignment_details):
        if assignment_basics["exists"]:
            sql = '''UPDATE assignments
                     SET title = ?, visible = ?, introduction = ?, date_updated = ?, start_date = ?, due_date = ?, allow_late = ?, late_percent = ?, view_answer_late = ?, enable_help_requests = ?, has_timer = ?, hour_timer = ?, minute_timer = ?, restrict_other_assignments = ?, allowed_ip_addresses = ?, allowed_external_urls = ?
                     WHERE course_id = ?
                       AND assignment_id = ?'''

            self.execute(sql, [assignment_basics["title"], assignment_basics["visible"], assignment_details["introduction"], assignment_details["date_updated"], assignment_details["start_date"], assignment_details["due_date"], assignment_details["allow_late"], assignment_details["late_percent"], assignment_details["view_answer_late"], assignment_details["enable_help_requests"], assignment_details["has_timer"], assignment_details["hour_timer"], assignment_details["minute_timer"], assignment_details["restrict_other_assignments"], assignment_details["allowed_ip_addresses"], assignment_details["allowed_external_urls"], assignment_basics["course"]["id"], assignment_basics["id"]])
        else:
            sql = '''INSERT INTO assignments (course_id, title, visible, introduction, date_created, date_updated, start_date, due_date, allow_late, late_percent, view_answer_late, enable_help_requests, has_timer, hour_timer, minute_timer, restrict_other_assignments, allowed_ip_addresses, allowed_external_urls)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

            assignment_basics["id"] = self.execute(sql, [assignment_basics["course"]["id"], assignment_basics["title"], assignment_basics["visible"], assignment_details["introduction"], assignment_details["date_created"], assignment_details["date_updated"], assignment_details["start_date"], assignment_details["due_date"], assignment_details["allow_late"], assignment_details["late_percent"], assignment_details["view_answer_late"], assignment_details["enable_help_requests"], assignment_details["has_timer"], assignment_details["hour_timer"], assignment_details["minute_timer"], assignment_details["restrict_other_assignments"], assignment_details["allowed_ip_addresses"], assignment_details["allowed_external_urls"]])
            assignment_basics["exists"] = True

        self.update_when_content_updated(assignment_basics["course"]["id"])

        return assignment_basics["id"]

    def save_exercise(self, exercise_basics, exercise_details):
        cursor = self.conn.cursor()
        cursor.execute("BEGIN")
        if "what_students_see_after_success" not in exercise_details:
            exercise_details["what_students_see_after_success"] = 1

        try:
            if exercise_basics["exists"]:
                sql = '''
                    UPDATE exercises
                    SET title = ?, visible = ?, solution_code = ?, solution_description = ?, hint = ?,
                        max_submissions = ?, credit = ?, data_files = ?, back_end = ?,
                        instructions = ?, output_type = ?, allow_any_response = ?,
                        what_students_see_after_success = ?, starter_code = ?,
                        date_updated = ?, enable_pair_programming = ?, verification_code = ?,
                        weight = ?
                    WHERE course_id = ?
                      AND assignment_id = ?
                      AND exercise_id = ?'''

                cursor.execute(sql, [exercise_basics["title"], exercise_basics["visible"], str(exercise_details["solution_code"]), exercise_details["solution_description"], exercise_details["hint"], exercise_details["max_submissions"], exercise_details["credit"], json.dumps(exercise_details["data_files"]), exercise_details["back_end"], exercise_details["instructions"], exercise_details["output_type"], exercise_details["allow_any_response"], exercise_details["what_students_see_after_success"], exercise_details["starter_code"], exercise_details["date_updated"], exercise_details["enable_pair_programming"], exercise_details["verification_code"], exercise_details["weight"], exercise_basics["assignment"]["course"]["id"], exercise_basics["assignment"]["id"], exercise_basics["id"]])

                sql = '''DELETE FROM test_outputs
                         WHERE test_id IN (
                           SELECT test_id
                           FROM tests
                           WHERE course_id = ?
                             AND assignment_id = ?
                             AND exercise_id = ?)'''

                cursor.execute(sql, [exercise_basics["assignment"]["course"]["id"], exercise_basics["assignment"]["id"], exercise_basics["id"]])

                sql = '''DELETE FROM tests
                         WHERE course_id = ?
                           AND assignment_id = ?
                           AND exercise_id = ?'''

                cursor.execute(sql, [exercise_basics["assignment"]["course"]["id"], exercise_basics["assignment"]["id"], exercise_basics["id"]])

                for title in exercise_details["tests"]:
                    # Only saves 'jpg_output' if it isn't blank.
                    jpg_output = exercise_details["tests"][title]["jpg_output"]
                    #if jpg_output != "" and jpg_output.strip() == BLANK_IMAGE.strip():
                    if jpg_output != "" and jpg_output.strip() == BLANK_IMAGE:
                        jpg_output = ""

                    sql = '''INSERT INTO tests (course_id, assignment_id, exercise_id, title, before_code, after_code, instructions, can_see_test_code, can_see_expected_output, can_see_code_output, txt_output, jpg_output)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

                    test_id = cursor.execute(sql, [exercise_basics["assignment"]["course"]["id"], exercise_basics["assignment"]["id"], exercise_basics["id"], title, exercise_details["tests"][title]["before_code"], exercise_details["tests"][title]["after_code"], exercise_details["tests"][title]["instructions"], exercise_details["tests"][title]["can_see_test_code"], exercise_details["tests"][title]["can_see_expected_output"], exercise_details["tests"][title]["can_see_code_output"], exercise_details["tests"][title]["txt_output"], jpg_output])
            else:
                sql = '''INSERT INTO exercises (course_id, assignment_id, title, visible, solution_code, solution_description, hint, max_submissions, credit, data_files, back_end, instructions, output_type, allow_any_response, what_students_see_after_success, starter_code, date_created, date_updated, enable_pair_programming, verification_code, weight)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

                cursor.execute(sql, [exercise_basics["assignment"]["course"]["id"], exercise_basics["assignment"]["id"], exercise_basics["title"], exercise_basics["visible"], str(exercise_details["solution_code"]), exercise_details["solution_description"], exercise_details["hint"], exercise_details["max_submissions"], exercise_details["credit"], json.dumps(exercise_details["data_files"]), exercise_details["back_end"], exercise_details["instructions"], exercise_details["output_type"], exercise_details["allow_any_response"], exercise_details["what_students_see_after_success"], exercise_details["starter_code"], exercise_details["date_created"], exercise_details["date_updated"], exercise_details["enable_pair_programming"], exercise_details["verification_code"], exercise_details["weight"]])

                exercise_basics["id"] = cursor.lastrowid
                exercise_basics["exists"] = True

                for title in exercise_details["tests"]:
                    sql = '''INSERT INTO tests (course_id, assignment_id, exercise_id, title, before_code, after_code, instructions, txt_output, jpg_output, can_see_test_code, can_see_expected_output, can_see_code_output)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

                    test_id = cursor.execute(sql, [
                        exercise_basics["assignment"]["course"]["id"],
                        exercise_basics["assignment"]["id"],
                        exercise_basics["id"],
                        title,
                        exercise_details["tests"][title]["before_code"],
                        exercise_details["tests"][title]["after_code"],
                        exercise_details["tests"][title]["instructions"],
                        exercise_details["tests"][title]["txt_output"],
                        exercise_details["tests"][title]["jpg_output"],
                        exercise_details["tests"][title]["can_see_test_code"],
                        exercise_details["tests"][title]["can_see_expected_output"],
                        exercise_details["tests"][title]["can_see_code_output"],
                    ])

            cursor.execute("COMMIT")
        except self.conn.Error:
            cursor.execute("ROLLBACK")
            raise

        cursor.close()

        self.update_when_content_updated(exercise_basics["assignment"]["course"]["id"])

        return exercise_basics["id"]

    def save_presubmission(self, course_id, assignment_id, exercise_id, user_id, code):
        sql = '''INSERT OR REPLACE INTO presubmissions (course_id, assignment_id, exercise_id, user_id, code)
                 VALUES (?, ?, ?, ?, ?)'''

        self.execute(sql, [course_id, assignment_id, exercise_id, user_id, code])

    async def save_submission(self, course_id, assignment_id, exercise_id, user_id, code, passed, date, exercise_details, test_outputs, score, partner_id):
        sql = '''INSERT INTO submissions (course_id, assignment_id, exercise_id, user_id, code, passed, date, partner_id)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)'''
        submission_id = self.execute(sql, (course_id, assignment_id, exercise_id, user_id, code, passed, date, partner_id,))

        sql_statements = []
        params_list = []

        for test_title, test_dict in test_outputs.items():
            if test_dict["jpg_output"] != "":
                #if test_dict["jpg_output"].strip() == BLANK_IMAGE.strip():
                if test_dict["jpg_output"].strip() == BLANK_IMAGE:
                    test_dict["jpg_output"] = ""

            sql_statements.append('''INSERT INTO test_outputs (test_id, submission_id, txt_output, jpg_output)
                      VALUES (?, ?, ?, ?)''')

            params_list.append((exercise_details["tests"][test_title]["test_id"], submission_id, test_dict["txt_output"], test_dict["jpg_output"],))

        self.execute_multiple(sql_statements, params_list)

        self.save_exercise_score(course_id, assignment_id, exercise_id, user_id, score)
        self.save_presubmission(course_id, assignment_id, exercise_id, user_id, code)

        return submission_id
    
        # sql = '''INSERT INTO submissions (course_id, assignment_id, exercise_id, user_id, code, passed, date, partner_id)
        #          VALUES (?, ?, ?, ?, ?, ?, ?, ?)'''

        # submission_id = self.execute(sql, [int(course_id), int(assignment_id), int(exercise_id), user_id, code, passed, date, partner_id])

        # #TODO: Execute this all in one transaction
        # #      https://stackoverflow.com/questions/54289555/how-do-i-execute-an-sqlite-script-from-within-python
        # if len(test_outputs) > 0:
        #     test_sql = '''INSERT INTO test_outputs (test_id, submission_id, txt_output, jpg_output)
        #                   VALUES (?, ?, ?, ?)'''

        #     for test_title, test_dict in test_outputs.items():
        #         if test_dict["jpg_output"] != "":
        #             #if test_dict["jpg_output"].strip() == BLANK_IMAGE.strip():
        #             if test_dict["jpg_output"].strip() == BLANK_IMAGE:
        #                 test_dict["jpg_output"] = ""

        #         self.execute(test_sql, [exercise_details["tests"][test_title]["test_id"], submission_id, test_dict["txt_output"], test_dict["jpg_output"]])

        # self.save_presubmission(course_id, assignment_id, exercise_id, user_id, code)

        # return submission_id

    def save_help_request(self, course_id, assignment_id, exercise_id, user_id, code, txt_output, jpg_output, student_comment, date):
        sql = '''INSERT INTO help_requests (course_id, assignment_id, exercise_id, user_id, code, txt_output, jpg_output, student_comment, approved, date, more_info_needed)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

        self.execute(sql, (course_id, assignment_id, exercise_id, user_id, code, txt_output, jpg_output, student_comment, 0, date, 0,))

    def update_help_request(self, course_id, assignment_id, exercise_id, user_id, student_comment):
        sql = '''UPDATE help_requests
                 SET student_comment = ?, more_info_needed = ?, suggestion = ?, approved = ?
                 WHERE course_id = ?
                   AND assignment_id = ?
                   AND exercise_id = ?
                   AND user_id = ?'''
        self.execute(sql, (student_comment, 0, None, 0, course_id, assignment_id, exercise_id, user_id,))

    def delete_help_request(self, course_id, assignment_id, exercise_id, user_id):
        sql = '''DELETE FROM help_requests
                 WHERE course_id = ?
                   AND assignment_id = ?
                   AND exercise_id = ?
                   AND user_id = ?'''
        self.execute(sql, (course_id, assignment_id, exercise_id, user_id,))

    def save_help_request_suggestion(self, course_id, assignment_id, exercise_id, user_id, suggestion, approved, suggester_id, approver_id, more_info_needed):
        sql = '''UPDATE help_requests
                 SET suggestion = ?, approved = ?, suggester_id = ?, approver_id = ?, more_info_needed = ?
                 WHERE course_id = ?
                   AND assignment_id = ?
                   AND exercise_id = ?
                   AND user_id = ?'''

        self.execute(sql, (suggestion, approved, suggester_id, approver_id,  more_info_needed, course_id, assignment_id, exercise_id, user_id,))

    async def copy_course(self, existing_course_basics, new_course_title):
        sql = '''INSERT INTO courses (title, introduction, visible, passcode, allow_students_download_submissions, date_created, date_updated)
                 SELECT ?, introduction, visible, passcode, allow_students_download_submissions, date_created, date_updated
                 FROM courses
                 WHERE course_id = ?'''

        new_course_id = self.execute(sql, (new_course_title, existing_course_basics['id'],))

        for assignment_basics in self.get_assignments(existing_course_basics):
            sql = '''INSERT INTO assignments (course_id, title, visible, introduction, date_created, date_updated, start_date, due_date, allow_late, late_percent, view_answer_late, enable_help_requests, has_timer, hour_timer, minute_timer, restrict_other_assignments, allowed_ip_addresses, allowed_external_urls)
                     SELECT ?, title, visible, introduction, date_created, date_updated, start_date, due_date, allow_late, late_percent, view_answer_late, enable_help_requests, has_timer, hour_timer, minute_timer, restrict_other_assignments, allowed_ip_addresses, allowed_external_urls
                     FROM assignments
                     WHERE course_id = ?
                       AND assignment_id = ?'''

            new_assignment_id = self.execute(sql, (new_course_id, existing_course_basics['id'], assignment_basics[0],))

            sql = '''SELECT exercise_id
                     FROM exercises
                     WHERE course_id = ?
                       AND assignment_id = ?'''

            old_exercise_ids = [row["exercise_id"] for row in self.fetchall(sql, (existing_course_basics['id'], assignment_basics[0],))]

            for exercise_id in old_exercise_ids:
                sql = '''INSERT INTO exercises (course_id, assignment_id, title, visible, solution_code, solution_description, hint, max_submissions, credit, data_files, back_end, instructions, output_type, what_students_see_after_success, starter_code, date_created, date_updated, enable_pair_programming, verification_code, weight, allow_any_response)
                         SELECT ?, ?, title, visible, solution_code, solution_description, hint, max_submissions, credit, data_files, back_end, instructions, output_type, what_students_see_after_success, starter_code, date_created, date_updated, enable_pair_programming, verification_code, weight, allow_any_response
                         FROM exercises
                         WHERE course_id = ?
                           AND assignment_id = ?
                           AND exercise_id = ?'''

                new_exercise_id = self.execute(sql, (new_course_id, new_assignment_id, existing_course_basics['id'], assignment_basics[0], exercise_id))

                sql = '''INSERT INTO tests (course_id, assignment_id, exercise_id, title, before_code, after_code, instructions, txt_output, jpg_output, can_see_test_code, can_see_expected_output, can_see_code_output)
                         SELECT ?, ?, ?, title, before_code, after_code, instructions, txt_output, jpg_output, can_see_test_code, can_see_expected_output, can_see_code_output
                         FROM tests
                         WHERE course_id = ?
                           AND assignment_id = ?
                           AND exercise_id = ?'''

                self.execute(sql, (new_course_id, new_assignment_id, new_exercise_id, existing_course_basics['id'], assignment_basics[0], exercise_id))

        sql = '''INSERT INTO permissions (user_id, role, course_id)
                 SELECT user_id, role, ?
                 FROM permissions
                 WHERE course_id = ?'''

        self.execute(sql, (new_course_id, existing_course_basics['id'],))

        self.update_when_content_updated(new_course_id)

    def copy_assignment(self, course_id, assignment_id, new_title):
        sql = '''INSERT INTO assignments (course_id, title, visible, introduction, date_created, date_updated, start_date, due_date, allow_late, late_percent, view_answer_late, enable_help_requests, has_timer, hour_timer, minute_timer, restrict_other_assignments, allowed_ip_addresses, allowed_external_urls)
                 SELECT course_id, ?, visible, introduction, date_created, date_updated, start_date, due_date, allow_late, late_percent, view_answer_late, enable_help_requests, has_timer, hour_timer, minute_timer, restrict_other_assignments, allowed_ip_addresses, allowed_external_urls
                 FROM assignments
                 WHERE course_id = ?
                   AND assignment_id = ?'''

        new_assignment_id = self.execute(sql, (new_title, course_id, assignment_id,))

        sql = '''SELECT exercise_id
                 FROM exercises
                 WHERE course_id = ?
                   AND assignment_id = ?'''

        old_exercise_ids = [row["exercise_id"] for row in self.fetchall(sql, (course_id, assignment_id,))]

        for exercise_id in old_exercise_ids:
            sql = '''INSERT INTO exercises (course_id, assignment_id, title, visible, solution_code, solution_description, hint, max_submissions, credit, data_files, back_end, instructions, output_type, what_students_see_after_success, starter_code, date_created, date_updated, enable_pair_programming, verification_code, weight, allow_any_response)
                     SELECT course_id, ?, title, visible, solution_code, solution_description, hint, max_submissions, credit, data_files, back_end, instructions, output_type, what_students_see_after_success, starter_code, date_created, date_updated, enable_pair_programming, verification_code, weight, allow_any_response
                     FROM exercises
                     WHERE course_id = ?
                       AND assignment_id = ?
                       AND exercise_id = ?'''

            new_exercise_id = self.execute(sql, (new_assignment_id, course_id, assignment_id, exercise_id))

            sql = '''INSERT INTO tests (course_id, assignment_id, exercise_id, title, before_code, after_code, instructions, txt_output, jpg_output, can_see_test_code, can_see_expected_output, can_see_code_output)
                     SELECT course_id, ?, ?, title, before_code, after_code, instructions, txt_output, jpg_output, can_see_test_code, can_see_expected_output, can_see_code_output
                     FROM tests
                     WHERE course_id = ?
                       AND assignment_id = ?
                       AND exercise_id = ?'''

            self.execute(sql, (new_assignment_id, new_exercise_id, course_id, assignment_id, exercise_id))

        self.update_when_content_updated(course_id)

    def update_user(self, user_id, user_dict):
        self.set_user_dict_defaults(user_dict)

        sql = '''UPDATE users
                 SET name = ?, given_name = ?, family_name = ?, locale = ?, email_address = ?
                 WHERE user_id = ?'''

        self.execute(sql, (user_dict["name"], user_dict["given_name"], user_dict["family_name"], user_dict["locale"], user_dict["email_address"], user_id,))

        self.update_when_content_updated("user")

    def update_user_settings(self, user_id, theme, use_auto_complete, use_studio_mode, enable_vim):
        sql = '''UPDATE users
                 SET ace_theme = ?, use_auto_complete = ?, use_studio_mode = ?, enable_vim = ?
                 WHERE user_id = ?'''
        self.execute(sql, (theme, use_auto_complete, use_studio_mode, enable_vim, user_id))

        self.update_when_content_updated("user")

    def remove_user_submissions(self, user_id):
        sql = '''SELECT submission_id
                 FROM submissions
                 WHERE user_id = ?'''

        submissions = self.fetchall(sql, (user_id,))

        if submissions:
            sql = '''DELETE FROM scores
                     WHERE user_id = ?'''
            self.execute(sql, (user_id,))

            sql = '''DELETE FROM submissions
                     WHERE user_id = ?'''
            self.execute(sql, (user_id,))

            return True
        else:
            return False

    def delete_user(self, user_id):
        sql = '''DELETE FROM users
                  WHERE user_id = ?'''

        self.execute(sql, (user_id,))

        self.update_when_content_updated("user")

    def move_assignment(self, course_id, assignment_id, new_course_id):
        for table in ["assignments", "exercises", "tests"]:
            self.execute(f'''UPDATE {table}
                             SET course_id = ?
                             WHERE course_id = ?
                               AND assignment_id = ?''', (new_course_id, course_id, assignment_id))

        for table in ["help_requests", "presubmissions", "scores", "submissions", "user_assignment_starts"]:
            self.execute(f'''DELETE FROM {table}
                             WHERE course_id = ?
                               AND assignment_id = ?''', (course_id, assignment_id))
            
        self.update_when_content_updated(course_id)

    #TODO: Make this like move_assignment
    def move_exercise(self, course_id, assignment_id, exercise_id, new_assignment_id):
        self.execute('''UPDATE exercises
                        SET assignment_id = ?
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (new_assignment_id, course_id, assignment_id, exercise_id, ))

        self.execute('''UPDATE scores
                        SET assignment_id = ?
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (new_assignment_id, course_id, assignment_id, exercise_id, ))

        self.execute('''UPDATE submissions
                        SET assignment_id = ?
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (new_assignment_id, course_id, assignment_id, exercise_id, ))

        self.execute('''UPDATE presubmissions
                        SET assignment_id = ?
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (new_assignment_id, course_id, assignment_id, exercise_id, ))

        self.execute('''UPDATE scores
                        SET assignment_id = ?
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (new_assignment_id, course_id, assignment_id, exercise_id, ))

        self.execute('''UPDATE help_requests
                        SET assignment_id = ?
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (new_assignment_id, course_id, assignment_id, exercise_id, ))

        self.execute('''UPDATE tests
                        SET assignment_id = ?
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (new_assignment_id, course_id, assignment_id, exercise_id, ))
        
        self.update_when_content_updated(course_id)

    def copy_exercise(self, course_id, assignment_id, exercise_id, new_title):
        try:
            sql = '''INSERT INTO exercises (course_id, assignment_id, title, visible, solution_code, solution_description, hint, max_submissions, credit, data_files, back_end, instructions, output_type, what_students_see_after_success, starter_code, date_created, date_updated, enable_pair_programming, verification_code, weight, allow_any_response)
                     SELECT course_id, assignment_id, ?, visible, solution_code, solution_description, hint, max_submissions, credit, data_files, back_end, instructions, output_type, what_students_see_after_success, starter_code, date_created, date_updated, enable_pair_programming, verification_code, weight, allow_any_response
                     FROM exercises
                     WHERE course_id = ?
                         AND assignment_id = ?
                         AND exercise_id = ?'''

            new_exercise_id = self.execute(sql, (new_title, course_id, assignment_id, exercise_id, ))

            sql = '''INSERT INTO tests (course_id, assignment_id, exercise_id, title, before_code, after_code, instructions, txt_output, jpg_output, can_see_test_code, can_see_expected_output, can_see_code_output)
                     SELECT course_id, assignment_id, ?, title, before_code, after_code, instructions, txt_output, jpg_output, can_see_test_code, can_see_expected_output, can_see_code_output
                     FROM tests
                     WHERE course_id = ?
                       AND assignment_id = ?
                       AND exercise_id = ?'''

            self.execute(sql, (new_exercise_id, course_id, assignment_id, exercise_id, ))
        except:
            print(traceback.format_exc())

        self.update_when_content_updated(course_id)

    #TODO: Clean up all of these delete functions.
    def delete_exercise(self, course_id, assignment_id, exercise_id):
        self.execute('''DELETE FROM submissions
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (course_id, assignment_id, exercise_id, ))

        self.execute('''DELETE FROM presubmissions
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (course_id, assignment_id, exercise_id, ))

        self.execute('''DELETE FROM scores
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (course_id, assignment_id, exercise_id, ))

        self.execute('''DELETE FROM exercises
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (course_id, assignment_id, exercise_id, ))

        self.execute('''DELETE FROM tests
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (course_id, assignment_id, exercise_id, ))

        self.execute('''DELETE FROM test_outputs
                        WHERE submission_id IN (
                          SELECT submission_id
                          FROM submissions
                          WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?)''', (course_id, assignment_id, exercise_id, ))

        self.execute('''DELETE FROM help_requests
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (course_id, assignment_id, exercise_id, ))
        
        self.update_when_content_updated(course_id)

    def delete_assignment(self, assignment_basics):
        course_id = assignment_basics["course"]["id"]
        assignment_id = assignment_basics["id"]

        self.execute('''DELETE FROM tests
                        WHERE course_id = ?
                          AND assignment_id = ?''', (course_id, assignment_id, ))

        self.execute('''DELETE FROM test_outputs
                        WHERE submission_id IN (
                          SELECT submission_id
                          FROM submissions
                          WHERE course_id = ?
                          AND assignment_id = ?)''', (course_id, assignment_id, ))

        self.execute('''DELETE FROM presubmissions
                        WHERE course_id = ?
                          AND assignment_id = ?''', (course_id, assignment_id, ))

        self.execute('''DELETE FROM submissions
                        WHERE course_id = ?
                          AND assignment_id = ?''', (course_id, assignment_id, ))

        self.execute('''DELETE FROM scores
                        WHERE course_id = ?
                          AND assignment_id = ?''', (course_id, assignment_id, ))

        self.execute('''DELETE FROM user_assignment_starts
                        WHERE course_id = ?
                          AND assignment_id = ?''', (course_id, assignment_id, ))

        self.execute('''DELETE FROM help_requests
                        WHERE course_id = ?
                          AND assignment_id = ?''', (course_id, assignment_id, ))

        self.execute('''DELETE FROM exercises
                        WHERE course_id = ?
                          AND assignment_id = ?''', (course_id, assignment_id, ))

        self.execute('''DELETE FROM assignments
                        WHERE course_id = ?
                          AND assignment_id = ?''', (course_id, assignment_id, ))
        
        self.update_when_content_updated(course_id)

    def delete_course(self, course_id):
        self.execute('''DELETE FROM presubmissions
                        WHERE course_id = ?''', (course_id, ))

        self.execute('''DELETE FROM tests
                        WHERE course_id = ?''', (course_id, ))

        self.execute('''DELETE FROM test_outputs
                        WHERE submission_id IN (
                          SELECT submission_id
                          FROM submissions
                          WHERE course_id = ?)''', (course_id, ))

        self.execute('''DELETE FROM submissions
                        WHERE course_id = ?''', (course_id, ))

        self.execute('''DELETE FROM exercises
                        WHERE course_id = ?''', (course_id, ))

        self.execute('''DELETE FROM assignments
                        WHERE course_id = ?''', (course_id, ))

        self.execute('''DELETE FROM courses
                        WHERE course_id = ?''', (course_id, ))

        self.execute('''DELETE FROM scores
                        WHERE course_id = ?''', (course_id, ))

        self.execute('''DELETE FROM course_registrations
                        WHERE course_id = ?''', (course_id, ))

        self.execute('''DELETE FROM help_requests
                        WHERE course_id = ?''', (course_id, ))

        self.execute('''DELETE FROM permissions
                        WHERE course_id = ?''', (course_id, ))

        self.execute('''DELETE FROM user_assignment_starts
                        WHERE course_id = ?''', (course_id, ))
        
        self.delete_content_updated(course_id)

    def delete_course_submissions(self, course_id):
        self.execute('''DELETE FROM submissions
                        WHERE course_id = ?''', (course_id, ))

        self.execute('''DELETE FROM scores
                        WHERE course_id = ?''', (course_id, ))

        self.execute('''DELETE FROM presubmissions
                        WHERE course_id = ?''', (course_id, ))

        self.execute('''DELETE FROM help_requests
                        WHERE course_id = ?''', (course_id, ))

        self.execute('''DELETE FROM user_assignment_starts
                        WHERE course_id = ?''', (course_id, ))

        self.execute('''DELETE FROM test_outputs
                        WHERE submission_id IN (
                          SELECT submission_id
                          FROM submissions
                          WHERE course_id = ?)''', (course_id, ))

    def delete_assignment_submissions(self, assignment_basics):
        course_id = assignment_basics["course"]["id"]
        assignment_id = assignment_basics["id"]

        self.execute('''DELETE FROM submissions
                        WHERE course_id = ?
                          AND assignment_id = ?''', (course_id, assignment_id, ))

        self.execute('''DELETE FROM scores
                        WHERE course_id = ?
                          AND assignment_id = ?''', (course_id, assignment_id, ))

        self.execute('''DELETE FROM presubmissions
                        WHERE course_id = ?
                          AND assignment_id = ?''', (course_id, assignment_id, ))

        self.execute('''DELETE FROM help_requests
                        WHERE course_id = ?
                          AND assignment_id = ?''', (course_id, assignment_id, ))

        self.execute('''DELETE FROM user_assignment_starts
                        WHERE course_id = ?
                          AND assignment_id = ?''', (course_id, assignment_id, ))

        self.execute('''DELETE FROM test_outputs
                        WHERE submission_id IN (
                          SELECT submission_id
                          FROM submissions
                          WHERE course_id = ?
                          AND assignment_id = ?)''', (course_id, assignment_id, ))

    async def delete_exercise_submissions(self, course_id, assignment_id, exercise_id):
        self.execute('''DELETE FROM submissions
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (course_id, assignment_id, exercise_id, ))

        self.execute('''DELETE FROM scores
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (course_id, assignment_id, exercise_id, ))

        self.execute('''DELETE FROM presubmissions
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (course_id, assignment_id, exercise_id, ))

        self.execute('''DELETE FROM help_requests
                        WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?''', (course_id, assignment_id, exercise_id, ))

        self.execute('''DELETE FROM test_outputs
                        WHERE submission_id IN (
                          SELECT submission_id
                          FROM submissions
                          WHERE course_id = ?
                          AND assignment_id = ?
                          AND exercise_id = ?)''', (course_id, assignment_id, exercise_id, ))

    # async def create_course_scores_text(self, course_basics):
    #     out_file_text = "Assignment_ID\tAssignment_Title\tStudent_ID\tScore\n"

    #     for assignment_basics in self.get_assignments(course_basics):
    #         assignment_scores = self.get_assignment_scores(course_basics, assignment_basics)

    #         for score_dict in assignment_scores:
    #             out_file_text += f"{assignment_basics['assignment_id']}\t{assignment_basics['title']}\t{score_dict[0]}\t{score_dict[1]['percent_passed']}\n"

    #     return out_file_text

    async def create_assignment_scores_text(self, course_basics, assignment_basics):
        #TODO: Save the title(s) rather than ID(s).
        out_file_text = "Course\tAssignment\tStudent_ID\tScore\tWhen_Passed\tLast_Submission\n"
        scores = self.get_assignment_scores(course_basics, assignment_basics)

        course_id = course_basics["title"]
        assignment_id = assignment_basics["title"]

        for student in scores:
            out_file_text += f"{course_id}\t{assignment_id}\t{student[0]}\t{student[1]['percent_passed']}\t{student[1]['when_passed']}\t{student[1]['last_submission_time']}\n"

        return out_file_text

#    def export_data(self, course_basics, table_name, output_tsv_file_path):
#        if table_name == "submissions":
#            sql = '''SELECT c.title, a.title, e.title, s.user_id, s.submission_id, s.code, s.txt_output, s.jpg_output, s.passed, s.date
#                    FROM submissions s
#                    INNER JOIN courses c
#                      ON c.course_id = s.course_id
#                    INNER JOIN assignments a
#                      ON a.assignment_id = s.assignment_id
#                    INNER JOIN exercises e
#                      ON e.exercise_id = s.exercise_id
#                    WHERE s.course_id = ?'''
#
#        else:
#            sql = f"SELECT * FROM {table_name} WHERE course_id = ?"
#
#        rows = []
#        for row in self.fetchall(sql, (course_basics["id"],)):
#            row_values = []
#            for x in row:
#                if type(x) is datetime:
#                    x = str(x)
#                row_values.append(x)
#
#            rows.append(row_values)
#
#        with open(output_tsv_file_path, "w") as out_file:
#            out_file.write(json.dumps(rows))

#    def create_zip_file_path(self, descriptor):
#        temp_dir_path = "/database/tmp/{}".format(create_id())
#        zip_file_name = f"{descriptor}.zip"
#        zip_file_path = f"{temp_dir_path}/{zip_file_name}"
#        return temp_dir_path, zip_file_name, zip_file_path
#
#    def zip_export_files(self, temp_dir_path, zip_file_name, zip_file_path, descriptor):
#        os.system(f"cp ../VERSION {temp_dir_path}/{descriptor}/")
#        os.system(f"cd {temp_dir_path}; zip -r -qq {zip_file_path} .")
#
#    def create_export_paths(self, temp_dir_path, descriptor):
#        os.makedirs(temp_dir_path)
#        os.makedirs(f"{temp_dir_path}/{descriptor}")
#
#    def remove_export_paths(self, zip_file_path, tmp_dir_path):
#        if os.path.exists(zip_file_path):
#            os.remove(zip_file_path)
#
#        if os.path.exists(tmp_dir_path):
#            shutil.rmtree(tmp_dir_path, ignore_errors=True)

    async def get_student_pairs(self, course_id, user_name):
        # Uses the week of the year as a seed.
        seed = datetime.utcnow().isocalendar().week

        # Gets student names registered in a course (will add obscured emails to the end of the name in the case of duplicate names)
        students = list(self.get_partner_info(course_id, False).keys())

        # Randomizes students using seed
        random.Random(seed).shuffle(students)

        if len(students) == 0:
            pairs = []
        elif len(students) % 2 == 0:
            pairs = [[students[i], students[i + 1]] for i in range(0, len(students), 2)]
        else:
            # Create pairs for everyone except the last student.
            pairs = [[students[i], students[i + 1]] for i in range(0, len(students) - 1, 2)]

            # This code creates a trio.
            #pairs[-1].append(students[-1])

            # This code puts one person on their own.
            pairs.extend([[students[-1]]])

        # Indicates which pair the user is in.
        pairs = [{'is_user': True, 'pair': pair} if user_name in pair else {'is_user': False, 'pair': pair} for pair in pairs]

        return pairs

    def get_next_prev_student_ids(self, course_id, student_id):
        sql = '''SELECT u.user_id
                 FROM users u
                 INNER JOIN course_registrations cr
                   ON u.user_id = cr.user_id
                 WHERE cr.course_id = ?
                 ORDER BY u.name'''

        user_ids = [row["user_id"] for row in self.fetchall(sql, (course_id, ))]

        student_index = user_ids.index(student_id)

        prev_student_id = None
        next_student_id = None

        if student_index > 0:
            prev_student_id = user_ids[student_index - 1]
        if student_index < len(user_ids) - 1:
            next_student_id = user_ids[student_index + 1]

        return prev_student_id, next_student_id

    def get_submissions_student(self, course_id, student_id):
        sql = '''SELECT a.title AS assignment_title,
                        a.introduction AS assignment_introduction,
                        e.title AS exercise_title,
                        e.instructions AS exercise_instructions,
                        s.code,
                        max(s.date)
                 FROM submissions s
                 INNER JOIN exercises e
                   ON s.exercise_id = e.exercise_id
                 INNER JOIN assignments a
                   ON s.assignment_id = a.assignment_id
                 WHERE s.course_id = ?
                   AND s.user_id = ?
                   AND s.passed = 1
                   AND e.visible = 1
                   AND a.visible = 1
                   AND a.has_timer = 0
                   AND e.back_end != "not_code"
                 GROUP BY a.assignment_id, e.exercise_id'''

        submissions = []
        for row in self.fetchall(sql, (course_id, student_id,)):
            submission = {}
            for x in ["assignment_title", "assignment_introduction", "exercise_title", "exercise_instructions", "code"]:
                submission[x] = row[x]

            submissions.append(submission)

        return sort_list_of_dicts_nicely(submissions, ["assignment_title", "exercise_title"])