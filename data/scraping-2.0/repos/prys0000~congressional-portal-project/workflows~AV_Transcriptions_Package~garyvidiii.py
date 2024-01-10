import openai
import os
import re
import time
import dateparser  # You may need to install this library for date parsing
from dateparser import parse as dateparser_parse
import spacy  # You may need to install spaCy for named entity recognition
import pandas as pd
import sys
from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer
from dateutil.parser import parse as dateparser_parse
import logging
import speech_recognition as sr
from moviepy.editor import VideoFileClip

### EMAIL japryse@ou.edu for script
