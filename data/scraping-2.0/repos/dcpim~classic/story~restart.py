#!/usr/bin/python3
# This script restarts the story

import os
import openai
import connix
import run

print(connix.header())
form = connix.form()

if not run.validate(False):
	run.error("Unauthorized.")

# Add reset to db
run.sql("INSERT INTO ai_story (username, sentence, date) VALUES (%s, %s, %s);", "System", "<hr>", connix.now())

# Done
run.done()
