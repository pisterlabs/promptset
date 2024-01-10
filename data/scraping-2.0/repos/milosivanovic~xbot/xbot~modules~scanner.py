import re
import time
import random
import json
import urllib.request
import urllib.error
import urllib.parse
import io
import PythonSed

from . import cleverbot
from . import openai

def scan(bot, message = None):
	results = []

	if message:
		bot.remote['message'] = message

	message_lowercase = bot.remote['message'].lower()

	# scan for youtube links and show title
	for code in re.findall('(?:youtube\.com\/watch\?|youtu\.be/)(?:[A-Za-z0-9-_\.&%#=]*v=)?([A-Za-z0-9-_]+)', bot.remote['message']):
		results.append(youtube_title(code))

	# someone is talking to the bot
	if re.search('^%s(?:\:|,)' % re.escape(bot.nick.lower()), message_lowercase):
		if bot.remote['nick'].lower() in bot.inv['banned']:
			return
		#bot._sendq(("NOTICE", bot.remote['nick']), "This feature has been disabled.")

		#if 'cleverbot' not in bot.inv: bot.inv['cleverbot'] = {}
		#if bot.remote['receiver'] not in bot.inv['cleverbot']:
		#	bot.inv['cleverbot'][bot.remote['receiver']] = cleverbot.CleverBot()
		#query = bot.remote['message'][len(bot.nick)+2:]
		#results.append("%s: %s" % (bot.remote['nick'], re.compile('cleverbot', re.IGNORECASE).sub(bot.nick, bot.inv['cleverbot'][bot.remote['receiver']].query(query))))
		if 'openai' not in bot.inv: bot.inv['openai'] = {}
		if bot.remote['receiver'] not in bot.inv['openai']:
			bot.inv['openai'][bot.remote['receiver']] = openai.OpenAIChat(bot)
		query = bot.remote['message'][len(bot.nick)+2:]
		results.append("%s: %s" % (bot.remote['nick'], re.compile('openai', re.IGNORECASE).sub(bot.nick, bot.inv['openai'][bot.remote['receiver']].ask(query))))

	# sed replace
	if bot.remote['message'].startswith("s/"):
		out = io.StringIO()
		message_stringio = io.StringIO(bot.previous['message'])
		sed = PythonSed.Sed()
		sed.regexp_extended = True
		try:
			sed.load_string(bot.remote['message'])
			sed_result = sed.apply(message_stringio, output=out)
			if len(sed_result):
				pre_append = "%s meant: %s" % (bot.remote['nick'], sed_result[0])
				if len(pre_append) > 429:
					pre_append = "%s..." % pre_append[:426]
				results.append(pre_append)
			else:
				if bot.remote['message'].count('/') == 2:
					results.append("%s: You're a dumdum." % bot.remote['nick'])
		except PythonSed.SedException as e:
			results.append(str(e))
		except IndexError:
			pass

	# per 10% chance, count uppercase and act shocked
	#if len(bot.remote['message']) > 2 and random.random() > 0.9:
	#	if count_upper(bot.remote['message']) > 80:
	#		time.sleep(4)
	#		results.append(random.choice([':' + 'O' * random.randint(1, 10), 'O' * random.randint(1, 10) + ':']))

	# per 0.01% chance, butt into someone's conversation
	'''if random.random() > 0.999:
		if not bot.remote['message'].startswith("\x01"):
			words = bot.remote['message'].split()
			if len(words) > 2:
				for n in range(random.randint(1, 3)):
					if random.random() > 0.5:
						words[random.randint(1, len(words)-1)] = "butt"
					else:
						for m, word in enumerate(words):
							if len(word) > 4 and m > 0:
								if random.random() > 0.3:
									words[m] = words[m][:-4] + "butt"

				results.append(' '.join(words))'''

	if 'gface' in bot.remote['message']:
		results.append('\x28\x20\xE2\x89\x96\xE2\x80\xBF\xE2\x89\x96\x29')

	results = [result for result in results if result is not None]
	try: return '\n'.join(results)
	except TypeError: return None

def youtube_title(code):
	try:
		# try with embed json data (fast)
		url = urllib.parse.quote_plus('https://www.youtube.com/watch?v=%s' % code)
		title = json.load(urllib.request.urlopen('https://www.youtube.com/oembed?url=%s' % url, timeout = 5))['title']
	except json.JSONDecodeError:
		# json data didn't return a title? forget about it
		title = None
	except urllib.error.HTTPError as error:
		# embed request not allowed? fallback to HTML (slower)
		if error.code == 401:
			import lxml.html
			try:
				title = lxml.html.document_fromstring(urllib.request.urlopen('https://www.youtube.com/watch?v=%s' % code, timeout = 5).read().decode('utf-8')).xpath("//title/text()")[0].replace(' - YouTube', '')
				#[0].split("\n")[1].strip()
			except IndexError:
				title = None
		else:
			title = None
			if error.code != 404:
				raise

	if title:
		if title != "YouTube - Broadcast Yourself.":
			return "YouTube: \x02%s\x02" % title

	return None

def count_upper(str):
	n = s = 0
	for c in str:
		z = ord(c)
		if (z >= 65 and z <= 90) or z == 33:
			n += 1
		if z == 32:
			s += 1

	return float(n) / (len(str)-s) * 100
