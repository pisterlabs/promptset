import logging
import imaplib
import email
from email import policy
import openai
import eons
import re
from markdownify import markdownify
from ebbs import Builder

class get_email(Builder):
	def __init__(this, name = "Get Email"):
		super().__init__(name)
		this.requiredKWArgs.append('server')
		this.requiredKWArgs.append('username')
		this.requiredKWArgs.append('password')
		
		this.optionalKWArgs['port'] = 993
		this.optionalKWArgs['ssl'] = True
		this.optionalKWArgs['folder'] = 'INBOX'
		this.optionalKWArgs['search'] = '(UNSEEN)'
		this.optionalKWArgs['delete'] = False
		this.optionalKWArgs['mark_as_read'] = False
		this.optionalKWArgs['summarize'] = False
		this.optionalKWArgs['openai_api_key'] = None
		this.optionalKWArgs['openai_engine'] = 'davinci'
		this.optionalKWArgs['openai_max_tokens'] = 500
		this.optionalKWArgs['openai_temperature'] = 0.9
		this.optionalKWArgs['openai_prompt'] = "Please summarize this email in one sentence:"
		
		this.emails = []
		

	def Build(this):
		if this.ssl:
			this.mail = imaplib.IMAP4_SSL(this.server, this.port)
		else:
			this.mail = imaplib.IMAP4(this.server, this.port)
		this.mail.login(this.username, this.password)
		this.mail.select(this.folder)
		_, data = this.mail.search(None, this.search)
		for num in data[0].split():
			_, rawMessage = this.mail.fetch(num, '(RFC822)')
			message = email.message_from_bytes(rawMessage[0][1], policy=policy.default)
			try:
				body = message.get_body(preferencelist=('plain', 'html')).as_string()
				body = this.StripHeaders(body)
				body = this.StripGoogleGroupFooter(body)
				body = markdownify(body, convert=['p', 'b', 'i', 'strong', 'em', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'ul', 'ol', 'li', 'br', 'hr', 'blockquote', 'code', 'pre'])
			except:
				body = None

			summary = None
			if (this.summarize and body):
				summary = this.GetEmailSummary(body)

			this.emails.append(eons.util.DotDict({
				'from': message['From'],
				'to': message['To'],
				'subject': message['Subject'],
				'body': body,
				'summary': summary
			}))
			
			# logging.info('Message %s: %s' % (num, msg['Subject']))
			if (this.mark_as_read):
				this.mail.store(num, '+FLAGS', '\\Seen')
			if (this.delete):
				this.mail.store(num, '+FLAGS', '\\Deleted')
		if (this.delete):
			this.mail.expunge()
		this.mail.close()
		this.mail.logout()

		return this.emails
	

	def StripHeaders(this, message):
		for iter,line in enumerate(message.split('\n')):
			if (not re.match(r'^\s.*$', line) and not re.match(r'[a-zA-Z-]+:.*$', line)):
				logging.debug(f"Trimming the first {iter} lines from message to remove headers")
				return '\n'.join(message.split('\n')[iter:])

	def StripGoogleGroupFooter(this, message):
		for iter,line in enumerate(message.split('\n')):
			if (re.match(r'^You received this message because.*$', line)):
				return '\n'.join(message.split('\n')[:iter])


	def GetEmailSummary(this, message):
		if this.openai_api_key is None:
			raise Exception('OpenAI API Key is not set')
		if (len(message) > 2048):
			message = message[:2048]
		logging.debug(f"Getting summary for: {message}")
		openai.api_key = this.openai_api_key
		response = openai.Completion.create(
			engine = this.openai_engine,
			prompt = f"{this.openai_prompt} \n{message}",
			max_tokens = this.openai_max_tokens,
			temperature = this.openai_temperature,
			top_p = 1,
			n = 1,
			stream = False,
			logprobs = None,
			stop = None,
			presence_penalty = 0,
			frequency_penalty = 0,
			best_of = 1,
		)
		return response['choices'][0]['text']