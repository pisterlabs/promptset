import openai
import json
import logging
import time
from flask import Flask, request, render_template, render_template_string, jsonify, make_response, redirect, url_for, make_response

app = Flask(__name__)
openai.api_key = '###'

conversation_context = ''
finished = False
guess = None

html_content = '''
<!DOCTYPE html>
<html lang="en">
<script type="text/javascript" src="https://livejs.com/live.js"></script>
<head>
	<meta charset="UTF-8">
	<title>Chat with GPT-3</title>
	<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.1/jquery.min.js"></script>
</head>
<script>
function myFunction() {
var message= $('#message').val();
  $.ajax({
			  url: "/join",
			  type: "POST",
			  data: {message:message}
		  }).done;
};
function restartConversation() {
  $.ajax({
			  url: "/join",
			  type: "POST",
			  data: {message:'##RESTART##'}
		  }).done;
};

</script>
<body>
	<p>
		<h2>Chat with GPT-3!</h2><br>This service would not have been possible without the great work of <a href="https://openai.com/blog/openai-api/">OpenAI</a>. <br>Developed by Lucas Vergeest	  <a href="https://www.linkedin.com/in/lucasvergeest/">
		 <img alt="LinkedIn" src="https://veldboereenhoorn.nl/wp-content/uploads/2018/08/Linkedin-logo-1-550x550-300x300.png"
		 width="30" height="30"></a><br><br>
		Message: <input type="text" style="font-size: 20px; width: 500px;" autofocus onkeydown="if (event.keyCode == 13) { myFunction(); return false; }" id="message" name="message">   <button id="clicked" onclick="myFunction()">Submit</button>  <button id="restart" onclick="restartConversation()">Restart</button>
	</p>
	<div class="show-data" >
	</div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'HEAD'])
def start():
		if guess == 'human' and answer == 'machine':
			return render_template('human_machine.html')
		elif guess == 'machine' and answer == 'machine':
			return render_template('machine_machine.html')
		elif finished:
			return render_template('question.html')
		else:
			try:
				return render_template_string(html_content_new)
			except:
				return render_template_string(html_content)


@app.route('/join', methods=['GET', 'POST'])
def message():
	human = False
	input_text = request.form['message']
	ip_address = request.remote_addr
	resp = make_response(render_template_string(html_content))

	if input_text == '##RESTART##':
		resp.set_cookie('conversation_log', '')
		conversation_context = ''
	else:
		print('A message was posted: ' + input_text + ' | IP address: ' + ip_address)
		old_conversation_context = request.cookies.get('conversation_log')
		if old_conversation_context == None:
			conversation_context = 'You: "How are you today?"\nLea: "I am fine, thanks, how about yourself?"\nYou: "Not too bad either."\nLea: "I am very glad to hear that!"\n'
		else:
			conversation_context = old_conversation_context

		# Write chat logs to file
		log_file = open('log_file_{}'.format(ip_address), 'w+')
		log_file.write(conversation_context)

		context_lines = conversation_context.split('\n')
		if len(context_lines) > 14:
			context_lines_used = context_lines[-14:]
		else:
			context_lines_used = context_lines
		context_lines_used = '\n'.join(context_lines_used)
		raw_text = context_lines_used + '\nYou: "' + input_text + '"\nLea: "'

		response = openai.Completion.create(engine="davinci", temperature=0.8, frequency_penalty=0.5, prompt=raw_text, max_tokens=40)
		response_text = response.choices[0].text
		response_text = response_text.split('\n')[0]
		delay_time = len(response_text)
		# time.sleep(3 + (delay_time / 10))
	
		new_context = '\nYou: "' + input_text + '"\nLea: "' + response_text
		conversation_context += new_context
		conversation_context_display = conversation_context.split('\n')[4:]
		if len(conversation_context_display) >= 32:
			conversation_context_display = conversation_context_display[-32:]
		conversation_context_display = '\n'.join(conversation_context_display)
		resp.set_cookie('conversation_log', conversation_context)
	
		conversation_context_display = conversation_context_display.replace('\n', '<br>')
		global html_content_new
		html_content_new = html_content.split('\n')
		html_content_new.insert(-3, conversation_context_display)
		html_content_new = '\n'.join(html_content_new)
	return resp, human

@app.route('/GuessHuman', methods=['POST'])
def GuessHumanFunction():
	global guess, answer
	guess = 'human'
	answer = 'machine'
	return guess, answer

@app.route('/GuessMachine', methods=['POST'])
def GuessMachineFunction():
	global guess, answer
	guess = 'machine'
	answer = 'machine'
	return guess, answer

app.run(debug=True, host='0.0.0.0', port=8080)
#serve(app, host='0.0.0.0', port=8080)
