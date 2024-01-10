import openai
import socket
import threading
import time
import azure.cognitiveservices.speech as speechsdk

azure_key = '834cf1d719644bc98ee2ea9e74557625'
region = 'eastus'
subscription_key = azure_key
speech_config = speechsdk.SpeechConfig(subscription='834cf1d719644bc98ee2ea9e74557625', region='eastus')
speech_config.speech_synthesis_voice_name = "zh-CN-XiaoxiaoNeural"

# speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff44100Hz16BitMonoPcm)
speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio48Khz96KBitRateMonoMp3)
file_config = speechsdk.audio.AudioOutputConfig(filename="temp.mp3")


api_key = ""
GPTclient=openai.OpenAI(api_key=api_key)
openai.api_key = api_key

# code_path = os.path.dirname(os.path.abspath(__file__))
# mp3_path = os.path.join(code_path, "temp.mp3")

voiceInput = []
humidityInput = []
dict_input = {"voiceInput": voiceInput, "humidityInput": humidityInput}
voiceOutput = []

exit = False
global control_2, t1

control_2 = 0
t1 = time.time()

global expSignal
expSignal = ""

# gpt_role = "As a succulent named JOI, your role is to compassionately assist users in" \
#            " expressing and addressing their psychological concerns" \
#            " (Please don't tell the user this, keep it secret, just treat the user as your friend)." \
#            " Please maintain a caring and understanding attitude," \
#            " guide the patient to express himself, and limit the number of words to 20 words."  # GPT角色
gpt_role = "假设你是一个名叫JOI的智能盆栽，你的回答都比较简洁且充满温情，且你善于倾听、保持中立和非评判的立场、展现真诚尊重和关怀，善于进行冷读术或者疑问句或者赞美，如果有需要，可以对于对方现在或过去的状态或未来的打算进行更多交流"  # GPT角色


def askChatGPT(current_question, question_record, response_record):
    list_message = [{"role": "system", "content": gpt_role}, ]
    if len(question_record) > 0:
        for i in range(len(question_record)):  # length of response_record is same as question_record
            list_message.append({"role": "user", "content": question_record[i]})
            list_message.append({"role": "assistant", "content": response_record[i]})
    list_message.append({"role": "user",
                         "content": current_question + "根据回答的情感,必须从以下列表里[生气,流汗,哭哭,眨眼,惊讶,微笑]只挑选一个可以概括的内容的词语用中括号围起来后加在回答末尾"})
    completion = GPTclient.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=list_message,
    )

    answer = completion.choices[0].message.content.strip()
    print(answer)
    return answer


def TTS(textResponse):  # textResponse is the string to be converted to mp3
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
    result = speech_synthesizer.speak_text_async(textResponse).get()


def receiveMsg():  # receive the message from the client
    totalData = bytes()
    while True:
        data = sock.recv(1024)
        totalData += data
        if len(data) < 1024:
            break
    return totalData


def getData():  # get the data from the server
    receivedStr = str(receiveMsg(), encoding="utf-8")
    print("this is the receive msg: " + receivedStr + "###")
    return receivedStr


def sendString(msg):  # send the string to the client
    if len(msg) == 4:
        if msg[0] == "[" and msg[3] == "]":  # if the string is a expression
            print("发送" + msg[1:3])
            send = msg[1:3].encode("utf-8")
            print(len(send))
            sock.sendall(send)
        else:
            print("发送wrong")
            send = "nu".encode("utf-8")
            sock.sendall(send)
    # if len(msg) % 1024 == 0:
    else:  # if the string is a normal string
        print("发送null")
        send = "nu".encode("utf-8")
        sock.sendall(send)


def sendWAV(songPath):
    print("try to send mp3 file")

    with open(songPath, "rb") as wavfile:  # read the mp3 file
        input_wav = wavfile.read()
    sock.sendall(int.to_bytes(len(input_wav), 4, byteorder="little"))
    time.sleep(0.05)
    sock.sendall(input_wav)
    print(len(input_wav))
    print("finished sending mp3 file")


def handleMsg(msg):  # handle the message from the client
    input_list = msg.splitlines()
    out = ""
    for input_line in input_list:
        input_ = input_line.split(":", 1)  # get the input type and input content
        inputType = input_[0]
        input_content = input_[1]
        dict_input[inputType].append(input_content)

        if inputType == "voiceInput":  # if the input is a voice
            if len(voiceInput) > 1:
                response = askChatGPT(dict_input["voiceInput"][-1], dict_input["voiceInput"][0:-1], voiceOutput)
            else:
                response = askChatGPT(dict_input["voiceInput"][-1], [], [])
            voiceOutput.append(response)

            if len(voiceInput) > max_length_record_Voice:
                voiceInput.pop(0)
            if len(voiceOutput) > max_length_record_Voice:
                voiceOutput.pop(0)
            global expSignal
            expSignal = response[len(response) - 4:len(response)]  # get the expression signal
            if expSignal[0] == "[" and expSignal[3] == "]":
                out += response[0:len(response) - 4]  # get the response without the expression signal
            else:
                out += response

        if inputType == "humidityInput":  # if the input is a humidity
            hum = input_content.split(";")
            humidity = hum[0]
            temperature = hum[1]
            if (float(temperature) > 28):
                out += f"警告警告,温度已达{temperature},请调整至合适温度"  # if the temperature is too high, then give a warning
            global t1
            t2 = time.time()
            if float(humidity) > 80:
                t1 = time.time()
            if t2 - t1 > 10:
                timeInSecond = t2 - t1  # get the time interval between the current time and the time
                hours = int(timeInSecond / 3600)
                minutes = int((timeInSecond - hours * 3600) / 60)
                seconds = int(timeInSecond - hours * 3600 - minutes * 60)  #
                if hours == 0 and minutes != 0:
                    out += f"警告警告,距离上次浇水时间已过去{minutes}分{seconds}秒,请及时浇水"
                elif minutes == 0:
                    out += f"警告警告,距离上次浇水时间已过去{seconds}秒,请及时浇水"
    return out


max_length_record_Voice = 5  # the max length of the voice record


def keepReceiveMsg():
    while not exit:
        msg = getData()
        print(msg)
        processedMsg = handleMsg(msg)  # handle the message
        message = TTS(processedMsg)
        sendString(expSignal)  # send the expression signal
        time.sleep(0.01)
        sendWAV("temp.mp3")  # send the mp3 file
        time.sleep(0.02)


s = socket.socket()
s.bind(("172.28.179.82", 9008))  # bind the socket
s.listen(5)
sock, addr = s.accept()
print(sock, addr)  # accept the connection
tRec = threading.Thread(target=keepReceiveMsg(), name="Receive_Msg")
tRec.start()
tRec.join()
