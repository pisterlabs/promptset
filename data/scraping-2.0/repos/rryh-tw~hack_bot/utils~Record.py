from pydub.playback import play
from pydub.silence  import split_on_silence, detect_nonsilent
from datetime       import datetime
import dotenv
from pydub          import AudioSegment


import speech_recognition as sr
import discord
import openai
import numpy              as np
import glob
import os


INFINITY = 2_147_483_647
openai.api_key = str(os.getenv('OPENAI_TOKEN'))

def prompt_openai(word):
    try:
        completion = openai.Completion.create(engine="text-davinci-003",temperature= 0.5, prompt=word,max_tokens=1024)
        return completion.choices[0].text
    except openai.error.InvalidRequestError:
        print("[*] Prompt to many words ..., Cutting down")
        return prompt_openai("\n".join(word.split('\n')[:-1]))



# 用時間排序對話 來製作給 openai 的文字
def create_prompt(all_result, all_time, recorded_users):
    all_index  = [0 for _ in recorded_users]
    start_time = [INFINITY for _ in recorded_users]

    stop_index = [len(all_result[i]) for i in range(len(recorded_users)) ]

    prompt_text = "Q:以下是使用者們使用discord開會後的錄音經由語音辨識後的結果，請優化後並總結它\n"

    while all_index!=stop_index:
        
        for i in range(len(recorded_users)):
            if ( start_time[i] == INFINITY ):
                if (all_index[i]<stop_index[i]):
                    start_time[i] = all_time[i][all_index[i]][0] 
        # print(all_index, stop_index)
        add_index = np.argmin(start_time)
        prompt_text = prompt_text + recorded_users[add_index] + " : " + all_result[ add_index ][ all_index[add_index] ] + "\n"

        all_index[add_index] += 1
        start_time[add_index] = INFINITY
    return prompt_text + "A:"




class StopRecordSave():
    def __init__(self,savefolder,name=None, client=None):
        self.client = client
        self.savefolder  = savefolder
        self.start_time = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        self.name = name
        os.makedirs(savefolder,exist_ok=True)

        

    async def once_done(self, sink: discord.sinks, channel: discord.TextChannel, *args):
        recorded_users = [  # A list of recorded users
            (await self.client.fetch_user(user_id)).name
            for user_id, audio in sink.audio_data.items()
        ]
        # for user_id, audio in sink.audio_data.items():
    
        # await sink.vc.disconnect()


        end_time = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        day_folder =  os.path.join(self.savefolder, f'{self.start_time}~{end_time}')
        os.makedirs(day_folder,exist_ok=True)

        if (self.name):
            with open(day_folder+"/name.txt",'w') as f:
                f.write(self.name)



        all_result = []  # save recognize words
        all_time   = []  # save recognize words time
        for user_id, audio in sink.audio_data.items():

            this_file = os.path.join(day_folder,f'{user_id}.wav')

            audio = AudioSegment.from_raw(audio.file, format="wav", sample_width=2,frame_rate=48000,channels=2)
            audio.export(this_file, format='wav')

            result, timeline = speech_to_text(this_file)
            print(user_id,":",result)

            all_result.append(result)
            all_time.append(timeline)

        openai_prompt     = create_prompt(all_result, all_time, recorded_users)
        conclusion_result = prompt_openai(openai_prompt)
        this_file = os.path.join(day_folder,f'conclusion.txt')
        print(conclusion_result)
        with open(this_file,'w') as f:
            f.write(conclusion_result)

        # await channel.send(":speaking_head: :speech_balloon: The record is summarised, you can use /check_record command to check it." )
        await channel.send(conclusion_result)
            # speech_to_tex(a)
        # discordfiles = [discord.File(audio.file, f"{user_id}.{sink.encoding}") for user_id, audio in sink.audio_data.items()]  # List down the files.
        # await channel.send(f"Finished recording audio for: \n{', '.join(recorded_users)}", files=discordfiles) 




# 停止按鈕

class StopRecordButton(discord.ui.View):
    def __init__(self, voice_channel, text_channel, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.voice_channel = voice_channel
        self.text_channel  = text_channel


    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        await self.message.edit(content="You took too long! Disabled all the components.", view=self)

    @discord.ui.button(label="停止錄音", style=discord.ButtonStyle.primary)
    async def first_button_callback(self, button, interaction):

        await interaction.response.send_message('====== Stop recording ======')
        self.voice_channel.stop_recording()
        # await ctx.delete()
        # await self.text_channel 




def speech_to_text(path):
    def match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    r = sr.Recognizer() 
    sound = AudioSegment.from_wav(path, format="wav")
    chunks = split_on_silence(sound,
        min_silence_len = 300,
        silence_thresh = sound.dBFS-20,
        keep_silence=100,
    )

    # normalized_sound = match_target_amplitude(sound, -20.0)
    nonsilent_data = detect_nonsilent(sound, min_silence_len=500, silence_thresh=-50, seek_step=50)


    folder_name = "audio-chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    
    whole_text = []
    time_text  = []
    
    # for i, audio_chunk in enumerate(chunks, start=1):
    for i, chunk in enumerate(nonsilent_data, start=1):
        this_time = [chunk_ for chunk_ in chunk]
        time_text.append(this_time)
        audio_chunk  = sound[this_time[0]:this_time[1]]

        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")

        with sr.AudioFile(chunk_filename) as source:
            try:
                audio_listened = r.record(source)
                try:
                    text = r.recognize_google(audio_listened, language = 'zh-tw', show_all=True)
                    if text['alternative'][0]['confidence'] < 0.7:
                        text['alternative'][0]['transcript'] = "*inaudible*"
                    text = text['alternative'][0]['transcript']
                except sr.UnknownValueError as e:
                    text = "*inaudible*"
                else:                
                    whole_text.append(text)
            except:
                whole_text.append('')
                continue
    return whole_text, time_text

class CheckRecordMenu():
    def __init__(self, time_arr, folder_arr, isfile=True, client=None, *args, **kwargs):
        self.client = client
        self.label_arr, discrip_arr, emoji_arr =  self.get_record_time(time_arr, folder_arr)

        options = [ discord.SelectOption(label=self.label_arr[i],description=discrip_arr[i],emoji = emoji_arr[i])for i in range(len(time_arr))]
        
        # self.embed = discord.Embed(title="Help panel!", description="Your Desc")

        self.select = discord.ui.Select(
            placeholder = "All recordings you can access",
            min_values  = 1, 
            max_values  = 1,
            options = options
            )

        if (not isfile):
            self.select.callback = self.callback
        else:
            self.select.callback = self.callback2
        self.view = discord.ui.View()
        self.view.add_item(self.select)

        self.folder_arr = folder_arr

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        await self.message.edit(content="You took too long! Disabled all the components.", view=self)


    def get_record_time(self,time_arr, folder_arr):
        self.start_arr, self.end_arr = [], []
        label_arr, discrip_arr, emoji_arr = [], [], []
        public_index, private_index = 1,1

        for num, folder_name in enumerate(time_arr,start=1):
            name_file = os.path.join(folder_arr[num-1],'name.txt')

            if (not os.path.isfile(name_file)):
                if('public' in folder_arr[num-1]):
                    label_arr.append(  f"Public_record_{public_index}")
                    public_index += 1
                else:
                    label_arr.append(  f"Private_record_{private_index}")
                    private_index += 1
            else:
                label_arr.append( open(name_file,'r').read())

            if('public' in folder_arr[num-1]):
                emoji_arr.append("🟢")
            else:
                emoji_arr.append("🔴")


            start, end = folder_name.split("~")
            start_time = datetime.strptime(start,'%y-%m-%d-%H-%M-%S')
            end_time   = datetime.strptime(end  ,'%y-%m-%d-%H-%M-%S')

            self.start_arr.append(start_time.strftime('%Y-%m-%d  %H:%M:%S'))
            self.end_arr.append(end_time.strftime('%Y-%m-%d  %H:%M:%S'))

            total_time = (end_time-start_time).total_seconds()
            
            discrip_arr.append(f"Start from {start_time.strftime('%Y-%m-%d  %H:%M:%S')} ({total_time}s)")
        return label_arr, discrip_arr, emoji_arr

    async def callback2(self, interaction):
        which_chosen = self.label_arr.index(self.select.values[0])
        # print(which_chosen)
        all_wav_files  = glob.glob(self.folder_arr[which_chosen]+"/*.wav")
        recorded_users = []
        user_id_arr    = []
        for all_wav in all_wav_files:
            this_file = os.path.basename(all_wav)[:-4]
            recorded_users.append((await self.client.fetch_user(this_file)).name)
            user_id_arr.append(this_file)
            await interaction.channel.send(f'<@{this_file}>', file=discord.File(all_wav))

            
        if (len(user_id_arr)==1):
            await interaction.response.send_message(f"This recording is from {self.start_arr[which_chosen]} to {self.end_arr[which_chosen]}, only {len(user_id_arr)} people participated")

        else:
            await interaction.response.send_message(f"This recording is from {self.start_arr[which_chosen]} to {self.end_arr[which_chosen]}, a total of {len(user_id_arr)} people participated")

    async def callback(self, interaction):
        which_chosen = self.label_arr.index(self.select.values[0])
        # print(which_chosen)
        all_wav_files  = glob.glob(self.folder_arr[which_chosen]+"/*.wav")
        recorded_users = []
        user_id_arr    = []
        for all_wav in all_wav_files:
            this_file = os.path.basename(all_wav)[:-4]
            recorded_users.append((await self.client.fetch_user(this_file)).name)
            user_id_arr.append(this_file)
            # await interaction.channel.send(f'<@{this_file}>', file=discord.File(all_wav))


        conclusion_result_file = os.path.join(self.folder_arr[which_chosen], "conclusion.txt")
        if (not os.path.isfile(conclusion_result_file)):

            await interaction.channel.send("Sorry, we lost the conlusion, let me regenerate it!! it may take some time.")


            all_result = []  # save recognize words
            all_time   = []  # save recognize words time
            for user_id, this_file in zip(user_id_arr,all_wav_files):  
                result, timeline = speech_to_text(this_file)
                print(user_id,":",result)

                all_result.append(result)
                all_time.append(timeline)

            await interaction.response.defer()

            openai_prompt     = create_prompt(all_result, all_time, recorded_users)
            conclusion_result = prompt_openai(openai_prompt)
            this_file = os.path.join(self.folder_arr[which_chosen],f'conclusion.txt')
            print(conclusion_result)
            with open(this_file,'w') as f:
                f.write(conclusion_result)

            await interaction.followup.send(conclusion_result, ephemeral=True)
        else:

            with open(conclusion_result_file,'r') as f:
                await interaction.response.send_message(f.read(), ephemeral=True)



if __name__ == "__main__":
    # result = speech_to_text(r"D:\Carrer_hack\github_practice\recorded\1071431018701144165\23-02-09-00\518082603090182144.wav")
    # result = speech_to_text(r'output.wav')
    pass
    # print(result)


    # s1 = speech_to_text(r"D:\Carrer_hack\github_practice\recorded\1071431018701144165\23-02-09-17\471205246249467934.wav")
    # s2 = speech_to_text(r"D:\Carrer_hack\github_practice\recorded\1071431018701144165\23-02-09-17\518082603090182144.wav")

    # recorded_users = ["test1","test2"]

    # a1 = [s1[0] , s2[0]]
    # b1 = [s1[1] , s2[1]]
    # print(a1, b1)
    # # print([len(a1[i]) for i in range(len(recorded_users)) ])
    # prompt = create_prompt(a1,b1,recorded_users)
    # print(prompt)

    # print(prompt_openai(prompt))
