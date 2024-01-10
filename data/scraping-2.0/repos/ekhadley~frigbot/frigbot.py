import datetime, random, math, json, requests, time, os, numpy as np
from googleapiclient.discovery import build
from Zenon.zenon import zenon
import openai
from utils import *

class Frig:
    def __init__(self, keydir, configDir, chatid):
        self.last_msg_id = 0 # unique message id. Used to check if a new message has appeared
        self.loop_delay = 0.3 # delay in seconds between checking for new mesages
        self.chatid = chatid
        self.keydir = keydir
        self.configDir = configDir
        self.read_saved_state(configDir)

        self.client = zenon.Client(self.keys["discord"])
        
        self.lol = lolManager(self.keys["riot"], self.configDir, "summonerIDs.json")
        self.yt = ytChannelTracker(self.keys["youtube"], "UCqq5t2vi_G753e19j6U-Ypg", self.configDir, "lastFemboy.json") # femboy fishing channelid

        self.gifsearch("arcane", 300)
        
        self.commands = {"!help":self.help_resp, # a dict of associations between commands (prefaced with a '!') and the functions they call to generate responses.
                         "!commands":self.help_resp,
                         "!cmds":self.help_resp,
                         "!gpt4":self.gpt_resp,
                         "!gpt":self.gpt_resp,
                         "!arcane":self.arcane_resp,
                         "!faptime":self.faptime_resp,
                         "!lastfap":self.lastfap_resp,
                         "!fapfail":self.fapfail_resp,
                         "!lostfap":self.fapfail_resp,
                         "!rps":self.rps_resp,
                         "!fish":self.yt.forceCheckAndReport,
                         "!gif":self.random_gif_resp,
                         "!lp":self.lp_resp}

        self.echo_resps = [ # the static repsonse messages for trigger words which I term "echo" responses
                "This computer is shared with others including parents. This is a parent speaking to you to now. Not sure what this group is up to. I have told my son that role playing d and d games are absolutely forbidden in out household. We do not mind him having online friendships with local people that he knows for legitimate purposes. Perhaps this is an innocent group. But, we expect transparency in our son's friendships and acquaintances. If you would like to identify yourself now and let me know what your purpose for this platform is this is fine. You are welcome to do so.",
                           
                ["Do not go gentle into that good juckyard.", "Tetus should burn and rave at close of day.", "Rage, rage against the dying of the gamings.", "Though wise men at their end know gaming is right,", "Becuase their plays had got no karma they", "Do not go gentle into that good juckyard"]
                           ]

        self.echoes = {"nefarious":self.echo_resps[0], # these are the trigger words and their associated echo
                       "avatars":self.echo_resps[0],
                       "poem":self.echo_resps[1],
                       "poetry":self.echo_resps[1],
                       "tetus":self.echo_resps[1],
                       "juckyard":self.echo_resps[1]
                       }



    def read_saved_state(self, dirname):
        self.user_IDs = loadjson(self.configDir, "userIDs.json")
        self.rps_scores = loadjson(self.configDir, "rpsScores.json")
        self.lastfap =  dateload(self.configDir, "lastfap.txt")
        self.keys = loadjson(self.keydir, "keys.json")
        openai.api_key = self.keys["openai"]
        self.botname = self.user_IDs["FriggBot2000"]

    def arcane_resp(self, msg):
        delta = datetime.datetime(2024,11,20, 21, 5, 0) - datetime.datetime.now()
        years, days, hours, minutes, seconds = delta.days//365, delta.days, delta.seconds//3600, (delta.seconds%3600)//60, delta.seconds%60
        if years < 1: return f"arcane s2 comes out in approximately {days} days, {hours} hours, {minutes} minutes, and {seconds} seconds. hang in there."
        return f"arcane s2 comes out in approximately 1 year, {days-365} days, {hours} hours, {minutes} minutes, and {seconds} seconds. hang in there."

    def arcane_reference_resp(self, query="arcane", num=500):
        phrases = ["holy shit was that an arcane reference", "literal chills", "my honest reaction to that information:", "me rn:", "this is just like arcane fr", ""]
        return [random.choice(phrases), self.randomgif(query, num)]

    def gpt_resp(self, msg):
        print(f"{bold}{gray}[GPT]: {endc}{yellow}text completion requested{endc}")
        try:
            prompt = msg['content'].replace("!gpt", "").strip()
            completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
            resp = completion.choices[0].message.content
            print(f"{bold}{gray}[GPT]: {endc}{green}text completion generated {endc}")
            return resp
        except Exception as e:
            print(f"{bold}{gray}[GPT]: {endc}{red}text completion failed with exception:\n{e}{endc}")
            return "https://tenor.com/view/bkrafty-bkraftyerror-bafty-error-gif-25963379"

    def help_resp(self, msg):
        resp = f"commands:"
        for c in self.commands: resp += f"\n{c}"
        return resp

    def rps_resp(self, msg):
        rollname = msg["content"].replace("!rps", "").strip()
        authorid = msg["author"]["id"]
        if authorid+"w" not in self.rps_scores:
            print(f"{bold}{gray}[RPS]: {endc}{yellow}new RPS player found {endc}")
            self.rps_scores[authorid+"d"] = 0
            self.rps_scores[authorid+"w"] = 0
            self.rps_scores[authorid+"l"] = 0
        if rollname == "": return f"Your score is {self.rps_scores[authorid+'w']}/{self.rps_scores[authorid+'d']}/{self.rps_scores[authorid+'l']}"

        opts = ["rock", "paper", "scissors"]
        if rollname not in opts: return f"{rollname} is not an option. please choose one of {opts}"
        
        roll = opts.index(rollname)
        if authorid == self.user_IDs["Xylotile"]: botroll = random.choice([(roll+1)%3]*6 + [(roll+2)%3, roll])
        else: botroll = random.randint(0, 2)
        if roll == botroll: report = f"We both chose {opts[botroll]}"; self.rps_scores[authorid+"d"] += 1
        if (roll+2)%3 == botroll: report = f"I chose {opts[botroll]}. W"; self.rps_scores[authorid+"w"] += 1
        if (roll+1)%3 == botroll: report = f"I chose {opts[botroll]}. shitter"; self.rps_scores[authorid+"l"] += 1
        self.write_rps_scores()
        
        update = f"Your score is now {self.rps_scores[authorid+'w']}/{self.rps_scores[authorid+'d']}/{self.rps_scores[authorid+'l']}"
        return [report, update]

    def lp_resp(self, msg):
        summ = msg["content"].replace("!lp", "").strip()
        return self.lol.ranked_info(summ)

    def send(self, msg): # sends a string or list of strings as a message/messages in the chat
        if isinstance(msg, list):
            for m in msg: self.send(m)
        elif isinstance(msg, str) and msg != "":
            self.client.send_message(self.chatid, msg)

    def get_last_msg(self) -> str: # reads the most recent message in the chat, returns a json
        try:
            msg = self.client.get_message(self.chatid)
            return msg
        except Exception as e:
            print(f"{bold}{gray}[FRIG]: {endc}{red}message read failed with exception:\n{e}{endc}")
            return None
    
    def get_self_msg(self): # determines what the bot needs to send at any given instance based on new messages and timed messages
        msg = self.get_last_msg()
        if msg["id"] != self.last_msg_id and msg["author"]["id"] != self.botname:
            try:
                author = self.user_IDs[msg["author"]["global_name"]]
            except KeyError:
                print(f"{bold}{gray}[FRIG]: {endc}{yellow}new username '{msg['author']['global_name']}' detected. storing their ID. {endc}")
                self.user_IDs[msg["author"]["global_name"]] = msg["author"]["id"]
                with open(f"{self.configDir}userIDs.json", "w") as f:
                    f.write(json.dumps(self.user_IDs, indent=4))
            
            self.last_msg_id = msg["id"]
            return self.get_response_to_new_msg(msg)
        return self.get_timed_messages()

    def get_timed_messages(self): # this function checks all of the messages which are not responses or echoes, but required to happen after a set delay or specific time.
        if self.yt.checkLatestUpload(): return self.yt.reportVid()
        return ""

    def get_response_to_new_msg(self, msg): # determines how to respond to a newly detected message. 
        body = msg["content"].lstrip()
        if body.startswith("!"):
            try:
                command = body.split(" ")[0]
                print(f"{bold}{gray}[FRIG]: {endc}{yellow} command found: {command}{endc}")
                #self.client.typing_action(self.chatid, msg)
                return self.commands[command](msg)
            except KeyError as e:
                print(f"{bold}{gray}[FRIG]: {endc}{red} detected command '{command}' but type was unrecognized{endc}")
            return f"command: '{command}' was not recognized"
        else:
            return self.echo_resp(body)
        return ""

    def echo_resp(self, body, arcane_reference_prob=.10): # determines which, if any, (non command) response to respond with. first checks phrases then other conditionals
        bsplit = body.split(" ")
        for e in self.echoes:
            if e in bsplit:
                print(f"{bold}{gray}[FRIG]: {endc}{gray} issuing echo for '{e}'{endc}")
                return self.echoes[e]
        key = "arcane"
        state = 0
        if np.random.uniform() < arcane_reference_prob:
            for c in body:
                if c.lower() == key[state]: state += 1
                if state == 6: return self.arcane_reference_resp() 
        return ""

    def runloop(self):
        print(bold, cyan, "\nFrigBot started!", endc)
        while 1:
            try:
                resp = self.get_self_msg()
                self.send(resp)
                self.wait()
            except Exception as e:
                print(f"{red}, {bold}, [FRIG] CRASHED WITH EXCEPTION:\n{e}")
                time.sleep(3)

    def write_rps_scores(self):
        with open(f"{self.configDir}rpsScores.json", "w") as f:
            f.write(json.dumps(self.rps_scores, indent=4))
    
    def gifsearch(self, query, num):
        url = f"https://g.tenor.com/v2/search?q={query}&key={self.keys['tenor']}&limit={num}"
        r = requests.get(url)
        gets = json.loads(r.content)["results"]
        urls = [g["url"] for g in gets]
        return urls
    def randomgif(self, query, num):
        return random.choice(self.gifsearch(query, num))
    def random_gif_resp(self, msg, num=100):
        query = msg['content'].replace("!gif", "").strip()
        return self.randomgif(query, num)

    def faptime(self):
        delta = datetime.datetime.now() - self.lastfap
        days, hours, minutes, seconds = delta.days, delta.seconds//3600, (delta.seconds%3600)//60, delta.seconds%60
        return days, hours, minutes, seconds
    def faptime_resp(self, msg):
        days, hours, minutes, seconds = self.faptime()
        return f"Xylotile has not nutted in {days} days, {hours} hours, {minutes} minutes, and {seconds} seconds. stay strong."
    def lastfap_resp(self, msg):
        return f"Xylotile's last nut was on {self.lastfap.strftime('%B %d %Y at %I:%M%p')}"
    def fapfail_resp(self, msg):
        authorid = msg["author"]["id"]
        try:
            if int(authorid) != int(self.user_IDs["Xylotile"]): return f"You are not authorized to make Xylotile nut."
            else:
                days, hours, minutes, seconds = self.faptime()
                self.set_last_fap()
                return ["https://tenor.com/view/ambatukam-ambasing-ambadeblow-gif-25400729", f"Xylotile has just lost their nofap streak of {days} days, {hours} hours, {minutes} minutes, and {seconds} seconds."]
        except KeyError:
            print(bold, red, f"Xylotile's userID could not be found, so the fapstreak update could not be verified. thats not good! spam @eekay")
    def set_last_fap(self):
        datesave(datetime.datetime.now(), f"{self.configDir}lastfap.txt")
        self.lastfap = self.load_lastfap()

    def wait(self):
        time.sleep(self.loop_delay)


class lolManager: # this handles requests to the riot api
    def __init__(self, riotkey, saveDir, filename):
        self.savePath = f"{saveDir}/{filename}"
        self.riotkey = riotkey
        self.summonerIDs = loadjson(saveDir, filename)

    def load_player_ids(self):
        with open(self.savePath, 'r') as f:
            return json.load(f)
    
    def get_summoner_id(self, summonerName, region=None):
        try:
            return self.summonerIDs[str(summonerName)]
        except KeyError:
            print(f"{gray}{bold}[LOL]:{endc} {yellow}requested summonerID for new name:' {summonerName}'{endc}")
            url = f"https://{region}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{summonerName}?api_key={self.riotkey}"
            get = requests.get(url)
            if get.status_code == 200:
                self.summonerIDs[str(summonerName)] = get.json()["id"]
                print(f"{gray}{bold}[LOL]:{endc} {yellow}stored summonerID for new username: '{summonerName}'{endc}")
                self.store_player_ids()
                return self.summonerIDs[str(summonerName)]
            else:
                print(f"{gray}{bold}[LOL]:{endc} {red}summonerID for new username: '{summonerName}' could not be located{endc}")
                return None
    def store_player_ids(self):
        with open(self.savePath, "w") as f:
            f.write(json.dumps(self.summonerIDs, indent=4))

    def ranked_info(self, summonerName, region=None):
        region = "na1" if region is None else region
        summonerID = self.get_summoner_id(summonerName, region)
        url = f"https://{region}.api.riotgames.com/lol/league/v4/entries/by-summoner/{summonerID}?api_key={self.riotkey}"
        get = requests.get(url)
        if get.status_code == 200:
            report = self.parse_ranked_info(get.json(), summonerName)
            print(f"{gray}{bold}[LOL]: {endc}{green}ranked info acquired for '{summonerName}'{endc}")
            return report
        elif get.status_code == 403:
            print(f"{gray}{bold}[LOL]: {endc}{red}got 403 for name '{summonerName}'. key is probably expired. request url:\n{url}{endc}")
            return f"got 403 for name '{summonerName}'. key is probably expired. blame riot"
        else:
            print(f"{gray}{bold}[LOL]: {endc}{red}attempted ID for '{summonerName}' got: {get}. request url:\n{url}'{endc}")
            return "https://tenor.com/view/snoop-dog-who-what-gif-14541222"
    
    def parse_ranked_info(self, info, name):
        if info == []:
            if "dragondude" in name.lower(): return "ap is still a bitch (not on the ranked grind)"
            return f"{name} is not on the ranked grind"
        try:
            info = info[0]
            name = info["summonerName"]
            lp = info["leaguePoints"]
            wins = int(info["wins"])
            losses = int(info["losses"])
            winrate = wins/(wins+losses)

            tier = info["tier"].lower().capitalize()
            div = info["rank"]
            rankrep = f"in {tier} {div} at {lp} lp"

            rep = f"{name} is {rankrep} with a {winrate:.3f} wr over {wins+losses} games"
            return rep
        except ValueError: print(info); return f"got ranked info:\n'{info}',\n but failed to parse. (spam @eekay)"

class ytChannelTracker:
    def __init__(self, ytkey, channelID, configDir, filename, checkInterval=10800):
        self.checkInterval = checkInterval
        self.savePath = f"{configDir}/{filename}" # where on disk do we keep most recent video ID (not rly a log, just the most recent)
        self.channelID = channelID # the channelid (not the visible one) of the channel we are monitoring
        self.mostRecentVidId, self.lastCheckTime = self.readSave() # read the most recent video ID and time of last api request
        self.yt = build('youtube', 'v3', developerKey=ytkey) # initialize our client

    def getLatestVidId(self): # uses ytv3 api to get the time and id of most recent video upload from channel
        request = self.yt.channels().list(part='contentDetails', id=self.channelID)
        response = request.execute()
        playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        playlist = self.yt.playlistItems().list(part='contentDetails', playlistId=playlist_id, maxResults=1)
        plresp = playlist.execute()
        videoId = plresp['items'][0]['contentDetails']['videoId'].strip()
        changed = self.mostRecentVidId != videoId
        self.mostRecentVidId = videoId
        return changed, videoId

    def checkLatestUpload(self): # limits rate of checks, returns wether a new vid has been found, updates saved state
        if self.shouldCheck():
            changed, newest = self.getLatestVidId()
            self.recordNewRead(videoId=newest)
            return changed
        return False

    def reportVid(self): return f"new femboy fishing:\nurl=https://youtube.com/watch?v={self.mostRecentVidId}"
    def forceCheckAndReport(self, *args): # forces check, and just gives link
        changed, newest = self.getLatestVidId()
        self.recordNewRead(videoId=newest)
        return f"url=https://youtube.com/watch?v={self.mostRecentVidId}"

    def readSave(self): # reads stored videoID of most recent 
        with open(self.savePath, 'r') as f:
            save = json.load(f)
            videoId, lastread = save["videoId"], self.str2dt(save["lastCheckTime"])
        return videoId, lastread
    def recordNewRead(self, videoId=None): # writes the current most recent upload to disk
        with open(self.savePath, "r") as f:
            saved = json.load(f)
        saved["lastCheckTime"] = self.now()
        if videoId is not None: saved["videoId"] = videoId
        with open(self.savePath, "w") as f:
            f.write(json.dumps(saved, indent=4))

    def timeSinceCheck(self): # returns the amount of time since last 
        delta = datetime.datetime.now() - self.lastCheckTime
        sec = delta.days*24*60*60 + delta.seconds
        #print(f"{sec} sec since last check. check interval is {self.checkInterval} sec. checking in {self.checkInterval - sec}")
        return delta.days*24*60*60 + delta.seconds

    def shouldCheck(self): return self.timeSinceCheck() >= self.checkInterval

    def now(self): return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    def str2dt(self, dstr): return datetime.datetime.strptime(dstr, "%Y-%m-%dT%H:%M:%SZ")
