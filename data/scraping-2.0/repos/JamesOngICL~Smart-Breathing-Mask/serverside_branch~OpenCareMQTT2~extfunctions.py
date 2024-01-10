import sqlite3
import openai
from datetime import date
import rsa

def getimage(username):
      "Return image in this function"
      return 0

def getbalance(username):
      "Return balance in this function"
      return 0

def updatebalance(username, balanceamount):
      "Update balance in this function"
      return 0

def updatepassword(username, newpassword):
      return 0

def getabout(username):
      data = sqlite3.connect("data.sqlite3")
      cmd = data.cursor()
      username = "'"+username+"'"
      aboutme = cmd.execute("SELECT about FROM aboutme WHERE username="+username)
      dat = aboutme.fetchall()
      if len(dat)<1:
            cmd.execute("""INSERT INTO aboutme VALUES ("""+username+""",'');""")
            data.commit()
            aboutme = ""
      else:
            aboutme = dat[0][0]
      data.close()
      return aboutme

def updateabout(username,aboutme):
      data = sqlite3.connect("data.sqlite3")
      cmd = data.cursor()
      cmd.execute("UPDATE aboutme SET about = ? WHERE username = ?", (aboutme,username))
      data.commit()
      data.close()
      
def updatedevices(username,device):
      data = sqlite3.connect("devicedata.sqlite3")
      cmd = data.cursor()
      device = "'"+device+"'"
      username = "'"+username+"'"
      devicefound = cmd.execute("SELECT * FROM devicelist WHERE device="+device)
      devicefound = devicefound.fetchall()
      
      if len(devicefound)==0:
            return 0
      
      cmd.execute("""UPDATE devicelist SET username="""+username+""" WHERE device="""+device+";")
      data.commit()
      data.close()
      
      return 1

def getdevices(username):
      data = sqlite3.connect("devicedata.sqlite3")
      cmd = data.cursor()
      username = "'"+username+"'"
      devices = cmd.execute("SELECT device, model FROM devicelist WHERE username="+username)
      devices = devices.fetchall()       
      data.close()
      return devices

def updatereading(device,column,value):
      data = sqlite3.connect("devicedata.sqlite3")
      cmd = data.cursor()
      device = "'"+device+"'"
      devicefound = cmd.execute("SELECT * FROM devicelist WHERE device="+device)
      devicefound = devicefound.fetchall()
      
      if len(devicefound)==0:
            return 0
      
      cmd.execute("""UPDATE devicelist SET """+column+"""="""+str(value)+""" WHERE device="""+device)
      data.commit()
      data.close()
      
      return 1

def getreading(device,column):
      data = sqlite3.connect("devicedata.sqlite3")
      cmd = data.cursor()
      device = "'"+device+"'"
      devices = cmd.execute("""SELECT """+column+""" FROM devicelist WHERE device="""+device)
      readings = devices.fetchall()       
      data.close()
      return readings[0][0]

def getmodel(device):
      data = sqlite3.connect("devicedata.sqlite3")
      cmd = data.cursor()
      device = "'"+device+"'"
      devices = cmd.execute("""SELECT model FROM devicelist WHERE device="""+device)
      readings = devices.fetchall()       
      data.close()
      return readings[0][0]

def getsensors(model):
      if model=="1.0 Temperature Scanner":
            return ["temperaturereading","accelerometerreading","co2reading","heartratereading"]
      return []

def aiquery(query):
      # Set up the OpenAI API client
      openai.api_key = "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"
      # Set up the model and prompt
      model_engine = "text-davinci-002"
      prompt = "as a "+query+" tell me how to manage health in a few sentences, don't use however"

      # Generate a responseop
      completion = openai.Completion.create(
      engine=model_engine,
      prompt=prompt,
      max_tokens=1024,
      n=1,
      stop=None,
      temperature=0.5,
      )

      response = completion.choices[0].text
      
      return response

def updatesteps(device, latestgyroreading, threshold):
      val = float(getreading(device,"accelerometerreading")) - 1
      latestgyroreading = latestgyroreading - 1 
      print("Step Test:")
      print(val)
      print(latestgyroreading)
      if abs(val-latestgyroreading) > threshold and latestgyroreading*val<0:
            print("here3")
            print(int(str(date.today()).split("-")[2]))
            print(getreading(device,"date"))
            if int(str(date.today()).split("-")[2])==int(getreading(device,"date")):
                  print("here1")
                  currsteps = int(getreading(device,"dailystep"))
                  updatereading(device,"dailystep",currsteps+2)
            else:
                  print("here2")
                  updatereading(device,"dailystep",1)
                  updatereading(device,"date",int(str(date.today()).split("-")[2]))
                  
def leaderboard():
      data = sqlite3.connect("devicedata.sqlite3")
      cmd = data.cursor()
      devices = cmd.execute("""SELECT username,dailystep FROM devicelist""")
      readings = devices.fetchall()       
      data.close()
      dict1 = {}
      for i in range(len(readings)):
           if readings[i][0] is not None and len(readings[i][0])>0:
                 if readings[i][0] in dict1:
                       dict1[readings[i][0]]=dict1[readings[i][0]]+int(readings[i][1])
                 else:
                       dict1[readings[i][0]]=int(readings[i][1])
      sort = sorted(dict1.items(), key=lambda x: x[1], reverse=True)
      return sort

def getkey():
    return rsa.PrivateKey(17822536325291103101323819498187309223540366914795284790480275832907387155167793075777561150522889142593001669196266128814411559496746832319118549807122213726549864914547111915612810151458421841885576703978191424830064979840625261982526983948534265579315189645712708057829444744526097651478257973249834256433525747544157305555064689911278584472322809386995238704397789909958860784202773540915843232935732240372781175094908655965143500697948600899964661078620371137797175431929993230441845012688184755850715508366128027618816784376158765529688868977572400965001046656964586523083679277864216337830024186581266532501199, 65537, 1087784691108296266312087492450817658638043664787541986388163988764050057534998127822607757481904215487007441243649610376697838442208024921440929539473714922962593033831094613156709043835294373675058467978588670511623356567473351662879105479258084171037135642199838751107279536416137305734213238658894920670509595724739825211540234780479702424414398186471656024186099861884175085047371105513648384538030490885890729127379882285217550587103176098749434266378378526433347873952844996681661834413405305039767132944647856314657002219432670729195755883730241683914716559918139261770167776492415327852374428080783720161, 2474717752835152437728903791088824332879396363132938677892215286195269595522320667261401951521011060447577192398462001984631742634595981682778104366909393602797943647046558188691384830506615024071582384272368845688243454727304965796834135441180778875672244320778133305015139663019818305028646710303913001002335988268829481979563, 7201846071081347218990614101738894604362580621663195494372444956272496170044168090994253755038837950789743124293994301231299052663864577390210697611477569488744717927980041064667405690476421998563660321238251259191061318787055350306212197105995018397161175564447501462422475029856383473773)

                  