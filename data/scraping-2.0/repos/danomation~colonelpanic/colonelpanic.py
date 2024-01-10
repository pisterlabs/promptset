import os
#the up-to-date openai module as of 12-22-23
from openai import OpenAI
from random import randrange
#discordpy
import discord
import time

client = OpenAI(
    api_key = "youropenaiapikey"
)

history = []

sourceip = "192.168." + str(randrange(0, 255)) + "." + str(randrange(0, 255))
targetip = "192.168." + str(randrange(0, 255)) + "." + str(randrange(0, 255))
fakeip = "192.168." + str(randrange(0, 255)) + "." + str(randrange(0, 255))
fakeip1 = "192.168." + str(randrange(0, 255)) + "." + str(randrange(0, 255))
fakeip2 = "192.168." + str(randrange(0, 255)) + "." + str(randrange(0, 255))
fakeip3 = "192.168." + str(randrange(0, 255)) + "." + str(randrange(0, 255))
fakeip4 = "192.168." + str(randrange(0, 255)) + "." + str(randrange(0, 255))
print("SourceIP is " + sourceip)
print("TargetIP is " + targetip)
print("fake vms = " + fakeip + ", " + fakeip1 + ", " + fakeip2 + ", " + fakeip3 + ", " + fakeip4)


def sendgpt(message, author):
    messages = [
            {
                "role" : "system", "content" : "No commentary of any kind. You are a faux terminal on Ubuntu 22.04 hostname colonelpanic with ip " + sourceip + " as part of a hacking game. The user is logged in as root. If the user discovers the webserver on " + targetip + " he wins the game. The ONLY other ips for the /16 are " + fakeip + ", " + fakeip1 + ", " + fakeip2 + ", " + fakeip3 + ", " + fakeip4 + ". nmap always shows all of the 7 ips (self, target, and 5 fakes). Encase the output in triple tick marks.\\n."
            },
        ]
    messages += history[-20:]
    messages.append({"role" : "user", "content" : message })
    history.append({"role" : "user", "content" : message })
    response = client.chat.completions.create(
        user = author,
        model = "gpt-4",
        #model = "gpt-4-1106-preview",
        messages = messages,
        max_tokens=2048,
    )

    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": response.choices[0].message.content})
    return(reply)


def connect_discord_server():
    client = discord.Client(intents=discord.Intents.all())
    @client.event
    async def on_ready():
        print('colonelpanic session started'.format(client))
        # connect message
        await client.get_channel(yourchannelid).send("colonelpanic v0.1 connected \n ```SSH connection to " + sourceip + " was successful. \nUsing username \"root\".\nAuthenticating with public key \"key unidentified\"\nWelcome to Ubuntu 22.04.3 LTS (GNU/Linux 5.15.0-91-generic x86_64)\n\n * Documentation:  https://help.ubuntu.com\n * Management:     https://landscape.canonical.com\n * Support:        https://ubuntu.com/advantage\n\n  System information as of now\n\n  System load:              0.33349609375\n  Usage of /:               45.2% of 74.79GB\n  Memory usage:             31%\n  Swap usage:               0%\n  Processes:                165\n  Users logged in:          1\n  IPv4 address for eth0:    " + sourceip + "\n\n * Strictly confined Kubernetes makes edge and IoT secure. Learn how MicroK8s\n   just raised the bar for easy, resilient and secure K8s cluster deployment.\n\n   https://ubuntu.com/engage/secure-kubernetes-at-the-edge\n\nExpanded Security Maintenance for Applications is not enabled.\n\n12 updates can be applied immediately.\nTo see these additional updates run: apt list --upgradable\n\n16 additional security updates can be applied with ESM Apps.\nLearn more about enabling ESM Apps service at https://ubuntu.com/esm\n```")
    @client.event
    async def on_message(message):
        if message.channel.id == yourchannelid:
            if "!newgame" in str(message.content):
                global history
                global sourceip
                global targetip
                global fakeip
                global fakeip1
                global fakeip2
                global fakeip3
                global fakeip4
                history = []
                sourceip = "192.168." + str(randrange(0, 255)) + "." + str(randrange(0, 255))
                targetip = "192.168." + str(randrange(0, 255)) + "." + str(randrange(0, 255))
                fakeip = "192.168." + str(randrange(0, 255)) + "." + str(randrange(0, 255))
                fakeip1 = "192.168." + str(randrange(0, 255)) + "." + str(randrange(0, 255))
                fakeip2 = "192.168." + str(randrange(0, 255)) + "." + str(randrange(0, 255))
                fakeip3 = "192.168." + str(randrange(0, 255)) + "." + str(randrange(0, 255))
                fakeip4 = "192.168." + str(randrange(0, 255)) + "." + str(randrange(0, 255))
                await client.get_channel(yourchannelid).send("```SSH connection to " + sourceip + " successful. \nUsing username \"root\".\nAuthenticating with public key \"key unidentified\"\nWelcome to Ubuntu 22.04.3 LTS (GNU/Linux 5.15.0-91-generic x86_64)\n\n * Documentation:  https://help.ubuntu.com\n * Management:     https://landscape.canonical.com\n * Support:        https://ubuntu.com/advantage\n\n  System information as of now\n\n  System load:              0.33349609375\n  Usage of /:               45.2% of 74.79GB\n  Memory usage:             31%\n  Swap usage:               0%\n  Processes:                165\n  Users logged in:          1\n  IPv4 address for eth0:    " + sourceip + "\n\n * Strictly confined Kubernetes makes edge and IoT secure. Learn how MicroK8s\n   just raised the bar for easy, resilient and secure K8s cluster deployment.\n\n   https://ubuntu.com/engage/secure-kubernetes-at-the-edge\n\nExpanded Security Maintenance for Applications is not enabled.\n\n12 updates can be applied immediately.\nTo see these additional updates run: apt list --upgradable\n\n16 additional security updates can be applied with ESM Apps.\nLearn more about enabling ESM Apps service at https://ubuntu.com/esm\n```")
            else:
                themessage = "msg " + str(message.author) + ": " + message.content + ""
                print(themessage)
                if "colonelpanic" not in str(message.author.name):
                    getgpt = sendgpt(str(message.content), str(message.author))
                    maxlen = 2000
                    chunks = [getgpt[i:i+maxlen] for i in range(0, len(getgpt), maxlen)]
                    for chunk in chunks:
                        await message.reply(chunk)

    client.run('yourdiscordkey')

connect_discord_server()
