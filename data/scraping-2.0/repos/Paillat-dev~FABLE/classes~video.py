import os
import json
import yaml

from utils.openaicaller import openai
from utils.normalize_file import normalize_file as nf
from utils.config import bcolors
from utils.misc import clear_screen, open_explorer_here, realbcolors, printm
from utils.uploader import upload_video

from generators.script import generate_script
from generators.montage import mount, prepare
from generators.thumbnail import generate_thumbnail

class Video:
    def __init__(self, idea, parent):
        self.parent = parent # The parent class, which is a Channel class
        self.id = None
        self.url = None
        self.script = None
        self.path = None
        self.idea = idea
        self.title = self.idea['title']
        self.description = self.idea['description']
        self.metadata = None
    
    async def generate(self):
        normalized_title = await nf(self.idea['title'])
        self.path = os.path.join(self.parent.path, "videos", normalized_title)
        if not os.path.exists( self.path):
            os.makedirs( self.path)
        script = None
        if os.path.exists(os.path.join( self.path, "script.json")):
            if input("Video script already exists. Do you want to overwrite it ? (y/N) : ").lower() == "y":
                os.remove(os.path.join( self.path, "script.json"))
    
        if not os.path.exists(os.path.join( self.path, "script.json")):
            script_prompt = None
            if os.path.exists(os.path.join(self.parent.path, "script_prompt.txt")):
                with open(os.path.join(self.parent.path, "script_prompt.txt"), "r") as f:
                    script_prompt = f.read()
                    f.close()
            if script_prompt:
                printm("Using custom script prompt")
                script = await generate_script(self.idea['title'], self.idea['description'], script_prompt)
            else:
                printm("Using default script prompt")
                script = await generate_script(self.idea['title'], self.idea['description'])
            script = json.loads(script)
            with open(os.path.join( self.path, "script.json"), "w") as f:
                json.dump(script, f, indent=4)
                f.close()
        else:
            with open(os.path.join(self.path, "script.json"), "r") as f:
                script = json.load(f)
                f.close()
        await prepare( self.path)
        credits = await mount(self.path, script)
        self.metadata = {
            "title": self.idea['title'],
            "description": self.idea['description'] + "\n\n" + credits,
        }
        #if input("Do you want to generate a thumbnail ? (y/N) : ").lower() == "y":
        await generate_thumbnail( self.path, self.idea['title'], self.idea['description'])
        videoid = await upload_video( self.path, self.idea['title'], self.metadata['description'], 28, "", "private", self.parent.path)
        printm(f"Your video is ready! You can find it in { self.path}")
        video_meta_file = {
            "title": self.idea['title'],
            "description": self.metadata['description'],
            "id": videoid,
            "path":  self.path,
            "url": f"https://www.youtube.com/watch?v={videoid}",
        }
        self.url = video_meta_file['url']
        self.id = videoid
        with open(os.path.join( self.path, "video.yaml"), "w") as f:
            yaml.dump(video_meta_file, f)
            f.close()
        return self