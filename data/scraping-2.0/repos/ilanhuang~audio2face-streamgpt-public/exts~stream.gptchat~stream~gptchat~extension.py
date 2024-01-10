#Stream-GPT
#GNU - GLP Licence
#Copyright (C) <year>  <Huang I Lan & Erks - Virtual Studio>
#This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.

import omni.ext
import sys
sys.path.append("C:\\Users\\ERKS 2\\Documents\\Omniverse\\ov\\pkg\\audio2face-2022.2.1\\exts\\omni.audio2face.player\omni\\audio2face\\player\\scripts\\streaming_server")
import openai
import carb
from .window import AudioChatWindow

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class MyExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        openai.api_key = AudioChatWindow.get_openai_api_key()
        self._window = AudioChatWindow("VIRTUAL ASSISTANT", width=400, height=525)
        
    def on_shutdown(self):
        self._window.destroy()
        self._window = None
