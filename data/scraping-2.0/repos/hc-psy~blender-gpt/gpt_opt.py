import bpy
from bpy.types import Operator

import openai

from .gpt_gpt import chatgpt
from .gpt_cst import UI


class BLENDERGPT_OT_DEL_MSG(Operator):
    bl_idname = "gpt.del_msg"
    bl_label = "delete message"
    bl_description = "delete message"
    bl_options = {"REGISTER", "UNDO"}

    msg_idx: bpy.props.IntProperty(name="msg_idx", default=0)

    def execute(self, context):
        scene = context.scene
        history = scene.history
        history.remove(self.msg_idx)
        return {"FINISHED"}


class BLENDERGPT_OT_DEL_ALL_MSG(Operator):
    bl_idname = "gpt.del_all_msg"
    bl_label = "delete all messages"
    bl_description = "delete all messages"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        history = scene.history
        history.clear()
        return {"FINISHED"}


class BLENDERGPT_OT_GPT_CODE(Operator):
    bl_idname = "gpt.gpt_code"
    bl_label = "show GPT code"
    bl_description = "show GPT code"
    bl_options = {"REGISTER", "UNDO"}

    code: bpy.props.StringProperty(
        name="GPT Code", description="GPT Code", default="")

    def execute(self, context):

        txt_name = 'prompt_script.py'
        txt = bpy.data.texts.get(txt_name)

        if txt is None:
            txt = bpy.data.texts.new(txt_name)

        txt.clear()
        txt.write(self.code)

        txt_edit_area = None
        for area in bpy.context.screen.areas:
            if area.type == 'TEXT_EDITOR':
                txt_edit_area = area
                break

        if txt_edit_area is None:
            cxt_area = context.area
            for region in cxt_area.regions:
                if region.type == 'WINDOW':
                    bpy.ops.screen.area_split(
                        {'area': cxt_area, 'region': region}, direction='VERTICAL', factor=0.5)
                    break

            new_area = context.screen.areas[-1]
            new_area.type = 'TEXT_EDITOR'
            txt_edit_area = new_area

        txt_edit_area.spaces.active.text = txt

        return {"FINISHED"}


class BLENDERGPT_OT_SEND_MSG(Operator):
    bl_idname = "gpt.send_msg"
    bl_label = "send message"
    bl_description = "send message"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene

        # openai api key
        prf = context.preferences
        prf_openai_api_key = prf.addons["blender-gpt"].preferences.openai_key
        lan = prf.addons["blender-gpt"].preferences.language

        if not prf_openai_api_key:
            self.report({'ERROR'}, UI['error_no_api_key'][lan])
            return {'CANCELLED'}

        scene.on_finish = True
        # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

        if len(scene.history) == 0 or scene.history[-1].type == 'GPT':
            if scene.prompt_input == "":
                self.report({'ERROR'}, UI['error_no_prompt'][lan])
                scene.on_finish = False
                return {'CANCELLED'}

        try:
            code_exe_blender = chatgpt(context, api_key=prf_openai_api_key)
        except Exception as e:
            self.report({'ERROR'}, f"Error: {e}")
            scene.on_finish = False
            return {'CANCELLED'}

        if len(scene.history) == 0 or scene.history[-1].type == 'GPT':
            msg = scene.history.add()
            msg.type = 'USER'
            msg.content = scene.prompt_input
            scene.prompt_input = ""

        if code_exe_blender:
            msg = scene.history.add()
            msg.type = 'GPT'
            msg.content = code_exe_blender

            global_namespace = globals().copy()

        try:
            exec(code_exe_blender, global_namespace)
        except Exception as e:
            self.report({'ERROR'}, f"Error: {e}")
            scene.on_finish = False
            return {'CANCELLED'}

        scene.on_finish = False
        return {"FINISHED"}
