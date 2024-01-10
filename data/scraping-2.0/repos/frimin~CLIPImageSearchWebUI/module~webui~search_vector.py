from PIL import Image
import gradio as gr
import os
import numpy as np
import time
import torch
import json
import hashlib
from pathlib import Path

from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import FakeEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore
from module.search_vector_core import (
    clip_classifier, 
    clustering, 
    aesthetic_predictor, 
    deduplication,
    search_image_load_vecdb,
)
from module.webui.components.search import create_image_delete
from module.search_vector_core.search_state import SearchVectorPageState, open_and_try_resize_image
import subprocess
import uuid
from tqdm import tqdm
import module.webui.components.search_image_save as search_image_save
import module.utils.constants_util as constants_util
import numpy as np
import module.webui.components.create_embedding as create_embedding
from module.webui.components import on_load_search_target, set_search_target
from module.data import (
    get_clip_model, 
    get_vector_db_mgr, 
    get_webui_configs,
    get_cache_root,
    CLIPWarpper
)

local_state: SearchVectorPageState = SearchVectorPageState()

def get_compolent() -> SearchVectorPageState:
    return local_state

def on_load_page(search_target: dict, select_search_target: dict):
    cfg = get_webui_configs().get_cfg()

    search_target, select_search_target = on_load_search_target(search_target, select_search_target)

    return (
        search_target,
        select_search_target,
        cfg.search.default_top_k,
        )

def on_search_with_image(search_target: dict, 
                         select_search_target: list[str] | None, 
                         search_history: dict,
                         select_search_history: list[str] | None,
                         image: Image, 
                         top_k_number, 
                         page_size: float, 
                         progress = gr.Progress(track_tqdm=True)):
    vector_mgr = get_vector_db_mgr()
    if not image:
        raise gr.Error("未指定图片")
    if vector_mgr.is_empty():
        raise gr.Error("未加载任何库")
    cfg = get_webui_configs().get_cfg()
    max_top_k = cfg.search.max_top_k
    max_page_size = cfg.search.max_page_size
    if top_k_number > max_top_k:
        raise gr.Error(f"搜索参数的 Top K 不能超过 {max_top_k}")
    if page_size > max_page_size:
        raise gr.Error(f"搜索参数的分页大小不能超过 {max_page_size}")
    if select_search_target is None or len(select_search_target) == 0:
        raise gr.Error("没有选择查询目标")

    clip_model = get_clip_model()

    with clip_model.get_model() as m:
        clip_inputs = m.processor(images=image, return_tensors="pt", padding=True)
        clip_inputs["pixel_values"] = clip_inputs["pixel_values"].to(clip_model.device)
        image_features = m.model.get_image_features(**clip_inputs)
    embedding = image_features[0]
    embedding /= embedding.norm(dim=-1, keepdim=True) 

    msg = f"查询完毕"

    search_name = f"#{{n}} 对 {{target}} 图片查询"

    preview_image_with_label, preview_page_state, preview_search_name, search_history = local_state.search_with_save_page(
        embedding=embedding.tolist(),
        top_k=top_k_number,
        search_history=search_history,
        select_search_target=select_search_target,
        page_size=page_size,
        search_name=search_name
        )

    return local_state.update_viewer(
        page_state=preview_page_state,
        image_and_label=preview_image_with_label,
        search_target=search_target,
        search_history=search_history,
        select_search_name=preview_search_name,
        msg=msg,
        progress=progress,
    )

def on_search_with_prompt(search_target: dict, 
                          select_search_target: list[str] | None, 
                          search_history: dict,
                          select_search_history: list[str] | None,
                          prompt: str, 
                          top_k_number:float, 
                          page_size: float, 
                          progress = gr.Progress(track_tqdm=True)):
    vector_mgr = get_vector_db_mgr()

    if not prompt:
        raise gr.Error("未指定提示词")
    if vector_mgr.is_empty():
        raise gr.Error("未加载任何库")
    cfg = get_webui_configs().get_cfg()
    max_top_k = cfg.search.max_top_k
    max_page_size = cfg.search.max_page_size
    if top_k_number > max_top_k:
        raise gr.Error(f"搜索参数的 Top K 不能超过 {max_top_k}")
    if page_size > max_page_size:
        raise gr.Error(f"搜索参数的分页大小不能超过 {max_page_size}")
    if select_search_target is None or len(select_search_target) == 0:
        raise gr.Error("没有选择查询目标")

    print(f"search by prompt: {prompt}")

    clip_model = get_clip_model()
    with clip_model.get_model() as m:
        clip_inputs = m.processor(text=prompt, return_tensors="pt", padding=True)
        clip_inputs["input_ids"] = clip_inputs["input_ids"].to(clip_model.device)
        clip_inputs["attention_mask"] = clip_inputs["attention_mask"].to(clip_model.device)
        embedding = m.model.get_text_features(**clip_inputs)[0]
        embedding /= embedding.norm(dim=-1, keepdim=True) 

    search_name = f"#{{n}} 对 {{target}} 文本查询: {prompt}"

    preview_image_with_label, preview_page_state, preview_search_name, search_history = local_state.search_with_save_page(
        embedding=embedding.tolist(),
        top_k=top_k_number,
        search_history=search_history,
        select_search_target=select_search_target,
        page_size=page_size,
        search_name=search_name
        )

    msg = f"查询完毕"

    return local_state.update_viewer(
        page_state=preview_page_state,
        image_and_label=preview_image_with_label,
        search_target=search_target,
        search_history=search_history,
        select_search_name=preview_search_name,
        msg=msg,
        progress=progress,
    )

def page(block: gr.Blocks, args, top_elems):
    local_state.init()
    with gr.Blocks() as pageBlock:
        local_state.msg_text = msg_text = top_elems.msg_text
        local_state.image_file_with_lable_list = gr.State()
        local_state.image_select = gr.State()
        local_state.search_target = gr.State()

        with gr.Tab(label="提示查询"):
            with gr.Row():
                gr.Markdown("提供图片提示 (image prompt) 或 文本提示 (text prompt)，从一个或多个指定的查询目标中进行嵌入向量相似性搜索 (embedding similarity search)，一个或多个结果将被添加到历史查询中。")
            with gr.Row():
                #search_file = gr.File(label="上传Embedding或图片路径文件", file_types=[".json", ".txt"])
                #local_state.search_file = search_file
                search_image = gr.Image(label="图片提示", type="pil")
                search_text = gr.TextArea(label="文本提示", value="person, happy", info="查询提示文本，仅限于英文")
            
            with gr.Row():
                #gr.Button(value="仅浏览", variant="primary")
                top_k_number = gr.Number(label="Top K", value=0, info="查询结果数量")
                search_with_image_btn = gr.Button(value="图搜图", variant="primary")
                search_with_prompt_btn = gr.Button(value="文搜图", variant="primary")

        clip_classifier_compolents = clip_classifier.on_gui()
        repeat_query_compolents = clustering.on_gui()
        aesthetic_predictor_compolents = aesthetic_predictor.on_gui()
        deduplication_compolents = deduplication.on_gui()
        with gr.Row():
            local_state.select_search_target =  gr.Dropdown(multiselect=True, 
                label="查询目标", info="可以指定一个或者多个数据集目标创建查询",
                interactive=True,
            )
            local_state.page_size = page_size = gr.Number(label="分页大小", value=20, info="查询结果的每页大小")

        with gr.Row():
            local_state.search_history = gr.State()
            local_state.select_search_history = select_search_history = gr.Dropdown(label="查询历史", info="每次执行查询操作新的查询都会追加到最前")

        with gr.Row():
            local_state.image_gallery = gallery = gr.Gallery(label="查询浏览", columns=8, object_fit="contain", scale=5)

            with gr.Column():
                local_state.page_state = gr.State()

                with gr.Group():
                    local_state.first_page_btn = first_page_btn = gr.Button("首页")
                    local_state.last_page_btn = last_page_btn = gr.Button("尾页")
                    local_state.prev_page_btn = prev_page_btn = gr.Button("上一页")
                    local_state.next_page_btn = next_page_btn = gr.Button("下一页")
                with gr.Row():
                    local_state.page_index = gr.Number(label="当前页", value=1, interactive=True, min_width=60)
                    local_state.page_count = gr.Number(label="总页数", value=1, interactive=False, min_width=60)
                with gr.Group():
                    goto_page_btn = gr.Button("跳转到")
                    clear_search_btn = gr.Button("清空结果")
                    transfer_to_img_search_btn = gr.Button("发送到搜图")
                    open_img_folder_btn = gr.Button("打开目录")
                    save_select_image_btn = gr.Button("保存当前选中到输出")
                with gr.Row():
                    local_state.select_img_info = select_img_info = gr.Markdown("", visible=True)

        with gr.Accordion(open=False, label="查询结果处理"):
            with gr.Tab(label="查询导出"):
                with gr.Row():
                    save_to_outdir_copy_type = gr.Dropdown(label="保存模式", choices=["保存当前查询", "保存所有查询"], value="保存当前查询", type="index", interactive=True)
                    save_to_outdir_skip_img_filesize = gr.Number(label="跳过小于此文件大小", info="单位千字节(KB)", value=0, interactive=True)
                    format_choices = ["不修改", "JPEG", "PNG"]
                    save_to_outdir_format = gr.Dropdown(label="保存格式为", info="指定新的保存格式", choices=format_choices, value=format_choices[0], type="index", interactive=True)
                    save_to_outdir_quality = gr.Number(label="保存质量", info="仅保存为 JPEG 有效" , value=95, interactive=True)
                with gr.Row():
                    save_to_outdir_skip_img_pixel = gr.Number(label="跳过最小的边", info="当高或宽小于此像素时跳过,0为不启用", value=0, interactive=True)
                    save_to_outdir_skip_img_scale = gr.Number(label="跳过比例", info="当高宽比或宽高比大于此值时跳过,0为不启用,推荐值在[2,3]之间", value=0, interactive=True)
                    save_to_outdir_max_pixel = gr.Number(label="压缩最大边到", info="当高或宽大于此像素时等比缩小,0为不启用", value=0, interactive=True)
                with gr.Row():
                    save_to_outdir_start_page = gr.Number(label="起始页", value=1, interactive=True)
                    save_to_outdir_end_page = gr.Number(label="结束页", value=-1, interactive=True)
                    save_to_outdir_max_export_count = gr.Number(label="每个查询最大输出数量", info="输出每个查询前N个图片,0为不启用", value=0, interactive=True)
                    save_to_outdir_copy_same_name_ext = gr.Textbox(label="拷贝同名文件", info="拷贝同名的其它后缀文件" , value=".txt,.caption,.json", interactive=True)
                    save_to_outdir_random_new_name = gr.Checkbox(label="随机新的文件名称", info="避免多次输出后的长文件名" , value=False, interactive=True)
                with gr.Row():
                    default_save_path = os.path.join(Path.home(), "Pictures", "CLIPImageSearchWebUI")
                    save_to_outdir = gr.Textbox(label="保存结果图片到目录", scale=5, value=default_save_path)
                    save_to_outdir_btn = gr.Button("保存")

            create_image_delete(top_elems, local_state)
            search_image_load_vecdb_compolents = search_image_load_vecdb.on_gui()

    set_search_target([local_state.search_target, local_state.select_search_target])

    pageBlock.load(fn=on_load_page, inputs=[
        local_state.search_target,
        local_state.select_search_target
    ], 
    outputs=[
        local_state.search_target,
        local_state.select_search_target,
        top_k_number,
    ])

    image_viewer_outputs = local_state.get_image_viewer_outputs()

    search_with_image_btn.click(fn=on_search_with_image, inputs=[
        local_state.search_target,
        local_state.select_search_target,
        local_state.search_history,
        local_state.select_search_history,
        search_image, 
        top_k_number, 
        page_size
        ], outputs=image_viewer_outputs)
    search_with_prompt_btn.click(fn=on_search_with_prompt, inputs=[
        local_state.search_target, 
        local_state.select_search_target, 
        local_state.search_history,
        local_state.select_search_history,
        search_text, 
        top_k_number, 
        page_size
        ], outputs=image_viewer_outputs)

    # 翻页相关
    page_state_inputs = [ local_state.page_state ]

    def on_first_page(page_state, progress = gr.Progress()):
        if page_state is None:
            raise constants_util.NO_QUERY_RESULT_ERROR
        page_state["page_index"] = 1
        return local_state.update_viewer_page(page_state, progress)

    def on_last_page(page_state, progress = gr.Progress()):
        if page_state is None:
            raise constants_util.NO_QUERY_RESULT_ERROR
        page_state["page_index"] = page_state["page_count"]
        return local_state.update_viewer_page(page_state, progress)

    def on_prev_page(page_state, progress = gr.Progress()):
        if page_state is None:
            raise constants_util.NO_QUERY_RESULT_ERROR
        page_state["page_index"] -= 1
        page_state["page_index"] = max(page_state["page_index"], 1)
        return local_state.update_viewer_page(page_state, progress)

    def on_next_page(page_state, progress = gr.Progress()):
        if page_state is None:
            raise constants_util.NO_QUERY_RESULT_ERROR
        page_state["page_index"] += 1
        page_state["page_index"] = min(page_state["page_index"], page_state["page_count"])
        return local_state.update_viewer_page(page_state, progress)

    def on_goto_page(page_state, page_index: float, progress = gr.Progress()):
        if page_state is None:
            raise constants_util.NO_QUERY_RESULT_ERROR
        page_state["page_index"] = int(page_index)
        page_state["page_index"] = max(page_state["page_index"], 1)
        page_state["page_index"] = min(page_state["page_index"], page_state["page_count"])
        return local_state.update_viewer_page(page_state, progress)

    def on_select_history(page_state, search_history: dict, select_search_history: str, progress = gr.Progress()):
        search_id = None
        for search_name, id in search_history["search"]:
            if search_name == select_search_history:
                search_id = id
        if search_id is None:
            raise constants_util.INVALID_QUERT_RECORD_ERROR
        page_state = local_state.load_page_meta(search_id)
        page_state["page_index"] = 1
        return local_state.update_viewer_page(page_state, progress)

    first_page_btn.click(on_first_page, inputs=page_state_inputs, outputs=image_viewer_outputs)
    last_page_btn.click(on_last_page, inputs=page_state_inputs, outputs=image_viewer_outputs)
    prev_page_btn.click(on_prev_page, inputs=page_state_inputs, outputs=image_viewer_outputs)
    next_page_btn.click(on_next_page, inputs=page_state_inputs, outputs=image_viewer_outputs)
    goto_page_btn.click(on_goto_page, inputs=page_state_inputs + [ local_state.page_index ], outputs=image_viewer_outputs)
    select_search_history.select(on_select_history, inputs=page_state_inputs + [local_state.search_history, select_search_history], outputs=image_viewer_outputs)

    def on_clear_search():
        """清理搜索结果"""
        return local_state.update_viewer(None, [], msg="已清理", search_target=None, search_history=None)
    clear_search_btn.click(fn=on_clear_search, outputs=image_viewer_outputs)

    def on_select_img(image_file_with_lable_list: list[tuple[str, str]], evt: gr.SelectData):
        """选中图片时更新标签状态"""
        item = image_file_with_lable_list[evt.index]
        with Image.open(item[0]) as img:
            width, height = img.width, img.height
        text = f"标签: **{item[1]}**\n\n原始文件路径: **{item[0]}**\n\n分辨率：{width}x{height}"
        return (text, evt.index)
    gallery.select(fn=on_select_img, inputs=[local_state.image_file_with_lable_list], outputs=[select_img_info, local_state.image_select], show_progress=True)

    def on_transfer_to_img_search(image_file_with_lable_list: list[tuple[str, str]], select_index: float):
        if image_file_with_lable_list is None:
            raise constants_util.NO_QUERY_RESULT_ERROR
        select_index = int(select_index)
        if select_index < 0:
            return
        item = image_file_with_lable_list[select_index]
        filename_with_ext = item[0]
        filename_without_ext, _ = os.path.splitext(filename_with_ext)

        cache_root = os.path.join(get_cache_root().cache_root, "preview")
        hash_id = hashlib.sha1(filename_without_ext.encode('utf-8')).hexdigest()
        cache_file = os.path.join(cache_root, f"{hash_id}.jpg")
        image_file = open_and_try_resize_image(filename_with_ext, cache_file, local_state.cache_image_max_size, local_state.greater_than_size)
        return image_file

    transfer_to_img_search_btn.click(fn=on_transfer_to_img_search, inputs=[local_state.image_file_with_lable_list, local_state.image_select], outputs=[search_image])

    def on_open_folder(image_file_with_lable_list: list[tuple[str, str]], select_index: float):
        """"打开选中图片所在文件夹"""
        if image_file_with_lable_list is None:
            raise constants_util.NO_QUERY_RESULT_ERROR
        select_index = int(select_index)
        if select_index < 0:
            return
        item = image_file_with_lable_list[select_index]
        subprocess.Popen(f'explorer /select,"{item[0]}"')
    open_img_folder_btn.click(fn=on_open_folder, inputs=[local_state.image_file_with_lable_list, local_state.image_select])

    save_select_image_btn.click(fn=search_image_save.save_select_image, 
                                inputs=[save_to_outdir, local_state.image_file_with_lable_list, local_state.image_select, save_to_outdir_copy_same_name_ext],
                                outputs=[select_img_info])

    save_to_outdir_btn.click(fn=search_image_save.save_query_image_to_dir, inputs=page_state_inputs + [
        local_state.search_history,
        save_to_outdir_copy_type,
        save_to_outdir,
        save_to_outdir_start_page,
        save_to_outdir_end_page,
        save_to_outdir_max_export_count,
        save_to_outdir_copy_same_name_ext,
        save_to_outdir_random_new_name,
        save_to_outdir_skip_img_filesize,
        save_to_outdir_skip_img_pixel,
        save_to_outdir_skip_img_scale,
        save_to_outdir_max_pixel,
        save_to_outdir_format,
        save_to_outdir_quality
    ], outputs = [
        msg_text,
        save_to_outdir,
    ])

    clip_classifier.on_bind(search_state=local_state, compolents=clip_classifier_compolents)
    clustering.on_bind(search_state=local_state, compolents=repeat_query_compolents)
    aesthetic_predictor.on_bind(search_state=local_state, compolents=aesthetic_predictor_compolents)
    deduplication.on_bind(search_state=local_state, compolents=deduplication_compolents)
    #search_image_delete.on_bind(search_state=local_state, compolents=search_image_delete_compolents)
    search_image_load_vecdb.on_bind(search_state=local_state, compolents=search_image_load_vecdb_compolents)
