import os
import hashlib
import gradio as gr
import module.utils.constants_util as constants_util
import json
from PIL import Image
import time
import uuid
import torch
from langchain.vectorstores.faiss import FAISS
from tqdm import tqdm
from module.data import (
    get_vector_db_mgr, 
    get_webui_configs,
    get_cache_root,
    VectorDatabase, 
    VectorDatabaseManager
)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def open_and_try_resize_image(image_path, cache_file, max_side, greater_than_size) -> None:
    if os.path.exists(cache_file):
        return cache_file
    file_stats = os.stat(image_path)
    if file_stats.st_size <= greater_than_size:# 大于指定大小才建立缓存
        return image_path
    with Image.open(image_path) as image:
        width, height = image.size
        # 长款都小于指定尺寸，不调整了
        if width < max_side and height < max_side:
            return image_path
        if width > height:
            width_pct = max_side / width
            width = max_side
            height = height * width_pct
        else:
            height_pct = max_side / height
            width = width * height_pct
            height = max_side

        new_image = image.convert('RGB')
        new_image = new_image.resize((int(width), int(height)))
        with open(cache_file, "w") as f:
            new_image.save(cache_file, format="JPEG", quality=80, optimize=True)
            return cache_file
    return new_image

class SearchVectorPageState():
    msg_text:gr.Textbox()
    search_file: gr.File = None
    page_index: gr.Number()
    page_count: gr.Number()
    page_state: gr.State()
    page_select: gr.Number()
    image_gallery: gr.Gallery()
    # 当前页的原始文件路径和标签
    image_file_with_lable_list: gr.State() 
    image_select: gr.State()
    select_img_info: gr.Markdown()

    first_page_btn: gr.Button()
    last_page_btn: gr.Button()
    prev_page_btn: gr.Button()
    next_page_btn: gr.Button()
    select_search_target: gr.Dropdown()
    search_target: gr.State()
    search_history: gr.State()
    select_search_history: gr.Dropdown()
    search_count = 0
    page_size: gr.Number()

    def __init__(self) -> None:
        pass

    def init(self):
        cfg = get_webui_configs().get_cfg()
        self.cache_image_max_size: int = cfg.cache.image.max_size 
        self.greater_than_size: int = cfg.cache.image.greater_than_size

    def get_image_viewer_outputs(self):
        return [
            self.page_state,
            self.msg_text, 
            self.image_file_with_lable_list, 
            self.image_gallery,
            self.page_index,
            self.page_count,
            self.select_img_info,
            self.image_select,
            self.search_target,
            self.search_history,
            self.select_search_history,
        ]

    def open_image_or_create_cache(self, image_and_label: list[tuple[str, str]], progress: gr.Progress):
        preview_images = []
        raw_images = []
        labels = []

        cache_root = os.path.join(get_cache_root().cache_root, "preview")

        if not os.path.exists(cache_root):
            os.mkdir(cache_root)

        for filename, label in progress.tqdm(image_and_label, desc="创建图像缓存"):
            find_img = False

            hash_id = hashlib.sha1(filename.encode('utf-8')).hexdigest()
            cache_file = os.path.join(cache_root, f"{hash_id}.jpg")

            for image_ext in constants_util.IMAGE_EXTENSIONS:
                filename_with_ext = filename + image_ext
                if os.path.exists(filename_with_ext):
                    find_img=True
                    image_file = open_and_try_resize_image(filename_with_ext, cache_file, self.cache_image_max_size, self.greater_than_size)
                    preview_images.append(image_file)
                    raw_images.append(filename_with_ext)
                    break
        
            if find_img:
                labels.append(label)
            else:
                print(f"missing image file: {filename}")

        return list(zip(preview_images, labels)), list(zip(raw_images, labels))

    def load_page(self, search_id: str, page_index: int) -> list[tuple[str, str]]:
        cache_root = os.path.join(get_cache_root().cache_root, "search_id", search_id)
        pages_index_file = os.path.join(cache_root, "pages_index.json")

        if not os.path.exists(pages_index_file):
            return []

        with open(os.path.join(cache_root, "pages_index.json"), "r") as f:
            page_info = json.load(f)

        page_pos = page_info[int(page_index) - 1]

        with open(os.path.join(cache_root, "pages.json"), "r") as f:
            f.seek(page_pos[0])
            content = f.read(page_pos[1] - page_pos[0])
            return json.loads(content)

    def get_cache_root_path(self, search_id: str):
        return os.path.join(get_cache_root().cache_root, "search_id", search_id)

    def load_page_meta(self, search_id: str):
        cache_root = self.get_cache_root_path(search_id)
        with open(os.path.join(cache_root, "pages_meta.json"), "r") as f:
            return json.load(f)

    def save_pages(self, 
                   search_id: str, 
                   image_and_label: list[tuple[str, str]], 
                   page_size: int, 
                   indices = None,
                   db: VectorDatabase = None,
                   progress: gr.Progress = None) -> list[tuple[str, str]]:
        cache_root = os.path.join(get_cache_root().cache_root, "search_id", search_id)

        if not os.path.exists(cache_root):
            os.makedirs(cache_root)

        #pages = list(chunks(image_and_label, page_size))
        
        #for i, v in enumerate(progress.tqdm(tqdm(pages, desc="写出页缓存文件"), desc="创建分页缓存")):
        #    with open(os.path.join(cache_root, f"page_{i + 1}.json"), "w") as f:
        #        json.dump(v, f)

        n = 0
        first_page = None

        t0 = time.time()

        with open(os.path.join(cache_root, "pages.json"), "w") as f:
            page_info = []
            for v in chunks(image_and_label, page_size):
                if first_page is None:
                    first_page = v
                n+=1
                start_pos = f.tell()
                # 每次 json.dump 都会导致 flush，很慢
                json_string = json.dumps(v)
                del v
                f.write(json_string)
                end_pos = f.tell()
                page_info.append((start_pos, end_pos))

        with open(os.path.join(cache_root, "pages_index.json"), "w") as f:
            json.dump(page_info, f)

        with open(os.path.join(cache_root, "pages_meta.json"), "w") as f:
            json.dump({ "page_count": n, "search_id": search_id }, f)

        t1 = time.time()

        save_db_root = os.path.join(cache_root, "vecdb")

        save_index = 0

        #if indices is not None and len(indices) > 0:
        #    indices = [i for i in indices if i >= 0]

        if indices is not None and len(indices) > 0:
            loader = torch.utils.data.DataLoader(indices, batch_size=5000)
            for batch in tqdm(loader, desc="保存向量库"):
                batch_embeds = torch.tensor(db.db.index.reconstruct_batch(batch))

                data = []

                for i, j in enumerate(batch):
                    j = int(j)
                    doc_uuid = db.db.index_to_docstore_id[j]
                    doc = db.db.docstore.search(doc_uuid)
                    filename = doc.page_content 
                    if doc.metadata:
                        image_root = doc.metadata["root"]
                        filename_witout_ext = os.path.join(image_root, filename)
                    else:
                        filename_witout_ext = filename
                    data.append((filename_witout_ext, batch_embeds[i]))

                vectorstore_new = FAISS.from_embeddings(text_embeddings=data, embedding=VectorDatabase.fake_embeddings)
                vectorstore_new.save_local(os.path.join(save_db_root, f"{save_index}"))
                save_index += 1

                del vectorstore_new, data

        #print(f"缓存搜索结果分页, time={t1-t0}")

        return first_page, n

    def search_with_save_page(self,
                              embedding : list[float], 
                              top_k : int, 
                              search_history: dict, 
                              select_search_target: list[str],
                              page_size: int, 
                              search_name: str = "搜索 {n}"):

        vector_mgr = get_vector_db_mgr()
        """搜索并保存搜索结果页"""
        page_size = int(page_size)
        page_size = max(page_size, 1)
        page_size = min(page_size, 1000)

        top_k = int(max(top_k, 1))

        assert len(select_search_target) >= 1

        preview_page_state = None
        preview_image_with_label = None
        preview_search_name = None

        new_searchs = []

        for i, target in enumerate(tqdm(select_search_target, desc="查询项目")):
            search_id = str(uuid.uuid4())
            image_and_label, indices, db = vector_mgr.search(embedding=embedding, top_k=top_k, variant=target)
            self.search_count += 1
            cur_search_name = search_name.format(n=self.search_count, target=target)
            if i == 0:
                preview_image_with_label, page_count = self.save_pages(search_id, image_and_label, page_size=page_size, indices=indices, db=db)
                preview_page_state = { "search_id": search_id, "page_index": 1, "page_count": page_count }
                preview_search_name = cur_search_name
            else:
                # 仅保存
                self.save_pages(search_id, image_and_label, page_size=page_size, indices=indices, db=db)

            # 更新搜索结果列表
            new_searchs.append([cur_search_name, search_id])

        if search_history is None:
            search_history = { "search": [] }

        search_history["search"] = new_searchs + search_history["search"]

        # 仅返回第一个搜索项目的第一页的浏览内容
        return preview_image_with_label, preview_page_state, preview_search_name, search_history

    def update_viewer(self, 
                      page_state: dict,
                      image_and_label: list[tuple[str, str]], 
                      search_target: dict, 
                      search_history: dict,
                      select_search_name: str = None,
                      msg: str = "已完成", 
                      progress: gr.Progress = None):
        if search_history is None:
            search_history = { "search": [] }
        if page_state is not None:
            preview_images_with_label, raw_images_with_label = self.open_image_or_create_cache(image_and_label, progress=progress) 
            page_count = page_state["page_count"]
        else:
            select_search_name = ""
            page_state = None
            raw_images_with_label = preview_images_with_label = None
            page_count = 1

        return (
            # page_state
            page_state,            
            # msg_text
            msg,
            # image_file_with_lable_list
            raw_images_with_label,
            # image_gallery
            preview_images_with_label,
            # page_index
            "1",
            # page_count
            page_count,
            # select_img_info
            "",
            # image_select,
            -1,
            # search_target,
            search_target if search_target is not None else gr.update(),
            # search_history,
            search_history,
            # select_search_history,
            gr.Dropdown.update(choices=[i[0] for i in search_history["search"]], value=select_search_name),
        )

    def update_viewer_page(self, page_state: dict(), progress: gr.Progress):
        cur_page = self.load_page(page_state["search_id"], page_state["page_index"])

        preview_images_with_label, raw_images_with_label = self.open_image_or_create_cache(cur_page, progress=progress) 

        return (
            # page_state
            page_state,
            # msg_text
            "已更新页面",
            # image_file_with_lable_list
            raw_images_with_label,
            # image_gallery
            preview_images_with_label,
            # page_index
            page_state["page_index"],
            # page_count
            page_state["page_count"],
            # select_img_info
            "",
            # image_select
            -1,
            # select_search_target
            gr.Dropdown.update(),
            # search_history
            gr.update(),
            # select_search_history,
            gr.update(),
        )