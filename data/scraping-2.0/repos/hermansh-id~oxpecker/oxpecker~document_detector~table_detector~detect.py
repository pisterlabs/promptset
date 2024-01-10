from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from langchain.thirdparty.table_detector.util import PIL_to_cv, cv_to_PIL
from collections import Counter 
from itertools import tee, count
import torch
from transformers import DetrImageProcessor
import pandas as pd
import asyncio, string
import regex as re
import pytesseract
from time import time
import warnings


class TableExtractionPipeline():
    def __init__(self, 
                 ocr, 
                #  model_table, 
                 model_table_structure):
        self.colors = ["red", "blue", "green", "yellow", "orange", "violet"]
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        self.ocr = ocr
        
        # self.model_table = model_table
        self.model_table_structure = model_table_structure

    def sharpen_image(self, pil_img):

        img = PIL_to_cv(pil_img)
        sharpen_kernel = np.array([[-1, -1, -1], 
                                [-1,  9, -1], 
                                [-1, -1, -1]])

        sharpen = cv2.filter2D(img, -1, sharpen_kernel)
        pil_img = cv_to_PIL(sharpen)
        return pil_img


    def uniquify(self, seq, suffs = count(1)):
        not_unique = [k for k,v in Counter(seq).items() if v>1] 

        suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))  
        for idx,s in enumerate(seq):
            try:
                suffix = str(next(suff_gens[s]))
            except KeyError:
                continue
            else:
                seq[idx] += suffix

        return seq

    def binarizeBlur_image(self, pil_img):
        image = PIL_to_cv(pil_img)
        thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]

        result = cv2.GaussianBlur(thresh, (5,5), 0)
        result = 255 - result
        return cv_to_PIL(result)



    def td_postprocess(self, pil_img):
        img = PIL_to_cv(pil_img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 100), (255, 5, 255))
        nzmask = cv2.inRange(hsv, (0, 0, 5), (255, 255, 255))
        nzmask = cv2.erode(nzmask, np.ones((3,3)))
        mask = mask & nzmask
        new_img = img.copy()
        new_img[np.where(mask)] = 255
        return cv_to_PIL(new_img)

    def table_detector(self, image, THRESHOLD_PROBA):
        model = self.model_table
        model.overrides['conf'] = THRESHOLD_PROBA
        model.overrides['iou'] = 0.45
        model.overrides['agnostic_nms'] = False
        model.overrides['max_det'] = 1000
        with torch.no_grad():
            outputs = model.predict(image)

        probas = outputs[0].probs


        return (model, probas, outputs[0].boxes)


    def table_struct_recog(self, image, THRESHOLD_PROBA):
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        feature_extractor = DetrImageProcessor(do_resize=True, size=1000, max_size=1000)
        encoding = feature_extractor(image, return_tensors="pt").to(device)
        model = self.model_table_structure
        with torch.no_grad():
            outputs = model(**encoding)

        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > THRESHOLD_PROBA

        target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
        postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
        bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
        return (model, probas[keep], bboxes_scaled)
    
    def pytess(self, cell_pil_img):
        paddle_output=' '
        cell_cv_img=PIL_to_cv(cell_pil_img)
        height, width, channels = cell_cv_img.shape
        if height>=10 and width>=10:
            # hasiltesseract = pytesseract.image_to_string(cell_cv_img, lang='ind', config='--oem 3 --psm 1')
            # paddle_output = paddle_output + hasiltesseract + " "
            result = self.ocr.ocr(cell_cv_img,cls=True)
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    paddle_output=paddle_output+' '+line[1][0]
            paddle_output=paddle_output+' '
        return str(paddle_output)
    
    def add_padding(self, pil_img, top, right, bottom, left, color=(255,255,255)):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    def plot_results_detection(self, c1, model, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        numpy_image = np.array(pil_img)
        pil_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        for coor in boxes:
            x1, y1, x2, y2 = coor.xyxy.tolist()[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            object_image = pil_img[y1:y2, x1:x2]
            plt.imshow(object_image)
            c1.pyplot()


    def crop_tables(self, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        cropped_img_list = []

        numpy_image = np.array(pil_img)
        pil_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        for coor in boxes:
            x1, y1, x2, y2 = coor.xyxy.tolist()[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            object_image = pil_img[y1:y2, x1:x2].astype(np.uint8)
            object_image = Image.fromarray(object_image)
            cropped_img_list.append(object_image)

        return cropped_img_list
    
    def cropTable(self, pil_img, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        # cropped_img_list = []

        numpy_image = np.array(pil_img)
        pil_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        x1, y1, x2, y2 = boxes
        # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        object_image = pil_img[y1:y2, x1:x2].astype(np.uint8)
        object_image = Image.fromarray(object_image)
        

        return object_image

    def generate_structure(self, model, pil_img, prob, boxes, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom):
        rows = {}
        cols = {}
        idx = 0


        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

            xmin, ymin, xmax, ymax = xmin, ymin, xmax, ymax 
            cl = p.argmax()
            class_text = model.config.id2label[cl.item()]

            if class_text == 'table row':
                rows['table row.'+str(idx)] = (xmin, ymin-expand_rowcol_bbox_top, xmax, ymax+expand_rowcol_bbox_bottom)
            if class_text == 'table column':
                cols['table column.'+str(idx)] = (xmin, ymin-expand_rowcol_bbox_top, xmax, ymax+expand_rowcol_bbox_bottom)

            idx += 1

        return rows, cols

    def sort_table_featuresv2(self, rows:dict, cols:dict):
        rows_ = {table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(rows.items(), key=lambda tup: tup[1][1])}
        cols_ = {table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(cols.items(), key=lambda tup: tup[1][0])}

        return rows_, cols_

    def individual_table_featuresv2(self, pil_img, rows:dict, cols:dict):

        for k, v in rows.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            rows[k] = xmin, ymin, xmax, ymax, cropped_img

        for k, v in cols.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cols[k] = xmin, ymin, xmax, ymax, cropped_img

        return rows, cols


    def object_to_cellsv2(self, master_row:dict, cols:dict, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom, padd_left):
        cells_img = {}
        header_idx = 0
        row_idx = 0
        previous_xmax_col = 0
        new_cols = {}
        new_master_row = {}
        previous_ymin_row = 0
        new_cols = cols
        new_master_row = master_row
        for k_row, v_row in new_master_row.items():    
            _, _, _, _, row_img = v_row
            xmax, ymax = row_img.size
            xa, ya, xb, yb = 0, 0, 0, ymax
            row_img_list = []
            for idx, kv in enumerate(new_cols.items()):
                k_col, v_col = kv
                xmin_col, _, xmax_col, _, col_img = v_col
                xmin_col, xmax_col = xmin_col - padd_left - 10, xmax_col - padd_left
                xa = xmin_col
                xb = xmax_col
                if idx == 0:
                    xa = 0
                if idx == len(new_cols)-1:
                    xb = xmax
                xa, ya, xb, yb = xa, ya, xb, yb

                row_img_cropped = row_img.crop((xa, ya, xb, yb))
                row_img_list.append(row_img_cropped)

            cells_img[k_row+'.'+str(row_idx)] = row_img_list
            row_idx += 1

        return cells_img, len(new_cols), len(new_master_row)-1

    def clean_dataframe(self, df):
        for col in df.columns:
            df[col]=df[col].str.replace("'", '', regex=True)
            df[col]=df[col].str.replace('"', '', regex=True)
            df[col]=df[col].str.replace(']', '', regex=True)
            df[col]=df[col].str.replace('[', '', regex=True)
            df[col]=df[col].str.replace('{', '', regex=True)
            df[col]=df[col].str.replace('}', '', regex=True)
        return df

    def convert_df(self, df):
        return df.to_csv().encode('utf-8')


    def create_dataframe(self, cells_pytess_result:list, max_cols:int, max_rows:int):
        headers = cells_pytess_result[:max_cols]
        new_headers = self.uniquify(headers, (f' {x!s}' for x in string.ascii_lowercase))
        counter = 0
        cells_list = cells_pytess_result[max_cols:]
        df = pd.DataFrame("", index=range(0, max_rows), columns=new_headers)

        cell_idx = 0
        for nrows in range(max_rows):
            for ncols in range(max_cols):
                df.iat[nrows, ncols] = str(cells_list[cell_idx])
                cell_idx += 1

        for x, col in zip(string.ascii_lowercase, new_headers):
            if f' {x!s}' == col:
                counter += 1
        header_char_count = [len(col) for col in new_headers]

        df = self.clean_dataframe(df)
        
        return df

    def start_process_individu(self, image, bboxes_scaled, TSR_THRESHOLD, padd_top, padd_left, padd_bottom, padd_right, delta_xmin, delta_ymin, delta_xmax, delta_ymax, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom):
        
        # model, probas, bboxes_scaled = self.table_detector(image, THRESHOLD_PROBA=TD_THRESHOLD)

        # if len(bboxes_scaled) == 0:
        #     return []
        
        unpadded_table = self.cropTable(image, bboxes_scaled, delta_xmin, delta_ymin, delta_xmax, delta_ymax)

        
        list_df = []
        # for unpadded_table in cropped_img_list:

        table = self.add_padding(unpadded_table, padd_top, padd_right, padd_bottom, padd_left)

        model, probas, bboxes_scaled = self.table_struct_recog(table, THRESHOLD_PROBA=TSR_THRESHOLD)
        rows, cols = self.generate_structure(model, table, probas, bboxes_scaled, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom)
        rows, cols = self.sort_table_featuresv2(rows, cols)
        master_row, cols = self.individual_table_featuresv2(table, rows, cols)

        cells_img, max_cols, max_rows = self.object_to_cellsv2(master_row, cols, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom, padd_left)

        sequential_cell_img_list = []
        for k, img_list in cells_img.items():
            for img in img_list:
                sequential_cell_img_list.append(self.pytess(img))

        # cells_pytess_result = await asyncio.gather(*sequential_cell_img_list)
        

        df = self.create_dataframe(sequential_cell_img_list, max_cols, max_rows)
            # list_df.append(df)
        # return list_df
        return df.to_json(orient='table')