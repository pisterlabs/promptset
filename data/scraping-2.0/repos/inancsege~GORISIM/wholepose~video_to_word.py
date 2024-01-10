# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
import os
import time
import openai
import pprint
import sys
import imutils
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as f
# import torch.optim
# import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
from pose_hrnet import get_pose_net
# import coremltools as ct
from collections import OrderedDict
from config import cfg
from config import update_config

from PIL import Image, ImageOps
import numpy as np
import cv2
import pandas as pd
from utils import pose_process, plot_pose
from natsort import natsorted
import shutil

import cupy
from Resnet2plus1d import r2plus1d_18
from collections import Counter
import torchvision.transforms as transforms

import time

start_time = time.time()

path = "C:/xampp/htdocs/Bitirme/wholepose/Video/video_color.mp4"

cap = cv2.VideoCapture(path)

w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
if w_frame != h_frame:
    x, y, h, w = 0, int((h_frame - w_frame) / 2), w_frame, w_frame

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('C:/xampp/htdocs/Bitirme/wholepose/Video_Cropped/video_color.mp4', fourcc, fps, (512, 512))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            crop_frame = frame[y:y + h, x:x + w]
            crop_frame = cv2.resize(crop_frame, (512, 512), interpolation=cv2.INTER_AREA)
            # crop_frame = cv2.flip(crop_frame, -1)
            out.write(crop_frame)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
else:
    shutil.copy2(path, 'C:/xampp/htdocs/Bitirme/wholepose/Video_Cropped/')

model = r2plus1d_18(pretrained=True, num_classes=226)
# load pretrained
checkpoint = torch.load('C:/xampp/htdocs/Bitirme/rgb_final_finetuned.pth')
test_path = "C:/xampp/htdocs/Bitirme/wholepose/Frames/video"
labels = pd.read_csv('C:/xampp/htdocs/Bitirme/SignList_ClassId_TR_EN.csv', encoding='latin5')
# test_path = "F:/validation_frames/signer1_sample57"
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k[7:]  # remove 'module.'
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()
# if phase == 'Train':
#     model.fc1 = nn.Linear(model.fc1.in_features, num_classes)

# Export the model to ONNX format
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    model.cuda()
model = model.to(device)

# Initialize CUDA context
if torch.cuda.is_available():
    device_id = 0  # Choose the device you want to use
    cupy.cuda.runtime.setDevice(device_id)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

index_mirror = np.concatenate([
    [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16],
    [21, 22, 23, 18, 19, 20],
    np.arange(40, 23, -1), np.arange(50, 40, -1),
    np.arange(51, 55), np.arange(59, 54, -1),
    [69, 68, 67, 66, 71, 70], [63, 62, 61, 60, 65, 64],
    np.arange(78, 71, -1), np.arange(83, 78, -1),
    [88, 87, 86, 85, 84, 91, 90, 89],
    np.arange(113, 134), np.arange(92, 113)
]) - 1
assert (index_mirror.shape[0] == 133)

multi_scales = [512, 640]


def norm_numpy_totensor(img):
    img = img.astype(np.float32) / 255.0
    for i in range(3):
        img[:, :, :, i] = (img[:, :, :, i] - mean[i]) / std[i]
    return torch.from_numpy(img).permute(0, 3, 1, 2)


def stack_flip(img):
    img_flip = cv2.flip(img, 1)
    return np.stack([img, img_flip], axis=0)


def merge_hm(hms_list):
    assert isinstance(hms_list, list)
    for hms in hms_list:
        hms[1, :, :, :] = torch.flip(hms[1, index_mirror, :, :], [2])

    hm = torch.cat(hms_list, dim=0)
    # print(hm.size(0))
    hm = torch.mean(hms, dim=0)
    return hm


with torch.no_grad():
    # config = open(os.path.join(sys.path[0], "wholebody_w48_384x288.yaml"), "r")
    config = "C:/xampp/htdocs/Bitirme/wholepose/wholebody_w48_384x288.yaml"
    cfg.merge_from_file(config)

    # dump_input = torch.randn(1, 3, 256, 256)
    # newmodel = PoseHighResolutionNet()
    newmodel = get_pose_net(cfg, is_train=False)
    # print(newmodel)
    # dump_output = newmodel(dump_input)
    # print(dump_output.size())
    checkpoint = torch.load('C:/xampp/htdocs/Bitirme/wholepose/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth')
    # newmodel.load_state_dict(checkpoint['state_dict'])

    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'backbone.' in k:
            name = k[9:]  # remove module.
        if 'keypoint_head.' in k:
            name = k[14:]  # remove module.
        new_state_dict[name] = v
    newmodel.load_state_dict(new_state_dict)

    newmodel.cuda().eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    input_path = 'C:/xampp/htdocs/Bitirme/wholepose/Video_Cropped'
    paths = []
    names = []
    for root, _, fnames in natsorted(os.walk(input_path)):
        for fname in natsorted(fnames):
            path1 = os.path.join(root, fname)
            if 'depth' in fname:
                continue
            paths.append(path1)
            names.append(fname)
        # paths = paths[:4]
        # names = names[:4]
    step = 600
    start_step = 6
    # paths = paths[start_step*step:(start_step+1)*step]
    # names = names[start_step*step:(start_step+1)*step]
    # paths = paths[4200:]
    # names = names[4200:]
    # paths = paths[::-1]
    # names = names[::-1]
    for i, path in enumerate(paths):
        # if i > 1:
        #     break
        output_npy = 'C:/xampp/htdocs/Bitirme/wholepose/Npy/{}.npy'.format(names[i])

        if os.path.exists(output_npy):
            continue

        cap = cv2.VideoCapture(path)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # frame_width = 256
        # frame_height = 256

        # output_filename = os.path.join('out_test', names[i])

        # img = Image.open(image_path)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # writer = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc('M','P','4','V'), 5, (frame_width,frame_height))
        output_list = []
        counter = 0
        while cap.isOpened():
            success, img = cap.read()
            counter += 1
            if counter % 20 != 0:
                continue
            if not success:
                # If loading a video, use 'break' instead of 'continue'.
                break
            # img = cv2.resize(img, (512,512))
            # img = cv2.resize(img,(512,512),interpolation = cv2.INTER_AREA)
            # img = imutils.resize(img, 512)
            # img = imutils.resize(img,512,512)
            frame_height, frame_width = img.shape[:2]
            img = cv2.flip(img, flipCode=1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = Image.fromarray(img)
            # img = ImageOps.mirror(img)
            # img.thumbnail((512,512),Image.ANTIALIAS)
            out = []
            for scale in multi_scales:

                if scale != 512:
                    # print("x")
                    img_temp = cv2.resize(img, (scale, scale))

                else:

                    img_temp = img

                img_temp = stack_flip(img_temp)
                img_temp = norm_numpy_totensor(img_temp).cuda()
                # print(img_temp.shape)
                # print(img_temp)
                # img_temp = img_temp.transpose(0,1)
                # img_temp = img_temp.squeeze()
                # img_temp = img_temp.permute(1,0,2,3)
                hms = newmodel(img_temp)
                if scale != 512:
                    out.append(f.interpolate(hms, (frame_width // 4, frame_height // 4), mode='bilinear'))
                else:
                    out.append(hms)

            out = merge_hm(out)
            # print(out.size())
            # hm, _ = torch.max(out, 1)
            # hm = hm.cpu().numpy()
            # print(hm.shape)
            # np.save('hm.npy', hm)
            result = out.reshape((133, -1))
            result = torch.argmax(result, dim=1)
            # print(result)
            result = result.cpu().numpy().squeeze()

            # print(result.shape)
            y = result // (frame_width // 4)
            x = result % (frame_width // 4)
            pred = np.zeros((133, 3), dtype=np.float32)
            pred[:, 0] = x
            pred[:, 1] = y
            hm = out.cpu().numpy().reshape((133, frame_height // 4, frame_height // 4))

            pred = pose_process(pred, hm)
            pred[:, :2] *= 4.0
            # print(pred.shape)
            assert pred.shape == (133, 3)

            # print(arg.cpu().numpy())
            # np.save('npy/{}.npy'.format(names[i]), np.array([x,y,score]).transpose())
            output_list.append(pred)
            # img = np.asarray(img)
            # for j in range(133):
            #     img = cv2.circle(img, (int(x[j]), int(y[j])), radius=2, color=(255,0,0), thickness=-1)
            # img = plot_pose(img, pred)
            # cv2.imwrite('out/{}.png'.format(names[i]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        output_list = np.array(output_list)
        # print(output_list.shape)
        np.save(output_npy, output_list)
        cap.release()
        # writer.release()
        # break


def crop(image, center, radius, size=512):
    scale = 1.3
    radius_crop = (radius * scale).astype(np.int32)
    center_crop = (center).astype(np.int32)

    rect = (max(0, (center_crop - radius_crop)[0]), max(0, (center_crop - radius_crop)[1]),
            min(512, (center_crop + radius_crop)[0]), min(512, (center_crop + radius_crop)[1]))

    image = image[rect[1]:rect[3], rect[0]:rect[2], :]

    if image.shape[0] < image.shape[1]:
        top = abs(image.shape[0] - image.shape[1]) // 2
        bottom = abs(image.shape[0] - image.shape[1]) - top
        image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif image.shape[0] > image.shape[1]:
        left = abs(image.shape[0] - image.shape[1]) // 2
        right = abs(image.shape[0] - image.shape[1]) - left
        image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return image


selected_joints = np.concatenate(([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                  [91, 95, 96, 99, 100, 103, 104, 107, 108, 111],
                                  [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0)
folder = 'C:/xampp/htdocs/Bitirme/wholepose/Video_Cropped'  # 'train', 'test'
npy_folder = "C:/xampp/htdocs/Bitirme/wholepose/Npy"  # 'train_npy/npy3', 'test_npy/npy3'
out_folder = "C:/xampp/htdocs/Bitirme/wholepose/Frames"  # 'train_frames' 'test_frames'

for root, dirs, files in os.walk(folder, topdown=False):
    for name in files:
        if 'color' in name:
            # print(os.path.join(root, name))
            cap = cv2.VideoCapture(os.path.join(root, name))
            npy = np.load(os.path.join(npy_folder, name + '.npy')).astype(np.float32)
            npy = npy[:, selected_joints, :2]
            npy[:, :, 0] = 512 - npy[:, :, 0]
            xy_max = npy.max(axis=1, keepdims=False).max(axis=0, keepdims=False)
            xy_min = npy.min(axis=1, keepdims=False).min(axis=0, keepdims=False)
            assert xy_max.shape == (2,)
            xy_center = (xy_max + xy_min) / 2 - 20
            xy_radius = (xy_max - xy_center).max(axis=0)
            index = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    image = crop(frame, xy_center, xy_radius)
                else:
                    break
                index = index + 1
                image = cv2.resize(image, (256, 256))
                if not os.path.exists(os.path.join(out_folder, name[:-10])):
                    os.makedirs(os.path.join(out_folder, name[:-10]))
                cv2.imwrite(os.path.join(out_folder, name[:-10], '{:04d}.jpg'.format(index)), image)
                # print(os.path.join(out_folder, name[:-10], '{:04d}.jpg'.format(index)))

all_frames = []


# model = r2plus1d_18(pretrained=True, num_classes=225)
# model.load_state_dict(torch.load('D:/bitirme_dataset/final_models_finetuned/final_models_finetuned/rgb_final_finetuned.pth'))
# model = r2plus1d_18(pretrained=True, num_classes=6)

def float_argmax(tensor):
    # Flatten the tensor to a 1D array
    output_array = tensor.detach().cpu().numpy()
    flat_tensor = output_array.flatten()
    # Find the index of the largest element in the flattened tensor
    index = np.argmax(flat_tensor)
    # Return the value of the largest element in the tensor as a float
    return float(flat_tensor[index])


def most_common(arr):
    count = Counter(arr)
    return count.most_common(1)[0][0]


def read_images(folder_path):
    # assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder: " + str(folder_path)
    folder_path = folder_path.replace(os.sep, '/')
    images = []

    frame_indices = np.arange(len(os.listdir(folder_path))) + 1

    # for i in range(self.frames):
    for i in frame_indices:
        # print(folder_path)
        folder = os.path.join(folder_path + "/{:04d}.jpg").format(i)
        image = Image.open(folder)
        # image = Image.open(os.path.join(folder_path, '{:04d}.jpg').format(i))

        crop_box = (16, 16, 240, 240)
        image = image.crop(crop_box)
        # assert image.size[0] == 224
        image = np.float32(image)
        image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        images.append(image)

    # images = torch.stack(images, dim=0)
    # switch dimension for 3d cnn
    # images = images.permute(1, 0, 2, 3)
    # print(images.shape)
    return images


# Define the functions for preprocessing and postprocessing
def preprocess_frame(frame):
    # Resize the frame to a specific size
    frame = cv2.flip(frame, 3)

    frame = np.array(frame)
    # frame = cv2.resize(frame, (32,32),interpolation=cv2.INTER_LINEAR)
    # frame = frame.astype(np.float32)
    frame = np.float32(frame)
    # frame = cv2.resize(frame, (32,32),interpolation=cv2.INTER_LINEAR)
    # Convert the frame to a numpy array

    # frame = np.array(frame)

    # Normalize the frame
    frame = frame / 255.0

    # Add an additional dimension to the frame (since the model expects a 4D tensor as input)
    # frame = np.expand_dims(frame, axis=0)
    frame = np.expand_dims(frame, axis=0)

    return frame


def argmax(x):
    return max(range(len(x)), key=lambda i: x[i])


def get_top_5_values(predictions):
    sorted_indices = torch.argsort(predictions, descending=True)
    top_5_indices = sorted_indices[:5]
    # top_5_values = predictions[top_5_indices]
    return top_5_indices


c = 0


def process_predictions(predictions):
    # Extract the predicted class from the predictions
    predicted = torch.argmax(predictions)
    # toppredictions = get_top_5_values(predictions)
    # print(toppredictions)
    # print(predicted.item())
    # print(labels.loc[predicted.item()].iloc[1])
    return labels.loc[predicted.item()].iloc[1]


# Start capturing the video
# input_video = cv2.VideoCapture(test_path)#.read()
all_frames = read_images(test_path)

# all_frames = np.array(all_frames)

list_of_words = ["test"]
wordCount = 0
j = 0
for i in range(int(len(all_frames) / 10)):
    if j + 40 > len(all_frames):
        break
    tensor_frames = all_frames[j:j + 40]
    j += 20
    tensor_frames = np.array(tensor_frames)
    input_tensor = torch.tensor(tensor_frames)
    input_tensor = input_tensor.permute(1, 4, 0, 2, 3)
    input_tensor = input_tensor.to('cuda')
    predictions = model(input_tensor)

    word = process_predictions(predictions)
    if float_argmax(predictions) > 1 and word != list_of_words[wordCount]:
        list_of_words.append(word)
        wordCount += 1

list_of_words_str = ""
for words, _ in enumerate(list_of_words):
    if words == 0:
        continue
    list_of_words_str += list_of_words[words] + " "
# print(list_of_words_str)

print(list_of_words_str)
end_time = time.time()

shutil.rmtree('C:/xampp/htdocs/Bitirme/wholepose/Frames')
os.remove('C:/xampp/htdocs/Bitirme/wholepose/Npy/video_color.mp4.npy')

# cv2.imshow('Frame', frame)
# image, results = mediapipe_detection(frame, holistic)
# print(image)


# draw_styled_landmarks(image, results)

# cv2.imshow('OpenCV Feed', image)

cv2.destroyAllWindows()

video_path = 'C:/xampp/htdocs/Bitirme/wholepose/Video/video_color.mp4'
cropped_video_path = "C:/xampp/htdocs/Bitirme/wholepose/Video_Cropped/video_color.mp4"
cap.release()
# fd = os.open(folder_p)
os.remove(video_path)
os.remove(cropped_video_path)
