import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import open3d as o3d
import json
import copy
from typing import List
import math
from scipy.spatial.transform import Rotation as R
import random
import mesh_to_depth as m2d
from PIL import Image
from matplotlib import pyplot as plt
import argparse

import openai
from flask import Flask, redirect, render_template, request, url_for
from io import BytesIO
import urllib.request
import requests
import torch

from utils.iterative_utils import point_cloud_to_depth, convert_realsense_rgb_depth_to_o3d_pcl, generate_background_mesh, transform_depth, sample_hemisphere_point_uniform, look_at
from depth_completion_files.depth_completion import complete_depth, make_depth_model
from utils.filtering_method import filter_pointclouds

OPENAI_KEY = ""

obj_dict = {
    'AlphabetSoup': 'soup can',
    'BBQSauce': 'BBQ bottle',
    'Butter': 'stick of butter',
    'Cherries': 'can of cherries',
    'ChocolatePudding': 'pudding box',
    'Cookies': 'cookie box',
    'Corn': 'can of corn',
    'CreamCheese': 'cream cheese box',
    'GranolaBars': 'granola bar box',
    'GreenBean': 'can of green beans',
    'Ketchup': 'ketchup bottle',
    'MacaroniAndCheese': 'box of mac and cheese',
    'Mayo': 'mayo bottle',
    'Milk': 'milk carton',
    'Mushrooms': 'can of mushrooms',
    'Mustard': 'mustard bottle',
    'OrangeJuice': 'orange juice carton',
    'Parmesan': 'parmesan container',
    'Peaches': 'can of peaches',
    'PeasAndCarrots': 'peas and carrots can',
    'Pineapple': 'pineapple can',
    'Popcorn': 'popcorn box',
    'Raisins': 'raisins box',
    'SaladDressing': 'salad dressing bottle',
    'Spaghetti': 'box of spaghetti noodles',
    'TomatoSauce': 'tomato sauce can',
    'Tuna': 'tuna can',
    'Yogurt': 'yogurt container',
}


def inference_hope(visualize, use_sd, save_intermediate, use_gt_caption, use_auto_caption, num_views, context_threshold, scene_frames, pipe, model, vis_processors, data_dir, scenes, write_file_root, depth_complete, config):
    # Loop through all the scenes
    scene_num_count = 0
    for scene in scenes:
        frames = scene_frames[scene_num_count]
        scene_num_count += 1

        write_file_scene = write_file_root + scene

        if not os.path.exists(write_file_scene):
            os.mkdir(write_file_scene)


        for frame in frames:
            write_file = write_file_scene + "/" + str(frame)
            if not os.path.exists(write_file):
                os.mkdir(write_file)

            if not os.path.exists(write_file + "/extra/"):
                os.mkdir(write_file + "/extra/")

            prompt = "a photo of household objects on a table"

            def load_scene(scene,scene_num):
                camera_dict = json.load(open(data_dir + scene + "/" + str(scene_num).zfill(4) + ".json"))
                camera_intrinsics = np.asarray(
                    camera_dict["camera"]["intrinsics"]
                ).reshape((3, 3))
                scene_dir = data_dir + scene + f"/{scene_num:04d}_rgb.jpg"
                depth_dir = data_dir + scene + f"/{scene_num:04d}_depth.png"
                camera_extrinsics = np.eye(4)
                camera_extrinsics = np.asarray(
                    camera_dict["camera"]["extrinsics"]
                ).reshape((4, 4))
                depth_scale = 1
                color = cv2.imread(scene_dir)
                depth = cv2.imread(depth_dir, cv2.IMREAD_UNCHANGED) * depth_scale
                objects = camera_dict["objects"]
                return color, depth, camera_extrinsics, camera_intrinsics, scene_dir, depth_dir, depth_scale, objects


            # Load scene 1
            color_1, depth_1, c1_T_w, c1_k, scene_dir, depth_dir, depth_scale, objects = load_scene(scene, frame)

            image_shape = np.shape(color_1)

            if use_gt_caption:
                prompt = ""
                if len(objects) >= 10:
                    objects = objects[:10]
                for i in range(len(objects)):
                    obj_id = objects[i]['class']
                    print(obj_id)
                    obj_name = obj_dict[obj_id]
                    if i == len(objects) - 1:
                        prompt = prompt + "and " + obj_name + " on a table"
                    else:
                        prompt = "" + obj_name + ", " + prompt
                prompt = "a photo of " + prompt
                print(prompt)

            if use_auto_caption:
                color_image_prompt = Image.fromarray(color_1)
                color_image_prompt = vis_processors["eval"](color_image_prompt).unsqueeze(0).to(device)
                text = model.generate({"image": color_image_prompt})
                print(text)
                prompt = "a photo of " + text[0]


            if save_intermediate:
                cv2.imwrite(write_file + f"/img_original.png", color_1)
                np.save(write_file + f"/depth_original.npy", depth_1)
            np.save(write_file + f"/extrinsics.npy", c1_T_w)
            np.save(write_file + f"/intrinsics.npy", c1_k)
            
            # Get point cloud 1 in camera frame 1
            pcl_returned = convert_realsense_rgb_depth_to_o3d_pcl(color_1, depth_1, c1_k)

            # Threshold out background to obtain center point of foreground objects
            z_threshold = 1.25
            points = np.asarray(pcl_returned.points)
            mask = points[:,2] < z_threshold
            pcl_filtered = o3d.geometry.PointCloud()
            pcl_filtered.points = o3d.utility.Vector3dVector(points[mask])

            # Transform filtered point cloud to world frame
            pcl_filtered_w = copy.deepcopy(pcl_filtered).transform(np.linalg.inv(c1_T_w))

            # Get point cloud 1 in world frame
            points_w = copy.deepcopy(pcl_returned).transform(np.linalg.inv(c1_T_w))
            center = np.mean(pcl_filtered_w.points, 0)

            colors_point_w = copy.deepcopy(np.asarray(points_w.colors))
            colors_point_w[:, [2, 0]] = colors_point_w[:, [0, 2]]
            points_w_vis = copy.deepcopy(points_w)
            points_w_vis.colors = o3d.utility.Vector3dVector(colors_point_w)
            o3d.io.write_point_cloud(write_file + f"/input.ply", points_w_vis)

            color_list = np.asarray(pcl_returned.colors)
            points_list = np.asarray(pcl_returned.points)
            depth, image = point_cloud_to_depth(c1_k, points_list, color_list, image_shape[0], image_shape[1])

            if visualize:
                _, ax = plt.subplots(1,2)
                ax[0].imshow(image)
                ax[0].set_title("Color 1")
                ax[1].imshow(depth)
                ax[1].set_title("Depth 1")
                plt.show()

            # Camera intrinsics for square image
            Width =        np.shape(color_1)[1]
            Height =       np.shape(color_1)[1]
            PPX =          int((np.shape(color_1)[1]) / 2)
            PPY =          int((np.shape(color_1)[1]) / 2)
            Fx =           c1_k[0,0]
            Fy =           c1_k[1,1]
            square_c1_k = np.asarray([[Fx, 0, PPX], [0, Fy, PPY], [0, 0, 1]])

            # Change image to squares
            transformation = np.eye(4)
            depth_square, color_square = transform_depth(depth, image, c1_k, square_c1_k, transformation, Width, Height)

            if visualize:
                _, ax = plt.subplots(2,2)
                ax[0, 0].imshow(color_1)
                ax[0, 0].set_title("Color 1")
                ax[0, 1].imshow(depth_1)
                ax[0, 1].set_title("Depth 1")
                ax[1, 0].imshow(color_square)
                ax[1, 0].set_title("Depth 1 projected to Depth 2")
                ax[1, 1].imshow(depth_square)
                ax[1, 1].set_title("Color 1 projected to Color 2")
                plt.show()

            # Generate mesh in this camera frame
            print("Generating Mesh")
            mesh = generate_background_mesh(color_square, depth_square, PPX, PPY, Fx, Fy)

            w_T_c1 = np.linalg.inv(c1_T_w)

            center_of_cam = w_T_c1[:3,3]
            center_of_point = center

            radius = np.linalg.norm(center_of_cam - center_of_point)

            points_hemi = []
            for i in range(0,num_views):
                for j in range(0,10):
                    phi = (i / num_views) * 2 * math.pi
                    theta = (j / 10) * math.pi/4
                    point_hemi = sample_hemisphere_point_uniform(radius = radius, phi = phi, theta = theta)
                    point_hemi = point_hemi + (1,)
                    point_hemi = np.asarray(point_hemi)
                    point_hemi[2] = point_hemi[2] + radius
                    points_hemi.append(point_hemi)


            points_hemi = np.asarray(points_hemi)

            points_hemi_w = w_T_c1 @ points_hemi.transpose()
            points_hemi_w = points_hemi_w[:3].transpose()

            pt_hemi = o3d.geometry.PointCloud()
            pt_hemi.points = o3d.utility.Vector3dVector(points_hemi_w)
            points_hemi_w = pt_hemi.points
            
            points_np = np.array([
                w_T_c1[:3,3],
                w_T_c1[:3,3] + [0.001, 0.001, 0.001]
            ])

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points_np)

            point_cloud_hemi = o3d.geometry.PointCloud()
            point_cloud_hemi.points = o3d.utility.Vector3dVector(points_hemi_w)

            views_locations = np.asarray(point_cloud_hemi.points)

            # loop through viewpoints
            axis_meshs = []
            pcds_before_filtering = []
            for view in range(0,num_views):

                view_scores = []
                axis_mesh_t_list = []
                for view_increment in range(0,10):
                    view_loc = views_locations[(view * 10) + view_increment]
                    look_at_matrix = look_at(view_loc, center, up_vector=np.array([0, 0, 1]))
                    axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    axis_mesh_t = copy.deepcopy(axis_mesh).transform(look_at_matrix)
                    axis_meshs.append(axis_mesh_t)

                    points_view_center = np.array([
                        view_loc,
                        center
                    ])

                    point_view_center = o3d.geometry.PointCloud()
                    point_view_center.points = o3d.utility.Vector3dVector(points_view_center)
                    point_view_center += point_view_center

                    points_w_t = copy.deepcopy(points_w)
                    points_w_t.transform(np.linalg.inv(look_at_matrix))

                    color_list = np.asarray(points_w_t.colors)
                    points_list = np.asarray(points_w_t.points)
                    depth, image = point_cloud_to_depth(c1_k, points_list, color_list, image_shape[0], image_shape[1])

                    view_scores.append(np.count_nonzero(depth) / (np.shape(depth)[0] * np.shape(depth)[1]))

                difference_array = np.absolute(np.asarray(view_scores)- context_threshold)
                index_view = difference_array.argmin()

                view_loc = views_locations[(view * 10) + index_view]
                look_at_matrix = look_at(view_loc, center, up_vector=np.array([0, 0, 1]))
                axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                axis_mesh_t = copy.deepcopy(axis_mesh).transform(look_at_matrix)
                axis_meshs.append(axis_mesh_t)

                points_view_center = np.array([
                    view_loc,
                    center
                ])

                point_view_center = o3d.geometry.PointCloud()
                point_view_center.points = o3d.utility.Vector3dVector(points_view_center)
                point_view_center += point_view_center

                points_w_t = copy.deepcopy(points_w)
                points_w_t.transform(np.linalg.inv(look_at_matrix))

                color_list = np.asarray(points_w_t.colors)
                points_list = np.asarray(points_w_t.points)
                depth, image = point_cloud_to_depth(c1_k, points_list, color_list, image_shape[0], image_shape[1])

                transformation = np.eye(4)
                depth_square_new_view, color_square_new_view = transform_depth(depth, image, c1_k, square_c1_k, transformation, Width, Height)

                # Rotate mesh to new camera frame
                mesh_transformed = copy.deepcopy(mesh).transform(np.linalg.inv(c1_T_w))
                mesh_transformed = copy.deepcopy(mesh_transformed).transform(np.linalg.inv(look_at_matrix))

                # Capture depth map of generated mesh
                params = []
                HEIGHT, WIDTH = np.shape(depth_square_new_view)[0], np.shape(depth_square_new_view)[1]
                aspect_ratio = WIDTH / HEIGHT

                xfov = 2 * np.arctan(HEIGHT / (2 * Fy))

                params.append({
                    'cam_pos': [0, 0, 0], 'cam_lookat': [0, 0, 1], 'cam_up': [0, -1, 0],
                    'x_fov': xfov,  # End-to-end field of view in radians
                    'near': 0.1, 'far': 1000000,
                    'height': HEIGHT, 'width': WIDTH,
                    'is_depth': True,  # If false, output a ray displacement map, i.e. from the mesh surface to the camera center.
                })

                vertices = np.asarray(mesh_transformed.vertices).astype(np.float32) # An array of shape (num_vertices, 3) and type np.float32.
                faces = np.asarray(mesh_transformed.triangles).astype(np.uint32)  # An array of shape (num_faces, 3) and type np.uint32.

                depth_maps = m2d.mesh2depth(vertices, faces, params, empty_pixel_value=np.nan)

                # Filter out original depth map using new depth map
                inds = depth_square_new_view > depth_maps[0]
                depth_square_new_view[inds] = 0
                color_square_new_view[inds] = np.array([np.nan, np.nan, np.nan])

                if visualize:
                    _, ax = plt.subplots(2,2)
                    ax[0, 0].imshow(color_square)
                    ax[0, 0].set_title("Color 1")
                    ax[0, 1].imshow(depth_square)
                    ax[0, 1].set_title("Depth 1")
                    ax[1, 0].imshow(depth_maps[0])
                    ax[1, 0].set_title("Depth 1 projected to Depth 2")
                    ax[1, 1].imshow(depth_square_new_view)
                    ax[1, 1].set_title("Color 1 projected to Color 2")
                    plt.show()

                # Turn image into mask
                mask_3 = np.copy(color_square_new_view)

                mask_3[np.isnan(mask_3)] = -1
                black_pixels = np.where(
                    (mask_3[:, :, 0] != -1) & 
                    (mask_3[:, :, 1] != -1) & 
                    (mask_3[:, :, 2] != -1)
                )

                mask_3[black_pixels] = [0, 0, 0]
                white_pixels = np.where(
                    (mask_3[:, :, 0] == -1) & 
                    (mask_3[:, :, 1] == -1) & 
                    (mask_3[:, :, 2] == -1)
                )
                mask_3[white_pixels] = [255, 255, 255]
                mask = mask_3[:,:,0]

                # Fill small holes of input image
                color_square_new_view_filled = cv2.inpaint((color_square_new_view * 255).astype(np.uint8), mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)

                color_square_new_view_filled = cv2.cvtColor(color_square_new_view_filled, cv2.COLOR_BGR2RGB)
                color_square_new_view_filled_vis = np.copy(color_square_new_view_filled)

                color_square_new_view_filled = Image.fromarray(np.uint8(color_square_new_view_filled))
                byte_stream = BytesIO()
                color_square_new_view_filled.save(byte_stream, format='PNG')
                byte_array = byte_stream.getvalue()

                new_layer = np.ones((np.shape(color_square_new_view)[0],np.shape(color_square_new_view)[0])) * 255
                new_layer[np.asarray(mask_3)[:,:,0] == 255] = 0

                kernel = np.ones((5,5),np.uint8)
                new_layer = cv2.morphologyEx(new_layer, cv2.MORPH_CLOSE, kernel)
                for kernel_i in range(0,2):
                    new_layer = cv2.erode(new_layer, kernel) 

                new_array = np.vstack((np.asarray(mask_3).transpose(2,0,1), np.asarray([new_layer]))).transpose((1,2,0))

                color_square_new_view_filled_vis[np.where(new_layer==0)] = [0,255,0]

                if use_sd:
                    mask_for_sd = new_array[:,:,3]
                    mask_for_sd = np.asarray([mask_for_sd, mask_for_sd, mask_for_sd]).transpose(1,2,0)
                    mask_for_sd = np.where(mask_for_sd == 255, 1, 0)
                    mask_for_sd = np.where((mask_for_sd==0)|(mask_for_sd==1), mask_for_sd^1, mask_for_sd)
                    mask_for_sd = np.where(mask_for_sd == 1, 255, 0)
                    mask_for_sd = mask_for_sd.astype(int)
                    mask_for_sd = Image.fromarray(mask_for_sd.astype(np.uint8))
                
                cv2.imwrite('imgs/mask.png', new_array)
                mask = Image.open("imgs/mask.png")

                if use_sd:
                    image_sd = pipe(prompt=prompt, image=color_square_new_view_filled, mask_image=mask_for_sd).images[0]
                    image_sd = np.array(image_sd)
                    filled_in_imgs = [image_sd]
                else:
                    byte_stream = BytesIO()
                    mask.save(byte_stream, format='PNG')
                    mask_byte_array = byte_stream.getvalue()

                    app = Flask(__name__)
                    openai.api_key = OPENAI_KEY
                    
                    PROMPT = prompt

                    result_bool = None
                    while result_bool == None:
                        try:
                            #Make your OpenAI API request here
                            num = 1
                            filled_in_imgs = []
                            response = openai.Image.create_edit(
                                image=byte_array,
                                mask=mask_byte_array,
                                prompt=PROMPT,
                                n=num,
                                size="512x512"
                            )

                            for i in range(0,num):
                                image_url = response['data'][i]['url']
                                urllib.request.urlretrieve(image_url, write_file + "/extra/" + f"{view}_inpainted_{i}_{prompt}.png")
                                response_image = requests.get(image_url)
                                filled_in_img = np.array(Image.open(BytesIO(response_image.content)))
                                filled_in_imgs.append(filled_in_img)

                            result_bool = 1

                        except openai.error.Timeout as e:
                            #Handle timeout error, e.g. retry or log
                            print(f"OpenAI API request timed out: {e}")
                            pass
                        except openai.error.APIError as e:
                            #Handle API error, e.g. retry or log
                            print(f"OpenAI API returned an API Error: {e}")
                            pass
                        except openai.error.APIConnectionError as e:
                            #Handle connection error, e.g. check network or log
                            print(f"OpenAI API request failed to connect: {e}")
                            pass
                        except openai.error.InvalidRequestError as e:
                            #Handle invalid request error, e.g. validate parameters or log
                            print(f"OpenAI API request was invalid: {e}")
                            pass
                        except openai.error.AuthenticationError as e:
                            #Handle authentication error, e.g. check credentials or log
                            print(f"OpenAI API request was not authorized: {e}")
                            pass
                        except openai.error.PermissionError as e:
                            #Handle permission error, e.g. check scope or log
                            print(f"OpenAI API request was not permitted: {e}")
                            pass
                        except openai.error.RateLimitError as e:
                            #Handle rate limit error, e.g. wait or log
                            print(f"OpenAI API request exceeded rate limit: {e}")
                            pass

                filled_in_img_best = filled_in_imgs[0]
                incomplete_depth = depth_square_new_view

                output_depth, new_intrinsics, outputImgHeight, outputImgWidth, occ, normal = complete_depth(filled_in_img_best, incomplete_depth, square_c1_k, depth_complete, config)

                if save_intermediate:
                    _, ax = plt.subplots(2,4)
                    ax[0, 0].imshow(cv2.cvtColor(np.float32(color_square), cv2.COLOR_BGR2RGB))
                    ax[0, 1].imshow(cv2.cvtColor(np.float32(color_square_new_view), cv2.COLOR_BGR2RGB))
                    ax[1, 0].imshow(color_square_new_view_filled_vis)
                    ax[1, 1].imshow(filled_in_img_best)
                    ax[0, 2].imshow(occ)
                    ax[1, 2].imshow(normal)
                    ax[0, 3].imshow(incomplete_depth)
                    ax[1, 3].imshow(output_depth)

                    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                    plt.savefig(write_file + f"/grid_{view}.png", dpi=300)

                    cv2.imwrite(write_file + "/extra/" + f"{view}_color_square_new_view_filled_vis.png", color_square_new_view_filled_vis)
                    cv2.imwrite(write_file + "/extra/" + f"{view}_filled_in_img_best.png", filled_in_img_best)
                    cv2.imwrite(write_file + "/extra/" + f"{view}_occ.png", occ)
                    cv2.imwrite(write_file + "/extra/" + f"{view}_normal.png", normal)
                    np.save(write_file + "/extra/" + f"{view}_output_depth.npy", output_depth)
                    np.save(write_file + "/extra/" + f"{view}_intrinsics.npy", square_c1_k)
                    np.save(write_file + "/extra/" + f"{view}_c1_T_w.npy", c1_T_w)
                    np.save(write_file + "/extra/" + f"{view}_view_matrix.npy", look_at_matrix)


                filled_in_depth_resized = cv2.resize(output_depth, (640,640))
                filled_in_img_resized = cv2.resize(filled_in_img_best, (640,640))
                if save_intermediate:
                    cv2.imwrite(write_file + f"/img_{view}_0.png", cv2.cvtColor(filled_in_img_resized, cv2.COLOR_BGR2RGB))
                pcd1 = convert_realsense_rgb_depth_to_o3d_pcl(filled_in_img_resized, filled_in_depth_resized * 1000, square_c1_k)

                pcd1, ind = pcd1.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

                pcd1_w = copy.deepcopy(pcd1).transform(look_at_matrix)

                pcds_before_filtering.append(pcd1_w)

                if save_intermediate:
                    o3d.io.write_point_cloud(write_file + f"/pred_view_{view}.ply", pcd1_w)
            
            input_pcd = points_w_vis
            pcds = []
            for view in range(0,num_views):
                pcd = pcds_before_filtering[view]

                #filter original out
                dists = pcd.compute_point_cloud_distance(input_pcd)
                dists = np.asarray(dists)
                ind = np.where(dists > 0.002)[0]
                pcd_added = pcd.select_by_index(ind)

                pcds.append(pcd_added)

                if view == 0:
                    ind_new = np.where(dists <= 0.002)[0]
                    new_color_input_pcd = pcd.select_by_index(ind_new)


            all_added_pcd = filter_pointclouds(pcds)

            o3d.io.write_point_cloud(write_file + "/input.ply", new_color_input_pcd)
            o3d.io.write_point_cloud(write_file + "/added.ply", all_added_pcd)
            o3d.io.write_point_cloud(write_file + "/final.ply", new_color_input_pcd + all_added_pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script configuration")
    
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--use_sd", action="store_true", help="Use Stable Diffusion")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate steps")
    parser.add_argument("--use_gt_caption", action="store_true", help="Use ground truth caption")
    parser.add_argument("--use_auto_caption", action="store_true", help="Use automatic captioning")
    
    parser.add_argument("--num_views", type=int, default=10, help="Number of views")
    parser.add_argument("--context_threshold", type=float, default=0.4, help="Context threshold")

    parser.add_argument("--data_dir", type=str, default="hope-dataset/hope_video/", help="Directory containing data")
    parser.add_argument("--write_file_root", type=str, default="results/", help="Root directory for writing files")

    # Parsing arguments
    args = parser.parse_args()

    scene_frames = [
        [0, 32, 70, 182, 349], # Scene 0000
        [0, 86, 116, 153, 228], # Scene 0001
        [0, 86, 98, 115, 243], # Scene 0002
        [0, 62, 86, 113, 132], # Scene 0003
        [0, 22, 43, 66, 85], # Scene 0004
        [0, 31, 47, 148, 166], # Scene 0005
        [0, 15, 38, 87, 108], # Scene 0006
        [0, 14, 35, 85, 118], # Scene 0007
        [0, 22, 32, 43, 106], # Scene 0008
        [0, 26, 38, 52, 95], # Scene 0009
    ]

    if args.use_sd:
        from diffusers import StableDiffusionInpaintPipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
        pipe.to("cuda")
    else:
        pipe = None


    if args.use_auto_caption:
        from lavis.models import load_model_and_preprocess
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    else:
        model = None
        vis_processors = None

    scenes = os.listdir(args.data_dir)
    scenes.sort()

    depth_complete, config = make_depth_model()

    inference_hope(args.visualize, args.use_sd, args.save_intermediate, args.use_gt_caption, args.use_auto_caption, args.num_views, args.context_threshold, scene_frames, pipe, model, vis_processors, args.data_dir, scenes, args.write_file_root, depth_complete, config)


