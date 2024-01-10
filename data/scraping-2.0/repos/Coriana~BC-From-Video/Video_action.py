# NOTE: this is _not_ the original code of IDM!
# As such, while it is close and seems to function well,
# its performance might be bit off from what is reported
# in the paper.

from argparse import ArgumentParser
import pickle
import cv2
import numpy as np
import json
import torch as th
import ffmpeg 

from openai_vpt.agent import ENV_KWARGS, resize_image, AGENT_RESOLUTION, MineRLAgent


KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

MESSAGE = """
This script will take a video, predict actions for its frames and
and show them with a cv2 window.

Press any button the window to proceed to the next frame.
"""

# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
CAMERA_SCALER = 360.0 / 2400.0

def comma_sep(v, arr_str):
    return ','.join([act for act in arr_str.split(' ') if v[act][0]>0])
    
def action_pretty(a):
    cam=f"cam[{a['camera'][0][0]:+7.2f}, {a['camera'][0][1]:+7.2f}]"
    hot=[f"hot.{i}" for i in range(1,10) if a[f'hotbar.{i}'][0]>0]
    acts = comma_sep(a, 'use attack drop inventory')
    moves= comma_sep(a, 'forward back left right')
    mods = comma_sep(a, 'jump sneak sprint')
    return f"{cam} \t {acts} \t {moves} \t {mods} \t {','.join(hot)}"
    #print(action_pretty(v))

def cam_to_pix(c):
  return int(c/10.0 * 66.5) # Linear relationship(!) from earlier calibration tests
  
def main(video_path):
    model = 'foundation-model-1x.model'
    weights = '7_180p_bkup.weights'
   # print(MESSAGE)
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    policy_kwargs["img_shape"] = [320, 180, 3]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w,h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    mid_x, mid_y = int(w//2), int(h//2)
    frame_size = (640, 360)

    # Obtain frame size information using get() method
    # Initialize video writer object
    prev_kb, prev_buttons, new_kb, new_buttons, kb , buttons = [], [],[], [],[], [],
    started = 0
    framenum = 0
    cam_1 = 0
    cam_2 = 0
    FutureFrames = []
    out_path = video_path[:-4] # remove .mp4
    output = cv2.VideoWriter(out_path +'-predict.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)
    while(cap.isOpened()):
    
        ret, frame = cap.read()
        pov = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #cv2.imshow("MineRL IDM model predictions", pov)
        #cv2.waitKey(0)
        action = agent.get_action(dict(pov=pov))
        
        # print(action)
        action_send = []
        for y, (action_name, action) in enumerate(action.items()):
            
            if action_name == 'camera':
                cam_in = action[0]
                cam_1 = cam_in[:-1]     #drop 1/2
                cam_2 = cam_in[-1:]     #drop the other 1/2
                cam=f"{int(cam_1*7.2)}, {int(cam_2*7.2)}"
                #print(cam_in)
                action_send.append(cam)
            else:
                if action[0] == 1:  
                    action_send.append(1)
                    #print(action_name)
                    #kb.append(action_name)                            
                    cv2.putText(
                        frame,
                        f"{action_name}",
                        (10, 25 + y * 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 170, 0),
                        2
                    ) 
                else:
                    action_send.append(0)
                            
        print(action_send)                    
        pix_pitch, pix_yaw = cam_to_pix(cam_1), cam_to_pix(cam_2)
        frame = cv2.arrowedLine(frame, (mid_x,mid_y), (mid_x+pix_yaw, mid_y+pix_pitch), (255, 170, 0), 5)
                            
        #outframe = resize_image(frame[..., ::-1], frame_size)
        output.write(frame)
        #cv2.imwrite('frame'+ str(framenum) + '.png', frame[..., ::-1])
        #cv2.imshow("MineRL IDM model predictions", frame[..., ::-1])
        #cv2.waitKey(0)
            
    print(f"Fames left unprocessed at end of the video: {Num_frames}")
    cap.release()
    output.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    parser = ArgumentParser("Run IDM on MineRL recordings.")

    parser.add_argument("--video-path", type=str, required=True, help="Path to a .mp4 file (Minecraft recording).")

    args = parser.parse_args()

    main(args.video_path)
