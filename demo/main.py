import argparse
import tempfile
import os
import base64
import cv2
import mmengine
import numpy as np
import torch
from datetime import datetime
from mmengine import DictAction


from mmaction.apis import (detection_inference, inference_recognizer,
                           init_recognizer, pose_inference)


from fastapi import FastAPI, HTTPException
import time
app = FastAPI()

# ------------------------------------------------------------------
def frame_extract(folder_path):
    files = os.listdir(folder_path)
    frame_paths = []
    frames = []
    for file in files:
        frame_path = f'{folder_path}/{file}'
        frame = cv2.imread(frame_path)

        frame_paths.append(frame_path)
        frames.append(frame)
    return frame_paths, frames


# -----------------------------------------------------------------

def decode_base64_to_image(encoded_string):
    decoded_data = base64.b64decode(encoded_string)
    np_data = np.frombuffer(decoded_data, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    
    return image

# -----------------------------------------------------------------

def decode_data(encoded_frames: list) -> list:
    frames = []
    for frame in encoded_frames:
        frames.append(decode_base64_to_image(frame))
        
    return frames


# ----------------------------------------------------------------
def save_frames(frames: list, save_dir: str):
    i = 0
    for frame in frames:
        cv2.imwrite(f'{save_dir}/frame_{i}.jpg', frame)
        i+=1

# ----------------------------------------------------------------
def save_video(frames: list, action_label: str, datetime, score: float):

    fps = 10
    video_name =f'{datetime}_{action_label}_{score}.mp4'

    dir_path = f'adl-agitation-results/{datetime}_{action_label}_{score}'
    os.makedirs(dir_path)
    save_frames(frames,dir_path)

    frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]
    out = cv2.VideoWriter(f'{dir_path}/{video_name}', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    for frame in frames:

        cv2.putText(frame, action_label, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)
    out.release()


# ----------------------------------------------------------------
config_path = 'configs/skeleton/posec3d/custom_skeleton.py'
checkpoint = 'checkpoints/posec3d_run2.pth'
det_config = 'demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py'
det_checkpoint = ('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth')
det_cat_id = 0
pose_config = 'demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'
pose_checkpoint = ('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth')
det_score_thr = 0.9

label_map_path = 'tools/data/skeleton/label_map_custom.txt'
device = 'cuda:0'



label_map = [x.strip() for x in open(label_map_path).readlines()]



@app.post("/adl-agitation-inference")
async def main(data: dict):

    total_start_time = time.time()
    
    #create a temporary directory to store the frames
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_dir_path = tmp_dir.name
    
    #decode the frames from base64 string to np array
    
    num_frames = len(data['encoded_frames'])
    if num_frames == 0:
        raise HTTPException(status_code=400, detail='Encoded Frames were not found. Length of the received frames is 0')
    
    
    try:
        frames = decode_data(data['encoded_frames'])
    except:
        raise HTTPException(status_code=422, detail='Could not decode the strings received. Please ensure that the sent strings are encoded properly')
    
    #save the frame to tmp_dir
    save_frames(frames, tmp_dir_path)
   
    #get the frame paths and frames from the frames saved
    frame_paths, frames_yeet = frame_extract(tmp_dir_path)
    
    
    h, w, _ = frames[0].shape
    
    
    start_time_human_det = time.time()
    # Get Human detection results.
    det_results, _ = detection_inference(det_config, det_checkpoint,
                                         frame_paths, det_score_thr,
                                         det_cat_id, device)
    print('Human Detections Time', time.time() - start_time_human_det)
    torch.cuda.empty_cache()

    start_time_pose_detection = time.time()
    # Get Pose estimation results.
    pose_results, pose_data_samples = pose_inference(pose_config,
                                                     pose_checkpoint,
                                                     frame_paths, det_results,
                                                     device)
    
    print('Skeletal Detections Time:', time.time() - start_time_pose_detection)

    torch.cuda.empty_cache()

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frames)
    num_person = max([len(x['keypoints']) for x in pose_results])

    num_keypoint = 17
    keypoint = np.zeros((num_frames, num_person, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_frames, num_person, num_keypoint),
                              dtype=np.float16)
    for i, poses in enumerate(pose_results):
        keypoint[i] = poses['keypoints']
        keypoint_score[i] = poses['keypoint_scores']

    fake_anno['keypoint'] = keypoint.transpose((1, 0, 2, 3))
    fake_anno['keypoint_score'] = keypoint_score.transpose((1, 0, 2))

    config = mmengine.Config.fromfile(config_path)
    #config.merge_from_dict(cfg_options)


    model = init_recognizer(config, checkpoint, device)

    start_time_model = time.time()
    result = inference_recognizer(model, fake_anno)

    print('PoseC3D Time:', time.time() - start_time_model)
    
    max_pred_index = result.pred_score.argmax().item()
    action_label = label_map[max_pred_index]

    print('Total Time: ', time.time() - total_start_time )
    #save_video
    save_video(frames,action_label, datetime.now(), result.pred_score[max_pred_index])



    
    #Write to a database etc
    NotImplemented
    
    

    
    tmp_dir.cleanup()


    return 200


