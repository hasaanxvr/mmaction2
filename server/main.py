import os
import base64
import cv2
import mmengine
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from server.utils import transform_adl_data
from server.database import database_connection
from mmaction.apis import (detection_inference, inference_recognizer,
                           init_recognizer, pose_inference)


from fastapi import FastAPI, HTTPException
import time
app = FastAPI()

# --- initiate a db connection for saving results ----

db_connection = database_connection()


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
    for encoded_frame in encoded_frames:
        decoded_frame = decode_base64_to_image(encoded_frame) 
        frames.append(decoded_frame)
        
    return frames


# ----------------------------------------------------------------
def save_frames(frames: list, save_dir: str):
    i = 0
    for frame in frames:
        cv2.imwrite(f'{save_dir}/frame_{i}.jpg', frame)
        i+=1

# ----------------------------------------------------------------
def save_video(frames: list, action_label: str, datetime, score: float):

    fps = 3
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
checkpoint = 'checkpoints/run3.pth'
#det_config = 'demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py'
#det_checkpoint = ('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
#                 'faster_rcnn_r50_fpn_2x_coco/'
#                 'faster_rcnn_r50_fpn_2x_coco_'
#                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth')

det_config = 'mmdetection/configs/yolox/yolox_s_8xb8-300e_coco.py'
det_checkpoint = 'mmdetection/configs/yolox/checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'

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
    #tmp_dir = tempfile.TemporaryDirectory()
    #tmp_dir_path = tmp_dir.name
    
    #decode the frames from base64 string to np array
    
    num_frames = len(data['encoded_frames'])
    if num_frames == 0:
        raise HTTPException(status_code=400, detail='Encoded Frames were not found. Length of the received frames is 0')
    
    
    try:
        frames = decode_data(data['encoded_frames'])
    except:
        raise HTTPException(status_code=422, detail='Could not decode the strings received. Please ensure that the sent strings are encoded properly')
    
    
    h, w, _ = frames[0].shape
    frame_loading_time = time.time() - total_start_time
    
    start_time_human_det = time.time()
    # Get Human detection results.
    det_results, _ = detection_inference(det_config, det_checkpoint,
                                         frames, det_score_thr,
                                         det_cat_id, device)
    
    end_time_human_det = time.time()
    
    torch.cuda.empty_cache()

    start_time_pose_detection = time.time()
    # Get Pose estimation results.
    pose_results, pose_data_samples = pose_inference(pose_config,
                                                     pose_checkpoint,
                                                     frames, det_results,
                                                     device)
    
    end_time_pose_detection = time.time() 
    

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

    end_time_model = time.time()

    #print('Save Frames: ', save_frames_end - save_frames_start)
    #print('Extract Frames: ', extract_frames_end - extract_frames_start)
    
    
    max_pred_index = result.pred_score.argmax().item()
    action_label = label_map[max_pred_index]



    print('Human Detections Time', end_time_human_det - start_time_human_det)
    print('Skeletal Detections Time:', end_time_pose_detection - start_time_pose_detection)
    print('PoseC3D Time:',end_time_model - start_time_model)
    print('Total Time: ', time.time() - total_start_time )


    # ----- save_video ------
    time_now = datetime.now()
    save_video(frames,action_label, time_now, result.pred_score[max_pred_index])


    ## ------ save csv for every class + score -------
    #df = pd.DataFrame()

    #df = pd.concat([df, pd.DataFrame([label_map]), pd.DataFrame([result.pred_score])], ignore_index=True)

    #df.to_csv(f'adl-agitation-results/{time_now}_{action_label}_{result.pred_score[max_pred_index]}/results.csv')
    
   


    ## -------- save csv for top N classes and atomic actions for analysis ------------
    N = 5
    result.pred_score = result.pred_score.cpu()

    atomic_actions = ['lie', 'sit', 'stand', 'walk']
    atomic_scores = [result.pred_score[18], result.pred_score[19], result.pred_score[20], result.pred_score[25]]

    atomic_df = pd.DataFrame({'Class Name': atomic_actions, 'Score': atomic_scores})
    atomic_df = atomic_df.sort_values(by='Score', ascending=False)

    top_n_indices = sorted(range(len(result.pred_score)), key=lambda i: result.pred_score[i], reverse=True)[:N] 
    top_n_scores = [result.pred_score[i] for i in top_n_indices] 
    top_n_labels = [label_map[i] for i in top_n_indices]

    df = pd.DataFrame({'Class Name': top_n_labels, 'Score': top_n_scores}) 
    
    empty_df = pd.DataFrame({'Class Name': [''] * 3, 'Score': [''] * 3})

    df_combined = pd.concat([df, empty_df, atomic_df])
    
    df_combined.to_csv(f'adl-agitation-results/{time_now}_{action_label}_{result.pred_score[max_pred_index]}/results_top5.csv', index = False)
    
    
    
    ## --------Write to database ---- 
    #data = transform_adl_data(df, atomic_df)
    #db_connection.write_to_database(data)
    
    


    
    #tmp_dir.cleanup()


    return 200


