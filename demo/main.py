import argparse
import tempfile
import os
import base64
import cv2
import mmengine
import numpy as np
import torch
from mmengine import DictAction

from mmaction.apis import (detection_inference, inference_recognizer,
                           init_recognizer, pose_inference)


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('video', help='video file/url')
    parser.add_argument('out_filename', help='output filename')
    parser.add_argument(
        '--config',
        default=('configs/skeleton/posec3d/'
                 'slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py'),
        help='skeleton model config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu60_xsub_keypoint/'
                 'slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth'),
        help='skeleton model checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='the category id for human detection')
    parser.add_argument(
        '--pose-config',
        default='demo/demo_configs/'
        'td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--label-map',
        default='tools/data/skeleton/label_map_ntu60.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args

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




def decode_base64_to_image(encoded_string):
    decoded_data = base64.b64decode(encoded_string)
    np_data = np.frombuffer(decoded_data, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    
    return image



config = 'configs/skeleton/custom_skeleton.py'
checkpoint = 'checkpoints\posec3d_run1.pth'
det_config = 'demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py'
det_checkpoint = ('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth')
det_cat_id = 0
pose_config = 'td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'
pose_checkpoint = ('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth')
det_score_thr = 0.9

label_map = 'tools/data/skeleton/label_map_custom.txt'
device = 'cpu'


from fastapi import FastAPI, HTTPException
app = FastAPI()




@app.post("/adl-agitation-inference")
async def main(data: dict):
    

    #args = parse_args()
    

    # modify the paths etc
    
    tmp_dir = tempfile.TemporaryDirectory()
    #frame_paths, frames = frame_extract(args.video, args.short_side,
    #                                    tmp_dir.name)

    frame_paths, frames = frame_extract('checkpoints/test_frames')
    num_frame = len(frame_paths)
    h, w, _ = frames[0].shape
    
    import time
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
        total_frames=num_frame)
    num_person = max([len(x['keypoints']) for x in pose_results])

    num_keypoint = 17
    keypoint = np.zeros((num_frame, num_person, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_frame, num_person, num_keypoint),
                              dtype=np.float16)
    for i, poses in enumerate(pose_results):
        keypoint[i] = poses['keypoints']
        keypoint_score[i] = poses['keypoint_scores']

    fake_anno['keypoint'] = keypoint.transpose((1, 0, 2, 3))
    fake_anno['keypoint_score'] = keypoint_score.transpose((1, 0, 2))

    config = mmengine.Config.fromfile(config)
    #config.merge_from_dict(cfg_options)


    model = init_recognizer(config, checkpoint, device)
    start_time_model = time.time()


    result = inference_recognizer(model, fake_anno)
    print(result)

    print(time.time() - start_time_model)



    return 200


