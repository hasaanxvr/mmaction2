import os
import subprocess

videos = os.listdir('demo/test_videos')



for video in videos:
    video_path = f'demo/test_videos/{video}'
    try:
        result = subprocess.run(["python",'demo/demo_skeleton.py', f'{video_path}', f'demo/test_results_skeleton/{video}', '--config', 'configs/skeleton/posec3d/custom_skeleton.py', '--checkpoint', 'checkpoints/posec3d_run2.pth', '--label-map', 'tools/data/skeleton/label_map_custom.txt'])
        
    except:
        print('could not run the subprocess')
    
    exit()