import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

def gen_video_kpts(video_path, det_dim=416, num_peroson=1, gen_output=False):
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tự động tải file trọng số YOLOv8m-pose.pt nếu chưa có trong máy
    model = YOLO('yolov8m-pose.pt')
    
    kpts_result = []
    scores_result = []
    
    # Loop over frames
    for ii in tqdm(range(video_length)):
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Áp dụng ByteTrack để theo dõi (tracker="bytetrack.yaml")
        results = model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)
        
        # Khung xương rỗng [17 khớp x 2 toạ độ]
        kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
        scores = np.zeros((num_peroson, 17), dtype=np.float32)
        
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints) > 0:
            r = results[0]
            if r.boxes.id is not None:
                track_ids = r.boxes.id.cpu().numpy()
                xy = r.keypoints.xy.cpu().numpy()
                conf = r.keypoints.conf.cpu().numpy()
                
                # Sắp xếp lại dữ liệu theo Track ID để tránh việc bị nhảy lộn xộn các xương khi nhiều người đổi chỗ
                sorted_indices = np.argsort(track_ids)
                
                for i in range(min(len(sorted_indices), num_peroson)):
                    idx = sorted_indices[i]
                    kpts[i] = xy[idx]
                    scores[i] = conf[idx]
            else:
                # Nếu chỉ vừa phát hiện mà tracker chưa kịp cấp ID
                xy = r.keypoints.xy.cpu().numpy()
                conf = r.keypoints.conf.cpu().numpy()
                for i in range(min(len(xy), num_peroson)):
                    kpts[i] = xy[i]
                    scores[i] = conf[i]
                
        kpts_result.append(kpts)
        scores_result.append(scores)
        
    cap.release()
    
    keypoints = np.array(kpts_result)
    scores_array = np.array(scores_result)
    
    # Transpose [video_length, num_peroson, 17, 2] -> [num_peroson, video_length, 17, 2] do input data của 3D Network yêu cầu
    keypoints = keypoints.transpose(1, 0, 2, 3) 
    scores_array = scores_array.transpose(1, 0, 2)
    
    return keypoints, scores_array
