import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.yolov8_pose.gen_kpts import gen_video_kpts as yolo_pose
import os 
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy

sys.path.append(os.getcwd())
from demo.lib.utils import normalize_screen_coordinates, camera_to_world
from model.MotionAGFormer import MotionAGFormer

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    # Cố định khung quan sát trong Thế giới (Mét)
    # Vì người cao ~1.7m và có thể di chuyển quanh tâm, ta để lề rộng một chút
    ax.set_xlim3d([-1.5, 1.5])
    ax.set_ylim3d([-1.5, 1.5])
    ax.set_zlim3d([0, 2.0]) # Sàn là 0, đỉnh là 2 mét

    # Vẽ mặt sàn Grid để mắt người có mốc tham chiếu
    x_grid, y_grid = np.meshgrid(np.linspace(-1.5, 1.5, 5), np.linspace(-1.5, 1.5, 5))
    ax.plot_wireframe(x_grid, y_grid, np.zeros_like(x_grid), color='lightgrey', alpha=0.3, lw=0.5)

    ax.set_xlabel('X (m)', fontsize=8)
    ax.set_ylabel('Y (m)', fontsize=8)
    ax.set_zlabel('Z (m)', fontsize=8)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in range(len(I)):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=3, color = lcolor if LR[i] else rcolor)

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)


def get_pose2D(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    keypoints, scores = yolo_pose(video_path, num_peroson=1)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    
    # Add conf score to the last dim
    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir + 'keypoints.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)


def img2video(video_path, output_dir, action_name=None, predicted_score=None):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 25 # Fallback

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Đọc ảnh từ thư mục 'all/' (Nơi chứa ảnh ghép Input + Reconstruction)
    names = sorted(glob.glob(os.path.join(output_dir + 'all/', '*.png')))
    if not names:
        print(f"[Error] No images found in {output_dir + 'all/'} to generate video.")
        return

    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    # Tên video output
    video_basename = os.path.basename(video_path).split('.')[0]
    out_video_path = os.path.join(output_dir, f"{video_basename}_demo.mp4")
    videoWrite = cv2.VideoWriter(out_video_path, fourcc, fps, size) 

    print(f"Combining {len(names)} images into video: {out_video_path}")
    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()
    print(f"Video generated successfully at: {out_video_path}")


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def resample(n_frames):
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints):
    clips = []
    n_frames = keypoints.shape[1]
    if n_frames <= 243:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 243:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
    return clips, downsample

def turn_into_h36m(keypoints):
    new_keypoints = np.zeros_like(keypoints)
    new_keypoints[..., 0, :] = (keypoints[..., 11, :] + keypoints[..., 12, :]) * 0.5
    new_keypoints[..., 1, :] = keypoints[..., 11, :]
    new_keypoints[..., 2, :] = keypoints[..., 13, :]
    new_keypoints[..., 3, :] = keypoints[..., 15, :]
    new_keypoints[..., 4, :] = keypoints[..., 12, :]
    new_keypoints[..., 5, :] = keypoints[..., 14, :]
    new_keypoints[..., 6, :] = keypoints[..., 16, :]
    new_keypoints[..., 8, :] = (keypoints[..., 5, :] + keypoints[..., 6, :]) * 0.5
    new_keypoints[..., 7, :] = (new_keypoints[..., 0, :] + new_keypoints[..., 8, :]) * 0.5
    new_keypoints[..., 9, :] = keypoints[..., 0, :]
    new_keypoints[..., 10, :] = (keypoints[..., 1, :] + keypoints[..., 2, :]) * 0.5
    new_keypoints[..., 11, :] = keypoints[..., 6, :]
    new_keypoints[..., 12, :] = keypoints[..., 8, :]
    new_keypoints[..., 13, :] = keypoints[..., 10, :]
    new_keypoints[..., 14, :] = keypoints[..., 5, :]
    new_keypoints[..., 15, :] = keypoints[..., 7, :]
    new_keypoints[..., 16, :] = keypoints[..., 9, :]

    return new_keypoints


def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data

@torch.no_grad()
def get_pose3D(video_path, output_dir):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 16, 3, 128, 512, 3
    args.mlp_ratio, args.act_layer = 4, nn.GELU
    args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
    args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    args.hierarchical = False
    args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 2, 1
    args.use_tcn, args.graph_only = False, False
    args.n_frames = 243
    args = vars(args)

    ## Reload 
    model = nn.DataParallel(MotionAGFormer(**args))
    if torch.cuda.is_available():
        model = model.cuda()

    # Put the pretrained model of MotionAGFormer in 'checkpoint/'
    model_path = sorted(glob.glob(os.path.join('checkpoint', 'motionagformer-b-h36m.pth.tr')))[0]

    pre_dict = torch.load(model_path, map_location='cpu' if not torch.cuda.is_available() else None, weights_only=False)
    model.load_state_dict(pre_dict['model'], strict=True)

    model.eval()

    ## input
    keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']
    # keypoints = np.load('demo/lakeside3.npy')
    # keypoints = keypoints[:240]
    # keypoints = keypoints[None, ...]
    # keypoints = turn_into_h36m(keypoints)
    

    clips, downsample = turn_into_clips(keypoints)


    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ## 3D
    print('\nGenerating 2D pose image...')
    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        if img is None:
            continue
        img_size = img.shape

        input_2D = keypoints[0][i]

        image = show2Dpose(input_2D, copy.deepcopy(img))

        output_dir_2D = output_dir +'pose2D/'
        os.makedirs(output_dir_2D, exist_ok=True)
        cv2.imwrite(output_dir_2D + str(('%04d'% i)) + '_2D.png', image)

    
    print('\nGenerating 3D pose...')
    skeleton_3d_seq = []
    global_foot_anchor = None
    initial_2d_hip_y = None  # Vị trí hông 2D mốc để tính độ rơi
    
    for idx, clip in enumerate(clips):
        input_2D_raw = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0]) 
        
        # Lấy độ rơi từ 2D: Hông (khớp 0), trục Y (index 1)
        hip_2d_y = input_2D_raw[:, :, 0, 1] # (B, T)
        if initial_2d_hip_y is None:
            initial_2d_hip_y = hip_2d_y[0, 0]
        
        # Độ sụt pixel 2D (dương = rơi xuống)
        descent_2d = hip_2d_y - initial_2d_hip_y

        input_2D_aug = flip_data(input_2D_raw)
        input_2D = torch.from_numpy(input_2D_raw.astype('float32'))
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32'))
        if torch.cuda.is_available():
            input_2D = input_2D.cuda()
            input_2D_aug = input_2D_aug.cuda()

        output_3D_non_flip = model(input_2D) 
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]
            descent_2d = descent_2d[:, downsample]

        # 1. Phóng đại (Rescaling): Biến Hip-height từ 0.23m (tí hon) thành ~0.9m (người thực)
        # 0.9 / 0.2317 ≈ 3.88
        output_3D = output_3D * 3.88

        # 2. Root Recovery: Bơm độ rơi 2D vào 3D Camera-space (Y+ là xuống)
        # Hệ số quy đổi 2D-screen sang 3D-meters ≈ 3.5
        recovery_factor = 3.5
        output_3D[:, :, :, 1] += torch.from_numpy(descent_2d).to(output_3D.device).unsqueeze(-1) * recovery_factor

        # 3. Fixed Foot-Anchor: Neo vào mốc bàn chân của frame đầu tiên
        foot_mid_per_frame = (output_3D[:, :, 3, :] + output_3D[:, :, 6, :]) / 2.0
        if global_foot_anchor is None:
            global_foot_anchor = foot_mid_per_frame[0, 0, :].clone()
            print(f"  [Root-Recovery] Initial Hip height: {output_3D[0,0,0,1].item():.4f}")
            print(f"  [Root-Recovery] Foot Anchor set at: {global_foot_anchor.cpu().numpy()}")
        
        output_3D = output_3D - global_foot_anchor[None, None, None, :]

        # ====================================================================
        # World-Space Rotation & AI Alignment
        # ====================================================================
        rot_val = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot_q = np.array(rot_val, dtype='float32')
        output_3D_np = output_3D[0].cpu().detach().numpy()  # (T, V, 3)
        # Xoay sang World-space (Z-up)
        world_3d_all = camera_to_world(output_3D_np, R=rot_q, t=0)

        for j, post_out_world in enumerate(world_3d_all):
            # 1. Chuẩn bị cho AI: Swap trục (X, Z, Y) -> AI-Y là trục thẳng đứng (Up)
            # NTU-RGB+D AI mong đợi Y-up. Trong World-space, Z là Up.
            # Sau khi swap: ai_pose[:, 1] (Y) = world_3d[:, 2] (Z_up)
            ai_pose = post_out_world[:, [0, 2, 1]].copy()
            skeleton_3d_seq.append(ai_pose)
            
            # DEBUG: Theo dõi trục Rơi thực tế (AI-Y)
            if j % 20 == 0:
                print(f"  [DEBUG-AI] Frame {idx*243+j}: Hip AI-Y = {ai_pose[0, 1]:.4f}")
        
            # 2. Chuẩn bị cho Visualization: Giữ hệ tọa độ Tuyệt đối (Mét)
            # Quan trọng: KHÔNG dùng min(Z) hay max_value nữa để thấy được quỹ đạo rơi thực
            post_out = post_out_world.copy()

            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05) 
            ax = plt.subplot(gs[0], projection='3d')
            show3Dpose(post_out, ax)

            output_dir_3D = output_dir +'pose3D/'
            os.makedirs(output_dir_3D, exist_ok=True)
            str(('%04d'% (idx * 243 + j)))
            plt.savefig(output_dir_3D + str(('%04d'% (idx * 243 + j))) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
            plt.close(fig)
        

        
    print('Generating 3D pose successful!')

    action_name, predicted_score = None, None
    print('\nStarting Action Classification (In-Memory)...')
    try:
        from classify_stgcnpp import classify_raw_keypoints, NTU60_XSUB_CLASSES
        skeleton_3d_tensor = np.array(skeleton_3d_seq)
        config_path = 'pyskl/configs/stgcn++/stgcn++_ntu60_xsub_3dkp/j.py'
        checkpoint_path = 'pretrained/j.pth'
        
        top_results, _ = classify_raw_keypoints(
            keypoints=skeleton_3d_tensor, config_path=config_path, checkpoint_path=checkpoint_path,
            device='cuda:0' if torch.cuda.is_available() else 'cpu', clip_len=100, num_clips=10, tta=True
        )
        predicted_class_id = top_results[0][0]
        predicted_score = top_results[0][1]
        action_name = NTU60_XSUB_CLASSES[predicted_class_id] if predicted_class_id < len(NTU60_XSUB_CLASSES) else "Unknown"
        print(f"\n>>>> ACTION DETECTED: {action_name.upper()} (Score: {predicted_score:.4f}) <<<<\n")
        
        # In Top 5 để debug xem FALLING đang xếp hạng mấy
        print("--- TOP 5 PREDICTIONS ---")
        for rank, (cls_id, score) in enumerate(top_results[:5]):
            cls_name = NTU60_XSUB_CLASSES[cls_id] if cls_id < len(NTU60_XSUB_CLASSES) else "Unknown"
            print(f"  #{rank+1}: {cls_name} ({score:.4f})")
        print("-------------------------")
    except Exception as e:
        print(f"Action classification failed: {e}")

    ## all
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = cv2.imread(image_2d_dir[i])
        image_3d = cv2.imread(image_3d_dir[i])

        ## logo
        image_3d = image_3d[10:-10, 10:-10]

        ## resize
        image_2d = cv2.resize(image_2d, (960, 540))
        image_3d = cv2.resize(image_3d, (960, 540))

        ## concat
        image_all = np.concatenate([image_2d, image_3d], axis=1)

        ## text
        font = cv2.FONT_HERSHEY_SIMPLEX
        if action_name:
            # Chuyển tên hành động thành màu xanh nếu là Falling
            color = (0, 255, 0) if action_name.lower() == 'falling' else (0, 255, 255)
            cv2.putText(image_all, f'Action: {action_name.upper()} ({predicted_score:.2f})', (20, 50), font, 1, color, 2, cv2.LINE_AA)
        
        cv2.putText(image_all, 'Input', (200, 50), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_all, 'Reconstruction', (1200, 50), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        output_dir_all = output_dir +'all/'
        os.makedirs(output_dir_all, exist_ok=True)
        cv2.imwrite(output_dir_all + str(('%04d'% i)) + '_all.png', image_all)

    print('Generating demo successful!')
        
    return action_name, predicted_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = './demo/video/' + args.video
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = './demo/output/' + video_name + '/'

    get_pose2D(video_path, output_dir)
    action_name, predicted_score = get_pose3D(video_path, output_dir)
    img2video(video_path, output_dir, action_name, predicted_score)
    print('Generating demo successful!')


