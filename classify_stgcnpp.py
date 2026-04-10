"""
classify_stgcnpp.py — Action classification with ST-GCN++ using pyskl

Loads a .npz skeleton file (e.g. from demo.py output) and classifies
the action using the ST-GCN++ model trained on NTURGB+D 120 (XSub).

Usage:
    cd MotionAGFormer
    python classify_stgcnpp.py \\
        --input ./demo/output/fall1/skeleton_3d_raw/keypoints.npz \\
        --config /path/to/pyskl/configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py \\
        --checkpoint /path/to/stgcnpp_ntu120_xsub_hrnet_j.pth

Input format:
    .npz file with key 'reconstruction':
    - (T, 25, 3) NTURGB+D 25 joints (recommended — no conversion needed)
    - (T, 17, 3) H36M 17 joints (will be converted to NTURGB+D via joint_converter)
    - (T, 17, 3) H36M 3D raw (camera-to-world) — will be converted to NTURGB+D

The script auto-detects the format and converts if necessary.
python classify_stgcnpp.py \
    --input ./demo/output/fall1/skeleton_3d_ntu/keypoints.npz \
    --config pyskl/configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py \
    --checkpoint /path/to/stgcnpp_ntu120_xsub_hrnet_j.pth \
    --gpu 0

"""

import argparse
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'pyskl'))

# ============================================================
# Python 3.12 compatibility fixes
# ============================================================
try:
    import importlib.machinery as _ml
    if not hasattr(_ml.FileFinder, 'find_module'):
        def _fs_find_mod(self, name, path=None):
            return None
        _ml.FileFinder.find_module = _fs_find_mod
except Exception:
    pass

try:
    import pkgutil as _pu
    if not hasattr(_pu, 'ImpImporter'):
        class _DummyImp:
            def __init__(self, path=None):
                self.path = path
            def find_module(self, n, p=None):
                return None
            def load_module(self, n):
                raise ImportError(n)
        _pu.ImpImporter = _DummyImp
except Exception:
    pass

# ============================================================
# Project imports
# ============================================================
from pyskl.apis import init_recognizer
from pyskl.datasets.pipelines import Compose
from mmcv.parallel import collate

# Joint converter for H36M → NTURGB+D
from joint_converter import h36m_to_nturgbd


# ============================================================
# Pipeline helpers (inline — avoids config dependency)
# ============================================================
def build_pipeline(cfg, clip_len=100, num_clips=10):
    """
    Build a test pipeline from cfg, overriding clip_len and num_clips.
    Skips GenSkeFeat (not needed — input is already 3D NTURGB+D).

    Returns a Compose pipeline.
    """
    pipeline_cfg = []
    for step in cfg.data.test.pipeline:
        step = dict(step)
        if step['type'] == 'UniformSample':
            step['clip_len'] = clip_len
            step['num_clips'] = num_clips
        elif step['type'] == 'GenSkeFeat':
            # Skip: we already have 3D keypoints (M,T,V,C), no feat gen needed.
            # GenSkeFeat would rename keypoint→j, breaking BatchNorm.
            continue
        elif step['type'] == 'PreNormalize2D':
            # Skip: raw 3D data, no 2D normalization needed.
            continue
        pipeline_cfg.append(step)
    return Compose(pipeline_cfg)


def detect_format(keypoints):
    """
    Detect skeleton format from shape.

    Returns:
        'nturgb+d': 25 joints
        'h36m':     17 joints (H36M format)
        'coco':     17 joints (COCO format, NOT supported for 3D ST-GCN++)
        Unknown
    """
    V = keypoints.shape[1]
    if V == 25:
        return 'nturgb+d'
    elif V == 17:
        return 'h36m'  # assume H36M (not COCO) — user should use skeleton_3d_ntu
    return 'unknown'


# ============================================================
# Main classification function
# ============================================================
def classify_raw_keypoints(keypoints, config_path, checkpoint_path=None,
                           device='cuda:0', clip_len=100, num_clips=10,
                           tta=True, frame_dir='memory_stream'):
    """
    Classify action from an in-memory numpy array.
    """
    print(f"[classify] Loaded memory tensor | shape: {keypoints.shape}")

    # --- Detect format ---
    fmt = detect_format(keypoints)
    print(f"[classify] Detected format: {fmt}")

    # --- Convert H36M 17 → NTURGB+D 25 if needed ---
    if fmt == 'h36m':
        print("[classify] Converting H36M 17 → NTURGB+D 25 joints...")
        keypoints = h36m_to_nturgbd(keypoints)
        print(f"[classify] After conversion: {keypoints.shape}")
        # Toạ độ đã được xoay và chuẩn hoá Y-up từ vis.py
        print(f"[classify] Hip Y at frame 0 = {keypoints[0, 0, 1]:.4f}")
    elif fmt != 'nturgb+d':
        raise ValueError(
            f"Unsupported skeleton format. "
            f"Expected 25 joints (NTURGB+D) or 17 joints (H36M). Got {fmt}."
        )

    T, V, C = keypoints.shape
    assert C == 3, f"Expected 3D keypoints (x,y,z), got C={C}"
    assert V == 25, f"Expected 25 joints, got V={V}"
    print(f"[classify] Skeleton: {T} frames, {V} joints, {C}D")

    # --- Load model ---
    print("[classify] Loading ST-GCN++ model...")
    if num_clips > 1 or tta:
        print(f"[classify] TTA enabled: {num_clips} clips")
    model = init_recognizer(config_path, checkpoint_path, device=device)
    print("[classify] Model loaded.")

    # Check num_classes from model
    if hasattr(model, 'cls_head') and hasattr(model.cls_head, 'fc_cls'):
        num_classes_model = model.cls_head.fc_cls.out_features
    elif hasattr(model, 'classifier'):
        num_classes_model = model.classifier.out_features
    else:
        # Try from config
        num_classes_model = model.cfg.model.cls_head.num_classes
    print(f"[classify] Model num_classes: {num_classes_model}")

    # --- Build pipeline ---
    cfg = model.cfg
    num_clips_actual = num_clips if (tta and num_clips > 1) else 1
    pipeline = build_pipeline(cfg, clip_len=clip_len, num_clips=num_clips_actual)

    # --- Prepare results dict ---
    # Pipeline expects (M, T, V, C) — add person dim (M=1) and reshape
    results = dict(
        keypoint=keypoints.astype(np.float32).reshape(1, T, V, C),
        total_frames=T,
        start_index=0,
        label=-1,
        frame_dir=frame_dir,
    )

    # --- Apply pipeline ---
    results = pipeline(results)
    # After pipeline: keypoint is torch.Tensor (num_clips, M, T, V, C)
    keypoint_tensor = collate([results], samples_per_gpu=1)['keypoint']
    keypoint_tensor = keypoint_tensor.to(device)

    print(f"[classify] Pipeline output shape: {keypoint_tensor.shape}")

    # --- Inference ---
    with torch.no_grad():
        scores = model(return_loss=False, **({'keypoint': keypoint_tensor}))

    print(f"[classify] Raw scores shape: {scores.shape}")
    # scores: (num_clips, num_classes) or (1, num_classes)
    if isinstance(scores, torch.Tensor):
        if num_clips_actual > 1:
            scores = scores.mean(dim=0)  # Average across clips
        else:
            scores = scores[0]  # Single clip
        scores = scores.cpu().numpy()  # (num_classes,)
    else:
        # Fallback for numpy
        if num_clips_actual > 1:
            scores = scores.mean(axis=0)
        else:
            scores = scores[0]

    num_classes = scores.shape[0]
    results_top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    return results_top, num_classes

def classify_skeleton(input_path, config_path, checkpoint_path=None,
                     device='cuda:0', clip_len=100, num_clips=10,
                     tta=True):
    data = np.load(input_path)
    keypoints = data['reconstruction']
    frame_dir = os.path.splitext(os.path.basename(input_path))[0]
    return classify_raw_keypoints(keypoints, config_path, checkpoint_path, device, clip_len, num_clips, tta, frame_dir)


# NTU60 XSub class names
NTU60_XSUB_CLASSES = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", "pickup",
    "throw", "sit down", "stand up", "clapping", "reading", "writing", "tear up paper",
    "wear jacket", "take off jacket", "wear a shoe", "take off a shoe", "wear on glasses",
    "take off glasses", "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving",
    "kicking something", "reach into pocket", "hopping", "jump up", "make a phone call",
    "play with phone/tablet", "typing on a keyboard", "pointing to something with finger",
    "taking a selfie", "check time (from watch)", "rub two hands together", "nod head/bow",
    "shake head", "wipe face", "salute", "put the palms together", "cross hands in front",
    "sneeze/cough", "staggering", "falling", "touch head (headache)", "touch chest (stomachache/heart pain)",
    "touch back (backache)", "touch neck (neckache)", "nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm", "punching/slapping other person", "kicking other person",
    "pushing other person", "pat on back of other person", "point finger at the other person",
    "hugging other person", "giving something to other person", "touch other person's pocket",
    "handshaking", "walking towards each other", "walking apart from each other"
]

# ============================================================
# Entry Point
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classify action from skeleton file using ST-GCN++'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to .npz skeleton file (key: reconstruction)')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to pyskl ST-GCN++ config file '
                             '(e.g. pyskl/configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py)')
    parser.add_argument('--checkpoint', '-w', type=str, default=None,
                        help='Path to ST-GCN++ checkpoint (.pth). '
                             'If not provided, uses random weights.')
    parser.add_argument('--gpu', '-g', type=str, default='0',
                        help='GPU id (default: 0)')
    parser.add_argument('--clip-len', type=int, default=100,
                        help='Frames per clip (default: 100)')
    parser.add_argument('--num-clips', type=int, default=10,
                        help='Number of clips for TTA (default: 10, set to 1 to disable TTA)')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable TTA (use num_clips=1)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top predictions to show (default: 5)')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("ST-GCN++ Action Classification")
    print(f"  Input:    {args.input}")
    print(f"  Config:   {args.config}")
    print(f"  Checkpoint: {args.checkpoint or '(random weights)'}")
    print(f"  Device:   {device}")
    print("=" * 60)

    try:
        top_results, num_classes = classify_skeleton(
            input_path=args.input,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=device,
            clip_len=args.clip_len,
            num_clips=args.num_clips,
            tta=not args.no_tta,
        )

        print(f"\nTop {args.top_k} predictions ({num_classes} classes):")
        print("-" * 40)
        for rank, (cls_id, score) in enumerate(top_results[:args.top_k], 1):
            class_name = NTU60_XSUB_CLASSES[cls_id] if cls_id < len(NTU60_XSUB_CLASSES) else f"Unknown {cls_id}"
            print(f"  #{rank:2d}  Class {cls_id:3d} ({class_name})  Score: {score:.4f}")
        print("-" * 40)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


