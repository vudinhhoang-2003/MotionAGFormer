"""
Convert skeleton joint formats.

This module provides functions to convert between different skeleton joint formats.
Currently supports: H36M 17-joint (3D) -> NTURGB+D 25-joint (XSub format).
"""

import numpy as np


def h36m_to_nturgbd(h36m_keypoints):
    """
    Convert H36M 17-joint 3D skeleton to NTURGB+D 25-joint 3D skeleton.

    H36M 17 joints (0-based, as used in this codebase):
        0:  bottom_torso   1:  left_hip       2:  left_knee     3:  left_foot
        4:  right_hip      5:  right_knee     6:  right_foot    7:  center_torso
        8:  upper_torso    9:  neck_base     10: center_head  11: right_shoulder
       12:  right_elbow   13:  right_hand    14: left_shoulder 15: left_elbow
       16:  left_hand

    NTURGB+D 25 joints (0-based, XSub format):
        0:  base_of_spine        1:  middle_of_spine       2:  neck
        3:  head                 4:  left_shoulder         5:  left_elbow
        6:  left_wrist          7:  left_hand             8:  right_shoulder
        9:  right_elbow         10: right_wrist           11: right_hand
       12:  left_hip            13: left_knee             14: left_ankle
       15:  left_foot           16: right_hip             17: right_knee
       18:  right_ankle        19: right_foot            20: spine
       21:  tip_of_left_hand   22: left_thumb            23: tip_of_right_hand
       24:  right_thumb

    Mapping from H36M -> NTURGB+D:
        h36m[0]  -> ntu[0]   (bottom_torso  -> base_of_spine)
        h36m[7]  -> ntu[1]   (center_torso  -> middle_of_spine)
        h36m[9]  -> ntu[2]   (neck_base     -> neck)
        h36m[10] -> ntu[3]   (center_head   -> head)
        h36m[14] -> ntu[4]   (left_shoulder  -> left_shoulder)
        h36m[15] -> ntu[5]   (left_elbow     -> left_elbow)
        h36m[16] -> ntu[7]   (left_hand      -> left_hand)
        h36m[11] -> ntu[8]   (right_shoulder -> right_shoulder)
        h36m[12] -> ntu[9]   (right_elbow    -> right_elbow)
        h36m[13] -> ntu[11]  (right_hand     -> right_hand)
        h36m[1]  -> ntu[12]  (left_hip       -> left_hip)
        h36m[2]  -> ntu[13]  (left_knee      -> left_knee)
        h36m[3]  -> ntu[14]  (left_foot      -> left_ankle)
        h36m[4]  -> ntu[16]  (right_hip      -> right_hip)
        h36m[5]  -> ntu[17]  (right_knee     -> right_knee)
        h36m[6]  -> ntu[18]  (right_foot     -> right_ankle)
        ntu[6, 10, 15, 19, 20, 21, 22, 23, 24] are estimated from neighbors.

    Args:
        h36m_keypoints: np.ndarray of shape (..., 17, 3) or (..., 17, 4).
            The last two dims are (x, y, z) or (x, y, z, conf).

    Returns:
        np.ndarray of shape (..., 25, 3) or (..., 25, 4) — same format as input
        but with 25 joints.
    """
    h36m = np.asarray(h36m_keypoints, dtype=np.float32)

    # Detect input format (has confidence score or not)
    has_conf = h36m.shape[-1] == 4

    # Handle different input shapes: (T, 17, 3), (M, T, 17, 3), (17, 3), etc.
    orig_shape = h36m.shape
    flat = h36m.reshape(-1, orig_shape[-2], orig_shape[-1])  # (N, 17, C)

    N = flat.shape[0]
    C = flat.shape[-1]

    # Allocate output: (N, 25, C)
    ntu = np.zeros((N, 25, C), dtype=np.float32)

    # --- Direct mappings from H36M to NTURGB+D ---
    # Spine / torso chain
    ntu[:, 0, :3] = flat[:, 0, :3]           # bottom_torso  -> base_of_spine
    ntu[:, 1, :3] = flat[:, 7, :3]           # center_torso  -> middle_of_spine
    ntu[:, 2, :3] = flat[:, 9, :3]           # neck_base     -> neck
    ntu[:, 3, :3] = flat[:, 10, :3]          # center_head   -> head

    # Upper body — left arm
    ntu[:, 4, :3] = flat[:, 14, :3]           # left_shoulder
    ntu[:, 5, :3] = flat[:, 15, :3]          # left_elbow
    ntu[:, 7, :3] = flat[:, 16, :3]          # left_hand

    # Upper body — right arm
    ntu[:, 8, :3] = flat[:, 11, :3]           # right_shoulder
    ntu[:, 9, :3] = flat[:, 12, :3]          # right_elbow
    ntu[:, 11, :3] = flat[:, 13, :3]         # right_hand

    # Lower body — left leg
    ntu[:, 12, :3] = flat[:, 1, :3]           # left_hip
    ntu[:, 13, :3] = flat[:, 2, :3]           # left_knee
    ntu[:, 14, :3] = flat[:, 3, :3]          # left_foot (ankle level)

    # Lower body — right leg
    ntu[:, 16, :3] = flat[:, 4, :3]           # right_hip
    ntu[:, 17, :3] = flat[:, 5, :3]           # right_knee
    ntu[:, 18, :3] = flat[:, 6, :3]           # right_foot (ankle level)

    # --- Estimate missing NTURGB+D joints ---
    # ntu[6]  left_wrist      ≈ midpoint of left_elbow and left_hand
    ntu[:, 6, :3] = (ntu[:, 5, :3] + ntu[:, 7, :3]) / 2.0

    # ntu[10] right_wrist     ≈ midpoint of right_elbow and right_hand
    ntu[:, 10, :3] = (ntu[:, 9, :3] + ntu[:, 11, :3]) / 2.0

    # ntu[20] spine           ≈ midpoint of base_of_spine and middle_of_spine
    # H36M upper_torso (index 8) is a good reference for the upper spine
    ntu[:, 20, :3] = (ntu[:, 0, :3] + flat[:, 8, :3]) / 2.0

    # ntu[15] left_foot       ≈ left_ankle (H36M foot ≈ ankle)
    ntu[:, 15, :3] = ntu[:, 14, :3]

    # ntu[19] right_foot      ≈ right_ankle
    ntu[:, 19, :3] = ntu[:, 18, :3]

    # Hand tips and thumbs (not available in H36M → approximate with hand/wrist)
    # ntu[21] tip_of_left_hand  ≈ left_hand
    ntu[:, 21, :3] = ntu[:, 7, :3]
    # ntu[23] tip_of_right_hand ≈ right_hand
    ntu[:, 23, :3] = ntu[:, 11, :3]
    # ntu[22] left_thumb        ≈ left_hand
    ntu[:, 22, :3] = ntu[:, 7, :3]
    # ntu[24] right_thumb       ≈ right_hand
    ntu[:, 24, :3] = ntu[:, 11, :3]

    # --- Copy / estimate confidence scores ---
    if has_conf:
        # Direct confidence mappings
        ntu[:, 0, 3] = flat[:, 0, 3]          # bottom_torso
        ntu[:, 1, 3] = flat[:, 7, 3]          # center_torso
        ntu[:, 2, 3] = flat[:, 9, 3]          # neck_base
        ntu[:, 3, 3] = flat[:, 10, 3]          # center_head
        ntu[:, 4, 3] = flat[:, 14, 3]          # left_shoulder
        ntu[:, 5, 3] = flat[:, 15, 3]          # left_elbow
        ntu[:, 7, 3] = flat[:, 16, 3]          # left_hand
        ntu[:, 8, 3] = flat[:, 11, 3]          # right_shoulder
        ntu[:, 9, 3] = flat[:, 12, 3]          # right_elbow
        ntu[:, 11, 3] = flat[:, 13, 3]         # right_hand
        ntu[:, 12, 3] = flat[:, 1, 3]          # left_hip
        ntu[:, 13, 3] = flat[:, 2, 3]          # left_knee
        ntu[:, 14, 3] = flat[:, 3, 3]          # left_foot
        ntu[:, 16, 3] = flat[:, 4, 3]          # right_hip
        ntu[:, 17, 3] = flat[:, 5, 3]          # right_knee
        ntu[:, 18, 3] = flat[:, 6, 3]          # right_foot

        # Estimated: average of source joint confidences
        ntu[:, 6, 3]  = (ntu[:, 5, 3] + ntu[:, 7, 3]) / 2.0           # left_wrist
        ntu[:, 10, 3] = (ntu[:, 9, 3] + ntu[:, 11, 3]) / 2.0         # right_wrist
        ntu[:, 20, 3] = (ntu[:, 0, 3] + flat[:, 8, 3]) / 2.0         # spine
        ntu[:, 15, 3] = ntu[:, 14, 3]                                    # left_foot
        ntu[:, 19, 3] = ntu[:, 18, 3]                                    # right_foot
        ntu[:, 21, 3] = ntu[:, 7, 3]                                     # tip_of_left_hand
        ntu[:, 23, 3] = ntu[:, 11, 3]                                    # tip_of_right_hand
        ntu[:, 22, 3] = ntu[:, 7, 3]                                     # left_thumb
        ntu[:, 24, 3] = ntu[:, 11, 3]                                    # right_thumb

    # Reshape back to original leading dimensions
    out_shape = orig_shape[:-2] + (25,) + (C,)
    ntu = ntu.reshape(out_shape)

    return ntu


def ntu_label_order():
    """
    Return the NTURGB+D 25-joint label names in index order (0-based).

    Returns:
        list of 25 strings: joint names.
    """
    return [
        'base_of_spine',          # 0
        'middle_of_spine',        # 1
        'neck',                   # 2
        'head',                   # 3
        'left_shoulder',          # 4
        'left_elbow',             # 5
        'left_wrist',             # 6
        'left_hand',              # 7
        'right_shoulder',         # 8
        'right_elbow',            # 9
        'right_wrist',            # 10
        'right_hand',             # 11
        'left_hip',               # 12
        'left_knee',              # 13
        'left_ankle',             # 14
        'left_foot',              # 15
        'right_hip',              # 16
        'right_knee',             # 17
        'right_ankle',            # 18
        'right_foot',             # 19
        'spine',                  # 20
        'tip_of_left_hand',       # 21
        'left_thumb',             # 22
        'tip_of_right_hand',      # 23
        'right_thumb',            # 24
    ]