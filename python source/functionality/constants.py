"""Constants"""

# Program directory roots
# - the roots of each directory must be set manually
DATA_ROOT = "D:\\ML-Data\\Facial Keypoints Detection\\"
MODELS_ROOT = "D:\\ML-Models\\Facial Keypoints Detection\\"
PROGRAM_DATA_ROOT = "C:\\Users\\42077\\Documents\\FAV\\MPV\\semestralky\\3\\Facial Keypoints Detection\\python source\\data\\"
PROGRAM_FUNC_ROOT = "C:\\Users\\42077\\Documents\\FAV\\MPV\\semestralky\\3\\Facial Keypoints Detection\\python source\\functionality\\"
PROGRAM_ROOT = "../"

# Project global constants
IMAGE_SIZE = (96, 96)
Y_LENGTH = 30
BATCH_SIZE = 35

# Data mappings
FEATURES_MAPPING = {
    "left_eye_center_x": 0,
    "left_eye_center_y": 1,
    "right_eye_center_x": 2,
    "right_eye_center_y": 3,
    "left_eye_inner_corner_x": 4,
    "left_eye_inner_corner_y": 5,
    "left_eye_outer_corner_x": 6,
    "left_eye_outer_corner_y": 7,
    "right_eye_inner_corner_x": 8,
    "right_eye_inner_corner_y": 9,
    "right_eye_outer_corner_x": 10,
    "right_eye_outer_corner_y": 11,
    "left_eyebrow_inner_end_x": 12,
    "left_eyebrow_inner_end_y": 13,
    "left_eyebrow_outer_end_x": 14,
    "left_eyebrow_outer_end_y": 15,
    "right_eyebrow_inner_end_x": 16,
    "right_eyebrow_inner_end_y": 17,
    "right_eyebrow_outer_end_x": 18,
    "right_eyebrow_outer_end_y": 19,
    "nose_tip_x": 20,
    "nose_tip_y": 21,
    "mouth_left_corner_x": 22,
    "mouth_left_corner_y": 23,
    "mouth_right_corner_x": 24,
    "mouth_right_corner_y": 25,
    "mouth_center_top_lip_x": 26,
    "mouth_center_top_lip_y": 27,
    "mouth_center_bottom_lip_x": 28,
    "mouth_center_bottom_lip_y": 29
}

FEATURES_MAPPING_2 = {
    0: "left_eye_center_x",
    1: "left_eye_center_y",
    2: "right_eye_center_x",
    3: "right_eye_center_y",
    4: "left_eye_inner_corner_x",
    5: "left_eye_inner_corner_y",
    6: "left_eye_outer_corner_x",
    7: "left_eye_outer_corner_y",
    8: "right_eye_inner_corner_x",
    9: "right_eye_inner_corner_y",
    10: "right_eye_outer_corner_x",
    11: "right_eye_outer_corner_y",
    12: "left_eyebrow_inner_end_x",
    13: "left_eyebrow_inner_end_y",
    14: "left_eyebrow_outer_end_x",
    15: "left_eyebrow_outer_end_y",
    16: "right_eyebrow_inner_end_x",
    17: "right_eyebrow_inner_end_y",
    18: "right_eyebrow_outer_end_x",
    19: "right_eyebrow_outer_end_y",
    20: "nose_tip_x",
    21: "nose_tip_y",
    22: "mouth_left_corner_x",
    23: "mouth_left_corner_y",
    24: "mouth_right_corner_x",
    25: "mouth_right_corner_y",
    26: "mouth_center_top_lip_x",
    27: "mouth_center_top_lip_y",
    28: "mouth_center_bottom_lip_x",
    29: "mouth_center_bottom_lip_y"
}

LABEL_FLIP_MAPPING = {
    0: 2,
    2: 0,
    1: 3,
    3: 1,
    4: 8,
    8: 4,
    5: 9,
    9: 5,
    6: 10,
    10: 6,
    7: 11,
    11: 7,
    12: 16,
    16: 12,
    13: 17,
    17: 13,
    20: 20,
    22: 24,
    24: 22,
    23: 25,
    25: 23,
    26: 28,
    28: 26,
    27: 29,
    29: 27,
    14: 18,
    18: 14,
    15: 19,
    19: 15,
    21: 21
}
