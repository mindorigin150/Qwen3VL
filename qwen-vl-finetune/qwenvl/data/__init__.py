import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

INTERLEAVED_CO3D_TRAIN = {
    "annotation_path": "/run/determined/NAS1/public/lixinyuan/interleaved-co3d/qwen3_sft/data/all_tasks_train.jsonl",
    "data_path": "",
}

INTERLEAVED_CO3D_VAL = {
    "annotation_path": "/run/determined/NAS1/public/lixinyuan/interleaved-co3d/qwen3_sft/data/all_tasks_val.jsonl",
    "data_path": "",
}

# --- view synthesis ---
VIEW_SYNTHESIS_TRAIN = {
    "annotation_path": "/run/determined/NAS1/public/lixinyuan/interleaved-co3d/qwen3_sft/data/view_synthesis_train.jsonl",
    "data_path": "",
}

VIEW_SYNTHESIS_VAL = {
    "annotation_path": "/run/determined/NAS1/public/lixinyuan/interleaved-co3d/qwen3_sft/data/view_synthesis_val.jsonl",
    "data_path": "",
}

# --- camera pose ---
CAMERA_POSE_TRAIN = {
    "annotation_path": "/run/determined/NAS1/public/lixinyuan/interleaved-co3d/qwen3_sft/data/camera_pose_train.jsonl",
    "data_path": "",
}

CAMERA_POSE_VAL = {
    "annotation_path": "/run/determined/NAS1/public/lixinyuan/interleaved-co3d/qwen3_sft/data/camera_pose_val.jsonl",
    "data_path": "",
}

# --- identity matching ---
IDENTITY_MATCHING_TRAIN = {
    "annotation_path": "/run/determined/NAS1/public/lixinyuan/interleaved-co3d/qwen3_sft/data/identity_matching_train.jsonl",
    "data_path": "",
}

IDENTITY_MATCHING_VAL = {
    "annotation_path": "/run/determined/NAS1/public/lixinyuan/interleaved-co3d/qwen3_sft/data/identity_matching_val.jsonl",
    "data_path": "",
}

# --- point matching ---
POINT_MATCHING_TRAIN = {
    "annotation_path": "/run/determined/NAS1/public/lixinyuan/interleaved-co3d/qwen3_sft/data/point_matching_train.jsonl",
    "data_path": "",
}

POINT_MATCHING_VAL = {
    "annotation_path": "/run/determined/NAS1/public/lixinyuan/interleaved-co3d/qwen3_sft/data/point_matching_val.jsonl",
    "data_path": "",
}

# --- depth estimation ---
DEPTH_ESTIMATION_TRAIN = {
    "annotation_path": "/run/determined/NAS1/public/lixinyuan/interleaved-co3d/qwen3_sft/data/depth_estimation_train.jsonl",
    "data_path": "",
}

DEPTH_ESTIMATION_VAL = {
    "annotation_path": "/run/determined/NAS1/public/lixinyuan/interleaved-co3d/qwen3_sft/data/depth_estimation_val.jsonl",
    "data_path": "",
}


# data_dict = {
#     "cambrian_737k": CAMBRIAN_737K,
#     "cambrian_737k_pack": CAMBRIAN_737K_PACK,
#     "mp_doc": MP_DOC,
#     "clevr_mc": CLEVR_MC,
#     "videochatgpt": VIDEOCHATGPT,
# }

data_dict = {
    "interleaved-co3d-train": INTERLEAVED_CO3D_TRAIN,
    "interleaved-co3d-val": INTERLEAVED_CO3D_VAL,
    "view-synthesis-train": VIEW_SYNTHESIS_TRAIN,
    "view-synthesis-val": VIEW_SYNTHESIS_VAL,
    "camera-pose-train": CAMERA_POSE_TRAIN,
    "camera-pose-val": CAMERA_POSE_VAL,
    "identity-matching-train": IDENTITY_MATCHING_TRAIN,
    "identity-matching-val": IDENTITY_MATCHING_VAL,
    "point-matching-train": POINT_MATCHING_TRAIN,
    "point-matching-val": POINT_MATCHING_VAL,
    "depth-estimation-train": DEPTH_ESTIMATION_TRAIN,
    "depth-estimation-val": DEPTH_ESTIMATION_VAL,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
