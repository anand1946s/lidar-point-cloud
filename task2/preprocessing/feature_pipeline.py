import numpy as np
import open3d as o3d
import copy

from config import VOXEL_SIZE, REGISTRATION_METHOD

from task1.preprocessing.fpfh import compute_fpfh, get_initial_transform
from task1.preprocessing.iss import extract_iss_keypoints
from task1.preprocessing.harris3d import extract_harris3d, get_harris3d_initial_transform
from task1.preprocessing.sift3d import extract_sift3d, get_sift3d_initial_transform
from task1.preprocessing.shot import extract_shot_keypoints, prepare_shot_registration_data


def prepare_registration_inputs(
    feature_method,
    registration_method,
    source,
    target
):
    pipeline_data = {
        "registration_source": source,
        "registration_target": target,
        "initial_transform": np.eye(4),
        "source_fpfh": None,
        "target_fpfh": None,
    }

    # -------------------
    # WHOLE SET
    # -------------------
    if feature_method == "WHOLE_SET":
        if registration_method == "FGR":
            source_fpfh = compute_fpfh(source, voxel_size=VOXEL_SIZE)
            target_fpfh = compute_fpfh(target, voxel_size=VOXEL_SIZE)
            pipeline_data["source_fpfh"] = source_fpfh
            pipeline_data["target_fpfh"] = target_fpfh

    # -------------------
    # NONE
    # -------------------
    elif feature_method == "NONE":
        pass

    # -------------------
    # FPFH
    # -------------------
    elif feature_method == "FPFH":
        source_fpfh = compute_fpfh(source, voxel_size=VOXEL_SIZE)
        target_fpfh = compute_fpfh(target, voxel_size=VOXEL_SIZE)
        initial_result = get_initial_transform(
            source,
            target,
            source_fpfh,
            target_fpfh,
            voxel_size=VOXEL_SIZE
        )
        pipeline_data["source_fpfh"] = source_fpfh
        pipeline_data["target_fpfh"] = target_fpfh
        pipeline_data["initial_transform"] = initial_result.transformation

    # -------------------
    # ISS
    # -------------------
    elif feature_method == "ISS":
        source_iss = extract_iss_keypoints(source)
        target_iss = extract_iss_keypoints(target)

        source_fpfh = compute_fpfh(source_iss, voxel_size=VOXEL_SIZE)
        target_fpfh = compute_fpfh(target_iss, voxel_size=VOXEL_SIZE)

        initial_result = get_initial_transform(
            source_iss,
            target_iss,
            source_fpfh,
            target_fpfh,
            voxel_size=VOXEL_SIZE
        )

        pipeline_data["initial_transform"] = initial_result.transformation
        if registration_method == "FGR":
            pipeline_data["registration_source"] = source_iss
            pipeline_data["registration_target"] = target_iss
            pipeline_data["source_fpfh"] = source_fpfh
            pipeline_data["target_fpfh"] = target_fpfh

    # -------------------
    # HARRIS3D
    # -------------------
    elif feature_method in {"HARRIS", "HARRIS3D", "HARRIS_3D"}:
        # Scale coordinates down by a factor of 10 so that Harris3D's hardcoded 1cm
        # search radius corresponds to a 10cm search radius in the original room-scale coordinates.
        scale_factor = 10.0
        source_scaled = copy.deepcopy(source).scale(1.0 / scale_factor, center=(0, 0, 0))
        target_scaled = copy.deepcopy(target).scale(1.0 / scale_factor, center=(0, 0, 0))

        source_harris_scaled = extract_harris3d(source_scaled)
        target_harris_scaled = extract_harris3d(target_scaled)

        # Scale keypoints back up to original space
        source_harris = source_harris_scaled.scale(scale_factor, center=(0, 0, 0))
        target_harris = target_harris_scaled.scale(scale_factor, center=(0, 0, 0))

        pipeline_data["initial_transform"] = get_harris3d_initial_transform(
            source_harris,
            target_harris,
            threshold=VOXEL_SIZE * 2.0
        )

        if registration_method == "FGR":
            pipeline_data["registration_source"] = source_harris
            pipeline_data["registration_target"] = target_harris
            pipeline_data["source_fpfh"] = compute_fpfh(source_harris, voxel_size=VOXEL_SIZE)
            pipeline_data["target_fpfh"] = compute_fpfh(target_harris, voxel_size=VOXEL_SIZE)

    # -------------------
    # SIFT3D
    # -------------------
    elif feature_method in {"SIFT", "SIFT3D", "SIFT_3D", "3DSIFT", "3D_SIFT"}:
        source_sift = extract_sift3d(source)
        target_sift = extract_sift3d(target)

        pipeline_data["initial_transform"] = get_sift3d_initial_transform(
            source_sift,
            target_sift,
            threshold=VOXEL_SIZE * 2.0
        )

        if registration_method == "FGR":
            pipeline_data["registration_source"] = source_sift
            pipeline_data["registration_target"] = target_sift
            pipeline_data["source_fpfh"] = compute_fpfh(source_sift, voxel_size=VOXEL_SIZE)
            pipeline_data["target_fpfh"] = compute_fpfh(target_sift, voxel_size=VOXEL_SIZE)

    # -------------------
    # SHOT
    # -------------------
    elif feature_method == "SHOT":
        # Scale coordinates down by a factor of 10 so that SHOT's hardcoded 2cm/5cm
        # search radii correspond to 20cm/50cm in the original room-scale coordinates.
        scale_factor = 10.0
        source_scaled = copy.deepcopy(source).scale(1.0 / scale_factor, center=(0, 0, 0))
        target_scaled = copy.deepcopy(target).scale(1.0 / scale_factor, center=(0, 0, 0))

        # Downsample keypoints in scaled space.
        # Use a voxel size of 0.02 (corresponds to 0.20m in room-scale coordinates)
        # to ensure a moderate number of keypoints (~2000), making descriptor extraction
        # and RANSAC registration extremely fast.
        kp_voxel_size = 0.02
        source_shot_keypoints_scaled = extract_shot_keypoints(source_scaled, voxel_size=kp_voxel_size)
        target_shot_keypoints_scaled = extract_shot_keypoints(target_scaled, voxel_size=kp_voxel_size)

        source_shot_clean_scaled, source_shot_features = prepare_shot_registration_data(
            source_scaled, source_shot_keypoints_scaled
        )
        target_shot_clean_scaled, target_shot_features = prepare_shot_registration_data(
            target_scaled, target_shot_keypoints_scaled
        )

        # Scale keypoints back up
        source_shot_clean = source_shot_clean_scaled.scale(scale_factor, center=(0, 0, 0))
        target_shot_clean = target_shot_clean_scaled.scale(scale_factor, center=(0, 0, 0))

        initial_result = get_initial_transform(
            source_shot_clean,
            target_shot_clean,
            source_shot_features,
            target_shot_features,
            voxel_size=VOXEL_SIZE
        )

        pipeline_data["initial_transform"] = initial_result.transformation
        if registration_method == "FGR":
            pipeline_data["registration_source"] = source_shot_clean
            pipeline_data["registration_target"] = target_shot_clean
            pipeline_data["source_fpfh"] = source_shot_features
            pipeline_data["target_fpfh"] = target_shot_features

    else:
        raise ValueError(f"Unsupported feature method: {feature_method}")

    return pipeline_data
