import numpy as np

from preprocessing.fpfh import (
    compute_fpfh,
    get_initial_transform,
)

from preprocessing.shot import (
    extract_shot_keypoints,
    prepare_shot_registration_data,
)

from preprocessing.iss import (
    extract_iss_keypoints
)

from preprocessing.harris3d import (
    extract_harris3d,
    get_harris3d_initial_transform,
)

from preprocessing.sift3d import (
    extract_sift3d,
    get_sift3d_initial_transform,
)


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

        # FGR still needs descriptors even when using the full cloud.
        if registration_method == "FGR":

            source_fpfh = compute_fpfh(source)

            target_fpfh = compute_fpfh(target)

            pipeline_data["source_fpfh"] = source_fpfh
            pipeline_data["target_fpfh"] = target_fpfh

    # -------------------
    # NONE (Use whole point cloud, no initial alignment)
    # -------------------
    elif feature_method == "NONE":
        pass # Do nothing, just return the raw point clouds and identity transform
    # -------------------
    # FPFH
    # -------------------

    elif feature_method == "FPFH":

        source_fpfh = compute_fpfh(source)

        target_fpfh = compute_fpfh(target)

        initial_result = get_initial_transform(
            source,
            target,
            source_fpfh,
            target_fpfh,
            voxel_size=0.005
        )

        pipeline_data["source_fpfh"] = source_fpfh
        pipeline_data["target_fpfh"] = target_fpfh

        pipeline_data["initial_transform"] = (
            initial_result.transformation
        )

    # -------------------
    # ISS
    # -------------------

    elif feature_method == "ISS":

        source_iss = (
            extract_iss_keypoints(source)
        )

        target_iss = (
            extract_iss_keypoints(target)
        )

        source_fpfh = compute_fpfh(
            source_iss
        )

        target_fpfh = compute_fpfh(
            target_iss
        )

        initial_result = get_initial_transform(
            source_iss,
            target_iss,
            source_fpfh,
            target_fpfh,
            voxel_size=0.003
        )

        pipeline_data["registration_source"] = (
            source_iss
        )

        pipeline_data["registration_target"] = (
            target_iss
        )

        pipeline_data["source_fpfh"] = (
            source_fpfh
        )

        pipeline_data["target_fpfh"] = (
            target_fpfh
        )

        pipeline_data["initial_transform"] = (
            initial_result.transformation
        )

    # -------------------
    # HARRIS3D
    # -------------------

    elif feature_method in {"HARRIS", "HARRIS3D", "HARRIS_3D"}:

        source_harris = extract_harris3d(source)
        target_harris = extract_harris3d(target)

        pipeline_data["initial_transform"] = (
            get_harris3d_initial_transform(
                source_harris,
                target_harris,
            )
        )

        if registration_method == "FGR":
            pipeline_data["registration_source"] = source_harris
            pipeline_data["registration_target"] = target_harris
            pipeline_data["source_fpfh"] = compute_fpfh(source_harris)
            pipeline_data["target_fpfh"] = compute_fpfh(target_harris)

    # -------------------
    # SIFT3D
    # -------------------

    elif feature_method in {"SIFT", "SIFT3D", "SIFT_3D", "3DSIFT", "3D_SIFT"}:

        source_sift = extract_sift3d(source)
        target_sift = extract_sift3d(target)

        pipeline_data["initial_transform"] = (
            get_sift3d_initial_transform(
                source_sift,
                target_sift,
            )
        )

        if registration_method == "FGR":
            pipeline_data["registration_source"] = source_sift
            pipeline_data["registration_target"] = target_sift
            pipeline_data["source_fpfh"] = compute_fpfh(source_sift)
            pipeline_data["target_fpfh"] = compute_fpfh(target_sift)

    # -------------------
    # SHOT
    # -------------------

    elif feature_method == "SHOT":

        if registration_method == "FGR":

            source_shot = extract_shot_keypoints(source)
            target_shot = extract_shot_keypoints(target)

            source_shot, source_shot_features = (
                prepare_shot_registration_data(source, source_shot)
            )
            target_shot, target_shot_features = (
                prepare_shot_registration_data(target, target_shot)
            )

            pipeline_data["registration_source"] = source_shot
            pipeline_data["registration_target"] = target_shot
            pipeline_data["source_fpfh"] = source_shot_features
            pipeline_data["target_fpfh"] = target_shot_features

        else:

            source_shot = extract_shot_keypoints(source)
            target_shot = extract_shot_keypoints(target)

            pipeline_data["registration_source"] = source_shot
            pipeline_data["registration_target"] = target_shot
        
    else:

        raise ValueError(
            f"Unsupported feature method: "
            f"{feature_method}"
        )

    return pipeline_data
