import copy
import open3d as o3d


def visualize_registration(source, target, transformation):

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 1, 0])

    source_temp.transform(transformation)

    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        window_name="Registration Result"
    )