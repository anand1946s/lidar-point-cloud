"""Dataset boundary for registration experiments.

Registration code receives only ``CloudPair`` objects and never inspects a
dataset path, extension, or naming convention.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class CloudPair:
    source_id: str
    target_id: str
    source: o3d.geometry.PointCloud
    target: o3d.geometry.PointCloud
    ground_truth: Optional[np.ndarray] = None


class DatasetHandler:
    """Base class for a dataset adapter.

    A new dataset implements only identifier resolution/loading and its
    default pairs; algorithms deliberately have no dataset-specific hooks.
    """
    name: str

    def __init__(self, root: Path):
        self.root = Path(root)

    def load_pair(self, source_id: str, target_id: str) -> CloudPair:
        return CloudPair(source_id, target_id, self.load_cloud(source_id), self.load_cloud(target_id))

    def load_cloud(self, cloud_id: str) -> o3d.geometry.PointCloud:
        raise NotImplementedError

    def default_pairs(self) -> Iterable[tuple[str, str]]:
        return ()

    @staticmethod
    def _validate(cloud: o3d.geometry.PointCloud, path: Path):
        if not cloud.has_points():
            raise ValueError(f"Point cloud is empty or unreadable: {path}")
        return cloud
