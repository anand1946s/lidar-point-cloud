"""Built-in dataset adapters.  Keep format and naming differences here."""
from pathlib import Path
import numpy as np
import open3d as o3d
from .base import DatasetHandler


class PlyDatasetHandler(DatasetHandler):
    suffix = ".ply"

    def load_cloud(self, cloud_id):
        requested = Path(str(cloud_id))
        path = requested if requested.is_file() else self.root / (requested.name if requested.suffix else f"{cloud_id}{self.suffix}")
        if not path.is_file():
            raise FileNotFoundError(f"{self.name} cloud not found: {path}")
        return self._validate(o3d.io.read_point_cloud(str(path)), path)


class StanfordBunnyDataset(PlyDatasetHandler):
    name = "Bunny"
    def default_pairs(self):
        return (("bun000", "bun045"), ("bun045", "bun090"), ("bun000", "bun315"))


class StanfordDragonDataset(PlyDatasetHandler):
    name = "Dragon"
    def default_pairs(self):
        return (("dragonStandRight_0", "dragonStandRight_24"),
                ("dragonStandRight_24", "dragonStandRight_48"),
                ("dragonStandRight_48", "dragonStandRight_72"),
                ("dragonStandRight_72", "dragonStandRight_96"))


class StanfordBuddhaDataset(PlyDatasetHandler):
    name = "Buddha"
    def default_pairs(self):
        return (("happySideRight_0", "happySideRight_24"),
                ("happySideRight_24", "happySideRight_48"),
                ("happySideRight_48", "happySideRight_72"),
                ("happySideRight_72", "happySideRight_96"))


class ThreeDMatchDataset(PlyDatasetHandler):
    name = "3DMatch"
    def default_pairs(self):
        return tuple((f"cloud_bin_{i}", f"cloud_bin_{i + 1}") for i in range(4))


class ScanNetDataset(PlyDatasetHandler):
    name = "ScanNet"
    def default_pairs(self):
        return (
            ("scene0027_02_vh_clean", "scene0027_00_vh_clean"),
            ("scene0027_01_vh_clean", "scene0027_00_vh_clean"),
            ("scene0024_02_vh_clean", "scene0024_00_vh_clean"),
            ("scene0046_02_vh_clean", "scene0046_00_vh_clean"),
        )


class PavinDataset(DatasetHandler):
    name = "PAVIN"

    def load_cloud(self, cloud_id):
        requested = Path(str(cloud_id))
        path = requested if requested.is_file() else self.root / (requested.name if requested.suffix else f"{cloud_id}.csv")
        if not path.is_file():
            raise FileNotFoundError(f"PAVIN cloud not found: {path}")
        skiprows = 0
        with path.open() as stream:
            for index, line in enumerate(stream):
                if "DATA" in line:
                    skiprows = index + 1
                    break
        points = np.loadtxt(path, skiprows=skiprows)
        points = np.atleast_2d(points)[:, :3]
        points = points[~np.all(points == 0, axis=1)]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        return self._validate(cloud, path)

    def default_pairs(self):
        return (("scan300", "scan400"), ("scan400", "scan500"),
                ("scan500", "scan600"), ("scan600", "scan700"))


DATASET_HANDLERS = {
    "BUNNY": StanfordBunnyDataset,
    "DRAGON": StanfordDragonDataset,
    "BUDDHA": StanfordBuddhaDataset,
    "3DMATCH": ThreeDMatchDataset,
    "3DMATCH2": ThreeDMatchDataset,
    "SCANNET": ScanNetDataset,
    "SCANNET2": ScanNetDataset,
    "PAVIN": PavinDataset,
}
