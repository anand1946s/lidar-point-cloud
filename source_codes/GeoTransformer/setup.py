import os
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

ext_cls = CUDAExtension if (torch.cuda.is_available() and (os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH'))) else CppExtension

setup(
    name='geotransformer',
    version='1.0.0',
    ext_modules=[
        ext_cls(
            name='geotransformer.ext',
            sources=[
                'geotransformer/extensions/extra/cloud/cloud.cpp',
                'geotransformer/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'geotransformer/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'geotransformer/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'geotransformer/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'geotransformer/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
