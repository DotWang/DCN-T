from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pair_wise_distance',
    ext_modules=[
        CUDAExtension('pair_wise_distance_cuda', [
            'pair_wise_distance_cuda_source.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })