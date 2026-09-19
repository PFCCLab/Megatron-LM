# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "megatron.core.datasets.helpers_cpp",
            sources=["megatron/core/datasets/helpers.cpp"],
            language="c++",
            # Deterministic build flags so the compiled .so is byte-identical
            # across rebuilds even when invoked directly (not via build.sh):
            #   -frandom-seed stabilizes symbol/gensym mangling,
            #   -ffile-prefix-map strips build paths from the object,
            #   -g0 drops debug info; link flags drop the random ELF build-id
            #   (-Wl,--build-id=none) and strip symbols (-Wl,-s).
            extra_compile_args=[
                "-O3",
                "-Wall",
                "-std=c++17",
                "-ffile-prefix-map=.=.",
                "-frandom-seed=helpers_cpp",
                "-g0",
            ],
            extra_link_args=["-Wl,--build-id=none", "-Wl,-s"],
            optional=True,
        )
    ]
)
