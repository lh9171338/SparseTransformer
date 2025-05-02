#python3 setup.py install
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from distutils.sysconfig import get_config_vars


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)

if os.getenv("ENABLE_BF16", "0") == "1":
    print("BF16 is enabled")
    cxx_args = ["-g", "-DENABLE_BF16"]
    nvcc_args = ["-O2", "-DENABLE_BF16"]
else:
    cxx_args = ["-g"]
    nvcc_args = ["-O2"]

setup(
    name="sptr",
    version="1.1.0",
    author="lh9171338",
    author_email="lihao2015@whu.edu.cn",
    description="SparseTransformer CUDA ops",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/lh9171338/SparseTransformer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=["sptr"],
    ext_modules=[
        CUDAExtension("sptr.sptr_cuda", [
            "sptr/src/pointops_api.cpp",
            "sptr/src/attention/attention_cuda.cpp",
            "sptr/src/attention/attention_cuda_kernel.cu",
            "sptr/src/precompute/precompute.cpp",
            "sptr/src/precompute/precompute_cuda_kernel.cu",
            "sptr/src/rpe/relative_pos_encoding_cuda.cpp",
            "sptr/src/rpe/relative_pos_encoding_cuda_kernel.cu",
            ],
        extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args}
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
