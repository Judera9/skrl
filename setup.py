from setuptools import setup, find_packages

setup(
    name="skrl",
    version="1.4.3",
    description="Modular and flexible library for reinforcement learning on PyTorch",
    author="Toni-SM",
    license="MIT License",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "tensorboard",
        "tqdm",
    ],
    # extras_require={
    #     "torch": ["torch>=1.10"],
    #     "all": ["torch>=1.10", "flax>=0.9.0", "optax"],
    #     "tests": ["pytest", "pytest-html", "pytest-cov", "hypothesis"],
    # },
    url="https://github.com/Toni-SM/skrl",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)