from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

classifiers = [
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Games/Entertainment :: Puzzle Games",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

install_requires = [
    "numpy==1.19.5",
    "tqdm==4.51.0",
    "gym==0.18.0",
    "stable_baselines3==1.1.0",
    "opencv_python=4.5.1.48",
    "matplotlib==3.4.2",
    "dataclasses==0.8",
    "Pillow==8.4.0",
]

setup(
    name="gym_simplifiedtetris",
    version="0.2.0",
    author="Oliver Overend",
    author_email="ollyoverend10@gmail.com",
    url="https://github.com/OliverOverend/gym-simplifiedtetris",
    description="Simplified Tetris environments compliant with OpenAI Gym's API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(where="gym_simplifiedtetris"),
    install_requires=install_requires,
    classifiers=classifiers,
    package_dir={"": "gym_simplifiedtetris"},
    keywords="tetris, gym, openai-gym, reinforcement-learning, research, reward-shaping",
)
