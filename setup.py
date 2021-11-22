from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

setup(
    name="gym_simplifiedtetris",
    version="0.1.7",
    author="Oliver Overend",
    author_email="ollyoverend10@gmail.com",
    url="https://github.com/OliverOverend/gym-simplifiedtetris",
    description="Creates simplified Tetris environments for OpenAI Gym",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.7",
    packages=find_packages(where="gym_simplifiedtetris"),
    install_requires=[
        "numpy",
        "gym",
        "opencv-python",
        "imageio",
        "matplotlib",
        "pillow",
        "stable-baselines3",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Puzzle Games",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "gym_simplifiedtetris"},
    keywords="tetris, gym, openai-gym, reinforcement-learning, research",
)
