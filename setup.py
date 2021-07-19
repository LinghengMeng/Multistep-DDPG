from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The Multistep-DDPG repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

setup(
    name='mddpg',
    py_modules=['mddpg'],
    version='0.1',
    install_requires=[
        'joblib',
        'gym',
        'numpy',
        'pybullet',
        'tensorflow>=1.8.0, <2.0',
        'pandas',
        'psutil',
        'tqdm'
    ],
    description="The Effect of Multi-step Methods on Overestimation in Deep Reinforcement Learning",
    author="Lingheng Meng",
)