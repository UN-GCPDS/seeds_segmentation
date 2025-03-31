# filepath: /setup.py
from setuptools import setup, find_packages

setup(
    name="seeds_segmentation",
    version="0.0.0",
    description="Librería para segmentación de semillas",
    author="Camilo Pelaez Garcia",
    author_email="cpelaezg@unal.edu.co",
    license='Simplified BSD License',
    url="https://github.com/UN-GCPDS/seeds_segmentation",
    packages=find_packages(),
    install_requires=[
        'scikit-image',
        'matplotlib',
        'gdown',
        'opencv-python',
        'scikit-learn'
    ],
)