# Author: Adrien Corenflos

"""Install filterflow."""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='filterflow',
    version='0.1',
    description='Differentiable Particle Filtering.',
    author='Adrien Corenflos, James Thornton',
    author_email='adrien.corenflos@gmail.com',
    url='https://github.com/JTT94/filterflow',
    packages=find_packages(),
    install_requires=requirements,
)
