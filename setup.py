from setuptools import setup, find_packages

setup(
    name="mtcspy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'networkx==3.1',
        'pytest==7.4.0',
    ],
    author="Enrico Bottazzi",
)
