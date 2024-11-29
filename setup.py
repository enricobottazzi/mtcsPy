from setuptools import setup, find_packages

setup(
    name="mtcspy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'networkx==3.2',
        'pandas==2.2.2',
        'pytest==7.4.0',
        'PuLP==2.8.0',
    ],
    author="Enrico Bottazzi",
)
