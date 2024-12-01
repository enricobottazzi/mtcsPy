from setuptools import setup, find_packages

setup(
    name="mtcspy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'networkx @ git+https://github.com/enricobottazzi/networkx.git@main',
        'numpy==1.24.1',
        'pandas==2.2.2',
        'pytest==7.4.0',
        'PuLP==2.8.0',
        # Pin all transitive dependencies
        'six==1.16.0',
        'pytz==2024.1',
        'iniconfig==1.1.1',
        'tzdata==2024.1',
        'tomli==2.1.0',
        'zipp==3.21.0',
        'importlib-metadata==8.4.0',
        'more-itertools==10.5.0'
    ],
    python_requires='>=3.8',
    author="Enrico Bottazzi"
)