from setuptools import setup, find_packages

setup(
    name='experimentator',
    author='Gabriel Van Zandycke',
    author_email="gabriel.vanzandycke@hotmail.com",
    url="https://github.com/gabriel-vanzandycke/experimentator",
    licence="LGPL",
    python_requires='>=3.6',
    description="My deep-learning experimentation framework",
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "mlworkflow",
        "dill",
        "numpy",
        "matplotlib",
        "ipywidgets",
        "tqdm",
        "IPython"
    ],
)
