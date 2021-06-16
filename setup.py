from setuptools import setup, find_packages

setup(
    name='experimentator',
    author='Gabriel Van Zandycke',
    author_email="gabriel.vanzandycke@hotmail.com",
    url="https://github.com/gabriel-vanzandycke/experimentator",
    licence="LGPL",
    python_requires='>=3.6',
    description="My deep-learning experimentation framework",
    version='2.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['experiment=experimentator:main'],
    },
    install_requires=[
        "mlworkflow>=0.3.9",
        "dill",
        "numpy",
        "matplotlib",
        "ipywidgets",
        "tqdm",
        "IPython",
        "pandas"
    ],
)
