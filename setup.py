from setuptools import setup, find_packages

setup(
    name='experimentator',
    author='Gabriel Van Zandycke',
    author_email="gabriel.vanzandycke@hotmail.com",
    url="https://github.com/gabriel-vanzandycke/experimentator",
    licence="LGPL",
    python_requires='>=3.8',
    description="My deep-learning experimentation framework",
    version='2.2.5',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['experiment=experimentator:main'],
    },
    install_requires=[
        "mlworkflow>=0.4.0",
        "dill",
        "numpy",
        "matplotlib",
        "ipywidgets",
        "tqdm",
        "IPython",
        "pandas",
        "aleatorpy>=0.2.0",
        "python-constraint",
        "pyconfyg",
    ],
    extras_requires={
        "tensorflow": ["tensorflow>=2.4"],
        "wandb": ["wandb>=0.12.0"]
    }
)
