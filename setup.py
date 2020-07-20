import os
import setuptools
import versioneer

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
VERSION = """
# This file is auto-generated with the version information during setup.py installation.
__version__ = '{}'
"""

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cosmos",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Yerdos Ordabayev",
    author_email="ordabayev@brandeis.edu",
    description="Bayesian analysis of the single-molecule image data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ordabayevy/cosmos",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy", "scipy", "pandas", "matplotlib", "tqdm", "scikit-learn", "future", "configparser",
        "torch", "torchvision", "cliff",
        "pyro-ppl", "tb-nightly", "voila", "ipyfilechooser",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        "console_scripts": ["cosmos=cosmos.app:main"],
        "cosmos.app": [
            "fit=cosmos.fit:Fit",
            "view=cosmos.view:View",
        ]
    }
)
