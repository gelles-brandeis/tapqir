import os
import setuptools

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
VERSION = """
# This file is auto-generated with the version information during setup.py installation.
__version__ = '{}'
"""

# Find cosmos version.
for line in open(os.path.join(PROJECT_PATH, 'cosmos', '__init__.py')):
    if line.startswith('version_prefix = '):
        version = line.strip().split()[2][1:-1]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cosmos",
    version=version,
    author="Yerdos Ordabayev",
    author_email="ordabayev@brandeis.edu",
    description="Bayesian analysis of the single-molecule image data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ordabayevy/cosmos",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy", "scipy", "pandas", "matplotlib", "tqdm", "scikit-learn", "jupyter", "future", "configparser",
        "torch",
        "pyro-ppl", "tb-nightly",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
