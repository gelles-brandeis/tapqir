# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import os

import setuptools

import versioneer

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
VERSION = """
# This file is auto-generated with the version information during setup.py installation.
__version__ = '{}'
"""

with open("README.rst", "r") as fh:
    long_description = fh.read()

# examples/tutorials
EXTRAS_REQUIRE = [
    "notebook",
]
# tests
TEST_REQUIRE = [
    "black[jupyter]",
    "flake8",
    "isort",
    "pytest",
    "pytest-xvfb",
]
# docs
DOCS_REQUIRE = [
    "IPython",
    "nbsphinx>=0.8.5",
    "pydata_sphinx_theme",
    "sphinx",
    "sphinx-autodoc-typehints",
    "sphinx-click",
    "sphinx-copybutton",
    "sphinx-gallery",
    "sphinx-panels",
]
DESKTOP_REQUIRE = ["voila"]

setuptools.setup(
    name="tapqir",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Yerdos Ordabayev",
    author_email="ordabayev@brandeis.edu",
    description="Bayesian analysis of co-localization single-molecule microscopy image data",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://tapqir.readthedocs.io",
    project_urls={
        "Documentation": "https://tapqir.readthedocs.io",
        "Source": "https://github.com/gelles-brandeis/tapqir",
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "colorama",
        "funsor==0.4.2",
        "future",
        "ipyevents",
        "ipyfilechooser",
        "ipympl",
        "ipywidgets",
        "matplotlib",
        "pandas",
        "protobuf>=3.9,<3.20",
        "pykeops>=2.0",
        "pyro-ppl==1.8.2",
        "pyyaml>=6.0",
        "scikit-learn",
        "scipy",
        "tensorboard",
        "torch==1.11.0",
        "typer",
    ],
    extras_require={
        "desktop": DESKTOP_REQUIRE,
        "extras": EXTRAS_REQUIRE,
        "test": EXTRAS_REQUIRE + TEST_REQUIRE,
        "docs": DOCS_REQUIRE,
        "dev": DESKTOP_REQUIRE + EXTRAS_REQUIRE + TEST_REQUIRE + DOCS_REQUIRE,
    },
    keywords="image-classification probabilistic-programming cosmos pyro",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "tapqir=tapqir.main:app",
            "tapqir-gui=tapqir.gui:app",
        ],
    },
)
