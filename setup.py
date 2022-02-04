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
    "sphinx-gallery",
    "sphinx-panels",
]
IN_COLAB = "COLAB_GPU" in os.environ

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
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "cmake>=3.18",
        "colorama",
        "funsor==0.4.1",
        "future",
        "ipyfilechooser",
        "ipympl",
        "ipywidgets",
        "matplotlib",
        "pandas",
        "pykeops==1.5",
        "pyro-ppl>=1.7.0",
        "pyyaml>=6.0",
        "scikit-learn",
        "scipy",
        "typer",
    ]
    + ([] if IN_COLAB else ["tensorboard", "voila"]),
    extras_require={
        "extras": EXTRAS_REQUIRE,
        "test": EXTRAS_REQUIRE + TEST_REQUIRE,
        "docs": DOCS_REQUIRE,
        "dev": EXTRAS_REQUIRE + TEST_REQUIRE + DOCS_REQUIRE,
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
