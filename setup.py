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
    "matplotlib",
]
# tests
TEST_REQUIRE = [
    "black",
    "flake8",
    "isort",
    "pytest",
    "pytest-qt",
    "pytest-xvfb",
]

setuptools.setup(
    name="tapqir",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Yerdos Ordabayev",
    author_email="ordabayev@brandeis.edu",
    description="Bayesian analysis of the single-molecule image data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ordabayevy/tapqir",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "cliff",
        "configparser",
        # "funsor @ git+git://github.com/pyro-ppl/funsor.git@master",
        "funsor==0.4.0",
        "future",
        "pandas",
        "pyqtgraph",
        "pyro-ppl @ git+git://github.com/pyro-ppl/pyro.git@dev",
        "PySide2",
        "scikit-learn",
        "scipy",
        "tb-nightly",
    ],
    extras_require={
        "extras": EXTRAS_REQUIRE,
        "test": EXTRAS_REQUIRE + TEST_REQUIRE,
        "dev": EXTRAS_REQUIRE
        + TEST_REQUIRE
        + [
            "nbsphinx",
            "sphinx",
            "sphinx_rtd_theme",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": ["tapqir=tapqir.commands.app:main"],
        "tapqir.app": [
            "config=tapqir.commands.config:Config",
            "fit=tapqir.commands.fit:Fit",
            "show=tapqir.commands.show:Show",
            "glimpse=tapqir.commands.glimpse:Glimpse",
            "elbo=tapqir.commands.elbo:ELBO",
            "matlab=tapqir.commands.matlab:Matlab",
        ],
    },
)
