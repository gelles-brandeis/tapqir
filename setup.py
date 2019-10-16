import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cosmos",
    version="0.0.1",
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
        "pyro-ppl==0.3.4", "tb-nightly",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
