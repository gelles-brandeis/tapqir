import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cosmos-ordabayevy",
    version="0.0.1",
    author="Yerdos Ordabayev",
    author_email="ordabayev@brandeis.edu",
    description="Bayesian Image Classification for CoSMoS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gelles-brandeis/BayesianImage",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
