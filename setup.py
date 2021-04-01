import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cluster-jzenn",
    version="0.1.0",
    author="Johannes Zenn",
    author_email="johannes.zenn@gmail.com",
    description="A clustering package for k-Means Clustering and Spectral Clustering.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jzenn/cluster",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "matplotlib", "scipy"],
    python_requires=">=3.6",
)
