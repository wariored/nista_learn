import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nista_learn",
    version="0.0.1", # Post-release
    author="Cheikh Tidjane Konteye",
    author_email="cheikh@cheikhkonteye.com",
    description="A small machine learning package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wariored/nista_learn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
