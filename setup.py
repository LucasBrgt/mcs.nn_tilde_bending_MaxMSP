import subprocess

import setuptools

with open("README.md", "r") as readme:
    readme = readme.read()

with open("requirements.txt", "r") as requirements:
    requirements = requirements.read()

setuptools.setup(
    name="nn_tilde",
    version=subprocess.check_output([
        "git",
        "describe",
        "--abbrev=0",
    ]).strip().decode(),
    author="Antoine CAILLON",
    author_email="caillon@ircam.fr",
    description="Set of tools to create nn_tilde compatible models.",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=['nn_tilde'],
    package_dir={'nn_tilde': 'python_tools'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements.split("\n"),
    python_requires='>=3.7',
)
