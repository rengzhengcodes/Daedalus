from setuptools import setup, find_packages
import os


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="daedalus",
    version="0.1",
    description="Daedalus automatic accelerator design tool framework.",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
    keywords="accelerator hardware energy estimation",
    author="Reng Zheng, Luc Gaitskell",
    author_email="rengz@mit.edu, lucg@mit.edu",
    license="MIT",
    packages=["daedalus"],
    install_requires=[
        "pyYAML >= 1.1",
        "pyfiglet",
        "ruamel.yaml >= 0.17.20",
        "deepdiff >= 6.2.3",
        "Jinja2 >= 3.1.3",
        "timeloopfe >= 0.0.1",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)