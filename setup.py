#!/usr/bin/env python
from setuptools import find_packages, setup  # type: ignore

setup(
    name="bibcat",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(),  # Other metadata like author, description, etc.
)
