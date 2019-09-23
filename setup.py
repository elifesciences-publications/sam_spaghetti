#!/usr/bin/env python
# -*- coding: utf-8 -*-

# {# pkglts, pysetup.kwds
# format setup arguments

from setuptools import setup, find_packages


short_descr = "SAM Sequence Primordia Alignment, GrowtH Estimation, Tracking & Temporal Indexation"
readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')


# find version number in src/sam_spaghetti/version.py
version = {}
with open("src/sam_spaghetti/version.py") as fp:
    exec(fp.read(), version)


setup_kwds = dict(
    name='sam_spaghetti',
    version=version["__version__"],
    description=short_descr,
    long_description=readme + '\n\n' + history,
    author="Guillaume Cerutti, ",
    author_email="guillaume.cerutti@inria.fr, ",
    url='https://github.com/Guillaume Cerutti/sam_spaghetti',
    license='cecill-c',
    zip_safe=False,

    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        ],
    tests_require=[
        "coverage",
        "mock",
        "nose",
        ],
    entry_points={},
    keywords='',
    test_suite='nose.collector',
)
# #}
# change setup_kwds below before the next pkglts tag

#setup_kwds['entry_points']["scripts"] = ["sam_experiment_detect = scripts/sam_experiment_detect_quantify_and_align.py"]

# do not change things below
# {# pkglts, pysetup.call
setup(**setup_kwds)
# #}
