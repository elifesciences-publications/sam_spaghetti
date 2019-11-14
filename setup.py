#!/usr/bin/env python
# -*- coding: utf-8 -*-

# {# pkglts, pysetup.kwds
# format setup arguments

from setuptools import setup, find_packages


short_descr = "SAM Sequence Primordia Alignment, GrowtH Estimation, Tracking & Temporal Indexation"
readme = open('README.rst').read()
history = open('HISTORY.rst').read()

# find packages
pkgs = find_packages('src')



setup_kwds = dict(
    name='sam_spaghetti',
    version="0.1.0",
    description=short_descr,
    long_description=readme + '\n\n' + history,
    author="Guillaume Cerutti",
    author_email="guillaume.cerutti@inria.fr",
    url='https://github.com/Guillaume Cerutti/sam_spaghetti',
    license='cecill-c',
    zip_safe=False,

    packages=pkgs,
    package_dir={'': 'src'},
    setup_requires=[
        ],
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

setup_kwds['entry_points']['console_scripts'] = ["sam_experiment_detect_quantify_and_align = sam_spaghetti.scripts.sam_experiment_detect_quantify_and_align:main"]

# do not change things below
# {# pkglts, pysetup.call
setup(**setup_kwds)
# #}
