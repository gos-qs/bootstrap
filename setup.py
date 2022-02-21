#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 20:42:07 2022

setup for bootstrap

@author: gos-qs
"""

from setuptools import setup

setup(
    name = 'bootstrap',
    version = '0.1.0',
    author = 'gos-qs',
    author_email = '',
    packages = [],
    scripts = [],
    url = '',
    license = 'LICENSE.md',
    description = 'compute bootstrapped metrics with confidence intervals',
    install_requires = [
        "numpy",
    ],
)
