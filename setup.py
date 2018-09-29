# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='func_helper',
    version='1.0.0',
    description='Helper functions for functional programming',
    long_description=readme,
    author='Fumitoshi Morisato',
    author_email='fmorisato@gmail.com',
    url='https://github.com/Fumipo-Theta/func_helper',
    install_requires=['functools'],
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
