# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
  name='aotsc',
  version='0.1.1',
  description='Adaptive Off-the-shelf classifier',
  long_description='LONG Description',
  author='Arun Reddy Nelakurthi',
  author_email='arunreddy.nelakurthi@gmail.com',
  url='https://github.com/arunreddy/otsc',
  license='MIT License',
  packages=find_packages(exclude=('tests', 'docs')),
  entry_points={
    'console_scripts': ['aotsc=otsc.cli:main'],
  }
)
