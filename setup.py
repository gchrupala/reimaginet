# encoding: utf-8
from setuptools import setup

setup(name='imaginet',
      version='0.4',
      description='Visually grounded word and sentence representations',
      url='https://github.com/gchrupala/reimaginet',
      author='Grzegorz Chrupała',
      author_email='g.chrupala@uvt.nl',
      license='MIT',
      packages=['imaginet'],
      install_requires=[
          'Theano',
          'funktional>=0.5'
                    ],
      zip_safe=False)
