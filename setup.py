# encoding: utf-8
from setuptools import setup

setup(name='imaginet',
      version='0.1',
      description='Visually grounded word and sentence representations',
      url='https://github.com/gchrupala/reimaginet',
      author='Grzegorz Chrupała',
      author_email='g.chrupala@uvt.nl',
      license='MIT',
      packages=['imaginet'],
      install_requires=[
          'Theano',
          'funktional'
                    ],
      zip_safe=False)
