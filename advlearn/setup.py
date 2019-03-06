"""Adversarial learning framework for testing machine learning systems"""

from distutils.core import setup
from advlearn import __version__

setup(
    name='adverserial-learn',
    version=__version__,
    author=('Christopher Frederickson, \
            Michael Moore, \
            Alexander Karavaltchev'),
    author_email=('fredericc0@students.rowan.edu, \
                    moorem6@students.rowan.edu, \
                    karavalta4@students.rowan.edu'),
    packages=['advlearn'],
    url='',
    license='LICENSE.txt',
    description='A framework for attacking machine learning algorithms',
    long_description=open('README.rst').read(),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
    ],
)
