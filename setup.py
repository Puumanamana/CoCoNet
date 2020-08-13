'''
Setup config for coconet-binning package
'''

from setuptools import setup

setup(name='coconet-binning',
      version='0.53',
      description='A contig binning tool from viral metagenomes',
      long_description=open('README.rst').read(),
      keywords='binning metagenomics deep learning virus clustering',
      license='Apache License 2.0',
      url='https://github.com/Puumanamana/CoCoNet',
      author='Arisdakessian Cedric',
      author_email='carisdak@hawaii.edu',
      zip_safe=False,
      packages=['coconet'],
      entry_points={'console_scripts': ['coconet=coconet.coconet:main']},
      test_requires=['pytest',
                     'pytest-cov'],
      python_requires='>=3.6',
      install_requires=[
          'argparse',
          'pyyaml==5.1.0',
          'numpy>=1.16',
          'pandas>=1.0',
          'h5py>=2.6',
          'scikit-learn>=0.21',
          'torch>=1.5',
          'Biopython>=1.7',
          'python-igraph>=0.7.1.post6',
          'leidenalg>=0.7.0',
          'tqdm>=4.40.0',
      ])
