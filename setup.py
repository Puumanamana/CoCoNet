'''
Setup config for coconet-binning package
'''

from setuptools import setup, find_packages

setup(name='coconet-binning',
      version='0.54',
      description='A contig binning tool from viral metagenomes',
      long_description=open('README.rst').read(),
      keywords='binning metagenomics deep learning virus clustering',
      license='Apache License 2.0',
      url='https://github.com/Puumanamana/CoCoNet',
      author='Arisdakessian Cedric',
      author_email='carisdak@hawaii.edu',
      zip_safe=False,
      packages=find_packages(),
      entry_points={'console_scripts': ['coconet=coconet.coconet:main']},
      test_requires=['pytest',
                     'pytest-cov'],
      python_requires='>=3.6',
      install_requires=[
          'argparse',
          'pyyaml==5.1.0',
          'numpy',
          'pandas>=1.0',
          'h5py',
          'scikit-learn',
          'torch>=1.0',
          'Biopython',
          'python-igraph>=0.7.1.post6',
          'leidenalg>=0.7.0',
          'tqdm>=4.40.0',
          'pysam>=0.16'
      ])
