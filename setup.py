"""
Setup config for coconet-binning package
"""

from setuptools import setup, find_packages

setup(name='coconet-binning',
      version='1.1.0',
      description='A contig binning tool from viral metagenomes',
      long_description_content_type='text/x-rst',
      long_description=open('README.rst').read(),
      keywords='binning metagenomics deep learning virus clustering',
      license='Apache License 2.0',
      url='https://github.com/Puumanamana/CoCoNet',
      author='Cedric Arisdakessian',
      author_email='carisdak@hawaii.edu',
      zip_safe=False,
      packages=find_packages(),
      entry_points={'console_scripts': ['coconet=coconet.coconet:main']},
      test_requires=['pytest',
                     'pytest-cov'],
      python_requires='>=3.6',
      install_requires=[
          'psutil',
          'pyyaml',
          'numpy',
          'pandas>=1.0',
          'h5py',
          'scikit-learn',
          'scipy',
          'scikit-bio >=0.5.6',
          'torch>=1.0',
          'Biopython',
          'python-igraph>=0.8',
          'pysam>=0.16',
          'pybind11',
          'hnswlib'
      ])
