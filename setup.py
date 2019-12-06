from setuptools import setup, find_packages

setup(name='coconet-binning',
      version='0.0.2.2',
      description='A contig binning tool from viral metagenomes',
      url='https://github.com/Puumanamana/CoCoNet',
      author='Arisdakessian Cedric',
      author_email='carisdak@hawaii.edu',
      license='Apache License 2.0',
      zip_safe=False,
      entry_points={'console_scripts': ['coconet=coconet.coconet:main']},
      packages=find_packages(),
      python_requires='>=3.6',
      test_requires=['pytest',
                     'pytest-cov'],
      install_requires=[
          'numpy',
          'pandas',
          'h5py',
          'sklearn',
          'torch',
          'pyyaml==5.1',
          'Biopython',
          'click',
          'python-igraph>=0.7.1.post6',
          'leidenalg>=0.7.0'
      ])
