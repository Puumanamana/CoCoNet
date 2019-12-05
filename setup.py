from setuptools import setup, find_packages

setup(name='coconet-binning',
      version='0.0.2.1',
      description='A contig binning tool from viral metagenomes',
      url='https://github.com/Puumanamana/CoCoNet',
      author='Arisdakessian Cedric',
      author_email='carisdak@hawaii.edu',
      license='Apache License 2.0',
      zip_safe=False,
      entry_points={'console_scripts': ['coconet=coconet.coconet:main']},
      packages=find_packages(),
      python_requires='>=3.6',
      install_requires=[
          'numpy',
          'pandas',
          'h5py',
          'sklearn',
          'torch',
          'progressbar',
          'Biopython',
          'click',
          'python-igraph',
          'leidenalg'
      ])
