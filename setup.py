from setuptools import setup, find_packages

setup(name='CoCoNet',
      version='1.0',
      description='A contig binning tool from viral metagenomes',
      long_description=open('README.md').read(),
      url='https://github.com/Puumanamana/CoCoNet',
      author='Arisdakessian Cedric',
      author_email='carisdak@hawaii.edu',
      license='Apache License 2.0',
      zip_safe=False,
      entryoint={'console_scripts':['coconet=coconet.coconet:main']},
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
      ])
