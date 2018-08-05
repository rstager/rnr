from setuptools import setup, Extension
import platform

version = '0.0.1'

setup(name='rnr',
      zip_safe=True,
      version=version,
      description='RNR.',
      long_description='RNR.',
      url='https://github.com/rstager/rnr',
      author='rkstager',
      install_requires=[
      ],
      author_email='rkstager@gmail.com',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering',
      ],
      license='apache',
      packages=[
          'rnr'
      ],
      ext_modules=[])
