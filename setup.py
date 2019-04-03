from distutils.core import setup

import numpy

import Cython.Compiler.Options
from Cython.Build import cythonize

Cython.Compiler.Options.get_directive_defaults()['linetrace'] = True
Cython.Compiler.Options.get_directive_defaults()['binding'] = True
setup(name='ABC Shadow',
      ext_modules=cythonize([
          "./abc_shadow/graph/*.pyx", "./abc_shadow/*.pyx"],
                            annotate=True),
      include_dirs=[numpy.get_include()])
