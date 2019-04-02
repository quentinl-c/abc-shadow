from distutils.core import setup
from Cython.Build import cythonize
import numpy
import Cython.Compiler.Options

Cython.Compiler.Options.get_directive_defaults()['linetrace'] = True
Cython.Compiler.Options.get_directive_defaults()['binding'] = True
setup(name='ABC Shadow',
      ext_modules=cythonize(["./abc_shadow/*.pyx",
                             "./abc_shadow/graph/*.pyx"],
                            annotate=True),
      include_dirs=[numpy.get_include()])
