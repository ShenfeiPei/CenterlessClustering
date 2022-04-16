import os
import numpy
from Cython.Build import cythonize

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('KSUMSX_eigen', parent_package, top_path)

    cpp_version = "c++11"

    if os.name == "nt":
        ext_comp_args = ['/openmp']
        ext_link_args = ['/openmp']

        library_dirs = []
        libraries = []
    else:
        ext_comp_args = ['-fopenmp']
        ext_link_args = ['-fopenmp']

        library_dirs = []
        libraries = ["m"]

    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

    file_path = os.path.dirname(__file__)
    eigen_path = os.path.join(file_path, "Egien400")

    config.add_extension('KSUMSX_',
                         sources=['KSUMSX_.pyx'],
                         include_dirs=[numpy.get_include(), eigen_path],
                         language="c++",

                         extra_compile_args=ext_comp_args,
                         extra_link_args=ext_link_args,
                         define_macros=define_macros,
                         libraries=libraries)

    config.ext_modules = cythonize(config.ext_modules, compiler_directives={'language_level': 3})

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
