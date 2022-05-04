import os
import numpy
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('KSUMS2', parent_package, top_path)

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

    config.add_extension('KSUMS_',
                         sources=['KSUMS_.pyx'],
                         include_dirs=[numpy.get_include()],
                         language="c++",
                         extra_compile_args=ext_comp_args,
                         extra_link_args=ext_link_args,
                         define_macros=define_macros,
                         libraries=libraries,
                         depends=["KSUMS_.pxd", "KSUMS_.pyx"]
                         )

    config.add_extension('Keep_order_',
                         sources=['Keep_order_.pyx'],
                         include_dirs=[numpy.get_include()],
                         language="c++",

                         extra_compile_args=ext_comp_args,
                         extra_link_args=ext_link_args,

                         library_dirs=library_dirs,
                         libraries=library_dirs,

                         define_macros=define_macros,
                         depends=["Keep_order_.pxd", "Keep_order_.pyx"]
                         )

    config.ext_modules = cythonize(config.ext_modules, compiler_directives={'language_level': 3})

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
