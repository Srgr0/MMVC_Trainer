from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the Cython extension
extensions = [
    Extension(
        "core",  # Module name
        ["core.pyx"],  # Source file
        include_dirs=[numpy.get_include()],  # Include NumPy headers
        language="c",  # Use C compiler for compatibility
        extra_compile_args=["-O3"],  # Optimization flags
    )
]

setup(
    name="mmvc_alignment",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': 3}),
    zip_safe=False,
)
