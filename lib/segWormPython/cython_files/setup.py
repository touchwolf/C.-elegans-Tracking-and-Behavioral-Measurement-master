# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 18:16:50 2015

@author: ajaver
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy


setup(ext_modules=cythonize("*_cython.pyx"),
      include_dirs=[numpy.get_include()])

# python3 setup.py build_ext --inplace

circCurvature_ext = [
    Extension(
        "circCurvature", sources=[
            "circCurvature.pyx", "c_circCurvature.c"], include_dirs=[
                numpy.get_include()])]
curvspace_ext = [
    Extension(
        "curvspace", sources=[
            "curvspace.pyx", "c_curvspace.c"], include_dirs=[
                numpy.get_include()])]
cleanWorm_cython_ext = [
	Extension(
		"cleanWorm_cython", sources=[
			"cleanWorm_cython.pyx", "cleanWorm_cython.c"], include_dirs=[
				numpy.get_include()])]
linearSkeleton_cython_ext = [
	Extension(
		"linearSkeleton_cython", sources=[
			"linearSkeleton_cython.pyx", "linearSkeleton_cython.c"], include_dirs=[
				numpy.get_include()])]
segWorm_cython_ext = [
	Extension(
		"segWorm_cython", sources=[
			"segWorm_cython.pyx", "segWorm_cython.c"], include_dirs=[
				numpy.get_include()])]
for ext_modules in [circCurvature_ext, curvspace_ext,cleanWorm_cython_ext,linearSkeleton_cython_ext,segWorm_cython_ext]:
    setup(cmdclass={'build_ext': build_ext}, ext_modules=ext_modules,)
