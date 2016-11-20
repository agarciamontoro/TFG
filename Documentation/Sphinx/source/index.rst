.. TFG documentation master file, created by
   sphinx-quickstart on Wed Aug 10 14:00:40 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TFG's documentation!
===============================

Contents:

.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Testing
=======

.. .. currentmodule:: raytracer

.. autoclass:: kerr.KerrMetric
    :members:

.. autoclass:: raytracer.Camera
    :members:

.. autoclass:: raytracer.RayTracer
    :members:

This is a test to know whether the figure automatic numbering really works. See :numref:`my-figure-ref` if everything's fine.

.. _my-figure-ref:
.. figure:: ../../../Res/sphinxTest.png
   :scale: 50 %
   :alt: map to buried treasure

   This is the caption of the figure (a simple paragraph).

   The legend consists of all elements after the caption.  In this
   case, the legend consists of this paragraph and the following
   table.

Doxygen testing
===============

.. doxygenfile:: solvers.cu

.. doxygenfunction:: advanceStep

Bibliography
============

.. bibliography:: ../../Report/Bibliography.bib
