Para-disk
===================================

**Para-Disk**  is a Python library for creating images of molecular gas and dust emission from disks (protoplanetary or debris) around stars.


Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.


Installation
************

To install para-disk clone the github repository:

.. code-block::
   git clone https://github.com/kevin-flaherty/disk_model3.git
   cd disk_model3

.. code-block::

    git clone https://github.com/richteague/disksurf.git
    cd disksurf
    pip install .

To guide you through **para-disk** we've created a :doc:`Guide` with step-by-step
instructions for running the code. Check out :doc:`model.ipynb` to see a description of the 
parametric model that underlies the code, and :doc:`under-the-hood` for a 
description of some of the mechanisms used in the code. :doc:`troubleshooting`
provides a guide to common troubleshooting issues with this code. 

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   under-the-hood
   model.ipynb
   troubleshooting

   ../Guide.ipynb
