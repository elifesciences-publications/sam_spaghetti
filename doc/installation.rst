.. _installation:

============
Installation
============

-----------------
Conda environment
-----------------

Creating a new environment (recommended)
----------------------------------------

.. code-block::

    conda create -n sam_spaghetti -c mosaic sam_spaghetti

.. note::

    The name of the new environment (here: 'sam_spaghetti') is defined by the option -n. You will have to activate it each time you want to use the library by typing :

    .. code-block::

        conda activate sam_spaghetti

In an existing Python 3.7 environment
--------------------------------------


.. code-block::

    conda install -c mosaic sam_spaghetti


About Conda
-----------

**Conda** is an open source package management system and environment management system that runs on Windows, macOS and Linux. Conda quickly installs, runs and updates packages and their dependencies.
Conda easily creates, saves, loads and switches between environments on your local computer.
See `<https://docs.conda.io/en/latest/>`_

**Miniconda** is a free minimal installer for conda.
It is a small, bootstrap version of Anaconda that includes only ``conda``, ``Python``, the packages they depend on, and a small number of other useful packages, including ``pip``, ``zlib`` and a few others.
See `<https://docs.conda.io/en/latest/miniconda.html>`_

-------------------
Developer procedure
-------------------

Download the sources of :code:`sam_spaghetti`
---------------------------------------------

.. code-block::

    git clone https://gitlab.inria.fr/mosaic/publications/sam_spaghetti.git
    cd sam_spaghetti


Create a new conda environment with all dependencies
----------------------------------------------------

.. code-block::

    conda env create -n sam_spaghetti-dev -f conda/env.yaml
    conda activate sam_spaghetti-dev


Install the :code:`sam_spaghetti` library
-----------------------------------------

.. code-block::

    python setup.py develop





