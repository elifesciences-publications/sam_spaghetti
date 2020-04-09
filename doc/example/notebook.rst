.. toctree::
   :maxdepth: 2

.. _example-notebook:

################################################################################
Replaying the notebook examples on your own installation
################################################################################

Prerequisites : installation and data
-------------------------------------

Make sure you have followed the **Developer Procedure** of the :ref:`installation` guidelines, and navigate to the directory where you have cloned the sources of :code:`sam_spaghetti`. There, you will be able to download an example set of image files that are already well referenced in the configuration files.

.. code-block::

    ./download_sam_data.sh

This will download the data and place the image files under the :code:`share/data/microscopy` folder of your directory

.. note::

    In case no data is downloaded, check whether :code:`wget` is missing and in that case install it using:

    .. code-block::

        conda install wget

    Then run the previous command again.

Run Jupyter and open a notebook example
---------------------------------------

With your installation of :code:`sam_spaghetti`, you will be able to replay the notebook examples displayed in ths documentation, and even to process custom data, provided you fulfill the configuration file requirements.

.. code-block::

    jupyter notebook doc/notebook

This command should open a new tab in your Web browser, where you will be able to select a notebook example, for instance :code:`sam_sequence_detect_quantify_and_align.ipynb`.

Run the notebook
----------------

Now, you can replay the pipeline performed by the notebook and see the displays associated with the different steps. To run it all at once select **Restart & Run All** in the **Kernel** menu of the notebook.

.. note::

    As some of the steps involve intensive computing, the execution might take some time, be greedy on your CPU resources, or possibly saturate your RAM. We recommend to run the notebooks on a 64GB RAM computer, or at least on a 32GB RAM.

.. note::

    Running the notebook will cause :code:`sam_spaghetti` to write several files on your system. Make sure you have enough space left (typically 2GB per image sequence) before you run it.
