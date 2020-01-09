========================
SAM Spaghetti
========================

.. {# pkglts, doc

.. #}

**SAM Sequence Primordia Alignment, GrowtH Estimation, Tracking & Temporal Indexation**

:Author: Guillaume Cerutti
:Contributors:  Christophe Godin, Jonathan Legrand, Carlos Galvan-Ampudia, Teva Vernoux

:Teams:  `RDP <http://www.ens-lyon.fr/RDP/>`_ Team Signal, Inria project team `Mosaic <https://team.inria.fr/mosaic/>`_

:Institutes: `Inria <http://www.inria.fr>`_, `INRA <https://inra.fr>`_, `CNRS <https://cnrs.fr>`_

:Language: Python

:Supported OS: Linux, MacOS

:Licence: `Cecill-C`

Description
-----------

.. image:: _static/auxin_map.png
    :width: 800px
    :align: center


This package provides scripts to reproduce the analysis pipelines described in the article `Temporal integration of auxin information for the regulation of patterning <https://www.biorxiv.org/content/10.1101/469718v2>`_ and used to reconstruct population averages of Shoot Apical Meristems (SAM) of *Arabidopsis thaliana* with quantitative gene expression and hormonal signal 2D maps. It essentially gives access to two major quantitative image analysis and geometrical interpretation pipelines:


+---------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
+---------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                   **Image quantification & alignment**                                        | Starting from microscopy acquisitions (CZI) of SAMs expressing an Auxin sensor (DII) and a CLV3 fluorescent reporter, this pipeline quantifies image intensity at cell level and performs an alignment of time lapse sequences into a common SAM reference frame.     |
+---------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                      **PIN image polarity analysis**                                          | Using microscopy acquisitions of SAMs expressing a fluorescent auxin carrier (PIN) and a cell wall staining, this pipeline estimates polarities at cell level. It can also use the result from the previous pipeline to superimpose aligned auxin and PIN information.|
+---------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Requirements
------------

- `timagetk <https://gitlab.inria.fr/mosaic/timagetk>`_
- `cellcomplex <https://gitlab.inria.fr/mosaic/cellcomplex)>`_
- `tissue_nukem_3d <https://gitlab.inria.fr/mosaic/tissue_nukem_3d>`_
- `tissue_paredes <https://gitlab.inria.fr/mosaic/tissue_paredes>`_

