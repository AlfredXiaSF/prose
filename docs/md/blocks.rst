Blocks
======

.. image:: ../_static/block.png
   :align: center
   :width: 200px


A ``Block`` is a single unit of processing acting on the ``Image`` object, reading and writing its attributes. Blocks documentation include the following information:

- |read|: the ``Image`` attributes read by the ``Block``
- |write|: the ``Image`` attributes written by the ``Block``
- |modify|: indicates that the ``Image.data`` is modified by the ``Block``

Detection
---------

.. currentmodule:: prose.blocks.detection

.. image:: ../_static/detection.png
   :align: center
   :height: 230px

.. autosummary::
   :template: blocksum.rst
   :nosignatures:

   AutoSourceDetection
   PointSourceDetection   
   DAOFindStars
   SEDetection
   TraceDetection

PSF
---

.. currentmodule:: prose.blocks.psf

.. autosummary::
   :template: blocksum.rst
   :nosignatures:

   MedianEPSF

Once an effective PSF is constructed (an :py:class:`~prose.Image` in :code:`Image.epsf`) it can be modeled with the following blocks:

.. autosummary::
   :template: blocksum.rst
   :nosignatures:

   Gaussian2D
   Moffat2D
   JAXGaussian2D
   JAXMoffat2D

Alignment & Geometry
--------------------

.. image:: ../_static/matching.png
   :align: center
   :height: 220px

.. autosummary::
   :template: blocksum.rst
   :nosignatures:

   ~prose.blocks.geometry.Trim
   ~prose.blocks.geometry.Cutouts
   ~prose.blocks.alignment.Align
   ~prose.blocks.alignment.AlignReferenceSources
   ~prose.blocks.alignment.AlignReferenceWCS

Centroiding
-----------

.. currentmodule:: prose.blocks.centroids

.. autosummary::
   :template: blocksum.rst
   :nosignatures:

   CentroidCOM
   CentroidQuadratic
   CentroidGaussian2D
   CentroidBallet

Photometry
----------

.. currentmodule:: prose.blocks.photometry

.. image:: ../_static/photometry.png
   :align: center
   :width: 220px

.. autosummary::
   :template: blocksum.rst
   :nosignatures:

   AperturePhotometry
   AnnulusBackground
   
Utils
-----

.. currentmodule:: prose.blocks.utils

.. image:: ../_static/utils.png
   :align: center
   :height: 190px
   
.. autosummary::
   :template: blocksum.rst
   :nosignatures:

   Get
   Calibration
   SortSources
   CleanBadPixels

All
---

.. include:: all_blocks.rst
