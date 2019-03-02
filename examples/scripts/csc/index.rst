Convolutional Sparse Coding
===========================

Basic Usage
-----------

Greyscale Images
^^^^^^^^^^^^^^^^

.. toc-start

* `Convolutional sparse coding of a greyscale image (ADMM solver) <cbpdn_gry.py>`__
* `Convolutional sparse coding of a greyscale image using the parallel ADMM solver <parcbpdn_gry.py>`__
* `Convolutional sparse coding of a greyscale image using the CUDA solver <cbpdn_cuda.py>`__
* `Convolutional sparse coding of a greyscale image (FISTA solver) <cbpdn_fista_gry.py>`__
* `Convolutional sparse coding of a greyscale image (constrained data fidelity) <cminl1_gry.py>`__
* `Convolutional sparse coding of a greyscale image (constrained penalty term) <cprjl1_gry.py>`__
* `Convolutional sparse coding with gradient penalty of a greyscale image using the CUDA solver <cbpdn_grd_cuda.py>`__

.. toc-end


Colour Images
^^^^^^^^^^^^^

.. toc-start

* `Convolutional sparse coding of a colour image with a colour dictionary <cbpdn_clr_cd.py>`__
* `Convolutional sparse coding of a colour image with a colour dictionary (FISTA solver) <cbpdn_fista_clr.py>`__
* `Convolutional sparse coding of a colour image with a greyscale dictionary <cbpdn_clr_gd.py>`__
* `Convolutional sparse coding of a colour image with a greyscale dictionary and a joint sparsity term <cbpdn_jnt_clr.py>`__
* `Convolutional sparse coding of a colour image with a product dictionary <cbpdn_clr_pd.py>`__

.. toc-end


Image Restoration Applications
------------------------------

Denoising (Gaussian White Noise)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toc-start

* `Remove Gaussian white noise from a greyscale image using convolutional sparse coding <gwnden_gry.py>`__
* `Remove Gaussian white noise from a colour image using convolutional sparse coding <gwnden_clr.py>`__

.. toc-end


Denoising (Impulse Noise)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. toc-start

* `Remove salt & pepper noise from a colour image using convolutional sparse coding with a colour dictionary <implsden_clr.py>`__
* `Remove salt & pepper noise from a colour image using convolutional sparse coding with an l1 data fidelity term and an l2 gradient term, with a colour dictionary <implsden_grd_clr.py>`__
* `Remove salt & pepper noise from a hyperspectral image using convolutional sparse coding with an l1 data fidelity term and an l2 gradient term, with a dictionary consisting of the product of a convolutional dictionary for the spatial axes and a standard dictionary for the spectral axis <implsden_grd_pd_dct.py>`__
* `Remove salt & pepper noise from a hyperspectral image using convolutional sparse coding with an l1 data fidelity term and an l2 gradient term, with a dictionary consisting of the product of a convolutional dictionary for the spatial axes and a PCA basis for the spectral axis <implsden_grd_pd_pca.py>`__


.. toc-end


Inpainting
^^^^^^^^^^

.. toc-start

* `Inpainting of randomly distributed pixel corruption with lowpass image components handled via non-linear filtering (greyscale image) <cbpdn_ams_gry.py>`__
* `Inpainting of randomly distributed pixel corruption with lowpass image components handled via gradient regularisation of an impulse dictionary filter (greyscale image) <cbpdn_ams_grd_gry.py>`__
* `Inpainting of randomly distributed pixel corruption (greyscale image) <cbpdn_md_gry.py>`__
* `Inpainting of randomly distributed pixel corruption (greyscale image) using the parallel ADMM solver <parcbpdn_md_gry.py>`__
* `Inpainting of randomly distributed pixel corruption (colour image) <cbpdn_ams_clr.py>`__

.. toc-end
