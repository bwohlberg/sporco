Convolutional Sparse Coding
---------------------------

gwnden_gry.py
  Remove Gaussian white noise from a greyscale image using convolutional sparse coding

gwnden_clr.py
  Remove Gaussian white noise from a colour image using convolutional sparse coding

implsden_clr.py
  Remove salt & pepper noise from a colour image using convolutional sparse coding with a colour dictionary

cbpdn_ams_gry.py
  Inpainting of randomly distributed pixel corruption with lowpass image components handled via non-linear filtering (greyscale image)

cbpdn_ams_grd_gry.py
  Inpainting of randomly distributed pixel corruption with lowpass image components handled via gradient regularisation of an impulse dictionary filter (greyscale image)

cbpdn_md_gry.py
  Inpainting of randomly distributed pixel corruption (greyscale image)

parcbpdn_md_gry.py
  Inpainting of randomly distributed pixel corruption (greyscale image) using the parallel ADMM solver

cbpdn_ams_clr.py
  Inpainting of randomly distributed pixel corruption (colour image)

cbpdn_gry.py
  Convolutional sparse coding of a greyscale image (ADMM solver)

parcbpdn_gry.py
  Convolutional sparse coding of a greyscale image using the parallel ADMM solver

cbpdn_cuda.py
  Convolutional sparse coding of a greyscale image using the CUDA solver

cbpdn_fista_gry.py
  Convolutional sparse coding of a greyscale image (FISTA solver)

cminl1_gry.py
  Convolutional sparse coding of a greyscale image (constrained data fidelity)

cprjl1_gry.py
  Convolutional sparse coding of a greyscale image (constrained penalty term)

cbpdn_grd_cuda.py
  Convolutional sparse coding with gradient penalty of a greyscale image using the CUDA solver

cbpdn_clr_cd.py
  Convolutional sparse coding of a colour image with a colour dictionary

cbpdn_clr_gd.py
  Convolutional sparse coding of a colour image with a greyscale dictionary

cbpdn_jnt_clr.py
  Convolutional sparse coding of a colour image with a greyscale dictionary and a joint sparsity term
