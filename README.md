# DL4H-Desmoking-Laparoscopy-Surgery-Images
Implemenation of paper https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9261326

IMPORTANT: Please run this in a python environment as one of the notebooks will patch the torchmetrics install in such a way that can break it for other uses.

How to run:

1. Please download these datasets:

    https://zenodo.org/record/4515433#.ZFf-EM7MIuU

    Cholec80 from here (for valid research purposes only):

    http://camma.u-strasbg.fr/datasets

2. Install conda environment from 'conda_env' folder

3. Open 'synthetic_smoke_prep.ipynb' and follow instructions to setup the dataset and prep it. You will need a few days of processing for this and about 1TB of disk space.

4. Open 'train_and_run.ipynb' and follow instructions to train the models and run validation/test metrics. This will take an additional 1-2 days depending on hardware. This is assuming a single GPU 3070 class or higher. Please see notebooks for additional runtime estimates. This may work with 4gb vram but I have only tested with a 3070 8gb. REFERENCES ARE AT THE END OF THIS NOTEBOOK.

5. If desired you can run real smoke data through the net of your choice and make a side-by-side comparison video using 'run_inference.ipynb'. Note that this is entirely qualitative since no metrics are possible on non-synthetically smoked images. There is no base truth to compare against. A slideshow video for synthetic desmoking is also included. This is technically beyond the scope of the paper due to the non-quantitative nature, but interesting to play with.

6. To see dark channel prior in action you can open 'try_dcp.ipynb'. It will show you the smokes vs dark channel prior massks as well as the image they are generated from.

Results:
The original paper could not be reproduced. Performance with or without dark channel prior appears to be almost identical with a slight edge to those nets that did not use dark channel prior. Due to lack of arguments that author code was run with, and a lack of details on exactly how the input dataset was created (such as how the smoke was generated with python-clouds), we cannot be certain why. There is a possiblity that the input dataset the author's used had very little noise and distortion applied resulting in very high absolute image metric scores (like PSNR). Without the exact datset the author's used we cannot do a good relative comparison.

The UNET architecture presented here performs about as well as the very best UNET architectures that Salazar et al. presented in their paper. Please see the python notebooks for included results.

Pre-trained models:
https://app.box.com/s/qpxotp2xr6rxn786helj7pwspzlsay3k

---Credits and Thanks---

In libs, patch, and data_augmentation includes:


python-clouds - from https://github.com/SquidDev/Python-Clouds - Modified and tested for python 3.9 to generate rgba images - depends on pygame


data_augmentation - courtesy of collaborator Dr. Florian Richter - https://florianengineering.com/about

trainer base classes - Dr. Florian Richter

Additional guidance and advice on implementation - Dr. Florian Richter



Cholec 80 dataset utilized, origin of data:

Andru Twinanda, Didier Mutter Sherif Shehata, Jacques
Marescaux, and Nicolas Padoy Michel De Mathelin.
2016. Endonet: A deep architecture for recognition
tasks on laparoscopic videos. IEEE Transactions on
Medical Imaging, 36.

Source paper citation:

Sebasti  ́an Salazar-Colores, Hugo Moreno Jim  ́enez,
C  ́esar Javier Ortiz-Echeverri, and Gerardo Flores.
2020. Desmoking laparoscopy surgery images using
an image-to-image translation guided by an embed-
ded dark channel. IEEE Access, 8:208898–208909.

Source paper github:
https://github.com/ssalazarcolores/Desmoking-laparoscopy-surgery-images-using-an-image-to-image-translation-guided-by-an-embedded-Dark-
