# CycleGAN
Re-implementation of the CycleGAN model proposed by Zhu et al. (See it [here](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html)). CycleGAN is a deep learning model for image-to-image translation in the absence of paired training data. In particular, CycleGAN learns mappings between two domains even if no supervision is available on these mappings. To this end, CycleGAN simultaneously trains and maintains four deep neural networks. Also, a new version of CycleGAN (in /CycleGAN-Sync folder) was developed that can train the model using synchronous distributed training on multiple GPUs.

## Applications
I used this model for object transfiguration (horse <-> zebra), artistic style transfer (Monet & Cezzane painting <-> real photo), and season transfer (summer <-> winter). Here are some of the results (data was obtained from [here](https://people.eecs.berkeley.edu/%7Etaesung_park/CycleGAN/datasets/)):

### Horse <-> Zebra
<p align="center"><img src="Images/h2z_train.png" width = "700" class="center"></p>
<p align="center"><img src="Images/h2z_test.png" width = "700" class="center"></p>
<p align="center"><img src="Images/z2h_train.png" width = "700" class="center"></p>
<p align="center"><img src="Images/z2h_test.png" width = "700" class="center"></p>

### Painitng <-> Photo
<p align="center"><img src="Images/photo2painting_train.png" width = "700" class="center"></p>
<p align="center"><img src="Images/monet2photo_train.png" width = "700" class="center"></p>
<p align="center"><img src="Images/artistic_style_test.png" width = "700" class="center"></p>

### Summer <-> Winter
<p align="center"><img src="Images/season_transfer_train.png" width = "700" class="center"></p>
<p align="center"><img src="Images/season_transfer_test.png" width = "700" class="center"></p>

## Train CycleGAN
Run the following command to see a full description of the input flags:

```console
python train.py --help
```
Example: To train a CycleGAN model on horse-to-zebra data set for 200 epochs,
without the identity loss, and without multiprocessing in data loading, run the
following:

```console
python train.py --l_idnt 0 --ds horse2zebra --nEpoch 200 --nw 0 > out.log
```

Running the above command creates the following three folders:
  1- data: contains the specified data set
  2- models: contains the saved models during training
  3- generated_images: contains the evaluation of model during training on 10 randomly sampled data

In order to continue training from a saved model (e.g. the model saved after
epoch 49), run the following command:

```console
python train.py --l_idnt 0 --ds horse2zebra --nEpoch 200 --nw 0 --resume True --last_epoch 49 > out.log
```

Training will be continued while last_epoch < nEpoch.
