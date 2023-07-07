# predict360user

Library to predict user behavior in 360 videos. It extends [Rondon 360-videos models/dataset collection](https://gitlab.com/miguelfromeror/head-motion-prediction). The library's main classes are:

* `Dataset`: stores the dataset in memory as a pandas.DataFrame and provides functions for data preprocessing, such as user clustering by entropy.

  |     | ds    | user    | video         | traces        |
  | --- | ----- | ------- | ------------- | ------------- |
  | 0   | david | david_0 | david_10_Cows | [[x,y,z],...] |

* `Trainer`: train and evaluate prediction models
* `Plot360`: plot viewport, trajectory or predictions for users using 3D visualization

### Datasets

| dataset                                  | users (u) | videos (v) | trajectories (u*v) |integrated |
| ---------------------------------------- | --------- | ---------- | ------------------ |---------- |
| Xu_PAMI_18 ([paper][Xu_PAMIN_18])        | 59        | 77         | 4,543              |yes        |
| Xu_CVPR_18 ([paper][Xu_CVPR_18])         | 34        | 209        | 7,106              |yes        |
| Nguyen_MM_18 ([paper][Nguyen_MM_18])     | 48        | 9          | 432                |yes        |
| Fan_NOSSDAV_17 ([paper][Fan_NOSSDAV_17]) | 39        | 9          | 300                |yes        |
| David_MMSys_18 ([paper][David_MMSys_18]) | 57        | 19         | 1,083              |yes        |
| total                                    |           |            | 12,451             |yes        |


### Models

| model                                                                 | method              | user input                | integrated |
| --------------------------------------------------------------------- | ------------------- | ------------------------- | ---------- |
| pos_only ([paper][Romero_PAMI_22], [code][Romero_PAMI_22_code])       | LSTM                | position, saliency        | yes        |
| Xu_PAMI_18 ([paper][Xu_PAMIN_18], [code][Xu_PAMIN_18_code])           | LSTM                | position, saliency        | no         |
| Xu_CVPR_18 ([paper][Xu_CVPR_18])                                      | LSTM                | gaze, gaze RGB            | yes        |
| Nguyen_MM_18 ([paper][Nguyen_MM_18])                                  | LSTM                | position, tiles, saliency | no         |
| Li_ChinaCom_18 ([paper][Li_ChinaCom_18])                              | LSTM                | tiles, saliency           | no         |
| Romero_PAMI_22 ([paper][Romero_PAMI_22], [code][Romero_PAMI_22_code]) | LSTM                | position, saliency        | yes        |
| DVMS_MMSYS_22 ([paper][DVMS_MMSYS_22], [code][DVMS_MMSYS_22_code])    | LSTM                | position, saliency        | no         |
| Chao_MMSP_21 ([paper][Chao_MMSP_21])                                  | Transformer         | position                  | no         |
| Wu_AAAI_20 ([paper][Chao_MMSP_21])                                    | SphericalCNN, RNN   | position                  | no         |
| Taghavi_NOSSDAV_20 ([paper][Taghavi_NOSSDAV_20])                      | Clustering          | position                  | no         |
| Petrangeli_AIVR_18 ([paper][Petrangeli_AIVR_18])                      | Spectral Clustering | position                  | no         |

[Petrangeli_AIVR_18]: https://ieeexplore.ieee.org/document/8613652
[Taghavi_NOSSDAV_20]: https://dl.acm.org/doi/10.1145/3386290.3396934
[Wu_AAAI_20]: https://ojs.aaai.org/index.php/AAAI/article/view/7377
[Wu_AAAI_20_code]: https://github.com/wuchlei/AAAI20-Viewport-Prediction
[Chao_MMSP_21]: https://ieeexplore.ieee.org/document/9733647
[Nguyen_MM_18]: https://dl.acm.org/doi/10.1145/3240508.3240669
[Xu_CVPR_18]: https://ieeexplore.ieee.org/document/8578657
[DVMS_MMSYS_22]: https://dl.acm.org/doi/abs/10.1145/3524273.3528176
[DVMS_MMSYS_22_code]: https://gitlabDVMS_/DVMS
[Romero_PAMI_22]: https://ieeexplore.ieee.org/document/9395242
[Romero_PAMI_22_code]: https://gitlabmiguelfromeror/head-motion-prediction
[Xu_PAMIN_18]: https://ieeexplore.ieee.org/document/8418756
[Xu_PAMIN_18_code]: https://github.com/YuhangSong/DHP
[Fan_NOSSDAV_17]: https://doi.org/10.1145/3204949.3208139  
[David_MMSys_18]: https://dl.acm.org/doi/10.1145/3083165.3083180  
[Li_ChinaCom_18]: https://eudl.eu/pdf/10.1007/978-3-030-06161-6_49  


## Usage

#### Requirements

The project was tested in linux and Windows-WSL. First follow [TensorFlow pip installation tutorial](https://www.tensorflow.org/install/pip). 
The use conda to dependencies described [environment.yml](environment.yml) as follow. 
```bash
conda env create -f environment.yml
conda activate p3u
```

#### Train and evaluate

To illustrate usage, the code below shows how to train/evaluate `pos_only` model.

```bash
python -m predict360user -train -dataset_name david -model_name pos_only
python -m predict360user -evaluate -dataset_name david -model_name pos_only
```

#### Documentation

See notebooks in [docs/](docs/) folder.

## Cite

If you use `predict360user` please consider citing it as:

  ```bibtex
  @misc{predict360user,
    author = {Guedes, Alan},
    title = {predict360user: library to predict user navigation in 360 videos},
    year = {2021},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/alanlivio/predict360user}}
  }
  ```

## A note on maintenance

This repository was born as part of the UK EPSR SpheryStream project. Its maintenance is limited by a research project's time and resources. Even if I want to automate all 360 user prediction models, I do not have the time to maintain the whole body of automation that a well-maintained package deserves. Any help is very welcome. Here is a quick guide to interacting with this repository:

* If you find a bug, please open an issue, and I will fix it as soon as possible.
* If you want to request a new feature, please open an issue, and I will consider it as soon as possible.
* If you want to contribute yourself, please open an issue first, we discuss the objective, plan a proposal, and open a pull request to act on it.

If you would like to be involved further in the development of this repository, please get in touch with me directly: aguedes at ucl dot ac dot uk.