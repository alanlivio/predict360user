# predict360user

Python library to predict user navigation in 360 videos. It extends [Rondon 360-videos models/dataset collection](https://gitlab.com/miguelfromeror/head-motion-prediction).

## Library

The library's main classes are:

* `Dataset`: stores the dataset in memory as a pandas.DataFrame and provides functions for data preprocessing, such as user clustering by entropy.

|     | ds    | user    | video         | traces       | traces_hmps |
| --- | ----- | ------- | ------------- | ------------- | ------------ |
| 0   | david | david_0 | david_10_Cows | [[x,y,z],...] | [[4x6],...]  |

* `Trainer`: train and perform prediction models from the Rondon collection
* `Plot360`: plot viewport, trajectory or predictions for users using 3D visualization

<div style="text-align:center"><img src="docs/requests.gif" width="300" ></div>

## Datasets

| dataset            | users (u) | videos (v) | trajectories (u*v) |
| ------------------ | --------- | ---------- | ------------------ |
| Xu_PAMI_18 [1]     | 59        | 77         | 4,543              |
| Xu_CVPR_18 [2]     | 34        | 209        | 7,106              |
| Nguyen_MM_18 [3]   | 48        | 9          | 432                |
| Fan_NOSSDAV_17 [4] | 39        | 9          | 300                |
| David_MMSys_18 [5] | 57        | 19         | 1,083              |
| total              |           |            | 12,451             |

## Models

| model              | category | user input      | video input | status |
| ------------------ | -------- | --------------- | ----------- | ------ |
| pos_only [7]       | LSTM     | position        | saliency    | OK     |
| Xu_PAMI_18 [1]     | LSTM     | position        | saliency    | WIP    |
| Xu_CVPR_18 [2]     | LSTM     | gaze            | gaze RGB    | WIP    |
| Nguyen_MM_18 [3]   | LSTM     | position, tiles | saliency    | WIP    |
| Li_ChinaCom_18 [6] | LSTM     | tiles           | saliency    | WIP    |
| Rondon_TRACK [7]   | LSTM     | position        | saliency    | WIP    |

## Requirements

The project dependencies are described [environment.yml](environment.yml) and [requirements.txt](requirements.txt), which the main one is [TensorFlow](https://www.tensorflow.org/install/pip). On Windows, you must install [SDK and MSVC](https://visualstudio.microsoft.com/visual-cpp-build-tools/) because of the [spherical_geometry](https://github.com/spacetelescope/spherical_geometry) package.

```bash
conda env create -f environment.yml
conda activate p3u
```

To fetch and patch the Rondon head_motion_prediction submodule, do

```bash
git submodule init
git submodule update
sed -i -e 's/import keras/import tensorflow.keras/g' -e 's/from keras/from tensorflow.keras/g'  ./predict360user/head_motion_prediction/*.py
```

## Train and evaluate

To illustrate usage, the code below shows how to train/evaluate one dataset using [Rondon repo](https://gitlab.com/miguelfromeror/head-motion-prediction) and using this project.

Rondon:

```bash
cd predict360user/head_motion_prediction
python training_procedure.py -train -dataset_name David_MMSys_18 -model_name pos_only
python training_procedure.py -evaluate -dataset_name David_MMSys_18 -model_name pos_only
```

predict360user:

```bash
python -m predict360user -train -dataset_name david -model_name pos_only
python -m predict360user -evaluate -dataset_name david -model_name pos_only
```

## Documentation

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

## References

[1] https://ieeexplore.ieee.org/document/8418756  
[2] https://ieeexplore.ieee.org/document/8578657  
[3] https://dl.acm.org/doi/10.1145/3240508.3240669  
[4] https://doi.org/10.1145/3204949.3208139  
[5] https://dl.acm.org/doi/10.1145/3083165.3083180  
[6] https://eudl.eu/pdf/10.1007/978-3-030-06161-6_49  
[7] https://arxiv.org/pdf/1911.11702.pdf  
