# predict360user

Python library to predict user navigation in 360 videos. It extends extends [Rondon 360-videos models/dataset collection](https://gitlab.com/miguelfromeror/head-motion-prediction).

## Classes

The main library classes are:

* `Dataset`: stores the dataset in memory as a pandas.DataFrame and provides functions for data preprocessing, such as user clustering by entropy.

|     | ds    | user    | video         | traject       | traject_hmps |
| --- | ----- | ------- | ------------- | ------------- | ------------ |
| 0   | david | david_0 | david_10_Cows | [[x,y,z],...] | [[4x6],...]  |

* `Trainer`: train and perform prediction models from Rondon collection
* `Plot360`: plot viewport, trajectory or predictions for users using 3D visualization

<div style="text-align:center"><img src="docs/requests.gif" width="300" ></div>

## Datasets

| dataset            | users (u) | videos (v) | trajects (u*v) |
| ------------------ | ---------- | --------- | -------------- |
| Xu_PAMI_18 [1]     | 58         | 76        | 4,350          |
| Xu_CVPR_18 [2]     | 30        | 208        | 6,240          |
| Nguyen_MM_18 [3]   | 48         | 11        | 528            |
| Fan_NOSSDAV_17 [4] | 25         | 10        | 250            |
| David_MMSys_18 [5] | 57         | 19        | 1,083          |
| total              |            |           | 12,451         |


## Models

| model              | category | user input      | video input |
| ------------------ | -------- | --------------- | ----------- |
| Xu_PAMI_18 [1]     | LSTM     | position        | saliency    |
| Xu_CVPR_18 [2]     | LSTM     | gaze            | gaze RGB    |
| Nguyen_MM_18 [3]   | LSTM     | position, tiles | saliency    |
| Li_ChinaCom_18 [6] | LSTM     | tiles           | saliency    |
| Rondon_TRACK [7]   | LSTM     | position        | saliency    |


## Requirements

To install TensorFlow with Cuda, please follow https://www.tensorflow.org/install/.
For instance, you can do the following command with conda.

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Then install requirements,

```bash
pip install -r requeriments.txt
```

Ensure predict360user/head_motion_prediction submodule and patch it.

```bash
git submodule init
git submodule update
sed -i -e 's/import keras/import tensorflow.keras/g' -e 's/from keras/from tensorflow.keras/g'  ./predict360user/head_motion_prediction/*.py
```

## Train and evaluate

To comparison, the code bellow shows how to train/evaluate one dataset using [Rondon repo](https://gitlab.com/miguelfromeror/head-motion-prediction) and using this project.

Rondon:

```bash
cd predict360user/head_motion_prediction
python training_procedure.py -train -dataset_name David_MMSys_18 -model_name pos_only
python training_procedure.py -evaluate -dataset_name David_MMSys_18 -model_name pos_only
```

predict360user:

```bash
python run.py -train -dataset_name david -model_name pos_only
python run.py -evaluate -dataset_name david -model_name pos_only
```

## Documentation

See notebooks at [docs/](docs/) folder.


## References

[1] https://ieeexplore.ieee.org/document/8418756
[2] https://ieeexplore.ieee.org/document/8578657
[3] https://dl.acm.org/doi/10.1145/3240508.3240669
[4] https://doi.org/10.1145/3204949.3208139
[5] https://dl.acm.org/doi/10.1145/3083165.3083180
[6] https://eudl.eu/pdf/10.1007/978-3-030-06161-6_49
[7] https://arxiv.org/pdf/1911.11702.pdf