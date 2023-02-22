# predict360user

This project extends [Rondon 360-videos models/dataset collection](https://gitlab.com/miguelfromeror/head-motion-prediction).
It is composed by these main classes:

* Dataset: stores the original dataset in memory as a pandas.DataFrame and provides functions for data preprocessing, such user clustering by entropy.

  |       | ds    | user    | video         | traject | traject_hmps |
  | ----- | ----- | ------- | ------------- | ------- | ------------ |
  | 0     | david | david_0 | david_10_Cows | [[...   | [[[...       |

* Plot360: plot users' trajectories with a Plotly 3d visualization
* Trainer: train and perform prediction models from Rondon collection

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

user360:

```bash
python run.py -train -dataset_name david -model_name pos_only
python run.py -evaluate -dataset_name david -model_name pos_only
```

## Notebooks

See [notebooks/](notebooks/) folder

![Alt Text](docs/requests.gif)
