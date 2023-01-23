# users360

This project extends [Rondon 360-videos models/dataset collection](https://gitlab.com/miguelfromeror/head-motion-prediction) with support to:

* access users' trajectories at datasets as a panda.DataFrame

| ds    | ds_user | ds_video      | traject | traject_hmps |
| ----- | ------- | ------------- | ------- | ------------ |
| david | david_0 | david_10_Cows | [[...   | [[[...       |

* view users' trajectories with a Plotly 3d visualization
* perform user clustering by entropy
* perform tileset analyses
* perform prediction models filtred by users clustering

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

Ensure users360/head_motion_prediction submodule and patch it.

```bash
git submodule init
git submodule update
sed -i -e 's/import keras/import tensorflow.keras/g' -e 's/from keras/from tensorflow.keras/g'  ./users360/head_motion_prediction/*.py
```

## Train and evaluate

To comparison, the code bellow shows how to train/evaluate one dataset using [Rondon repo](https://gitlab.com/miguelfromeror/head-motion-prediction) and using this project.

Rondon:

```bash
cd users360/head_motion_prediction
python -m user360.main -train -dataset_name David_MMSys_18 -model_name pos_only
python -m user360.main -evaluate -dataset_name David_MMSys_18 -model_name pos_only
```

user360:


```bash
python -m user360.main -train -dataset_name David_MMSys_18 -model_name pos_only
python -m user360.main -evaluate -dataset_name David_MMSys_18 -model_name pos_only
```

## Notebooks

See [notebooks/](notebooks/) folder

![Alt Text](docs/requests.gif)
