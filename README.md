# users360

This project extends [Rondon 360-videos models/dataset collection](https://gitlab.com/miguelfromeror/head-motion-prediction) with support to:

* access users' trajectories at datasets as a panda.DataFrame 
* view users' trajectories with a Plotly 3d visualization
* perform user clustering by entropy
* perform tileset analyses
* perform prediction models filtred by users clustering


* Requirements 

To install TensorFlow with Cuda, please follow https://www.tensorflow.org/install/.
For instance, you can do the following command with conda.

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Then install requirements,

```bash
pip install -r requeriments.txt
```

Ensure users360/head_motion_prediction submodule.

```bash
git submodule init
git submodule update
./patch_submodule.sh
```

## Usage users360/head_motion_prediction

train/evaluate:

```bash
cd users360/head_motion_prediction
python training_procedure.py -train -gpu_id 0 -dataset_name David_MMSys_18 -model_name pos_only -init_window 30 -m_window 5 -h_window 25
python training_procedure.py -evaluate -gpu_id 0 -dataset_name David_MMSys_18 -model_name pos_only -init_window 30 -m_window 5 -h_window 25
```

## Usage user360

train/evaluate:

```bash
python training_procedure.py -train -gpu_id 0 -dataset_name David_MMSys_18 -model_name pos_only -init_window 30 -m_window 5 -h_window 25
python training_procedure.py -evaluate -gpu_id 0 -dataset_name David_MMSys_18 -model_name pos_only -init_window 30 -m_window 5 -h_window 25
```

## Notebooks

See `notebooks/users360.ipynb`

![Alt Text](docs/requests.gif)
