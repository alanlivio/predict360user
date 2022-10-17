# users360

Python library to perform user behavior analysis on 360-degree videos from [Rondon datasets collection](https://gitlab.com/miguelfromeror/head-motion-prediction).

## Requeriments 

Install tensorfolow with cuda, please follow https://www.tensorflow.org/install/

```bash
pip install -r requeriments.txt
```

Ensure users360/head_motion_prediction submodule.

```bash
git submodule init
git submodule update
sed -i 's/^import keras$/import tensorflow.keras as keras/g' users360/head_motion_prediction/training_procedure.py
```

## Usage user360

train/evaluate:

```bash
python training_procedure.py -train -gpu_id 0 -dataset_name David_MMSys_18 -model_name pos_only -init_window 30 -m_window 5 -h_window 25
python training_procedure.py -evaluate -gpu_id 0 -dataset_name David_MMSys_18 -model_name pos_only -init_window 30 -m_window 5 -h_window 25
```

## Usage users360/head_motion_prediction

train/evaluate:

```bash
cd users360/head_motion_prediction
python training_procedure.py -train -gpu_id 0 -dataset_name David_MMSys_18 -model_name pos_only -init_window 30 -m_window 5 -h_window 25
python training_procedure.py -evaluate -gpu_id 0 -dataset_name David_MMSys_18 -model_name pos_only -init_window 30 -m_window 5 -h_window 25
```
## Notebooks

See `notebooks/users360.ipynb`

![Alt Text](docs/requests.gif)
