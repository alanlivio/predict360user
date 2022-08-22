# users360

Python library to perform user behavior analysis on 360-degree videos from [Rondon datasets collection](https://gitlab.com/miguelfromeror/head-motion-prediction).

## Requeriments 

Install tensorflor, please follow https://www.tensorflow.org/install/

```bash
pip install -r requeriments.txt
```

## Usage

make dataset:

```bash
python make_dataset.py
```

train:

```bash
python main.py --train -gpu_id 0 -model_name pos_only -dataset_name David_MMSys_18 -m_window 25 -i_window 25 -h_window 25
```

predict:

```bash
python main.py --evaluate -gpu_id 0 -model_name pos_only -dataset_name David_MMSys_18 -m_window 25 -i_window 25 -h_window 25
```

## Notebooks

See `notebooks/users360.ipynb`

![Alt Text](docs/requests.gif)
