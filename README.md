*A ML code to train and evaluate **residue–residue (R–R) contact** predictors for proteins.*

---

## Configuration

All experiment settings are written in YAML and are under `configs/`. Use the `--config` flag to select a file. Create variants to track experiments (e.g., different loss weights or model heads).

---

## Training
Use main.py as entry point. It accepts the following arguments: 
```
  -h, --help            show this help message and exit
  -d DATASET_PATH, --dataset_path DATASET_PATH
                        Path to the dataset.
  -s SAVING_PATH, --saving_path SAVING_PATH
                        Path where to save the results.
  -c CONFIG_NAME, --config_name CONFIG_NAME
                        name of the config file in configs directory.
  --device DEVICE       The device to use (integer) or cpu.
```

### Configs 

All experiment settings are written in YAML and are under `configs/`. Use the `--config_name` flag to select a file within the folder.
Create variants to track experiments (e.g., different loss weights or model heads).

For example, the training I performed for the project can be run using the following commands
```bash
python main.py -d ./int_dataset/ -s ./results/proposed_bce -c proposed_bce.yaml

python main.py -d ./int_dataset/ -s ./results/base_bce -c base_bce.yaml
```


## Project layout

```
ContactProject/
├─ configs/           # YAML experiment configs
├─ dataset.py         # custom protein dataset & loaders (sequences, contacts, masks)
├─ loss.py            # loss functions
├─ main.py            # entry point: train/eval
├─ model.py           # model wrappers
├─ utils.py           # metrics, logging, helpers (collate function)
└─ requirements.txt   # Python dependencies
```
