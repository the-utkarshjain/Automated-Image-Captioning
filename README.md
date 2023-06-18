# Image Captioning using Recurrent Neural Networks

Team members:
1. Utkarsh Jain
2. Marialena Sfyraki
3. Kaiyuan Wang
4. Merve Kilic

## Usage
### Downloading the dataset
Run the get_datasets.ipynb notebook to fetch the dataset.

### Running the experiments
To run the experiments, run one of the following commands on shell:

1.  `$ python main.py default` (for LSTM decoder)

2.  `$ python main.py rnn` (for RNN decoder)

3.  `$ python main.py a2` (for Architecture2)

  
* Define the configuration for your experiment. See `default.json` to see the structure and available options.

* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment

* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir. This can be configured in `contants.py`

* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training pr evaluate performance.


## Files

- main.py: Main driver class

- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.

- dataset_factory: Factory to build datasets based on config

- model_factory.py: Factory to build models based on config

- constants.py: constants used across the project

- file_utils.py: utility functions for handling files

- caption_utils.py: utility functions to generate bleu scores

- vocab.py: A simple Vocabulary wrapper

- coco_dataset: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset

- get_datasets.ipynb: A helper notebook to set up the dataset in your workspace
