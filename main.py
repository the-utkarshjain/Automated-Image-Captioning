import sys
import torch
import warnings
from experiment import Experiment

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

# Main Driver for your code. Either run `python main.py` which will run the experiment with default config
# or specify the configuration by running `python main.py custom`
if __name__ == "__main__":
    exp_name = 'default'

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    
    if exp_name not in ['default', 'rnn', 'a2']:
        raise NotImplementedError("Requested experiment not supported. Available options: {'default', 'rnn', 'a2'}")

    print("Running Experiment: ", exp_name)
    
    exp = Experiment(exp_name, verbose = False)
    
    exp.run()
    exp.test()