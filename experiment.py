import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import nltk
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name, verbose = False):
        config_data = read_file_in_dir('./configs/', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)
        
        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__lr = config_data['experiment']['learning_rate']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        
        self.__min_val_loss = float("inf")
        self.__best_model_epoch = 0
        self.__best_model = None  # Save your best model in this field and use this in test method.
        
        self.__verbose = verbose
        self.__device = torch.device('cpu')
        
        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        # Set these Criterion and Optimizers Correctly
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = optim.Adam(self.__model.parameters(), lr = self.__lr)

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])
            
            if os.path.exists(os.path.join(self.__experiment_dir, 'best_model.pt')):
                best_model_state_dict = torch.load(os.path.join(self.__experiment_dir, 'best_model.pt'))
                self.__best_model = best_model_state_dict['model']
                self.__best_model_epoch = best_model_state_dict['best_epoch']
                self.__min_val_loss = best_model_state_dict['min_val_loss']
                print(f"Best validation model loaded from epoch {self.__best_model_epoch} with val loss of {self.__min_val_loss}")

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__device = torch.device('cuda')
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            
            train_loss = self.__train()
            val_loss = self.__val()
               
            if self.__verbose:
                test_loss, bleu1, bleu4 = self.test()
                print(f"Epoch: {epoch+1}/{self.__epochs}, Train: {train_loss}, Val: {val_loss}")
                print(f"Test: {test_loss}, Bleu1: {bleu1}, Bleu4: {bleu4}\n")
            
            if val_loss < self.__min_val_loss:
                self.__min_val_loss = val_loss
                self.__best_model_epoch = epoch + 1
                self.__best_model = self.__model.state_dict()
                self.__save_best_model()
                
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = 0

        loss_print = 0
        for i, (images, captions, _) in enumerate(self.__train_loader):
            images = images.to(self.__device)
            captions = captions.to(self.__device)
           
            output = self.__model(images, captions, self.__device)
            
            loss = self.__criterion(output, captions)
            training_loss += loss.item()
            loss_print += loss.item()
            
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()
            
            if self.__verbose and ((i+1) % int(len(self.__train_loader)/10) == 0):
                print(f"Iteration: {i+1}/{len(self.__train_loader)}, Train: {loss_print*10/len(self.__train_loader)}")
                loss_print = 0
            
        return training_loss/len(self.__train_loader)

    # Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = 0

        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                images = images.to(self.__device)
                captions = captions.to(self.__device)

                output = self.__model(images, captions, self.__device)

                loss = self.__criterion(output, captions)
                val_loss += loss.item()

        return val_loss/len(self.__val_loader)

    # Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):       
        print(f"Running evaluations on test dataset. Using saved model from epoch {self.__best_model_epoch} with val loss of {self.__min_val_loss}")
        self.__model.load_state_dict(self.__best_model)
        
        self.__model.eval()
        test_loss = 0
        bleu1 = 0
        bleu4 = 0

        with torch.no_grad():
            for i, (images, captions, img_ids) in enumerate(self.__test_loader):
                images = images.to(self.__device)
                captions = captions.to(self.__device)

                output = self.__model(images, captions, self.__device)

                loss = self.__criterion(output, captions)
                test_loss += loss.item()
                
                generated_captions = self.__model.generate_captions(images, self.__generation_config, self.__device)
                
                bleu1_mini, bleu4_mini = self.__compute_bleu(generated_captions, img_ids, self.__coco_test)
                bleu1 += bleu1_mini/len(img_ids)
                bleu4 += bleu4_mini/len(img_ids)
        
        test_loss = test_loss/len(self.__test_loader)
        bleu1 = bleu1/len(self.__test_loader)
        bleu4 = bleu4/len(self.__test_loader)
        
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss, bleu1, bleu4)
        self.__log(result_str)

        return test_loss, bleu1, bleu4
 
    def __compute_bleu(self, generated_captions, img_ids, test_coco):
        bleu1_score = 0
        bleu4_score = 0
        
        for generated_caption, img_id in zip(generated_captions, img_ids):
                reference_captions = []
                
                for ann in test_coco.imgToAnns[img_id]:
                    caption = ann['caption']
                    tokens = nltk.tokenize.word_tokenize(str(caption).lower())
                    reference_captions.append(tokens)
                    
                bleu1_score += bleu1(reference_captions, generated_caption)
                bleu4_score += bleu4(reference_captions, generated_caption)
        
        return bleu1_score, bleu4_score
    
    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)
        
    def __save_best_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'best_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'best_epoch': self.__best_model_epoch, 'min_val_loss': self.__min_val_loss}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}/{}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, self.__epochs, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
    
    def return_members(self):
        """
        Function to return whatever class members you want to access
        """
        
        self.__model.load_state_dict(self.__best_model)
        return self.__device, self.__generation_config, self.__model, self.__vocab, self.__coco_test, self.__test_loader
