#!/bin/env python3

import glob
import os
import time
import torch
from datetime import timedelta
import numpy as np
import pandas as pd
#import tensorflow as tf
from IPython.utils.text import columnize
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import utils.utils as utils
#import utils.pos_utils as pos_utils
import utils.model_utils as model_utils
import data_preparation.data_preparation_pos as data_preparation_pos
import random as python_random
from transformers import BertTokenizer

metric_names = {"pos": "Accuracy"}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# For reproducibility:
np.random.seed(42)
python_random.seed(42)
#tf.random.set_seed(42)

def reconstruct_subwords(subword_locations, filtered_preds, logits):
    """Assemble subwords back into the original word in the global lists
    of tokens, labels and predictions, and select a predicted tag"""
    new_preds = []
    prev_end = 0

    for start, end in subword_locations:
        if len(set(filtered_preds[start:end])) > 1:
            # Subword predictions do not all agree
            temp = np.array([(M.max(), M.argmax()) for M in logits[start:end]])
            prediction = temp[temp[:, 0].argmax(), 1]
        else:
            prediction = filtered_preds[start]
        new_preds += filtered_preds[prev_end:start] + [prediction]
        prev_end = end

    # Last subword onwards
    new_preds += filtered_preds[prev_end:]

    return new_preds




class Trainer:
    def __init__(self, data_path, task, model_name, run_name, name_sub_info, use_class_weights=False):
        score_functions = {"pos": self.get_score_pos}

        # self.training_lang = training_lang
        self.data_path = data_path
        # self.lang_path = data_path + training_lang + "/"
        self.task = task
        if self.task == "pos":
            self.eval_info = {}
        self.metric = score_functions[task]
        self.use_class_weights = use_class_weights
        self.class_weights = None
        self.sub_info = name_sub_info
        if self.data_path == True:
                   self.data_path = model_utils.download_datasets(self.task, self.sub_info)

        # Model names
        self.model_name = model_name

        if run_name == '':
            self.save_model_name = model_name.replace("/", "_")
        else:
            self.save_model_name = run_name


    def build_model(self, max_length, train_batch_size, learning_rate, epochs, num_labels,
                    tagset=None, eval_batch_size=32):
        

        self.model, self.tokenizer = model_utils.create_model(self.model_name,  self.task, num_labels=num_labels)
        self.model = model_utils.compile_model(self.model, learning_rate)
        print("Successfully built", self.model_name)
        self.max_length = max_length
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_labels = num_labels
        if tagset:
            self.tagset = tagset
            self.label_map = {label: i for i, label in enumerate(tagset)}
        self.eval_batch_size = eval_batch_size

    def setup_checkpoint(self, checkpoints_path):
        self.checkpoint_dir = checkpoints_path +  self.task + "/" + self.sub_info  + "/"
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        suffix = ""
        self.suffix = suffix
        self.checkpoint_filepath = \
            self.checkpoint_dir + self.save_model_name + f"_{self.task}_checkpoint{suffix}.hdf5"
        print("Checkpoint file:", self.checkpoint_filepath)
        self.temp_weights_filepath = self.checkpoint_dir + self.save_model_name + "_temp.hdf5"
        print("Temp weights file:", self.temp_weights_filepath)

    def setup_eval(self, data, dataset_name):
        self.eval_info[dataset_name] = {}
        self.eval_info[dataset_name]["all_words"] = []
        self.eval_info[dataset_name]["all_labels"] = []
        self.eval_info[dataset_name]["real_tokens"] = []
        self.eval_info[dataset_name]["subword_locs"] = []
        acc_lengths = 0

        for i in range(len(data)):
            self.eval_info[dataset_name]["all_words"].extend(data[i]["tokens"])  # Full words
            self.eval_info[dataset_name]["all_labels"].extend(
                [self.label_map[label] for label in data[i]["tags"]])
            _, _, idx_map = self.tokenizer.subword_tokenize(data[i]["tokens"], data[i]["tags"])
            # Examples always start at a multiple of max_length
            # Where they end depends on the number of resulting subwords
            example_start = i * self.max_length
            example_end = example_start + len(idx_map)
            self.eval_info[dataset_name]["real_tokens"].extend(
                np.arange(example_start, example_end, dtype=int))
            # Get subword starts and ends
            sub_ids, sub_starts, sub_lengths = np.unique(idx_map, return_counts=True,
                                                         return_index=True)
            sub_starts = sub_starts[sub_lengths > 1] + acc_lengths
            sub_ends = sub_starts + sub_lengths[sub_lengths > 1]
            self.eval_info[dataset_name]["subword_locs"].extend(
                np.array([sub_starts, sub_ends]).T.tolist())
            acc_lengths += len(idx_map)

    def prepare_data(self, limit=None):
        datasets = {}
        dataset_names = ["train", "dev", "train_eval", "test"]

        dataset = None
        data = None
        for dataset_name in tqdm(dataset_names):
            # Load plain data and TF dataset
            if self.task == "pos":

                # if self.data_path == True:
                #    self.data_path = model_utils.download_datasets(self.task, self.sub_info)

                data, dataset = data_preparation_pos.load_dataset(
                    self.data_path, self.tokenizer, self.model, self.max_length,
                    tagset=self.tagset, dataset_name=dataset_name
                )
                if dataset_name != "train":
                    self.setup_eval(data, dataset_name)
            if dataset_name == "train":
                dataset, batches = model_utils.make_batches(
                    dataset, self.train_batch_size, repetitions=self.epochs, shuffle=True
                )
            else:
                dataset, batches = model_utils.make_batches(
                    dataset, self.eval_batch_size, repetitions=1, shuffle=False
                )
            datasets[dataset_name] = (dataset, batches, data)

        self.train_dataset, self.train_batches, self.train_data = datasets["train"]
        self.dev_dataset, self.dev_batches, self.dev_data = datasets["dev"]
        self.train_eval_dataset, self.train_eval_batches, self.train_eval_data = datasets[
            "train_eval"]

    def setup_training(self, load_previous_checkpoint=False):
        self.history = History(self)
        if load_previous_checkpoint:
            print("Loading from", self.checkpoint_filepath)
            self.history.load_from_checkpoint()
            self.model.load_weights(self.checkpoint_filepath)

    def reset_to_epoch_start(self):
        self.model.load_weights(self.temp_weights_filepath)

    def handle_oom(self, f, *args, **kwargs):
        output = None
        while True:
            try:
                output = f(*args, **kwargs)
            except tf.errors.ResourceExhaustedError:
                print("Out of memory, retrying...")
                if f == self.model.fit:
                    # Otherwise it will see some data more than once
                    print("Resetting to weights at epoch start")
                    self.reset_to_epoch_start()
                continue
            break
        return output

    def show_time(self, epoch):
        elapsed = time.time() - self.start_time
        print("{:<25}{:<25}".format("Elapsed:", str(timedelta(seconds=np.round(elapsed)))))
        remaining = elapsed / (epoch + 1 - self.history.start_epoch) * (
                self.epochs + self.history.start_epoch - (epoch + 1))
        print("{:<25}{:<25}".format("Estimated remaining:",
                                    str(timedelta(seconds=np.round(remaining)))))
        return elapsed, remaining

    def show_progress_bar(self, epoch):
        bar = tqdm(range(self.history.start_epoch, self.history.start_epoch + self.epochs),
                   ncols=750, bar_format="{l_bar}{bar}{n}/{total}")
        bar.update(epoch - self.history.start_epoch + 1)
        bar.refresh()
        # tqdm.write("") # So the bar appears

    def get_score_pos(self, preds, train_eval_data, dataset_name):
        # FIXME: get rid of this redundant "train eval_data" here
        filtered_preds = preds[0].argmax(axis=-1).flatten()[
            self.eval_info[dataset_name]["real_tokens"]].tolist()
        filtered_logits = \
            preds[0].reshape((preds[0].shape[0] * preds[0].shape[1], preds[0].shape[2]))[
                self.eval_info[dataset_name]["real_tokens"]]
        new_preds = reconstruct_subwords(
            self.eval_info[dataset_name]["subword_locs"], filtered_preds, filtered_logits
        )
        assert len(new_preds) == len(self.eval_info[dataset_name]["all_labels"])
        return (np.array(self.eval_info[dataset_name]["all_labels"]) == np.array(new_preds)).mean()

    def save_checkpoint(self, dev_score):
        print(
            f"Dev score improved from {self.history.best_dev_score:.4f} "
            f"to {dev_score:.4f}, saving to {self.checkpoint_filepath}")
        self.model.save_weights(self.checkpoint_filepath)

    def train(self):

        self.start_time = time.time()

        for epoch in range(self.history.start_epoch, self.history.start_epoch + self.epochs):
            print(f"Epoch: {epoch}")
            epoch_start = time.time()

            # Fit and evaluate
            hist = self.handle_oom(self.model.fit, self.train_dataset, epochs=1,
                                   steps_per_epoch=self.train_batches,
                                   class_weight=self.class_weights, verbose=1)
            loss = hist.history["loss"][0]
            print("Saving temp weights...")
            self.model.save_weights(self.temp_weights_filepath)
            train_preds = self.handle_oom(self.model.predict, self.train_eval_dataset,
                                          steps=self.train_eval_batches, verbose=1)
            dev_preds = self.handle_oom(self.model.predict, self.dev_dataset,
                                        steps=self.dev_batches, verbose=1)

            # Show progress
            _ = self.show_time(epoch)
            epoch_duration = time.time() - epoch_start

            # Calculate scores
            train_score = self.metric(train_preds, self.train_eval_data, "train_eval")
            dev_score = self.metric(dev_preds, self.dev_data, "dev")
            if dev_score > self.history.best_dev_score:
                self.save_checkpoint(dev_score)
                self.history.update_best_dev_score(train_score,
                                                   dev_score,
                                                   epoch,
                                                   epoch_duration,
                                                   dev_preds)

            # Update and show history
            self.history.update_hist(epoch, loss, train_score, dev_score, epoch_duration)
            self.history.show_hist()
            # self.history.plot()

    def make_definitive(self):
        rename_files = [self.checkpoint_filepath, self.history.log_filepath,
                        self.history.checkpoint_params_filepath]
        for file in rename_files:
            os.replace(file, file.replace("_checkpoint", "").replace(self.suffix, ""))

    def get_main_params(self):
        include = ["training_lang", "data_path", "task", "use_class_weights",
                   "model_name", "max_length", "train_batch_size", "eval_batch_size",
                   "learning_rate", "epochs", "num_labels", "checkpoint_filepath"]
        return {k: v for k, v in self.__dict__.items() if k in include}


class History:
    def __init__(self, trainer):
        # Dirs and files
        self.logs_dir = trainer.checkpoint_dir + "logs/"
        self.log_filepath = self.logs_dir + "{}_{}_checkpoint_log{}.tsv".format(
            trainer.save_model_name, trainer.task, trainer.suffix)
        self.checkpoint_params_filepath = self.logs_dir + "{}_{}_checkpoint_params{}.tsv".format(
            trainer.save_model_name, trainer.task, trainer.suffix)
        self.checkpoint_report_filepath = self.logs_dir + "{}_{}_checkpoint_report{}.tsv".format(
            trainer.save_model_name, trainer.task, trainer.suffix)
        if not os.path.isdir(self.logs_dir):
            os.makedirs(self.logs_dir)

        # History attributes
        self.epoch_list = []
        self.loss_list = []
        self.train_score_list = []
        self.dev_score_list = []
        self.total_time_list = []
        self.start_epoch = 0
        self.best_dev_score = 0
        self.best_dev_epoch = None
        self.best_dev_total_time = None

        # Other
        self.task = trainer.task
        self.dev_data = trainer.dev_data
        self.metric_name = metric_names[self.task]
        self.trainer_params = trainer.get_main_params()

    def load_from_checkpoint(self):
        log = pd.read_csv(self.log_filepath)
        end_index = log["dev_score"].argmax() + 1
        self.epoch_list = log["epoch"].values[:end_index].tolist()
        self.loss_list = log["loss"].values[:end_index].tolist()
        self.train_score_list = log["train_score"].values[:end_index].tolist()
        self.dev_score_list = log["dev_score"][:end_index].values.tolist()
        self.total_time_list = log["total_time"][:end_index].values.tolist()
        self.best_dev_score = self.dev_score_list[-1]
        self.best_dev_epoch = self.epoch_list[-1]
        self.start_epoch = self.epoch_list[-1] + 1
        print(f"Checkpoint dev score: {self.best_dev_score}")

    @staticmethod
    def convert_time(t):
        return str(timedelta(seconds=np.round(t)))

    def update_best_dev_score(self, train_score, dev_score, epoch, epoch_duration, dev_preds):
        self.best_dev_score = dev_score
        self.best_dev_epoch = epoch
        if self.total_time_list:
            new_time = self.total_time_list[-1] + epoch_duration
        else:
            new_time = epoch_duration
        self.best_dev_total_time = self.convert_time(new_time)
        # Parameters with which the score was obtained
        params = {**self.trainer_params,
                  **{"epoch": epoch, "train_score": train_score, "dev_score": dev_score,
                     "total_training_time": self.best_dev_total_time}}
        pd.DataFrame.from_dict(params, orient="index").to_csv(
            self.checkpoint_params_filepath, index=False
        )

    def update_hist(self, epoch, loss, train_score, dev_score, epoch_duration):
        self.epoch_list.append(epoch)
        self.loss_list.append(loss)
        self.train_score_list.append(train_score)
        self.dev_score_list.append(dev_score)
        if self.total_time_list:
            new_time = self.total_time_list[-1] + epoch_duration
        else:
            new_time = epoch_duration
        self.total_time_list.append(new_time)

        pd.DataFrame({"epoch": self.epoch_list,
                      "loss": self.loss_list,
                      "train_score": self.train_score_list,
                      "dev_score": self.dev_score_list,
                      "total_time": self.total_time_list,
                      "total_time_h:m:s": [self.convert_time(t) for t in self.total_time_list]}
                     ).to_csv(self.log_filepath, index=False, sep="\t")

    def show_hist(self):
        print("History:")
        print("Best dev score so far: {:.3f}".format(self.best_dev_score))
        print("{:<20}{:<20}{:<20}{:<20}".format("Epoch",
                                                "Loss",
                                                "Train score",
                                                "Dev score"))
        for epoch in self.epoch_list:
            # if epoch == self.best_dev_epoch:
            #    bold_code = ("\033[1m", "\033[0m") # Add bold to row where best dev score was found
            # else:
            #    bold_code = ("", "")
            print("{:<20}{:<20.3f}{:<20.3f}{:<20.3f}".format(
                self.epoch_list[epoch], self.loss_list[epoch],
                self.train_score_list[epoch],
                self.dev_score_list[epoch]))

    def get_best_dev(self):
        """Score, epoch, time"""
        return self.best_dev_score, self.best_dev_epoch, self.best_dev_total_time
