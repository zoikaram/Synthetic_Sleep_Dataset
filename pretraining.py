import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import random
import torch
import einops
import sklearn
import math
import torch.optim as optim
import pandas as pd
import torch.nn as nn
import torch
import einops
import copy
#import utils
#from utils.config import process_config
import utils_config

from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import random

from Dataloader import dotdict #, dataloader

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train = torch.load('/esat/biomeddata/data_drop/r0779205/thesis/Datasets/update21_x_train.pt')
y_train = torch.load('/esat/biomeddata/data_drop/r0779205/thesis/Datasets/update21_y_train.pt')
x_train1 = torch.load('/esat/biomeddata/data_drop/r0779205/thesis/Datasets/update21_2_x_train.pt')
y_train1 = torch.load('/esat/biomeddata/data_drop/r0779205/thesis/Datasets/update21_2_y_train.pt')
x_train = torch.cat((x_train,x_train1[:50]), axis=0)
del x_train1
y_train = torch.cat((y_train, y_train1[:50]), axis=0)
del y_train1

x_val = torch.load('/esat/biomeddata/data_drop/r0779205/thesis/Datasets/update21_x_val.pt')
y_val = torch.load('/esat/biomeddata/data_drop/r0779205/thesis/Datasets/update21_y_val.pt')



args = {"dmodel": 128,
        "heads": 8,
        "dim_feedforward": 128,
        "fc_inner": 1024,
        "num_classes": 5}

from sleepTransformer import SleepTransformer
args = dotdict(args)

sleepTransformer = SleepTransformer(args).to(device)

loss = torch.nn.CrossEntropyLoss()
#best_model = copy.deepcopy(sleepTransformer) #To keep the best one according to validation.
#Auto to copy de douleuei opote grafo auto
best_model = sleepTransformer
optimizer = optim.Adam(sleepTransformer.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=0.03)

loss.to(device)
sleepTransformer.to(device)

config_file = "fourier_transformer_eeg_nopos.json"
config = utils_config.load_config(config_file)

seed = config.training_params.seed
torch.backends.cudnn.enabled = False
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

logs = {"current_epoch":0,
        "current_step":0,
        "steps_no_improve":0,
        "saved_step": 0,
        "train_logs":{},
        "val_logs":{},
        "test_logs":{},
        "best_logs":{"val_loss":100}}

import wandb
wandb_run = wandb.init(reinit=True, project="sleep_transformers", entity="zoikaram")

import tqdm

sleepTransformer.train()

targets, preds, batch_loss, early_stop = [], [], [], False
val_loss = {"total":0}
saved_at_step, prev_epoch_time = 0, 0

for logs["current_epoch"] in range(logs["current_epoch"], config.early_stopping.max_epoch):
    #pbar = tqdm.tqdm(enumerate(dataloader.train_loader), desc="Training", leave=None, position=0)
    for batch_idx in range(y_train.shape[0]):
        # Get the data and labels to gpu
        data_eeg = x_train[batch_idx].to(device)
        labels = y_train[batch_idx].to(device)
        labels = einops.rearrange(labels, 'b m -> (b m)')

        #sleepTransformer.train()

        # Initialize optimizer to 0, so that we don't sum up previous gradients.
        optimizer.zero_grad()

        pred = sleepTransformer(data_eeg)
        total_loss = loss(pred, labels)
        total_loss.backward()

        # To keep them logged but without taking space in gpu
        total_loss = total_loss.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()

        optimizer.step()
        scheduler.step()

        batch_loss.append(total_loss)
        targets.append(labels)
        preds.append(pred)

        del labels, pred, total_loss

        # This is optional but saves much space in gpu_memory
        torch.cuda.empty_cache()

        #if batch_idx==50:
            #break

        pbar_message = "Training batch {0:d}/{1:d} steps no improve {2:d} with ".format(batch_idx,
                                                                                        y_train.shape[0] - 1,
                                                                                        logs["steps_no_improve"])
        mean_batch = np.array(batch_loss).mean(axis=0)
        pbar_message += "Total_loss: {:.3f} ".format(mean_batch)
        print(pbar_message)

        # Check if it's time to validate, we validate every config.early_stopping.validate_every (in sleeptransformer every 100 opt steps)
        if logs["current_step"] % config.early_stopping.validate_every == 0 and \
                logs["current_step"] // config.early_stopping.validate_every >= config.early_stopping.validate_after and \
                batch_idx != 0:
            tts = torch.cat(targets).cpu().numpy().flatten()
            total_preds = np.concatenate(preds, axis=0).argmax(axis=-1)

            train_metrics = {}
            train_metrics["train_loss"] = mean_batch
            train_metrics["train_acc"] = np.equal(tts, total_preds).sum() / len(tts)
            train_metrics["train_f1"] = f1_score(total_preds, tts, average="macro")
            train_metrics["train_k"] = cohen_kappa_score(total_preds, tts)
            train_metrics["train_perclassf1"] = f1_score(total_preds, tts, average=None)
            print(mean_batch, train_metrics['train_acc'], train_metrics["train_f1"], train_metrics["train_k"])

            # Valdiation
            sleepTransformer.eval()
            tts, v_preds, inits, v_batch_loss = [], [], [], []
            # hidden = None
            with torch.no_grad():
                for batch_idx in range(y_val.shape[0]):
                    data_eeg = x_val[batch_idx].to(device)
                    labels = y_val[batch_idx].to(device)
                    labels = einops.rearrange(labels, 'b m -> (b m)')

                    pred = sleepTransformer(data_eeg)
                    total_loss = loss(pred, labels)

                    v_batch_loss.append(total_loss)
                    tts.append(labels)
                    v_preds.append(pred)

                    del labels, pred, total_loss

                tts = torch.cat(tts).cpu().numpy()
                total_preds = torch.cat(v_preds, axis=0)
                total_preds = torch.argmax(total_preds,axis=-1)
                #total_preds = torch.cat(preds, axis=0).cpu().numpy()
                total_preds = total_preds.cpu().detach().numpy()
                #total_preds = torch.Tensor.numpy(total_preds, force=True)
                #total_preds = np.argmax(axis=-1)
                #print(total_preds)

                val_metrics = {}
                # mean_batch = np.array(batch_loss).mean(axis=0)
                mean_batch = sum(v_batch_loss) / len(v_batch_loss)
                val_metrics["val_loss"] = mean_batch
                val_metrics["val_acc"] = np.equal(tts, total_preds).sum() / len(tts)
                val_metrics["val_f1"] = f1_score(total_preds, tts, average="macro")
                val_metrics["val_k"] = cohen_kappa_score(total_preds, tts)
                val_metrics["val_perclassf1"] = f1_score(total_preds, tts, average=None)
                print(mean_batch, val_metrics['val_acc'], val_metrics["val_f1"], val_metrics["val_k"])

            logs["val_logs"][logs["current_step"]] = val_metrics
            logs["train_logs"][logs["current_step"]] = train_metrics
            wandb.log({"train": train_metrics})

            # Flag if its saved dont save it again on $save_every
            not_saved = True

            # If we have a better validation loss
            if (val_metrics["val_loss"] < logs["best_logs"]["val_loss"]):
                logs["best_logs"] = val_metrics

                step = int(logs["current_step"] / config.early_stopping.validate_every)
                message = "Epoch {0:d} step {1:d} with ".format(logs["current_epoch"], step)
                if "val_loss" in val_metrics:
                    message += "Val_loss : {:.6f} ".format(val_metrics["val_loss"])
                if "val_acc" in val_metrics:
                    message += "Acc: {:.2f} ".format(val_metrics["val_acc"] * 100)
                if "val_f1" in val_metrics:
                    message += "F1: {:.2f} ".format(val_metrics["val_f1"] * 100)
                if "val_k" in val_metrics:
                    message += "K: {:.4f} ".format(val_metrics["val_k"])
                if "val_perclassf1" in val_metrics:
                    message += "F1_perclass: {} ".format(
                        "{}".format(str(list((val_metrics["val_perclassf1"] * 100).round(2)))))
                print(message)

                best_model.load_state_dict(sleepTransformer.state_dict())
                logs["saved_step"] = logs["current_step"]
                logs["steps_no_improve"] = 0

                savior = {}
                savior["model_state_dict"] = sleepTransformer.state_dict()
                savior["best_model_state_dict"] = best_model.state_dict()
                savior["optimizer_state_dict"] = optimizer.state_dict()
                savior["logs"] = logs
                #savior["metrics"] = dataloader.metrics
                savior["configs"] = config

                try:
                    torch.save(best_model.state_dict(), "/esat/biomeddata/data_drop/r0779205/thesis/update21_model_500.pt")
                except:
                    raise Exception("Problem in model saving")

                not_saved = False
            else:
                logs["steps_no_improve"] += 1   

            training_cycle = (logs["current_step"] // config.early_stopping.validate_every)
            if not_saved and training_cycle % config.early_stopping.save_every == 0:
                # Some epochs without improvement have passed, we save to avoid losing progress even if its not giving new best
                savior = {}
                savior["model_state_dict"] = sleepTransformer.state_dict()
                savior["best_model_state_dict"] = best_model.state_dict()
                savior["optimizer_state_dict"] = optimizer.state_dict()
                savior["logs"] = logs
                #savior["metrics"] = dataloader.metrics
                savior["configs"] = config

                try:
                    torch.save(best_model.state_dict(), "/esat/biomeddata/data_drop/r0779205/thesis/update21_model_500.pt")
                except:
                    raise Exception("Problem in model saving")

                logs["saved_step"] = logs["current_step"]

            early_stop = False
            # We don't allow early stop to happen if n_steps_stop_after have not pass
            # If they have passed and the model doesnt show improvement in the last n_steps_stop then we can consider it converged and stop.
            if training_cycle > config.early_stopping.n_steps_stop_after and \
                    logs["steps_no_improve"] >= config.early_stopping.n_steps_stop:
                early_stop = True

            # if early stop our model has converged and we can finish
            if early_stop: break

            # reinitialize the lists that keep the variables for those 100 steps
            batch_loss, targets, preds = [], [], []
            sleepTransformer.train()

            saved_at_step = logs["saved_step"] // config.early_stopping.validate_every
            print(saved_at_step)


        logs["current_step"] += 1

    if early_stop: break