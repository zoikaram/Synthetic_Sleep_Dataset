import torch.nn as nn
import torch
import einops
import numpy as np
import torch.optim as optim
import copy
#import utils
#from utils.config import process_config
import utils_config

import torch
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import random
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

args = {"dmodel": 128,
        "heads": 8,
        "dim_feedforward": 128,
        "fc_inner": 1024,
        "num_classes": 5}

from Dataloader import dotdict
from sleepTransformer import SleepTransformer
args = dotdict(args)


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sleepTransformer = SleepTransformer(args).to(device)

loss = torch.nn.CrossEntropyLoss()
best_model = sleepTransformer
optimizer = optim.Adam(sleepTransformer.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=0.03)

loss.to(device)
sleepTransformer.to(device)

sleepTransformer.load_state_dict(torch.load('supervised_model.pt'))

config_file = "fourier_transformer_eeg_nopos.json"
config = utils_config.load_config(config_file)

import wandb
wandb_run = wandb.init(reinit=True, project="sleep_transformers", entity="zoikaram")

from Dataloader import dataloader

sleepTransformer.eval()
tts, preds, inits, batch_loss = [], [], [], []
# hidden = None
with torch.no_grad():
    for batch_idx, data in enumerate(dataloader.test_loader):
        data_eeg = data[0][0].to(device)
        labels = data[1].to(device)
        labels = einops.rearrange(labels, 'b m -> (b m)')

        pred = sleepTransformer(data_eeg)
        total_loss = loss(pred, labels)

        batch_loss.append(total_loss)
        tts.append(labels)
        preds.append(pred)

        #del labels, pred, total_loss, data
        #torch.cuda.empty_cache()

        print(batch_idx)




    tts = torch.cat(tts).cpu().numpy().flatten()
    total_preds = torch.cat(preds, axis=0)
    total_preds = torch.argmax(total_preds, axis=-1)
    # total_preds = torch.cat(preds, axis=0).cpu().numpy()
    total_preds = total_preds.cpu().detach().numpy()

    test_metrics = {}
    mean_batch = sum(batch_loss) / len(batch_loss)
    test_metrics["test_loss"]= mean_batch
    test_metrics["test_acc"] = np.equal(tts, total_preds).sum() / len(tts)
    test_metrics["test_f1"] = f1_score(total_preds, tts,average="macro")
    test_metrics["test_k"] = cohen_kappa_score(total_preds,tts)
    test_metrics["test_perclassf1"] = f1_score(total_preds, tts, average=None)
    classes = ("Wake", 'N1', 'N2', 'N3', 'REM')
    cf_matrix = confusion_matrix(total_preds, tts)
    cf_matrix = cf_matrix / cf_matrix.astype(np.float).sum(axis=0)
    print(cf_matrix)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])

wandb.log({"test_metrics": test_metrics})

print(test_metrics)


plt.figure()
sn.heatmap(df_cm, annot=True, cmap="Greens")
plt.title('Confusion Matrix for 40 patients')
plt.xlabel('True classes')
plt.ylabel('Estimated classes')
plt.savefig('cm.png')

vector = torch.arange(0, len(labels)).cpu()
pred_max = torch.argmax(pred, axis=-1)
plt.figure(figsize=(30,6))
plt.plot(vector, labels.cpu())
plt.plot(vector, pred_max.cpu(), color='magenta')
plt.title('Hypnogram')
plt.xlabel('Epochs')
plt.ylabel('Stages')
plt.legend(['Real data','Predictions Model3'])
plt.savefig('hypnogram.png')