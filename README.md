# Synthetic_Sleep_Dataset

A synthetic sleep dataset is generated in this repository in order to improve a sleep stage model performance. \\

In **Dataloader**, real sleep (EEG) data are shaped in a convenient format for training.\\
The .json file contains a configuration with parameters useful for the analysis.
In **Synthetic_data** EEG signals are generated based on labels we import from real data
**pretraining** contains the training process of the SleepTransformer model exclusively with synthetic data
**training_after_pre** contains the supervising phase, where the pretrained model is trained with the real data 
**test** provides the code where the performance is evaluated
**training** contains the code where the model is trained without pretraining
