# data params
import os 

local_path = os.path.abspath(os.getcwd())

data_params = {
    "path_save": os.path.join(local_path, "weights"),
    "path": os.path.join(local_path, "data/wav/"), 
    "sampling_rate": 16000,
    "max_seconds": 8, 
    "max_length": None,
    "utterences_per_speaker": 5, # M
    "full_data": True,
    "window_size": None,
    "step_size": None,
    "shuffle_speakers": True,
    "number_of_speakers": 10, # N
    "num_workers": 2,
    "dataloader_shuffle": True,
}

# model params

model_params = {
    "fc1_dim": 512,
    "fc2_dim": 512,
    "embeding": 256
}


# learning params

learning_params = {
    "block": True,
    "start_learning_feature_epoch": None,
    
    "optimizer": "belief", # "belief", "ranger_belief", "adam", adamW
    "lr": 3e-4, #
    "eplison_belief": 1e-16,
    "beta": [0.9, 0.999], # not used
    "weight_decouple": True, 
    "weight_decay": 1e-4,
    "rectify": True,
    #
    "add_sch": False,
    #
    "epochs": 10, #
}

hparams_encoder = {
    "model_params": model_params,
    "training": learning_params,
    "data_params": data_params,
}