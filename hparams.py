# data params
import os 

local_path = os.path.abspath(os.getcwd())

data_params = {
    "path_save": os.path.join(local_path, "weights"),
    "path": os.path.join(local_path, "data/wav/"), 
    "sampling_rate": 16000,
    "max_seconds": 8, 
    "max_length": None,
    "utterences_per_speaker": 10, # M
    "full_data": True,
    "window_size": None,
    "step_size": None,
    "shuffle_speakers": False,
    "number_of_speakers": 20, # N
    "num_workers": 4,
    "dataloader_shuffle": True,
}

# model params

model_params = {
    "enable_fc1": False,
    "fc1_dim": 768,
    "enable_fc2": True,
    "fc2_dim": 512,
    "embeding": 512
}


# learning params

learning_params = {
    "block": False,
    "start_learning_feature_epoch": 60,
    
    "optimizer": "adamW", # "belief", "ranger_belief", "adam", adamW
    "lr": 5e-3, #
    "eplison_belief": 1e-16,
    "beta": [0.9, 0.999], # not used
    "weight_decouple": True, 
    "weight_decay": 1e-4,
    "rectify": True,
    #
    "add_sch": True,
    #
    "epochs": 100, #
}

hparams_encoder = {
    "model_params": model_params,
    "training": learning_params,
    "data_params": data_params,
}