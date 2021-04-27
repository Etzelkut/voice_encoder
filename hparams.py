# data params
import os 

local_path = os.path.abspath(os.getcwd())

data_params = {
    "path_save": os.path.join(local_path, "weights"),
    "path_test": os.path.join(local_path, "test"),

    "path": os.path.join(local_path, "data/wav/"), 
    "sampling_rate": 16000,
    "max_seconds": 6, 
    "max_length": None,
    "utterences_per_speaker": 8, # M
    "full_data": True,
    "window_size": None,
    "step_size": None,
    "shuffle_speakers": False,
    "number_of_speakers": 6, # N
    "num_workers": 4,
    "dataloader_shuffle": True,
}

# model params

model_params = {
    "enable_fc1": False,
    "fc1_dim": 1,
    "enable_fc2": False,
    "fc2_dim": 1,
    "embeding": 1
}


# learning params

learning_params = {
    "block": False,
    "start_learning_feature_epoch": 0,
    
    "optimizer": "adamW", # "belief", "ranger_belief", "adam", adamW
    "lr": 1e-4, #
    "eplison_belief": 1e-16,
    "beta": [0.9, 0.999], # not used
    "weight_decouple": True, 
    "weight_decay": 1e-4,
    "rectify": True,
    #
    "add_sch": True,
    #
    "epochs": 30, #
}

hparams_encoder = {
    "model_params": model_params,
    "training": learning_params,
    "data_params": data_params,
}