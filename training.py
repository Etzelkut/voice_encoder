from dependency import *

from dataset import Dataset_vox
from model import Voice_Encoder_pl
from hparams import hparams_encoder


def testing():
    dataset_pl = Dataset_vox(hparams_encoder["data_params"])
    dataset_pl.prepare_data()
    dataset_pl.setup()

    for batch in dataset_pl.train_dataloader():
        x, mask, speakers = batch
        print(x, mask, speakers)
        print(x.shape, mask.shape)
        break
    del dataset_pl
    re_dict_check = hparams_encoder.copy()
    model = Voice_Encoder_pl(re_dict_check)
    hidden = model.forward(x, mask)
    print(hidden.shape)

def train(save_weights = hparams_encoder["data_params"]["path_save"], seed_v = 42):
    seed_v = seed_v
    root_dir = save_weights
    naming = "encoder_try_long_no"

    seed_e(seed_v)

    comet_logger = CometLogger(
        save_dir='/content/log/',
        api_key="23CU99n7TeyZdPeegNDlQ5aHf",
        project_name="encoder-voice",
        workspace="etzelkut",
        # rest_api_key=os.environ["COMET_REST_KEY"], # Optional
        experiment_name = naming, # Optional
    )
    #

    dataset_pl = Dataset_vox(hparams_encoder["data_params"])
    dataset_pl.prepare_data()
    dataset_pl.setup()
    steps_per_epoch = int(len(dataset_pl.train_dataloader()))
    print(steps_per_epoch)


    checkpoint_callback = ModelCheckpoint(
                                      monitor='val_loss',
                                      save_last=True, 
                                      dirpath= root_dir,#os.path.join(path, "/checkpoints"),
                                      filename= naming + 'l{epoch}.ckpt',
                                      save_top_k=1,
                                      mode='min',
                                      )

    every_epoch = CheckpointEveryNSteps(save_step_frequency = steps_per_epoch*2, use_modelcheckpoint_filename = True, pathh=root_dir)

    proj_a = Voice_Encoder_pl(hparams_encoder, steps_per_epoch = steps_per_epoch)

    trainer = Trainer(callbacks=[checkpoint_callback, every_epoch],
                        logger=comet_logger,
                        gpus=1,
                        profiler=True,
                        check_val_every_n_epoch=2,
                        #auto_lr_find=True, #set hparams
                        #gradient_clip_val=0.5,
                        #early_stop_callback=True,
                        max_epochs = hparams_encoder["training"]["epochs"],
                        progress_bar_refresh_rate = 0,
                        deterministic=True,)

    trainer.fit(proj_a, dataset_pl)
    print("saving")
    checkpoint_name = os.path.join(root_dir, naming + '.ckpt')
    trainer.save_checkpoint(checkpoint_name)   
    print("end saving")