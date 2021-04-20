from dependency import *

class Voice_Encoder_pl(pl.LightningModule):
    def __init__(self, re_dict, *args, **kwargs): #*args, **kwargs hparams, steps_per_epoch
        super().__init__()
        self.save_hyperparameters(re_dict)
        self.save_hyperparameters()

        self.model_params = self.hparams["model_params"]
        self.learning_params = self.hparams["training"]

        #self.swa_model = None
        #self.swa_mode = False

        #print("mixup set: ", self.learning_params["mixup"])
        #if self.learning_params["data_dropout"]:
        #    print("data_dropout activated")
        #    self.time_drop = torchaudio.transforms.TimeMasking(time_mask_param=self.learning_params["time_l"])
        # self.check_random_mixup = False
        self.feature_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.fc1 = nn.Linear(in_features = 768, out_features = self.model_params["fc1_dim"])
        self.avpool = nn.AvgPool1d(5, stride=3)
        self.fc2 = nn.Linear(in_features = self.model_params["fc1_dim"], out_features = self.model_params["fc2_dim"])
        self.fc3 = nn.Linear(in_features = self.model_params["fc2_dim"], out_features = self.model_params["embeding"])

        self.criterion = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='softmax')

    def forward(self, audio, attention_mask):
      if self.learning_params["block"] or (self.current_epoch < self.learning_params["start_learning_feature_epoch"]):
        self.feature_extractor.eval()
        with torch.no_grad():
          hidden = self.feature_extractor(audio, attention_mask).last_hidden_state
      else:
        self.feature_extractor.train()
        hidden = self.feature_extractor(audio, attention_mask).last_hidden_state

      hidden = self.fc1(hidden)
      hidden = hidden.transpose(1,2).contiguous()
      hidden = self.avpool(hidden)
      hidden = hidden.transpose(1,2).contiguous()
      hidden = self.fc2(hidden)
      hidden = torch.sum(hidden, dim = 1)
      hidden = self.fc3(hidden)

      return hidden
    

    def configure_optimizers(self):
        if self.learning_params["optimizer"] == "belief":
            optimizer =  AdaBelief(self.parameters(), lr = self.learning_params["lr"], eps = self.learning_params["eplison_belief"],
                                    weight_decouple = self.learning_params["weight_decouple"], 
                                    weight_decay = self.learning_params["weight_decay"], rectify = self.learning_params["rectify"])
        elif self.learning_params["optimizer"] == "ranger_belief":
            optimizer = RangerAdaBelief(self.parameters(), lr = self.learning_params["lr"], eps = self.learning_params["eplison_belief"],
                                       weight_decouple = self.learning_params["weight_decouple"],  weight_decay = self.learning_params["weight_decay"],)
        elif self.learning_params["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_params["lr"])
        elif self.learning_params["optimizer"] == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_params["lr"])        

        if self.learning_params["add_sch"]:
            #CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
            #MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)
            lr_scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer,
	                                                                        max_lr=self.learning_params["lr"],
	                                                                        steps_per_epoch=self.hparams.steps_per_epoch, #int(len(train_loader))
	                                                                        epochs=self.learning_params["epochs"],
	                                                                        anneal_strategy='linear'),
                        'name': 'lr_scheduler_lr',
                        'interval': 'step', # or 'epoch'
                        'frequency': 1,
                        }
            print("sch added")
            return [optimizer], [lr_scheduler]
        return optimizer
    
    def loss_function(self, x, speakers):
      # N, M, D: N - Number of speakers in a batch, M - Number of utterances for each speaker, D - d-vector
      b, d = x.shape
      speakers_number = speakers[-1] # N - Number of speakers in a batch, 

      x = x.view(speakers_number, -1, d)
      loss = self.criterion(x)

      return loss


    def training_step(self, batch, batch_idx):
        #also Manual optimization exist
        x, mask, speakers = batch
        output = self(x, mask)
        loss = self.loss_function(output, speakers)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True) # prog_bar=True
        return loss

    #copied
    def get_lr_inside(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


    def training_epoch_end(self, outputs):
        self.log('epoch_now', self.current_epoch, on_step=False, on_epoch=True, logger=True)
        (oppp) =  self.optimizers(use_pl_optimizer=True)
        self.log('lr_now', self.get_lr_inside(oppp), on_step=False, on_epoch=True, logger=True)


    def validation_step(self, batch, batch_idx):
        x, mask, speakers = batch
        output = self(x, mask)
        loss = self.loss_function(output, speakers)

        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True) #prog_bar=True,
        return {'val_loss': loss}

"""
    def test_step(self, batch, batch_idx):
        
        x, mask, speakers = batch
        output = self(x, mask)
        loss = self.loss_function(output, speakers)

        return {'test_loss': loss, #!!!!!!!!!!


    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        self.log('test_f1_score_weighted', f1_scored_w, on_step=False, on_epoch=True, logger=True) #prog_bar=True,
        self.log('test_f1_score_macro', f1_scored_m, on_step=False, on_epoch=True,  logger=True) #prog_bar=True,

"""