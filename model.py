from dependency import *



#copied

def get_centroids_prior(embeddings):
    centroids = []
    for speaker in embeddings:
        centroid = 0
        for utterance in speaker:
            centroid = centroid + utterance
        centroid = centroid/len(speaker)
        centroids.append(centroid)
    centroids = torch.stack(centroids)
    return centroids

def get_centroids(embeddings):
    centroids = embeddings.mean(dim=1)
    return centroids

def get_centroid(embeddings, speaker_num, utterance_num):
    centroid = 0
    for utterance_id, utterance in enumerate(embeddings[speaker_num]):
        if utterance_id == utterance_num:
            continue
        centroid = centroid + utterance
    centroid = centroid/(len(embeddings[speaker_num])-1)
    return centroid

def get_utterance_centroids(embeddings):
    """
    Returns the centroids for each utterance of a speaker, where
    the utterance centroid is the speaker centroid without considering
    this utterance
    Shape of embeddings should be:
        (speaker_ct, utterance_per_speaker_ct, embedding_size)
    """
    sum_centroids = embeddings.sum(dim=1)
    # we want to subtract out each utterance, prior to calculating the
    # the utterance centroid
    sum_centroids = sum_centroids.reshape(
        sum_centroids.shape[0], 1, sum_centroids.shape[-1]
    )
    # we want the mean but not including the utterance itself, so -1
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids

def get_cossim_prior(embeddings, centroids):
    # Calculates cosine similarity matrix. Requires (N, M, feature) input
    cossim = torch.zeros(embeddings.size(0),embeddings.size(1),centroids.size(0))
    for speaker_num, speaker in enumerate(embeddings):
        for utterance_num, utterance in enumerate(speaker):
            for centroid_num, centroid in enumerate(centroids):
                if speaker_num == centroid_num:
                    centroid = get_centroid(embeddings, speaker_num, utterance_num)
                output = F.cosine_similarity(utterance,centroid,dim=0)+1e-6
                cossim[speaker_num][utterance_num][centroid_num] = output
    return cossim

def get_cossim(embeddings, centroids):
    # number of utterances per speaker
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)

    # flatten the embeddings and utterance centroids to just utterance,
    # so we can do cosine similarity
    utterance_centroids_flat = utterance_centroids.view(
        utterance_centroids.shape[0] * utterance_centroids.shape[1],
        -1
    )
    embeddings_flat = embeddings.view(
        embeddings.shape[0] * num_utterances,
        -1
    )
    # the cosine distance between utterance and the associated centroids
    # for that utterance
    # this is each speaker's utterances against his own centroid, but each
    # comparison centroid has the current utterance removed
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)

    # now we get the cosine distance between each utterance and the other speakers'
    # centroids
    # to do so requires comparing each utterance to each centroid. To keep the
    # operation fast, we vectorize by using matrices L (embeddings) and
    # R (centroids) where L has each utterance repeated sequentially for all
    # comparisons and R has the entire centroids frame repeated for each utterance
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1)
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1],
        embeddings_expand.shape[-1]
    )
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(
        embeddings.size(0),
        num_utterances,
        centroids.size(0)
    )
    # assign the cosine distance for same speakers to the proper idx
    same_idx = list(range(embeddings.size(0)))
    cos_diff[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0], num_utterances)
    cos_diff = cos_diff + 1e-6
    return cos_diff

def calc_loss_prior(sim_matrix):
    # Calculates loss from (N, M, K) similarity matrix
    per_embedding_loss = torch.zeros(sim_matrix.size(0), sim_matrix.size(1))
    for j in range(len(sim_matrix)):
        for i in range(sim_matrix.size(1)):
            per_embedding_loss[j][i] = -(sim_matrix[j][i][j] - ((torch.exp(sim_matrix[j][i]).sum()+1e-6).log_()))
    loss = per_embedding_loss.sum()    
    return loss, per_embedding_loss

def calc_loss(sim_matrix):
    same_idx = list(range(sim_matrix.size(0)))
    pos = sim_matrix[same_idx, :, same_idx]
    neg = (torch.exp(sim_matrix).sum(dim=2) + 1e-6).log_()
    per_embedding_loss = -1 * (pos - neg)
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss

def normalize_0_1(values, max_value, min_value):
    normalized = np.clip((values - min_value) / (max_value - min_value), 0, 1)
    return normalized

def mfccs_and_spec(wav_file, wav_process = False, calc_mfccs=False, calc_mag_db=False):    
    sound_file, _ = librosa.core.load(wav_file, sr=hp.data.sr)
    window_length = int(hp.data.window*hp.data.sr)
    hop_length = int(hp.data.hop*hp.data.sr)
    duration = hp.data.tisv_frame * hp.data.hop + hp.data.window
    
    # Cut silence and fix length
    if wav_process == True:
        sound_file, index = librosa.effects.trim(sound_file, frame_length=window_length, hop_length=hop_length)
        length = int(hp.data.sr * duration)
        sound_file = librosa.util.fix_length(sound_file, length)
        
    spec = librosa.stft(sound_file, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
    mag_spec = np.abs(spec)
    
    mel_basis = librosa.filters.mel(hp.data.sr, hp.data.nfft, n_mels=hp.data.nmels)
    mel_spec = np.dot(mel_basis, mag_spec)
    
    mag_db = librosa.amplitude_to_db(mag_spec)
    #db mel spectrogram
    mel_db = librosa.amplitude_to_db(mel_spec).T
    
    mfccs = None
    if calc_mfccs:
        mfccs = np.dot(librosa.filters.dct(40, mel_db.shape[0]), mel_db).T
    
    return mfccs, mel_db, mag_db


#end


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


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
        
        #if self.model_params["enable_fc1"]:
        #    self.fc1 = nn.Linear(in_features = 768, out_features = self.model_params["fc1_dim"])
        #if self.model_params["enable_fc2"]:
        #    self.fc2 = nn.Linear(in_features = self.model_params["fc1_dim"], out_features = self.model_params["fc2_dim"])
        
        #self.avpool = nn.AvgPool1d(kernel_size = 5)
        #self.fc3 = nn.Linear(in_features = self.model_params["fc2_dim"], out_features = self.model_params["embeding"])

        self.fcf = nn.Linear(in_features = 768, out_features = 512)
        self.mish = Mish()
        self.fcf2 = nn.Linear(in_features = 512, out_features = 512)

        self.criterion = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='softmax')

        print(self.parameters())
        
        self.check_times = True
        self.check_enable_wav2vec = False


    def forward(self, audio, attention_mask):

        if self.learning_params["block"] or (self.current_epoch < self.learning_params["start_learning_feature_epoch"]):
            self.feature_extractor.eval()
            with torch.no_grad():
                hidden = self.feature_extractor(audio, attention_mask).last_hidden_state
        else:
            self.check_enable_wav2vec = True
            self.feature_extractor.train()
            hidden = self.feature_extractor(audio, attention_mask).last_hidden_state

        #if self.model_params["enable_fc1"]:
        #    hidden = self.fc1(hidden)

        #hidden = hidden.transpose(1,2).contiguous()
        #hidden = self.avpool(hidden)
        #hidden = hidden.transpose(1,2).contiguous()
        
        #if self.model_params["enable_fc2"]:
        #    hidden = self.fc2(hidden)
        
        hidden = torch.mean(hidden, dim = 1)
        #hidden = self.fc3(hidden)'

        hidden = self.fcf(hidden)
        hidden = self.fcf2(self.mish(hidden))

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
        
        if self.check_enable_wav2vec and self.check_times:
            self.check_times = False
            print("check!!!!!!!!!!!!!!!!!!!!!")


    def validation_step(self, batch, batch_idx):
        x, mask, speakers = batch
        output = self(x, mask)
        loss = self.loss_function(output, speakers)


        assert self.hparams["data_params"]["utterences_per_speaker"] % 2 == 0
        
        N = self.hparams["data_params"]["number_of_speakers"]
        M = self.hparams["data_params"]["utterences_per_speaker"]
        x = torch.reshape(x, (N, M, -1))
        mask = torch.reshape(mask, (N, M, -1))

        enrollment_batch, verification_batch = torch.split(x, int(x.size(1)/2), dim=1)
        enrollment_mask_batch, verification_mask_batch = torch.split(mask, int(mask.size(1)/2), dim=1)

        enrollment_batch = torch.reshape(enrollment_batch, (N * M//2, -1))
        verification_batch = torch.reshape(verification_batch, (N * M//2, -1))
        enrollment_mask_batch = torch.reshape(enrollment_mask_batch, (N * M//2, -1))
        verification_mask_batch = torch.reshape(verification_mask_batch, (N * M//2, -1))

        perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
        unperm = list(perm)
        for i,j in enumerate(perm):
            unperm[j] = i
            
        verification_batch = verification_batch[perm]
        verification_mask_batch = verification_mask_batch[perm]

        enrollment_embeddings = self(enrollment_batch, enrollment_mask_batch)
        verification_embeddings = self(verification_batch, verification_mask_batch)
        
        verification_embeddings = verification_embeddings[unperm]


        enrollment_embeddings = torch.reshape(enrollment_embeddings, (N, M//2, enrollment_embeddings.size(1)))
        verification_embeddings = torch.reshape(verification_embeddings, (N, M//2, verification_embeddings.size(1)))
        
        enrollment_centroids = get_centroids(enrollment_embeddings)
        
        sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

        # calculating EER
        diff = 1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0
        for thres in [0.01*i+0.5 for i in range(50)]:
            sim_matrix_thresh = sim_matrix>thres
            
            FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(N))])
            /(N-1.0)/(float(M/2))/N)

            FRR = (sum([M/2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(N))])
            /(float(M/2))/N)
            
            # Save threshold when FAR = FRR (=EER)
            if diff> abs(FAR-FRR):
                diff = abs(FAR-FRR)
                EER = (FAR+FRR)/2
                EER_thresh = thres
                EER_FAR = FAR
                EER_FRR = FRR

        #self.log('FRR', FRR, on_step=True, on_epoch=True, logger=True)
        #self.log('FAR', FAR, on_step=True, on_epoch=True, logger=True)
        #self.log('thres', thres, on_step=True, on_epoch=True, logger=True)
        self.log('EER', EER, on_step=False, on_epoch=True, logger=True)
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