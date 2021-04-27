from dependency import *

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


class VoxCeleb(Dataset):
    
    def __init__(self, path, sampling_rate = 16000, max_seconds = 8, utterences_per_speaker = 2, full_data = True, window_size = None, step_size = None, 
                 shuffle=True): # path = "/content/data/wav/"

      # 3 - 4.1 seconds max

      if full_data:
        assert utterences_per_speaker < 45 # do max 40
      else:
        assert utterences_per_speaker < (267/step_size) - 1 # do max 40 if step size = 2 sec and window = 4 sec
      
      self.sampling_rate = sampling_rate
      self.max_seconds = max_seconds

      self.M = utterences_per_speaker
      self.full_data = full_data
      
      self.window_size = window_size
      self.step_size = step_size

      self.shuffle = shuffle

      self.speakers = os.listdir(path)
      self.speakers_file_list = {}
      self.utterences = 0
      self.speakers_number = len(self.speakers)
      for i in range(len(self.speakers)):
        id_path = os.path.join(path, self.speakers[i])
        self.speakers_file_list[self.speakers[i]] = getListOfFiles(id_path)
        self.utterences += len(self.speakers_file_list[self.speakers[i]])
    
    def sample_audio(self, audio):
      seconds = audio.shape[0] / self.sampling_rate
      max_samples = int(self.sampling_rate * self.max_seconds)
      if seconds > self.max_seconds:
        start_audio = random.sample(range(0, audio.shape[0] - max_samples), 1)[0]
        audio = audio[start_audio:start_audio + max_samples]
      return audio

    def read_audio(self, path):
      audio, sr = sf.read(path)
      if sr != self.sampling_rate:
        print(path)
        raise Exception("sampling rate broken")
      audio = self.sample_audio(audio)
      return audio

    def __len__(self):
      return self.speakers_number
    
    def __getitem__(self, idx):
      
      if self.shuffle:
        selected_speaker = random.sample(self.speakers, 1)[0]  # select random speaker
      else:
        selected_speaker = self.speakers[idx]   

      if self.full_data:
        list_of_audio = random.sample(self.speakers_file_list[selected_speaker], self.M)
        list_of_audio = list(map(self.read_audio, list_of_audio))
      else:
        raise Exception("Only full data avaulable now")

        # load utterance spectrogram of selected speaker

        # select M utterances per speaker

        # utterances of a speaker [batch(M), n_mels, frames]

        # transpose [batch, frames, n_mels]
      return list_of_audio, [selected_speaker]*len(list_of_audio)


def collate_fn_vox(batch, processor, sampling_rate, max_length = None):
  
  speakers_number = len(batch)

  connected_audio_list = []
  speakers = []
  for list_of_audio, selected_speaker in batch:
    connected_audio_list += list_of_audio
    speakers+=selected_speaker
  
  input_values = processor(connected_audio_list, padding = True, max_length = max_length, return_attention_mask = True, sampling_rate = sampling_rate, return_tensors="pt")
  speakers.append(speakers_number)

  return input_values.input_values, input_values.attention_mask, speakers



class VoxCeleb_test(Dataset):
    
    def __init__(self, path, sampling_rate = 16000, max_seconds = 8,): # path = "/content/test/wav/", "/content/test/veri_test.txt"

      # 3 - 4.1 seconds max
      self.path = os.path.join(path, "wav")
      path_txt = os.path.join(path, "test_division.txt")
      with open(path_txt) as f:
        self.audio_pairs = [line.split() for line in f]
      
      self.sampling_rate = sampling_rate
      self.max_seconds = max_seconds

    def sample_audio(self, audio):
      seconds = audio.shape[0] / self.sampling_rate
      max_samples = int(self.sampling_rate * self.max_seconds)
      if seconds > self.max_seconds:
        start_audio = random.sample(range(0, audio.shape[0] - max_samples), 1)[0]
        audio = audio[start_audio:start_audio + max_samples]
      return audio

    def read_audio(self, path):
      audio, sr = sf.read(path)
      if sr != self.sampling_rate:
        print(path)
        raise Exception("sampling rate broken")
      audio = self.sample_audio(audio)
      return audio

    def __len__(self):
      return len(self.audio_pairs)
    
    def __getitem__(self, idx):
      label, path1, path2 = self.audio_pairs[idx]
      path1, path2 = os.path.join(self.path, path1), os.path.join(self.path, path2)
      audio1, audio2 = self.read_audio(path1), self.read_audio(path2)
      return audio1, audio2, label



def collate_fn_vox_test(batch, processor, sampling_rate = 16000, max_length = None):
  
  pairs_number = len(batch)

  connected_audio_list_first = []
  connected_audio_list_second = []
  labels = []

  max_ll = 0

  for audio1, audio2, label in batch:
    max_l = max(audio1.shape[0], audio2.shape[0])
    max_ll = max(max_ll, max_l)

    connected_audio_list_first.append(audio1)
    connected_audio_list_second.append(audio2)
    label = int(label)
    labels.append(label)

  if max_length is None:
    max_length = max_ll

  connected_audio_list_first += connected_audio_list_second
  input_values = processor(connected_audio_list_first, padding = True, max_length = max_length, return_attention_mask = True, sampling_rate = sampling_rate, return_tensors="pt")
  #input_values_second = processor(connected_audio_list_second, padding = True, max_length = max_length, return_attention_mask = True, sampling_rate = sampling_rate, return_tensors="pt")
  
  input_values, mask = input_values.input_values, input_values.attention_mask

  a, b =  torch.split(input_values, int(input_values.size(0)/2), dim=0)
  amask, bmask = torch.split(mask, int(input_values.size(0)/2), dim=0)

  labels.append(pairs_number)

  return (a, amask), (b, bmask), labels

  #return (input_values_first.input_values, input_values_first.attention_mask), (input_values_second.input_values, input_values_second.attention_mask), labels



class Dataset_vox(pl.LightningDataModule):
    def __init__(self, conf, *args, **kwargs): #*args, **kwargs hparams, steps_per_epoch
      super().__init__()
      self.hparams = conf
      self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def prepare_data(self):
      print("can add download here")
    
    def setup(self):

      dataset = VoxCeleb(self.hparams["path"], sampling_rate=self.hparams["sampling_rate"], max_seconds=self.hparams["max_seconds"], 
                         utterences_per_speaker=self.hparams["utterences_per_speaker"], full_data=self.hparams["full_data"], 
                         window_size=self.hparams["window_size"], step_size=self.hparams["step_size"], shuffle=self.hparams["shuffle_speakers"])

      #size_of_main = len(dataset)
      #self.dataset_train, self.dataset_val = torch.utils.data.random_split(dataset, 
      #                                        [int(size_of_main*0.9), size_of_main - int(size_of_main*0.9)], 
      #                                        generator=torch.Generator().manual_seed(42))
      
      self.dataset_train = dataset
      self.dataset_test = None
      self.dataset_val = VoxCeleb_test(path=self.hparams["path_test"], sampling_rate=self.hparams["sampling_rate"], max_seconds=self.hparams["max_seconds"])
    
    
    def train_dataloader(self):
      data_train = DataLoader(self.dataset_train, batch_size=self.hparams["number_of_speakers"], num_workers=self.hparams["num_workers"], 
                              shuffle=self.hparams["dataloader_shuffle"], 
                              collate_fn = lambda x:collate_fn_vox(x, self.processor, self.hparams["sampling_rate"], max_length = self.hparams["max_length"])
                              )
      return data_train

    def val_dataloader(self):
      val = DataLoader(self.dataset_val, batch_size=self.hparams["number_of_speakers"], num_workers=self.hparams["num_workers"], 
                              shuffle=False, 
                              collate_fn = lambda x:collate_fn_vox_test(x, self.processor, self.hparams["sampling_rate"], max_length = self.hparams["max_length"])
                              )
      return val

    def test_dataloader(self):
      test = self.dataset_test
      return test