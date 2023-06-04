import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import os
from cc_net.perplexity import MultiSentencePiece, DocLM

import nltk 
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

import cld3

import heapq
from nltk import ngrams
import re

import numpy as np
import torch
from open_clip import tokenizer
import open_clip

import numpy as np
import pandas as pd

from img2dataset import download
from clip_retrieval import clip_inference
import shutil
import os

from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import json

import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms

import clip


path_to_laion_model = "/p/project/ccstdl/gupta6/interleaved_dataset/laion2B-en/lm_sp"

class LM:
    def __init__(self, tokenizer, lm):
        self.tokenizer = tokenizer
        self.lm = lm

    def get_perplexity(self, text : str, language : str):
        data = {"raw_content" : text, "language" : language}
        data = self.tokenizer.do(data)
        return self.lm.do(data)['perplexity']

def load_perplexity_language_model(path_to_lm):
    if path_to_lm is not None:
        sp_path = os.path.join(path_to_lm, "en.sp.model")
        sp = MultiSentencePiece({"en": sp_path}, field="raw_content", output_field="tokenized", normalize=True)

        lm_path = os.path.join(path_to_lm, "en.arpa.bin")
        lm = DocLM({"en": lm_path}, field="tokenized", output_field="perplexity", normalize=False)

        return LM(sp, lm)


SEPERATOR = "###img###sep###"

def get_before_after_text(text):
    sep_span = re.search(SEPERATOR, text).span()

    # Remove urls and email ids - see pycld3 README
    url_re = r"\b(?:https?://|www\.)[a-z0-9-]+(\.[a-z0-9-]+)+(?:[/?].*)?"
    before_text = re.sub(url_re, "", text[:sep_span[0]].strip())
    after_text = re.sub(url_re, "", text[sep_span[1]:].strip())

    return before_text, after_text


def get_filtered_ngrams(text, ngram_range, lang, filter_by_lang=False, perplexity_lm=None):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')

    if lang == 'en':
        candidates = sent_tokenizer.tokenize(text)
    else:
        if not filter_by_lang:
            candidates = [text]
        else:
            return []

    filtered_candidates = []
    for i in range(len(candidates)):
        for n in range(*ngram_range):
            sent_ngrams = [" ".join(ngram) for ngram in ngrams(candidates[i].split(), n)]
            filtered_candidates.extend(list(sent_ngrams))

    if perplexity_lm is not None and lang == "en":
        perplexity_filtered_candidates = []

        for candidate in filtered_candidates:
            perplexity_filtered_candidates.append((candidate, perplexity_lm.get_perplexity(candidate, lang)))

        top_n_candidates = heapq.nlargest(20, perplexity_filtered_candidates, key=lambda x : -x[1])

        return [candidate[0] for candidate in top_n_candidates]

    return filtered_candidates

def get_n_grams(before_text, after_text, ngram_range = (3, 20)):
  perplexity_lm = load_perplexity_language_model(path_to_laion_model)

  
  before_lang = cld3.get_language(before_text) if before_text is not None else None
  after_lang = cld3.get_language(after_text) if after_text is not None else None

  candidates = []
  if before_lang is not None:
      candidates.extend(
          get_filtered_ngrams(
              before_text,
              ngram_range,
              before_lang.language,
              False,
              perplexity_lm,
              )
          )

  if after_lang is not None:
      candidates.extend(
          get_filtered_ngrams(
              after_text,
              ngram_range,
              after_lang.language,
              False,
              perplexity_lm,
              )
          )
  return candidates

def extract_sentences(text):
    # Split the text into sentences
    sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    result = []

    # Start from the end and go in reverse order
    for sentence in reversed(sentences):
        result.insert(0, sentence)
        
        # Check if the total length of sentences is at least 300 characters
        if sum(len(s) for s in result) >= 300:
            break

    return " ".join(result)

def find_before_after_imgs(row):
  before_after_dict = {}
  before_imgs = {}
  after_imgs = {}
  img_idx = 0
  for i, element in enumerate(row["Doc"]):
    if element.startswith("###img#"):
      before_after_dict[element] = {}
      if i > 0 and (row["Doc"][i-1].startswith("###img#") is False):
        before_imgs[element] = row["Doc"][i-1]
        before_after_dict[element]['before'] = extract_sentences(row["Doc"][i-1])
      else:
        before_after_dict[element]['before'] = None
      if i < len(row["Doc"])-1 and (row["Doc"][i+1].startswith("###img#") is False):
        after_imgs[element] = row["Doc"][i+1]
        before_after_dict[element]['after'] = extract_sentences(row["Doc"][i+1])
      else:
        before_after_dict[element]['after'] = None
      before_after_dict[element]['url'] = row['Imgs.'+element][2]
      before_after_dict[element]['n_grams'] = get_n_grams(before_after_dict[element]['before'], before_after_dict[element]['after'])
  return pd.Series(before_after_dict)

def process_text(df):
  result = df.apply(find_before_after_imgs, axis=1)
  melted = pd.melt(result.reset_index(), id_vars='index', var_name='column', value_name='value')
  melted = melted.dropna(subset=['value'])
  value_df = melted['value'].apply(pd.Series)
  final_df = pd.concat([melted, value_df], axis=1)
  final_df = final_df.drop('value', axis=1)
  final_df.sort_values('index')
  return final_df

def add_n_grams(df):
  df_n_grams = df[['n_grams', 'url']].reset_index().explode('n_grams')
  df_n_grams = df_n_grams.reset_index(drop=True)
  df_n_grams = df_n_grams.dropna(subset='n_grams')
  return df_n_grams

class ClipWrapper(torch.nn.Module):
    def __init__(self, device, model_name='ViT-L/14'):
        super(ClipWrapper, self).__init__()
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32', device=device, jit=True)
        self.clip_model.eval()

    def forward(self, text_tokens):
      with torch.no_grad():
        text_features = self.clip_model.encode_text(text_tokens).float()
        return text_features.cpu()

device = "cuda" if torch.cuda.is_available() else "cpu"
clip = ClipWrapper(device)

def compute_embeddings(texts):
  text_tokens = tokenizer.tokenize(list(texts)).to(device)
  return clip(text_tokens)

def calculate_batched_clip_embeddings(df, batch_size, text_column_name, embedding_column_name):
  indices = list(df.index.values)
  for i in range(0, len(indices), batch_size):
    if i + batch_size > len(indices):
      end = len(indices)
    else:
      end = i+batch_size
    current_batch = df.loc[indices[i:end],text_column_name]
    embeddings = compute_embeddings(current_batch).numpy()
    embeddings = pd.Series(embeddings.tolist())
    embeddings.index = indices[i:end]
    df.loc[indices[i:end], embedding_column_name] = embeddings

def get_image_urls(df):
  # select columns that start with "Imgs.###"
  cols_to_select = df.columns[df.columns.str.startswith('Imgs.###')]

  # create a new dataframe to store the split lists
  new_df = pd.DataFrame()

  # loop through the selected columns
  for col in cols_to_select:
      # select non-empty lists using boolean indexing
      non_empty_lists = df[col][df[col].apply(lambda x: len(x) > 0)]
      
      # split the lists into three separate columns
      split_lists = pd.DataFrame(non_empty_lists.tolist(), columns=['Id', 'format', 'url'])
      
      # append the split lists to the new dataframe
      new_df = pd.concat([new_df, split_lists], axis=0, ignore_index=True)
  new_df = new_df.drop_duplicates(subset=['url']).sample(frac=1)
  return new_df

def get_image_embedding(url, parquet_dfs, meta_data_df, all_embeddings):
  for p_df in parquet_dfs:
    if url in p_df.index:
      if p_df.loc[url, 'status'] == 'success':
        return all_embeddings[meta_data_df.loc[p_df.loc[url, 'key']]][0]
  return None

def dot_product(row, emb_column_1, emb_column2):
  a = np.array(row[emb_column_1])
  b = np.array(row[emb_column2])
  if a.all() and b.all():
    print(a.all(), b.all(), np.dot(a, b))
    return np.dot(a, b)
  else:
    return np.nan

def create_filtered_scores_pairs(group, threshold=0.3):
  final_dict = {}
  for _, row in group.iterrows():
    print(row)
    print(row['clip_score'])
    if row['clip_score']>=threshold:
      final_dict[row['n_grams']] = row['clip_score']
    return final_dict

# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def get_aesthetics_model():

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
  model = model .to(device)
  s = torch.load("/content/sac+logos+ava1-l14-linearMSE.pth?raw=true", map_location=torch.device(device))

  model.load_state_dict(s)

  model.to(device)
  model.eval()
  return model

def get_aesthetics_score(model, image_features, device):
  im_emb_arr = normalized(image_features.cpu().detach().numpy() )

  prediction = model(torch.from_numpy(im_emb_arr).to(device)) #.type(torch.cuda.FloatTensor))
  aesthetics_score = prediction.cpu().detach().numpy().tolist()[0][0]
  return aesthetics_score

class Normalization(nn.Module):
  def __init__(self, shape):
    super().__init__()
    self.register_buffer('mean', torch.zeros(shape))
    self.register_buffer('variance', torch.ones(shape))

  def forward(self, x):
    return (x - self.mean) / self.variance.sqrt()
    

class NSFWModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.norm = Normalization([768])
    self.linear_1 = nn.Linear(768, 64)
    self.linear_2 = nn.Linear(64, 512)
    self.linear_3 = nn.Linear(512, 256)
    self.linear_4 = nn.Linear(256, 1)
    self.act = nn.ReLU()
    self.act_out = nn.Sigmoid()

  def forward(self, x):
    x = self.norm(x)
    x = self.act(self.linear_1(x))
    x = self.act(self.linear_2(x))
    x = self.act(self.linear_3(x))
    x = self.act_out(self.linear_4(x))
    return x
  
def get_nsfw_model():
  path_to_model= "clip_autokeras_binary_nsfw.pth"
  nsfwmodel = NSFWModel()
  nsfwmodel_state_dict = torch.load(path_to_model)
  nsfwmodel.load_state_dict(nsfwmodel_state_dict)
  nsfwmodel.eval()

def get_nsfw_score(model, image_features, device):
  nsfw_prediction = model(torch.tensor(image_features))
  nsfw_score = nsfw_prediction.cpu().detach().numpy().tolist()[0][0]
  return nsfw_score

# Main function
def main():
  import pdb; pdb.set_trace()
  df = pd.read_parquet('compressed.parquet', engine='fastparquet')
  
  final_df = process_text(df.loc[2:4])
  final_df = final_df.rename(columns={"index": "original_index", "column": "image_number"})
  
  df_n_grams = add_n_grams(final_df)
  
  df_n_grams.loc[:, 'clip_embeddings'] = pd.Series(dtype='object')
  calculate_batched_clip_embeddings(df_n_grams.loc[0:100], 16, text_column_name='n_grams', embedding_column_name='clip_embeddings')
  image_url_df = get_image_urls(df)
  
  image_url_df.to_parquet('deduped_urls.parquet', engine='fastparquet')
  df_n_grams.to_parquet('df_n_grams.parquet', engine='fastparquet')
  final_df.to_parquet('final_df.parquet', engine='fastparquet')

  output_folder = '/p/project/ccstdl/gupta6/interleaved_dataset/imgs'
  img_embedding_folder_test = '/p/project/ccstdl/gupta6/interleaved_dataset/imgs_embs'

  download(
      processes_count=16,
      thread_count=64,
      url_list="deduped.parquet",
      image_size=256,
      output_folder=output_folder,
      output_format="webdataset",
      input_format="parquet",
      distributor="multiprocessing",
  )
  
  clip_inference(
        input_dataset="imgs/{00000..00003}.tar",
        output_folder=img_embedding_folder_test,
        input_format="webdataset",
        batch_size=256,
        enable_text=False,
        clip_model="ViT-L/14",
    )


  meta_data_df = pd.read_parquet('/p/project/ccstdl/gupta6/interleaved_dataset/imgs_embs/metadata/metadata_0.parquet', engine='fastparquet')
  meta_data_df.reset_index(inplace=True)
  meta_data_df.set_index('image_path', inplace=True)

  # Get all the parquet files

  parquet_df = pd.read_parquet('/p/project/ccstdl/gupta6/interleaved_dataset/imgs/00000.parquet', engine='fastparquet')
  parquet_df = parquet_df.set_index('url')

  parquet_df1 = pd.read_parquet('/p/project/ccstdl/gupta6/interleaved_dataset/imgs/00001.parquet', engine='fastparquet')
  parquet_df1 = parquet_df1.set_index('url')

  parquet_df2 = pd.read_parquet('/p/project/ccstdl/gupta6/interleaved_dataset/imgs/00002.parquet', engine='fastparquet')
  parquet_df2 = parquet_df2.set_index('url')

  parquet_df3 = pd.read_parquet('/p/project/ccstdl/gupta6/interleaved_dataset/imgs/00003.parquet', engine='fastparquet')
  parquet_df3 = parquet_df3.set_index('url')

  # Load all the embeddings
  all_embeddings = np.load('./p/project/ccstdl/gupta6/interleaved_dataset/imgs_embs/img_emb/img_emb_0.npy')

  temp = image_url_df.apply(lambda x: get_image_embedding(x['url'], [parquet_df,parquet_df1,parquet_df2,parquet_df3], meta_data_df, all_embeddings), axis=1)
  image_url_df = pd.concat([image_url_df, temp.rename("embeddings")], axis=1)

  image_url_df.reset_index(inplace=True)
  image_url_df.set_index('url', inplace=True)

  embeddings = df_n_grams.apply(lambda x: image_url_df.loc[x['url'], 'embeddings'], axis=1)

  df_n_grams = pd.concat([df_n_grams, embeddings.rename("embeddings")], axis=1)

  df_n_grams.apply(lambda x: dot_product(x, 'clip_embeddings', 'embeddings'), axis=1)

  df_n_grams = df_n_grams.groupby('url').apply(lambda group: pd.Series(create_filtered_scores_pairs(group))).reset_index()

  final_df = final_df.dropna(subset='before')
  final_df = final_df.dropna(subset='after')

  final_df.loc[:, 'before_emb'] = pd.Series(dtype='object')
  calculate_batched_clip_embeddings(final_df, 16, text_column_name='before', embedding_column_name='before_emb')

  final_df.loc[:, 'after_emb'] = pd.Series(dtype='object')
  calculate_batched_clip_embeddings(final_df, 16, text_column_name='after', embedding_column_name='after_emb')

  # temp = final_df.apply(lambda x: get_image_embedding(x['url'], [parquet_df,parquet_df1,parquet_df2,parquet_df3], meta_data_df, all_embeddings), axis=1)
  
  temp = final_df.apply(lambda x: image_url_df.loc[x['url'], 'embeddings'], axis=1)
  final_df = pd.concat([final_df, temp.rename('image_embeddings')], axis=1)

  final_df['before_score'] = final_df.apply(lambda x: dot_product(x, 0, 0), axis=1)
  final_df['after_score'] = final_df.apply(lambda x: dot_product(x, 0, 0), axis=1)


  def create_before_scores(group):
      return {row['image_number']: row['before_score'] for _, row in group.iterrows()}

  def create_after_scores(group):
      return {row['image_number']: row['after_score'] for _, row in group.iterrows()}

  # Concatenate the filtered n_grams to final_df, by extracting the n_grams from df_n_grams based on url found in final_df
  final_df = final_df.merge(df_n_grams, left_on='url', right_on='url', how='outer')
  
  # Filter out images based on aesthetics score
  aesthetics_model = get_aesthetics_model()
  final_df['aesthetics_score'] = final_df.apply(lambda x: get_aesthetics_score(aesthetics_model, x['image_embeddings'], device), axis=1)
  final_df = final_df[final_df['aesthetics_score'] >= 0.5]

  # Filter out images based on nsfw score
  nsfw_model = get_nsfw_model()
  final_df['nsfw_score'] = final_df.apply(lambda x: get_nsfw_score(nsfw_model, x['image_embeddings'], device), axis=1)
  final_df = final_df[final_df['nsfw_score'] <= 0.5]
  
  new_df = final_df.groupby('original_index').apply(lambda group: pd.Series({'before_scores': create_before_scores(group), 'after_scores': create_after_scores(group)})).reset_index()

  # Set 'index' as the main index
  new_df.set_index('original_index', inplace=True)

  concatenated_df = df.merge(new_df, left_index=True, right_index=True, how='outer')

  return concatenated_df 

# call main funciton
if __name__ == "__main__":
    main()