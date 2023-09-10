import logging
logging.captureWarnings(True)

import os

import nltk
import numpy as np
import pandas as pd
import plotly.express as px
from safetensors.numpy import save_file, load_file
from sentence_transformers import SentenceTransformer
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
import umap

model = SentenceTransformer('all-MiniLM-L6-v2')

tensors = "./tensors"
root = './data'
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

syms = os.listdir(root)

def read_file(file: str) -> str:
  with open(f"{root}/{file}", "r") as f:
    return f.read()

descrips = np.array([ read_file(file) for file in syms ])

def read_or_calc(file:str) -> np.ndarray:
  stf = f"{tensors}/{file}.safetensors"
  if os.path.exists(stf): return load_file(stf)["embedding"]
  else:
    print(file)
    data = np.mean(model.encode(tokenizer.tokenize(read_file(file))), axis=0)
    save_file({"embedding" : data}, stf)
    return data

def filter_noise_by(descrips, data):
  return np.array([x[0] for x in zip(data, descrips) if "does not have significant operations" not in x[1]])

def cleaned_data(descrips, syms, embeds):
  embeds = filter_noise_by(descrips, embeds)
  syms = filter_noise_by(descrips, syms)
  descrips = filter_noise_by(descrips, descrips)
  return descrips, syms, embeds

embeddings = np.array([ read_or_calc(file) for file in syms ])

descrips, syms, embeddings = cleaned_data(descrips, syms, embeddings)

print("embeddings loaded")
embeddings = normalize(embeddings, norm='l2')

def wrap_text(r):
  acc = []
  while r != "":
    acc.append(r[:120])
    r = r[120:]
  return "<br>".join(acc)

descrips = [ wrap_text(r) for r in descrips ]

clusters = SpectralClustering(n_clusters=150, n_jobs=-1).fit_predict(embeddings)

embeddings = umap.UMAP(metric="manhattan").fit_transform(embeddings)

addtl_info = {"sym": syms, "cluster": clusters, "descrip": descrips}
fig = px.scatter(
  pd.DataFrame({
    "x": embeddings[:, 0],
    "y": embeddings[:, 1],
    **addtl_info
  }),
  x="x",
  y="y",
  color="cluster",
  hover_name="sym",
  hover_data="descrip"
)

name = f"stocks"

fig.write_html(f"./{name}.html")
fig.write_json(f"./{name}.json")
fig.show()