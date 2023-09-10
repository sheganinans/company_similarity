import logging
logging.captureWarnings(True)

import os

import nltk
import numpy as np
import pandas as pd
from safetensors.numpy import save_file, load_file
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import normalize
import plotly.express as px

import hdbscan

import umap

from sentence_transformers import SentenceTransformer, util

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

def run_param_search():
  logging.captureWarnings(True)
  hdb = hdbscan.HDBSCAN(gen_min_span_tree=True).fit(embeddings)

  # specify parameters and distributions to sample from
  param_dist = {'min_samples': [3,5,10,15],
                'min_cluster_size':[3,5,10,15],
                'cluster_selection_method' : ['leaf', "eom"],
                'cluster_selection_epsilon' : [0.02,0.0225,0.025,0.0275,0.03],
                'metric' : ['manhattan', "euclidean"]
              }

  validity_scorer = make_scorer(hdbscan.validity.validity_index,greater_is_better=True)

  n_iter_search = 40
  random_search = RandomizedSearchCV(hdb
                                    ,param_distributions=param_dist
                                    ,n_iter=n_iter_search
                                    ,scoring=validity_scorer
                                    ,random_state=0x1337)

  random_search.fit(embeddings)
  print(f"Best Parameters {best}")
  print(f"DBCV score :{random_search.best_estimator_.relative_validity_}")

#run_param_search()

#best = random_search.best_params_
best = {'min_samples': 3, 'min_cluster_size': 3, 'metric': 'manhattan', 'cluster_selection_method': 'eom', 'cluster_selection_epsilon': 0.0275}

clusters = hdbscan.HDBSCAN(
  min_cluster_size=best["min_cluster_size"],
  min_samples=best["min_samples"],
  cluster_selection_method=best["cluster_selection_method"],
  cluster_selection_epsilon=best["cluster_selection_epsilon"],
  metric=best["metric"]).fit_predict(embeddings)

reducer = umap.UMAP(metric="manhattan")

embeddings = reducer.fit_transform(embeddings)

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