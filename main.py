import logging

from sklearn.neighbors import NearestNeighbors
logging.captureWarnings(True)
import os

import hdbscan
from matplotlib import pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
from safetensors.numpy import save_file, load_file
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.preprocessing import normalize
import umap
import kmapper as km
from sklearn.manifold import TSNE


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

descrips = np.array([ wrap_text(r) for r in descrips ])

print("running umap")

proj = umap.UMAP(metric="manhattan").fit_transform(embeddings)

print("generating viz")

for n in [5,10,25,50,75,100,150,200,250]:
  print(n)
  clusters_kmeans = KMeans(n_clusters=n).fit_predict(embeddings)
  clusters_spectral = SpectralClustering(n_clusters=n).fit_predict(embeddings)

  def gen(name:str, clusters: np.ndarray):
    addtl_info = {"sym": syms, "cluster": clusters, "descrip": descrips}
    fig = px.scatter(
      pd.DataFrame({
        "x": proj[:, 0],
        "y": proj[:, 1],
        **addtl_info
      }),
      x="x",
      y="y",
      color="cluster",
      hover_name="sym",
      hover_data="descrip"
    )

    fig.write_html(f"./charts/{name}_{n}.html")
    fig.write_json(f"./charts/{name}_{n}.json")
    fig.show()

  gen("kmeans", clusters_kmeans)
  gen("spectral", clusters_spectral)

print("running hdbscan")

best = {'min_samples': 3, 'min_cluster_size': 3, 'metric': 'manhattan', 'cluster_selection_method': 'eom', 'cluster_selection_epsilon': 0.0275}

clusters = hdbscan.HDBSCAN(
  min_cluster_size=best["min_cluster_size"],
  min_samples=best["min_samples"],
  cluster_selection_method=best["cluster_selection_method"],
  cluster_selection_epsilon=best["cluster_selection_epsilon"],
  metric=best["metric"]).fit_predict(embeddings)

addtl_info = {"sym": syms, "cluster": clusters, "descrip": descrips}
fig = px.scatter(
  pd.DataFrame({
    "x": proj[:, 0],
    "y": proj[:, 1],
    **addtl_info
  }),
  x="x",
  y="y",
  color="cluster",
  hover_name="sym",
  hover_data="descrip"
)

fig.write_html("./charts/hdbscan.html")
fig.write_json(f"./charts/hdbscan.json")
fig.show()

def filter_noise_by(descrips, data):
  return np.array([x[0] for x in zip(data, descrips) if x[1] != -1])

def cleaned_data(clusters, descrips, syms, embeds):
  embeds = filter_noise_by(clusters, embeds)
  syms = filter_noise_by(clusters, syms)
  descrips = filter_noise_by(clusters, descrips)
  clusters = filter_noise_by(clusters, clusters)
  return clusters, descrips, syms, embeds

print("filtering data based on hdbscan noise result")

clusters, descrips, syms, embeddings = cleaned_data(clusters, descrips, syms, embeddings)

print("rerunning umap")

proj = umap.UMAP(metric="manhattan").fit_transform(embeddings)

addtl_info = {"sym": syms, "cluster": clusters, "descrip": descrips}
fig = px.scatter(
  pd.DataFrame({
    "x": proj[:, 0],
    "y": proj[:, 1],
    **addtl_info
  }),
  x="x",
  y="y",
  color="cluster",
  hover_name="sym",
  hover_data="descrip"
)

fig.write_html("./charts/hdbscan_cleaned.html")
fig.write_json(f"./charts/hdbscan_cleaned.json")
fig.show()

def run_knn():
  print("running knn.")
  nbrs = NearestNeighbors(n_neighbors=2, metric="manhattan", n_jobs=-1).fit(embeddings)
  ds, _ = nbrs.kneighbors(embeddings)
  plt.plot(np.sort(ds, axis=0)[:, 1])
  plt.savefig(fname="./knn.png")
  plt.show()

#run_knn()

def run_kmap(n=150, o=.5, eps=720, pp=30):
  print("running kmap.")
  name = f"n{n}_o{o:.2f}_eps{eps:.3f}_pp{pp}"
  cl = DBSCAN(eps=eps, metric="manhattan", n_jobs=-1)
  mapper = km.KeplerMapper()
  graph = mapper.map(
    mapper.fit_transform(
      embeddings,
      scaler=None,
      projection=umap.UMAP(n_components=3, metric="manhattan")),
    clusterer=cl,
    cover=km.Cover(n, o))
  mapper.visualize(
    graph,
    nbins=50,
    path_html=f"./charts/{name}.html",
    title=name,
    custom_tooltips=descrips,
    include_searchbar=True)

run_kmap()