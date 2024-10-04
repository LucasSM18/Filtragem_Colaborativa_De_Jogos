import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

def getKeys(obj):
   item = [key for key in eval(obj)]
   return '|' + '|'.join(map(str,item)) + '|'

def checkGenre(genres, infos):
   items = str(genres).split(",")
   list = ""
   for x in items:
      filter = str(infos).find("|" + x.strip() + "|") 
      if filter == -1:
         list = list + str(x).strip() + "|"
   return str(infos) + list
##
##
print("Digite um jogo:")
game = input()
games = pd.read_csv("steamspy_data.csv", usecols=['appid', 'name', "developer", "publisher", 'genre', 'tags'], sep=",")
games["info"] = list(map(getKeys, games["tags"]))
games["info"] = "|" + games["developer"] + games["info"]
games.loc[games["developer"] != games["publisher"], "info"] = "|" + games["publisher"] + games["info"] 
list(map(checkGenre, games["genre"], games["info"]))

vec = TfidfVectorizer()
tfid = vec.fit_transform(games["info"].apply(lambda x: np.str_(x)))

sim = cosine_similarity(tfid)

sim_df2 = pd.DataFrame(sim, columns=games["name"], index=games["name"])

final_df = pd.DataFrame(sim_df2[game].sort_values(ascending=False).drop_duplicates(keep='first'))

final_df.columns = ["recomendações"]

desvio_padrao = np.std(final_df["recomendações"].head(20))

media = final_df["recomendações"].head(20).mean()

print(final_df.head(20), "\n")
print("\nMédia: ", media)
print("Desvio padrão: ", desvio_padrao)

#grafico de barras com as tags principais pra filtragem
list = final_df.head(20).merge(games, on="name", how="left")[["name", "info"]].head(20) 
array = list["info"].to_numpy()
tags = []
main = array[0].split("|")
main = [i for i in main if i != ""]
for x in array :
   newArr = x.split("|")
   tags = tags + [i for i in newArr if i != ""]
tags = [i for i in tags if i in array[0].split("|")]
unique, counts = np.unique(tags, return_counts=True)
count_sort_ind = np.argsort(-counts)
counts = counts[count_sort_ind][:6]
unique = unique[count_sort_ind][:6]
round = [np.round(x, 2) for x in counts * 100 / sum(counts)]
data = dict(zip(unique, round))
colors = ["#1a1aff", "#3333ff", "#4d4dff", "#6666ff", "#8080ff", "#9999ff"]
plt.bar(range(len(data)), data.values(), align="center", color=colors)
plt.title("Principais Categorias")
plt.grid(color="gray", linewidth=1, axis="y", alpha=0.5)
plt.gca().set_axisbelow(True)
xlocs = plt.xticks()
xlocs=[i for i in range(len(data))]
plt.xticks(xlocs, data.keys())
patches = [Patch(color=colors[i], label=v) for i, v in enumerate(data.keys())]
labels = ([key for key in data.keys()])
plt.legend(title="Categorias", prop={'size': 10}, labels=labels, handles=patches, bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0, fontsize=15, frameon=False)
for i, v in enumerate(data.values()):
   plt.text(xlocs[i] - 0.38, v + 0.05, str(v)+"%", weight='bold')
plt.tick_params(rotation=60)
plt.show()


