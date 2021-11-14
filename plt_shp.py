import numpy as np
import pandas as pd
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set(style="whitegrid",palette="pastel",color_codes=True)
sns.mpl.rc("figure",figsize=(6,10))

shp_path = "../wss_gsmsoil_CA_[2016-10-13]/spatial/gsmsoilmu_a_ca.shp"

sf = shp.Reader(shp_path)
with open("../frl-wine-producers-and-blenders-ca_mu.json") as f:
	df_ = pd.read_json(json.load(f))

MUKEY_dict = df_.MUKEY.value_counts().to_dict()

def read_shapefile(sf):
	fields = [x[0] for x in sf.fields][1:]
	records = sf.records()
	shps = [s.points for s in sf.shapes()]
	df = pd.DataFrame(columns=fields,data=records)
	df = df.assign(coords=shps)
	return df

df = read_shapefile(sf)
for i in df.MUKEY:
	if float(i) not in MUKEY_dict.keys():
		MUKEY_dict[float(i)] = 0
print(df.MUKEY.value_counts())

def plot_full(sf,x_lim = None,y_lim=None,figsize=(5,15)):
	plt.figure(figsize=figsize)
	for shape in sf.shapeRecords():
		x = [i[0] for i in shape.shape.points[:]]
		y = [i[1] for i in shape.shape.points[:]]
		MUKEY = float(shape.record[3])
		plt.plot(x,y,'k',linewidth=0.1)
		if (x_lim == None) & (y_lim == None):
			x0 = np.mean(x)
			y0 = np.mean(y)
			if MUKEY_dict[MUKEY] != 0.0:
				plt.text(x0,y0,MUKEY_dict[MUKEY],fontsize=10)
	if (x_lim != None) & (y_lim != None):
		plt.xlim(x_lim)
		plt.ylim(y_lim)

plot_full(sf)
plt.show()
