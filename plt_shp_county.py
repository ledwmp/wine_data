import numpy as np
import pandas as pd
import shapefile as shp
from shapely.geometry import shape,mapping,Point,Polygon,MultiPolygon
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set(style="white",palette="pastel",color_codes=True)
sns.mpl.rc("figure",figsize=(6,10))

shp_path = "../county/CA_counties.shp"
#shp_path = "../wss_gsmsoil_CA_[2016-10-13]/spatial/gsmsoilmu_a_ca.shp"

sf = shp.Reader(shp_path)
with open("../frl-wine-producers-and-blenders-ca_mu.json") as f:
	df_ = pd.read_json(json.load(f))
print(df_.lon)
df_["lon_lat"] = list(zip(df_["lon"],df_["lat"]))
def read_shapefile(sf):
	fields = [x[0] for x in sf.fields][1:]
	records = sf.records()
	shps = [s.points for s in sf.shapes()]
	df = pd.DataFrame(columns=fields,data=records)
	df = df.assign(coords=shps)
	return df

df = read_shapefile(sf)

def plot_full(sf,x_lim = None,y_lim=None,figsize=(5,15)):
	plt.figure(figsize=figsize)
	for shape in sf.shapeRecords():
		x = [i[0] for i in shape.shape.points[:]]
		y = [i[1] for i in shape.shape.points[:]]
		plt.plot(x,y,'k',linewidth=0.1)
	points_x,points_y = zip(*(Point(coord).coords[0] for coord in df_.lon_lat if coord != (np.nan,np.nan)))
	plt.scatter(points_x,points_y,s=1,c="k")
	plt.xlim(-125,-113.5)
	plt.ylim(32,42.5)

plot_full(sf)
plt.show()
