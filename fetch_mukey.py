import shapefile as shp
import numpy as np
from shapely.geometry import shape,mapping,Point,Polygon,MultiPolygon
import pandas as pd
import time
import glob
import json

path_tmp = "../frl-wine-producers-and-blenders-ca*.json"
all_files = glob.glob(path_tmp)
tmp_list = []
for i in all_files:
	with open(i) as f:
		region = i.split("-")[-1].split(".json")[0]
		tmp_list.append((json.load(f),region))
df,keys =  zip(*((pd.read_json(i),j) for i,j in tmp_list))
df = pd.concat(df,keys=keys).reset_index()
print(df.head())

df["lon_lat"] = list(zip(df["lon"],df["lat"]))

sf = shp.Reader("../wss_gsmsoil_CA_[2016-10-13]/spatial/gsmsoilmu_a_ca.shp")
shapes = [(shape(i.shape.__geo_interface__),i.record) for i in sf.shapeRecords()]

def return_mapunit(coord):
	point = Point(coord)
	for i,j in shapes:
		if point.within(i):
			print(j)
			return j[2],j[3]
	return np.nan,np.nan
df["MUSYM"],df["MUKEY"] = zip(*df["lon_lat"].map(return_mapunit))

with open("../frl-wine-producers-and-blenders-ca_mu.json","w") as f:
	json.dump(df.to_json(),f)
f.close()
