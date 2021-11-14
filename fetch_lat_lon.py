import requests
import numpy as np
import urllib.parse
import pandas as pd
import time
import glob
import json

path_tmp = "../frl-wine-producers-and-blenders-ca*.csv"
all_files = glob.glob(path_tmp)
"""
df = (pd.read_csv(f) for f in all_files)
df = pd.concat(df,ignore_index=True)
print(df.columns)
"""
for tmp in all_files:
	df = pd.read_csv(tmp)
	print(df.columns)
	remove_list = [" STE "," SUITE ", " SUITES ", " BLDG "," BLDGS "," UNIT "," UNITS "]
	df["STREET_"] = df["STREET"]
	for i in remove_list:
		df["STREET_"] = df["STREET_"].str.split(i).str[0]


	df["STREET_"] = df["STREET_"].map(lambda x: x.split(" & ")[1] if " & " in x else x)

	df["ADDRESS"] = df["STREET_"]+", "+df["CITY"]+", CA "+df["ZIP"].astype(str)
	print(df.ADDRESS)

	def fetch_openstreetmap(address,retry_count=0):
		url = 'https://nominatim.openstreetmap.org/search/'+\
				urllib.parse.quote(address)+'?format=json'
		print(address)
		print(url)
		try:
			response = requests.get(url).json()
			print(response)
			lat,lon = response[0]["lat"],response[0]["lon"]
			print(lat,lon)
			return lat,lon
		except:
			time.sleep(1.1)
			retry_count += 1
			if retry_count <= 1:
				return fetch_openstreetmap(address,retry_count)
			elif 1 < retry_count < 4:
				address = ", ".join([i.strip() for i in address.split(",")[1:]])
				return fetch_openstreetmap(address,retry_count)
			else:
				return np.nan,np.nan

	def return_lat_lon(address):
		blah = fetch_openstreetmap(address)
		print(blah)
		lat,lon = blah[0],blah[1]
		time.sleep(1.1)
		return lat,lon
	df["lat"],df["lon"] = zip(*df["ADDRESS"].map(return_lat_lon))
	print(df.head())
	with open(tmp.split(".csv")[0]+".json","a") as f:
		json.dump(df.to_json(),f)
	f.close()
	
	
	
	

#return_lat_lon(df.ADDRESS)


#address = 'Shivaji Nagar, Bangalore, KA 560001'
#url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'

#response = requests.get(url).json()
#print(response[0]["lat"])
#print(response[0]["lon"])
