import pandas as pd
import numpy as np
import json
import glob
from collections import Counter
import recordlinkage
from sklearn.model_selection import train_test_split
from recordlinkage.base import BaseCompareFeature
import difflib
import matplotlib.pyplot as plt


#load wine reviews into dataframe
scrape_list = []
path = "../scrape/winemag-data_*.json"
all_files = glob.glob(path)
for i in all_files:
    with open(i) as f:
        year = i.split("_")[-1].split(".json")[0]
        scrape_list.append((json.load(f),year))

df,keys = zip(*((pd.DataFrame(i),j) for i,j in scrape_list))
df_wine = pd.concat(df,keys=keys).reset_index()
df_wine["winery"] = df_wine["winery"].str.upper()
"""
tmp_list = [i.split(" ") for i in df_wine["winery"]]
tmp_list = [i for j in tmp_list for i in j]
for i,j in sorted(Counter(tmp_list).items(),key = lambda x:x[1]):
    print(i,j)
"""
#load winery data into dataframe
with open("../frl-wine-producers-and-blenders-ca_mu.json") as f:
    df_vine = pd.read_json(json.load(f))
df_vine.rename({"level_0":"region"},inplace=True,axis=1)
df_vine.drop("level_1",inplace=True,axis=1)

#need to populate column with OPERATING NAME, and if empty, then OWNER NAME
df_vine["OWNER_PARE_SPACE"] = df_vine["OPERATING NAME"]
mask = df_vine["OWNER_PARE_SPACE"] == " "
df_vine.loc[mask,"OWNER_PARE_SPACE"] = df_vine.loc[mask,"OWNER NAME"]

#remove list of company words commonly found in OWNER
def strip_company(tmp):
    """Function to clean business jargon out of business names
    """
    strip_list = [" LLC"," INC."," COMPANY"," INC"," GROUP",\
                " ENTERPRISES"," CORPORATION"," INCORPORATED",\
                " HOLDINGS"," BEVERAGE"," INVESTMENTS",\
                " LTD."," L.L.C."," LLC."," ASSOCIATES",\
                " PARTNERS", " CO.", " VENTURES"," PROPERTIES",\
                " INTERNATIONAL",","\
                ]
    for i in strip_list+strip_list:
        tmp = tmp.split(i)[0]
    return tmp

def clean_columns(tmp):
    """Function to concatenate and remove punctuation from business names
    """
    remove_list = [".",",","'",":","&","!"]
    tmp = "".join([i for i in tmp if i not in remove_list])
    return "".join(tmp.split(" "))

#deduplicate owners in winery dataframe
df_vine["OWNER_PARE"] = df_vine["OWNER_PARE_SPACE"].apply(clean_columns)
indexer = recordlinkage.index.SortedNeighbourhoodIndex(\
            left_on="OWNER_PARE",right_on="OWNER_PARE",window=5)
candidate_link = indexer.index(df_vine)
compare = recordlinkage.Compare()
compare.string("OWNER_PARE","OWNER_PARE",method="levenshtein")
compare.geo(left_on_lat = "lat",left_on_lng = "lon",\
                        right_on_lat = "lat",right_on_lng = "lon",\
                        method="gauss",offset=0.4,scale=0.2) #this is ~30mi radious
compare.exact("region","region")
compare_vectors = compare.compute(candidate_link,df_vine,df_vine)
print(compare_vectors[compare_vectors.sum(axis=1) >= 2.5])

#clean columns before trying to match names
mask = (df_vine["OPERATING NAME"] != " ")
df_test_train = df_vine.loc[mask,("OPERATING NAME","OWNER NAME")].copy(deep=True)
#df_test_train["OPERATING NAME"] = df_test_train["OPERATING NAME"].apply(strip_company)
df_test_train["OPERATING NAME PARE"] = df_test_train["OPERATING NAME"].apply(clean_columns)
#df_test_train["OWNER NAME"] = df_test_train["OWNER NAME"].apply(strip_company)
df_test_train["OWNER NAME PARE"] = df_test_train["OWNER NAME"].apply(clean_columns)

#make a multiindex of matching records
df_test_train = pd.concat((df_test_train["OWNER NAME"],df_test_train["OPERATING NAME"]),ignore_index=True).to_frame()
df_test_train = pd.concat((df_test_train[["OWNER NAME","OWNER NAME PARE"]],df_test_train[["OPERATING NAME","OPERATING NAME PARE"]].rename(\
                {"OPERATING NAME":"OWNER NAME","OPERATING NAME PARE":"OWNER NAME PARE"},axis=1)),ignore_index=True)
df_test_train.rename({"OWNER NAME":"w/SPACE","OWNER NAME PARE":"w/oSPACE"},axis=1,inplace=True)
owner_name = df_test_train.index.values[:int(len(df_test_train)/2)]
operating_name = df_test_train.index.values[int(len(df_test_train)/2):]
match_indices = pd.MultiIndex.from_arrays([operating_name,owner_name],names=["OPERATOR","OWNER"])
print(df_test_train)
print(match_indices)

class lc_substring(BaseCompareFeature):
    """
    """
    def _compute_vectorized(self,s1, s2):
        def substring(tmp):
            """Uses difflib to compute longest common substring
            """
            tmp_a,tmp_b = tmp.iloc[0],tmp.iloc[1]
            matcher = difflib.SequenceMatcher(None,tmp_a,tmp_b)
            match_out = matcher.find_longest_match(0,len(tmp_a),0,len(tmp_b))
            len_short = min(len(tmp_a),len(tmp_b))
            return match_out.size/len_short
        s = pd.concat((s1,s2),axis=1)
        out = s.apply(substring,axis=1)
        return out

#train lr classifier
indexer = recordlinkage.index.SortedNeighbourhoodIndex("w/oSPACE",window=3)
candidate_link = indexer.index(df_test_train)
compare = recordlinkage.Compare()
compare.string("w/oSPACE","w/oSPACE",method="jarowinkler")
compare.string("w/oSPACE","w/oSPACE",method="lcs")
compare.add(lc_substring("w/oSPACE","w/oSPACE"))
compare.add(lc_substring("w/SPACE","w/SPACE"))
compare_vectors = compare.compute(candidate_link,df_test_train,df_test_train)
compare_vectors.index = compare_vectors.index.set_names(["OPERATOR","OWNER"])
fig,axs = plt.subplots(1,len(compare_vectors.columns))
for i in range(len(compare_vectors.columns)):
    axs[i].hist(compare_vectors[i])
plt.show()

train,test = train_test_split(compare_vectors,test_size=0.2)
train_matches = train.index & match_indices
test_matches = test.index & match_indices
lr = recordlinkage.LogisticRegressionClassifier()
lr.fit(train,train_matches)
result_lr = lr.predict(test)
print(recordlinkage.confusion_matrix(test_matches,result_lr,len(test)))
print(recordlinkage.fscore(test_matches,result_lr))

#match winery names on bottle to wineries in CA
df_wine["winery_pare"] = df_wine["winery"].apply(clean_columns)
indexer = recordlinkage.index.SortedNeighbourhoodIndex(\
            left_on="winery_pare",right_on="OWNER_PARE",window=3)
candidate_link = indexer.index(df_wine,df_vine)
compare = recordlinkage.Compare()
compare.string("winery_pare","OWNER_PARE",method="jarowinkler")
compare.string("winery_pare","OWNER_PARE",method="lcs")
compare.add(lc_substring("winery_pare","OWNER_PARE"))
compare.add(lc_substring("winery","OWNER_PARE_SPACE"))
compare_vectors = compare.compute(candidate_link,df_wine,df_vine)
compare_vectors.index = compare_vectors.index.set_names(["OPERATOR","OWNER"])
result_lr = lr.predict(compare_vectors)
print(len(result_lr))
plt.hist(compare_vectors[2])
plt.show()
