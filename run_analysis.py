import pandas as pd
import numpy as np
import json
import glob
from collections import Counter
from collections import defaultdict
import recordlinkage
from sklearn.model_selection import train_test_split
from recordlinkage.base import BaseCompareFeature
import difflib
import matplotlib.pyplot as plt
import re
from load_GIS import load_GIS
from load_climate import load_climate

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
df_wine.drop("level_1",inplace=True,axis=1)
df_wine.rename({"level_0":"review_year"},inplace=True,axis=1)
"""
columns=['review_year', 'points', 'title', 'description', 'taster_name',
       'taster_twitter_handle', 'taster_photo', 'price', 'designation',
       'variety', 'region_1', 'region_2', 'province', 'country', 'winery']
"""
df_wine.drop(["description","taster_name","taster_twitter_handle",\
            "taster_photo","province","country"],\
            inplace=True,axis=1,\
            )


def find_vintage(tmp):
    """Extract vintage year from title
    """
    match = re.findall(r'([2-3][0][0-9]{2})',tmp)
    if len(match) == 1:
        return int(match[0])
    else:
        return np.nan

#extract vintage from title, keep only wines with vintage and points
df_wine["vintage"] = df_wine["title"].apply(find_vintage)
plt.hist(df_wine["vintage"],bins=25)
plt.show()
mask = (df_wine["vintage"] == np.nan) &\
        (df_wine["points"].astype(str).str.isdigit() == False)
df_wine = df_wine[~mask]
df_wine["points"] = df_wine["points"].astype(float)
df_wine["log_price"] = np.log2(df_wine["price"])
#df_wine.boxplot(column=["points"],by="vintage")
#plt.show()


#load winery data into dataframe
with open("../frl-wine-producers-and-blenders-ca_mu.json") as f:
    df_vine = pd.read_json(json.load(f))
df_vine.rename({"level_0":"region"},inplace=True,axis=1)
df_vine.drop("level_1",inplace=True,axis=1)


#need to populate column with OPERATING NAME, and if empty, then OWNER NAME
df_vine["OWNER_PARE"] = df_vine["OPERATING NAME"]
mask = df_vine["OWNER_PARE"] == " "
df_vine.loc[mask,"OWNER_PARE"] = df_vine.loc[mask,"OWNER NAME"]


#clean columns
def clean_columns(tmp):
    """Function to concatenate and remove punctuation from business names
    """
    remove_list = [".",",","'",":","&","!","-"]
    tmp = "".join([i for i in tmp if i not in remove_list])
    return "".join(tmp.split(" "))

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


#deduplicate owners in winery dataframe, save dups to new df to be classified later
df_vine["OWNER_PARE"] = df_vine["OWNER_PARE"].apply(clean_columns)
indexer = recordlinkage.index.SortedNeighbourhoodIndex(\
            left_on="OWNER_PARE",right_on="OWNER_PARE",window=3)
candidate_link = indexer.index(df_vine)
compare = recordlinkage.Compare()
compare.string("OWNER_PARE","OWNER_PARE",method="jarowinkler")
compare.string("OWNER_PARE","OWNER_PARE",method="lcs")
#compare.geo(left_on_lat = "lat",left_on_lng = "lon",\
#                        right_on_lat = "lat",right_on_lng = "lon",\
#                        method="gauss",offset=0.4,scale=0.2) #this is ~30mi radius for offset
compare.add(lc_substring("OWNER_PARE","OWNER_PARE"))
compare_vectors = compare.compute(candidate_link,df_vine,df_vine)
double_indices = compare_vectors[compare_vectors.sum(axis=1) >= 2.0].index
double_indices = list(set([i for j in double_indices.ravel() for i in j]))
df_vine_dup = df_vine.iloc[double_indices]
df_vine.drop(index=double_indices,inplace=True)

#clean columns before trying to match names
mask = (df_vine["OPERATING NAME"] != " ")
df_test_train = df_vine.loc[mask,("OPERATING NAME","OWNER NAME")].copy(deep=True)
df_test_train["OPERATING NAME"] = df_test_train["OPERATING NAME"].apply(clean_columns)
df_test_train["OWNER NAME"] = df_test_train["OWNER NAME"].apply(clean_columns)


#make a multiindex of matching records
df_test_train = pd.concat((df_test_train["OWNER NAME"],df_test_train["OPERATING NAME"]),ignore_index=True).to_frame()
owner_name = df_test_train.index.values[:int(len(df_test_train)/2)]
operating_name = df_test_train.index.values[int(len(df_test_train)/2):]
match_indices = pd.MultiIndex.from_arrays([operating_name,owner_name],names=["OPERATOR","OWNER"])


#train lr classifier
indexer = recordlinkage.index.SortedNeighbourhoodIndex(0,window=3)
candidate_link = indexer.index(df_test_train)
compare = recordlinkage.Compare()
compare.string(0,0,method="jarowinkler")
compare.string(0,0,method="lcs")
compare.add(lc_substring(0,0))
compare.exact(0,0)
compare_vectors = compare.compute(candidate_link,df_test_train,df_test_train)
compare_vectors.index = compare_vectors.index.set_names(["OPERATOR","OWNER"])
fig,axs = plt.subplots(1,len(compare_vectors.columns))
for i in range(len(compare_vectors.columns)):
    axs[i].hist(compare_vectors[i])
plt.show()

train,test = train_test_split(compare_vectors,test_size=0.2,random_state=42)
train_matches = train.index & match_indices
test_matches = test.index & match_indices
lr = recordlinkage.LogisticRegressionClassifier()
lr.fit(train,train_matches)
result_lr = lr.predict(test)
print(recordlinkage.confusion_matrix(test_matches,result_lr,len(test)))
print(recordlinkage.fscore(test_matches,result_lr))

class common_sets:
    """Class to cluster lists of tuples
    Args:
        List of tuples
    Returns:
        List of lists of similar items
    """
    def __init__(self,tuple_list):
        self.set_list = []
        for i in tuple_list:
            self.add_member(i)
    def add_member(self,tuple):
        unique = 0
        for i in self.set_list:
            if tuple[0] in i and tuple[1] not in i:
                i.append(tuple[1])
                unique = 1
                break
            elif tuple[1] in i and tuple[0] not in i:
                i.append(tuple[0])
                unique = 1
                break
            elif tuple[1] in i and tuple[0] in i :
                unique = 1
                break
        if unique == 0:
            self.set_list.append([tuple[0],tuple[1]])

#use lr classifier to deduplicate columns in dup dataframe, append to vine dataframe
indexer = recordlinkage.index.SortedNeighbourhoodIndex("OWNER_PARE")
candidate_link = indexer.index(df_vine_dup)
compare = recordlinkage.Compare()
compare = recordlinkage.Compare()
compare.string("OWNER_PARE","OWNER_PARE",method="jarowinkler")
compare.string("OWNER_PARE","OWNER_PARE",method="lcs")
compare.add(lc_substring("OWNER_PARE","OWNER_PARE"))
compare.exact("OWNER_PARE","OWNER_PARE")
compare_vectors = compare.compute(candidate_link,df_vine_dup,df_vine_dup)
result_lr = lr.predict(compare_vectors)
exact_match = compare_vectors.index[(compare_vectors[compare_vectors.columns[-1]] == 1)]
result_lr = result_lr.union(exact_match)
common_wineries = common_sets([i for i in result_lr.ravel()]).set_list

tmp_list = []
for i in common_wineries: #just keep the oldest winery owned by the same group, the first license number
    df_tmp = df_vine_dup.loc[i]
    df_tmp["license"] = df_tmp["PERMIT NUMBER"].str.split("-").str[-1].astype(int)
    keep_id = df_tmp.loc[df_tmp["license"] == min(df_tmp["license"])].index
    tmp_list.append(df_vine_dup.loc[keep_id])
df_vine_dedup = pd.concat(tmp_list,ignore_index=True)
df_vine = pd.concat((df_vine,df_vine_dedup),ignore_index=True)


#use lr classifier to match winery names on bottle to wineries in CA
df_wine["winery_pare"] = df_wine["winery"].apply(clean_columns)
indexer = recordlinkage.index.SortedNeighbourhoodIndex(\
            left_on="winery_pare",right_on="OWNER_PARE",window=3)
candidate_link = indexer.index(df_wine,df_vine)
compare = recordlinkage.Compare()
compare.string("winery_pare","OWNER_PARE",method="jarowinkler")
compare.string("winery_pare","OWNER_PARE",method="lcs")
compare.add(lc_substring("winery_pare","OWNER_PARE"))
compare.exact("winery_pare","OWNER_PARE")
compare_vectors = compare.compute(candidate_link,df_wine,df_vine)
compare_vectors.index = compare_vectors.index.set_names(["wine","winery"])
result_lr = lr.predict(compare_vectors)
exact_match = compare_vectors.index[(compare_vectors[compare_vectors.columns[-1]] == 1)]
result_lr = result_lr.union(exact_match) #append exact matches to result multiindex
out_index = compare_vectors.index.difference(result_lr) #unmatched wines
#for i in result_lr:
#    if count_wine[i[0]] > 1:
#        print(i)
#        print(compare_vectors.loc[pd.MultiIndex.from_arrays([[i[0]],[i[1]]]),:])
#        print(df_wine.iloc[i[0],df_wine.columns.get_loc("winery_pare")],df_vine.iloc[i[1],df_vine.columns.get_loc("OWNER_PARE")])


#only keep wine:winery pairs that score the highest if multiple wineries match with a wine
compare_vectors["match_score"] = compare_vectors.sum(axis=1)
count_wine = Counter([i[0] for i in result_lr.ravel()])
singly_positive = [i for i in result_lr.ravel() if count_wine[i[0]] == 1]
doubly_positive = [(i,compare_vectors.loc[i,"match_score"]) \
                        for i in result_lr.ravel() if count_wine[i[0]] > 1]
double_dict = defaultdict(list)
for i in doubly_positive:
    double_dict[i[0][0]].append(i)
doubly_positive = [max(j,key = lambda x: x[1])[0] for i,j in double_dict.items()]
positive = singly_positive+doubly_positive
positive = {i:j for i,j in positive}
df_wine_vine = df_wine.copy(deep=True)
df_wine_vine["winery_index"] = df_wine_vine.index.map(positive)
df_wine_vine = df_wine_vine.merge(df_vine,left_on="winery_index",right_on=df_vine.index)
df_GIS = load_GIS()
df_wine_vine = df_wine_vine.merge(df_GIS,left_on="MUKEY",right_on="mukey",)
"""
df_wine_vine.columns =
['review_year', 'points', 'title', 'price', 'designation', 'variety',
       'region_1', 'region_2', 'winery', 'vintage', 'log_price', 'winery_pare',
       'winery_index', 'region', 'PERMIT NUMBER', 'OWNER NAME',
       'OPERATING NAME', 'STREET', 'CITY', 'STATE', 'ZIP', 'COUNTY', 'STREET_',
       'ADDRESS', 'lat', 'lon', 'lon_lat', 'MUSYM', 'MUKEY', 'OWNER_PARE',
       'mukey', 'slope_l', 'slope_r', 'slope_h', 'tfact', 'wei', 'elev_l',
       'elev_r', 'elev_h', 'nirrcapcl', 'otherph', 'weg', 'drainagecl',
       'nirrcapscl', 'hydgrp', 'taxorder', 'taxsuborder', 'taxgrtgroup',
       'taxsubgrp', 'taxpartsize', 'taxtempcl']
"""
