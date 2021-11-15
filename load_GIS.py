import pandas as pd
import numpy as np

with open("../wss_gsmsoil_CA_[2016-10-13]/component_table.txt") as r:
        columns = [i.strip() for i in r]
"""
columns=
['comppct_l', 'comppct_r', 'comppct_h', 'compname', 'compkind', 'majcompflag', 'otherph', 'localphase', 'slope_l', 'slope_r', 'slope_h', 'slopelenusle_l', 'slopelenusle_r', 'slopelenusle_h', 'runoff', 'tfact', 'wei', 'weg', 'erocl', 'earthcovkind1', 'earthcovkind2', 'hydricon', 'hydricrating', 'drainagecl', 'elev_l', 'elev_r', 'elev_h', 'aspectccwise', 'aspectrep', 'aspectcwise', 'geomdesc', 'albedodry_l', 'albedodry_r', 'albedodry_h', 'airtempa_l', 'airtempa_r', 'airtempa_h', 'map_l', 'map_r', 'map_h', 'reannualprecip_l', 'reannualprecip_r', 'reannualprecip_h', 'ffd_l', 'ffd_r', 'ffd_h', 'nirrcapcl', 'nirrcapscl', 'nirrcapunit', 'irrcapcl', 'irrcapscl', 'irrcapunit', 'cropprodindex', 'constreeshrubgrp', 'wndbrksuitgrp', 'rsprod_l', 'rsprod_r', 'rsprod_h', 'foragesuitgrpid', 'wlgrain', 'wlgrass', 'wlherbaceous', 'wlshrub', 'wlconiferous', 'wlhardwood', 'wlwetplant', 'wlshallowwat', 'wlrangeland', 'wlopenland', 'wlwoodland', 'wlwetland', 'soilslippot', 'frostact', 'initsub_l', 'initsub_r', 'initsub_h', 'totalsub_l', 'totalsub_r', 'totalsub_h', 'hydgrp', 'corcon', 'corsteel', 'taxclname', 'taxorder', 'taxsuborder', 'taxgrtgroup', 'taxsubgrp', 'taxpartsize', 'taxpartsizemod', 'taxceactcl', 'taxreaction', 'taxtempcl', 'taxmoistscl', 'taxtempregime', 'soiltaxedition', 'castorieindex', 'flecolcomnum', 'flhe', 'flphe', 'flsoilleachpot', 'flsoirunoffpot', 'fltemik2use', 'fltriumph2use', 'indraingrp', 'innitrateleachi', 'misoimgmtgrp', 'vasoimgtgrp', 'mukey', 'cokey']

"""
df_comp = pd.read_csv("../wss_gsmsoil_CA_[2016-10-13]/tabular/comp.txt",delimiter="|",names = columns)
print(list(df_comp.iloc[1000,:]))