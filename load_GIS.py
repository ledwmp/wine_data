import pandas as pd
import numpy as np

def load_GIS():
    """Every mapunit is made of components, this component table describes soil
    composition of mapunits.
    Returns:
        Dataframe with each mapunit as a row with component averages
    """
    with open("../wss_gsmsoil_CA_[2016-10-13]/component_table.txt") as r:
            columns = [i.strip() for i in r]
    """
    columns=
    ['comppct_l', 'comppct_r', 'comppct_h', 'compname', 'compkind', 'majcompflag', 'otherph', 'localphase', 'slope_l', 'slope_r', 'slope_h', 'slopelenusle_l', 'slopelenusle_r', 'slopelenusle_h', 'runoff', 'tfact', 'wei', 'weg', 'erocl', 'earthcovkind1', 'earthcovkind2', 'hydricon', 'hydricrating', 'drainagecl', 'elev_l', 'elev_r', 'elev_h', 'aspectccwise', 'aspectrep', 'aspectcwise', 'geomdesc', 'albedodry_l', 'albedodry_r', 'albedodry_h', 'airtempa_l', 'airtempa_r', 'airtempa_h', 'map_l', 'map_r', 'map_h', 'reannualprecip_l', 'reannualprecip_r', 'reannualprecip_h', 'ffd_l', 'ffd_r', 'ffd_h', 'nirrcapcl', 'nirrcapscl', 'nirrcapunit', 'irrcapcl', 'irrcapscl', 'irrcapunit', 'cropprodindex', 'constreeshrubgrp', 'wndbrksuitgrp', 'rsprod_l', 'rsprod_r', 'rsprod_h', 'foragesuitgrpid', 'wlgrain', 'wlgrass', 'wlherbaceous', 'wlshrub', 'wlconiferous', 'wlhardwood', 'wlwetplant', 'wlshallowwat', 'wlrangeland', 'wlopenland', 'wlwoodland', 'wlwetland', 'soilslippot', 'frostact', 'initsub_l', 'initsub_r', 'initsub_h', 'totalsub_l', 'totalsub_r', 'totalsub_h', 'hydgrp', 'corcon', 'corsteel', 'taxclname', 'taxorder', 'taxsuborder', 'taxgrtgroup', 'taxsubgrp', 'taxpartsize', 'taxpartsizemod', 'taxceactcl', 'taxreaction', 'taxtempcl', 'taxmoistscl', 'taxtempregime', 'soiltaxedition', 'castorieindex', 'flecolcomnum', 'flhe', 'flphe', 'flsoilleachpot', 'flsoirunoffpot', 'fltemik2use', 'fltriumph2use', 'indraingrp', 'innitrateleachi', 'misoimgmtgrp', 'vasoimgtgrp', 'mukey', 'cokey']

    """
    #read components in from tabular folder
    df_comp = pd.read_csv("../wss_gsmsoil_CA_[2016-10-13]/tabular/comp.txt",delimiter="|",names = columns)
    df_comp.dropna(axis=1,how="all",inplace=True)
    """
    columns=
    ['comppct_r', 'compname', 'compkind', 'majcompflag', 'otherph',
       'slope_l', 'slope_r', 'slope_h', 'tfact', 'wei', 'weg', 'hydricrating',
       'drainagecl', 'elev_l', 'elev_r', 'elev_h', 'airtempa_l', 'airtempa_r',
       'airtempa_h', 'map_l', 'map_r', 'map_h', 'ffd_l', 'ffd_r', 'ffd_h',
       'nirrcapcl', 'nirrcapscl', 'irrcapcl', 'irrcapscl', 'rsprod_l',
       'rsprod_r', 'rsprod_h', 'frostact', 'initsub_l', 'initsub_r',
       'initsub_h', 'totalsub_l', 'totalsub_r', 'totalsub_h', 'hydgrp',
       'corcon', 'corsteel', 'taxclname', 'taxorder', 'taxsuborder',
       'taxgrtgroup', 'taxsubgrp', 'taxpartsize', 'taxpartsizemod',
       'taxceactcl', 'taxreaction', 'taxtempcl', 'taxmoistscl',
       'taxtempregime', 'soiltaxedition', 'mukey', 'cokey']
    """
    #drop components that are redundant with other categories,or will probably not be helpful
    #dropping the cokey here, this is the key that accesses other tables, will need if other tables used
    df_comp = df_comp[(df_comp["compkind"] == "Series")]
    df_comp.drop(['airtempa_l', 'airtempa_r','airtempa_h', 'map_l', 'map_r',\
                    'map_h', 'ffd_l', 'ffd_r', 'ffd_h','corcon', 'corsteel',\
                    'soiltaxedition','taxclname','compname','taxreaction',\
                    'compkind','taxpartsizemod','taxceactcl','majcompflag','cokey',\
                    'hydricrating','taxtempregime'],\
                    inplace=True,axis=1)
    #drop columns that are mostly np.nan
    df_comp.dropna(thresh=len(df_comp)/2,axis=1,inplace=True)

    def collapse_mukey(tmp):
        """Takes groupby mukey as input, takes weighted average of columns based
        on comppct_r for dtypes int and float
        """
        #for some reason these pcts don't add up
        tmp["pct_mukey"] = tmp["comppct_r"]/sum(tmp["comppct_r"])
        mask = ((tmp.dtypes == float) | (tmp.dtypes == int)) &\
         ((tmp.columns != "pct_mukey") & (tmp.columns != "mukey"))
        tmp.loc[:,mask] = tmp.loc[:,mask].multiply(tmp["pct_mukey"],axis=0)
        return tmp

    df_group_num = df_comp.groupby("mukey").apply(collapse_mukey)
    df_group_num = df_group_num.groupby("mukey").mean()
    mask = ((df_group_num.dtypes == float) | (df_group_num.dtypes == int))
    df_group_num = df_group_num.loc[:,mask]
    df_group_num.drop(["comppct_r","pct_mukey"],axis=1,inplace=True)
    def try_index_list(tmp):
        try:
            return tmp[0]
        except:
            return np.nan
    #this is sloppy and needs to be fixed with dedicated function
    df_group_obj = df_comp.groupby("mukey").agg(lambda x: try_index_list(x.value_counts().index))
    mask = (df_group_obj.dtypes == object)
    df_group_obj = df_group_obj.loc[:,mask]
    return df_group_num.merge(df_group_obj,left_index=True,right_index=True).reset_index()
if __name__ == "__main__":
    load_GIS()
