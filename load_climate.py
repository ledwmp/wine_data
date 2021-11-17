import pandas as pd
import numpy as np

def load_climate():
    """Load precipitation and temperature dataframes
    """
    df_pcp = pd.read_csv("../4-pcp.csv",delimiter=",",skiprows=3)
    df_tavg = pd.read_csv("../4-tavg.csv",delimiter=",",skiprows=3)
    df_pcp["Date"] = pd.to_datetime(df_pcp["Date"],format="%Y%m")
    df_tavg["Date"] = pd.to_datetime(df_tavg["Date"],format="%Y%m")
    mask = (df_pcp["Date"] >= pd.to_datetime("199501",format="%Y%m"))
    df_pcp = df_pcp[mask].reset_index()
    df_pcp.drop(['Rank','Anomaly (1901-2000 base period)',\
                        '1901-2000 Mean','index','Location ID'],axis=1,inplace=True)
    mask = (df_tavg["Date"] >= pd.to_datetime("199501",format="%Y%m"))
    df_tavg = df_tavg[mask].reset_index()
    df_tavg.drop(['Rank','Anomaly (1901-2000 base period)',\
                        '1901-2000 Mean','index','Location ID'],axis=1,inplace=True)
    df_tavg["Location"] = df_tavg["Location"].str.split(" County").str[0].str.upper()
    df_pcp["Location"] = df_pcp["Location"].str.split(" County").str[0].str.upper()

    return df_tavg,df_pcp



if __name__ == "__main__":
    load_climate()
