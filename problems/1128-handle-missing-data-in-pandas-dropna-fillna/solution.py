import pandas as pd

def solution(df):
    kc = [c for c in df.columns if df[c].isna().mean() <= 0.5]
    df = df[kc]

    df = df.loc[df.isna().mean(axis=1) <= 0.5]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].mean())
        else:
            mode = df[c].mode()
            if not mode.empty:
                df[c] = df[c].fillna(mode.iloc[0])

    
    return df
   

    pass