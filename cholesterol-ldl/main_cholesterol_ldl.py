import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

class Data(BaseModel):
    data: str

def inference(df: pd.DataFrame) -> pd.DataFrame:
    
    with open('cholesterol_ldl_human_pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    with open('cholesterol_ldl_human_pca', 'rb') as pickle_file:
        pca = pickle.load(pickle_file)

    df_pca = pca.transform(df)
    result = model.predict(df_pca[:,  0:10])
    df['target'] = result
    df['target'] = df['target'].replace({0: 'low', 1: 'ok', 2:'high'})

    return df['target']
    
app = FastAPI()

@app.post("/predictCholesterolLdl")
def predict(data: Data):
    x = json.loads(data.data)
    df = pd.DataFrame(x)
    result = inference(df)
    # print("Result:", result.shape)
    return list(result)