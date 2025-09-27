import numpy as np
import pandas as pd


def get_floor_mean(floor_range):
    floors = floor_range.split(' to ')
    return (floors[0] + floors[1])/2

def clean_flat_type(flat_type):
    x = x.lower().replace(" ", "-")
    return x

def preprocess(df):
    # MONTH
    out = df.apply(lambda r: pd.Series(r['MONTH'].split('-'),
                    index=['RESALE_YEAR', 'RESALE_MONTH']), axis=1) # in meters
    df = pd.concat([df, out], axis=1)

    # FLAT_TYPE
    df['FLAT_TYPE'] = df['FLAT_TYPE'].apply(clean_flat_type)
    mapping = {
        '1-room': 1, 
        '2-room': 2, 
        '3-room': 3,
        '4-room': 4, 
        '5-room': 5,
        'executive': 6,       
        'multi-generation': 7  
    }
    df['ROOMS_NUM'] = df['FLAT_TYPE'].map(mapping)

    # FLOOR_RANGE
    df['FLOOR_MEAN'] = df['FLOOR_RANGE'].apply(get_floor_mean)

    #


    return df