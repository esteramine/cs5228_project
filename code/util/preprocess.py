import numpy as np
import pandas as pd

def get_floor_avg(floor_range):
    floors = [int(x) for x in floor_range.split(' to ')]
    return int((floors[0] + floors[1]) / 2)

def update_flat_type(flat):
    if flat == "2 ROOM" or flat == "1 ROOM":
        return "1 / 2 ROOM"
    elif flat == "EXECUTIVE" or flat == 'MULTI-GENERATION':
        return 'EXECUTIVE & MULTI-GENERATION'
    else: 
        return flat

def update_model_type(flat):
    if flat == 'Model A-Maisonette' or flat == 'Improved-Maisonette' or flat == 'Premium Maisonette':
        return 'Maisonette'
    elif flat == 'Premium Apartment' or flat == 'Premium Apartment Loft':
        return 'Apartment'
    elif flat == 'Terrace' or flat == 'Multi Generation' or flat =='Adjoined flat':
        return "Terrace/Multi-Gen/Adjoined"
    elif flat == 'Model A2':
        return "Model A"
    elif flat == 'Type S1' or flat == 'Type S2':
        return "Type S"
    else: 
        return flat


def preprocess(df_original):
    df = df_original.copy()
    # MONTH
    df[['RESALE_YEAR', 'RESALE_MONTH']] = df['MONTH'].str.split('-', expand=True)
    df['RESALE_YEAR'] = df['RESALE_YEAR'].astype(int)
    df['RESALE_MONTH'] = df['RESALE_MONTH'].astype(int)

    # STREET
    df['STREET'] = df['STREET'].str.strip().str.lower()

    # FLAT_TYPE
    df['FLAT_TYPE'] = df['FLAT_TYPE'].replace({
        '5-room': '5 room',
        '4-room': '4 room',
        '3-room': '3 room',
        '2-room': '2 room',
        '1-room': '1 room',
    })

    # FLAT_AGE
    df['FLAT_AGE'] = df['RESALE_YEAR'] - df['LEASE_COMMENCE_DATA']

    # FLOOR_AVG
    df['FLOOR_AVG'] = df['FLOOR_RANGE'].apply(get_floor_avg)

    # drop unused data
    df = df.drop(['ECO_CATEGORY'], axis=1)
    # ['TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE', 'FLOOR_BUCKET', 'FLOOR_RANGE']

    return df