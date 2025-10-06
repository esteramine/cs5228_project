import numpy as np
import pandas as pd

def get_floor_avg(floor_range):
    floors = [int(x) for x in floor_range.split(' to ')]
    return int((floors[0] + floors[1]) / 2)


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