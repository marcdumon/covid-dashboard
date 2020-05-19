# --------------------------------------------------------------------------------------------------------
# 2020/05/13
# src - temp.py
# md
# --------------------------------------------------------------------------------------------------------

def load_ecdc_cases():
    df = pd.read_csv(data_path / 'external/ecdc.csv')
    # Cleanup
    df.rename(columns={'dateRep': 'date', 'countriesAndTerritories': 'location', 'continentExp': 'region'}, inplace=True)
    df = df.loc[:, ['date', 'location', 'region', 'cases']]
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df.sort_values(by=['location', 'date'], inplace=True)
    # print(df['date'])
    print(df)