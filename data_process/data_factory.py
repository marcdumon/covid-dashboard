# --------------------------------------------------------------------------------------------------------
# 2020/04/06
# src - data_factory.py
# md
# --------------------------------------------------------------------------------------------------------
import io
from pathlib import Path

import requests
import pandas as pd

data_path = Path('/media/md/Development/COVID-19/0_covid.v0/data/')


def download_ecdc():
    url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
    resp = requests.get(url, auth=(":", ":"))
    with open(data_path / 'external/ecdc.csv', 'w') as f:
        f.write(resp.text)


def download_sciensano():
    url = 'https://epistat.sciensano.be/Data/'
    datasets = dict()
    datasets['cases'] = 'COVID19BE_CASES_AGESEX.csv'  # DATE    PROVINCE    REGION  AGEGROUP    SEX CASES
    datasets['hosp'] = 'COVID19BE_HOSP.csv'  # .        DATE    PROVINCE    REGION  NR_REPORTING    TOTAL_IN    TOTAL_IN_ICU    TOTAL_IN_RESP   TOTAL_IN_ECMO   NEW_IN  NEW_OUT
    datasets['mort'] = 'COVID19BE_MORT.csv'  # .        DATE                REGION  AGEGROUP    SEX	DEATHS
    datasets['tests'] = 'COVID19BE_tests.csv'  # .      DATE	TESTS

    # columns = ['date', 'cases', 'new_in', 'new_out', 'total_in', 'total_icu', 'total_icu', 'total_ecmo', 'deaths', 'tests']
    # df = pd.DataFrame(columns=columns)
    df = pd.DataFrame()
    for ds in datasets:
        # Save datasets
        resp = requests.get(url + datasets[ds], auth=(":", ":"))
        with open(data_path / f'external/sciensano_{ds}.csv', 'w') as f:
            f.write(resp.text)
        # Consolidate and save all datasets into sciensano.csv
        dataset = pd.read_csv(io.StringIO(resp.text))
        dataset = dataset.groupby('DATE').sum()
        df = pd.concat([df, dataset], axis=1, sort=False)

    df.to_csv(data_path / 'external/sciensano.csv')


def load_ecdc():
    df = pd.read_csv(data_path / 'external/ecdc.csv')

    df.rename(columns={'dateRep': 'date', 'countriesAndTerritories': 'location', 'continentExp': 'region'}, inplace=True)
    df = df.loc[:, ['date', 'location', 'region', 'cases', 'deaths']]
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df.sort_values(by=['location', 'date'], inplace=True)
    return df


def load_ecdc_cases(): # Todo: Replace by load_ecdc
    df = pd.read_csv(data_path / 'external/ecdc.csv')
    df.rename(columns={'dateRep': 'date', 'countriesAndTerritories': 'location', 'continentExp': 'region'}, inplace=True)
    df = df.loc[:, ['date', 'location', 'region', 'cases']]
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df.sort_values(by=['location', 'date'], inplace=True)
    return df


def load_sciensano_cases():
    df = pd.read_csv(data_path / 'external/sciensano_cases.csv')
    df = df.groupby(['PROVINCE', 'REGION', 'DATE', ]).sum()
    df = pd.DataFrame(df.to_records())
    df.rename(columns={'DATE': 'date', 'PROVINCE': 'location', 'REGION': 'region', 'CASES': 'cases'}, inplace=True)
    df = df.loc[:, ['date', 'location', 'region', 'cases']]
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.sort_values(by=['location', 'date'], inplace=True)
    return df


# load_xxx to cleanup df and abstract data directories etc in notebooks
def load_ECDC():  # Todo: make obsolete
    df = pd.read_csv(data_path / 'external/ecdc.csv')
    # Cleanup
    df.rename(columns={'dateRep': 'date', 'countriesAndTerritories': 'country', 'geoId': 'geoid', 'popData2018': 'pop2018'}, inplace=True)
    df.drop(['countryterritoryCode'], axis=1, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df.sort_values(by=['country', 'date'], inplace=True)
    # Add totals
    for country in df['country']:
        # Todo: Refactor to make much faster + save for reuse
        df.loc[df['country'] == country, 'total_cases'] = df.loc[df['country'] == country, 'cases'].cumsum()
        df.loc[df['country'] == country, 'total_deaths'] = df.loc[df['country'] == country, 'deaths'].cumsum()
    return df


def load_sciensano(fname=''):  # Todo: make obsolete
    """
    fname:  cases, hosp, mort, tests, ''
    """
    if fname:
        df = pd.read_csv(data_path / f'external/sciensano_{fname}.csv')
    else:
        df = pd.read_csv(data_path / f'external/sciensano.csv')

    df.columns = map(str.lower, df.columns)  # Lower case column names
    df['date'] = pd.to_datetime(df['date'])
    return df


#
def add_days_deaths_bigger_n(df, n=0):
    day_n = df.loc[df['total_deaths'] > n, ['date']].min()[0]
    df_copy = df.copy(deep=True)  # Otherwise 'A value is trying to be set on a copy of a slice from a DataFrame.' warning
    df_copy[f'days_death_{n}'] = (df['date'] - day_n).dt.days
    return df_copy


def add_days_cases_bigger_n(df, n=0):
    day_n = df.loc[df['total_cases'] > n, ['date']].min()[0]
    df_copy = df.copy(deep=True)  # Otherwise 'A value is trying to be set on a copy of a slice from a DataFrame.' warning
    df_copy[f'days_case_{n}'] = (df['date'] - day_n).dt.days
    return df_copy


def consolidate_sciensano_cases():
    # consolidate sciensano_cases per date and province
    sciensano_cases = load_sciensano('sciensano_cases')
    sciensano_cases = sciensano_cases.groupby(by=['province', 'date']).sum()
    sciensano_cases = pd.pivot_table(sciensano_cases, values='cases', index='date', columns='province')
    sciensano_cases.to_csv(data_path / 'external/xxx.csv')


if __name__ == '__main__':
    # download_ecdc()
    # download_sciensano()
    # load_ecdc_cases()
    # load_sciensano_cases()
    load_ecdc()

    # print(load_sciensano())
    # x = load_ECDC()
    # x = x.loc[x['country'] == 'Belgium', ['date', 'deaths', 'total_deaths']]
    # print(x)

    # print(add_days_deaths_bigger_n(x, 10))
    # consolidate_sciensano_cases()
    pass
