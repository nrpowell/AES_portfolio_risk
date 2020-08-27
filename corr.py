from argparse import ArgumentParser
from datetime import datetime
from scipy.stats.stats import pearsonr

from constants import minimum_contract_date, maximum_contract_date, minimum_settlement_date

import copy
import math
import os
import requests
import urllib
import xlrd

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 


_OUTPUT_CORR_MATRIX = 1
_SAVE_DICT = 2

def createCorrelationMatrix(vert_currency, horz_currency, vert_currency_name, horz_currency_name):
    assert len(vert_currency.keys()) == len(horz_currency.keys()),f"{vert_currency_name} has {len(vert_currency.keys())} contracts; {horz_currency_name} has {len(horz_currency.keys())}"

    ## We want to order the contract dates (the keys of the two currency dicts) and then calculate the correlations...
    sorted_vert_contracts = sorted(list(vert_currency.keys()))
    sorted_horz_contracts = sorted(list(horz_currency.keys()))

    arr = np.zeros((len(sorted_vert_contracts), len(sorted_horz_contracts)))

    row_names = []
    col_names = []
    for i in range(0, len(sorted_vert_contracts)):
        date_tuple = sorted_vert_contracts[i]
        dt = datetime(*date_tuple)
        month_year = dt.strftime("%B %Y")
        row_names.append(month_year)
        col_names.append(month_year)

        vert_log_returns = vert_currency[sorted_vert_contracts[i]]
        for j in range(i, len(sorted_horz_contracts)):
            horz_log_returns = horz_currency[sorted_horz_contracts[j]]

            if len(vert_log_returns) != len(horz_log_returns):
                cutoff = len(vert_log_returns) if len(vert_log_returns) < len(horz_log_returns) else len(horz_log_returns)
                vert_log_returns = vert_log_returns[:cutoff]
                horz_log_returns = horz_log_returns[:cutoff]

            r = round(pearsonr(vert_log_returns, horz_log_returns)[0], 2)
            arr[i, j] = r
            arr[j, i] = r

    df = pd.DataFrame(arr, index=row_names, columns=col_names)
    return df


def main(function, directory):
    fp = os.getcwd() + f"/{directory}/historical_forwards_new.xlsx"
    wb = xlrd.open_workbook(fp)

    sheet = wb.sheet_by_index(0)

    col_currency    = 0
    col_setdate     = 4
    col_contract    = 5
    col_price       = 7

    current_date = (2020, 8, 10, 0, 0, 0)

    ## Syntax for getting the dates
    xlrd.xldate_as_tuple(sheet.cell_value(4,col_setdate), wb.datemode)

    ## Settable constants determining the size of the matrix
    #minimum_contract_date   = (2020, 8, 1, 0, 0, 0)
    #maximum_contract_date   = (2023, 8, 1, 0, 0, 0)
    #minimum_settlement_date = (2019, 6, 22, 0, 0, 0)

    log_returns_dict = {}

    all_settlement_dates = set()

    for i in range(1, sheet.nrows):
        settlement_date = xlrd.xldate_as_tuple(sheet.cell_value(i, col_setdate), wb.datemode)
        contract_date = xlrd.xldate_as_tuple(sheet.cell_value(i, col_contract), wb.datemode)

        if (settlement_date > contract_date) or (settlement_date < minimum_settlement_date) or (contract_date >= maximum_contract_date) or (contract_date < minimum_contract_date):
            continue
        currency = sheet.cell_value(i, col_currency)
        all_settlement_dates.add(settlement_date)

        price = sheet.cell_value(i, col_price)

        ## For FX data only - EURUSD is the inverse of the other exchange rates, which are in the form USDCOP, USDBRL etc
        if currency == "FX_EURUSD":
            price = (1.0 / price)
            currency = "FX_USDEUR"

        if currency not in log_returns_dict:
            print("-------------")
            log_returns_dict[currency] = {}
            
        contract_dict = log_returns_dict[currency]
        if contract_date not in contract_dict:
            contract_dict[contract_date] = [price]
        else:
            price_list = contract_dict[contract_date]
            price_list.append(price)
            contract_dict[contract_date] = price_list

        log_returns_dict[currency] = contract_dict

    price_dict = copy.deepcopy(log_returns_dict)

    sep = '\n'
    print(f"Sorted list of settlement dates looks like: {sep.join(map(str, sorted(list(all_settlement_dates))))}")
    ## Convert price list to sequence of log returns
    for key in log_returns_dict.keys():
        contract_dict = log_returns_dict[key]
        for contract in contract_dict.keys(): 
            price_list = contract_dict[contract]
            log_returns_list = []
            for p in range(1, len(price_list)):
                log_return = math.log(price_list[p-1]/price_list[p])
                    
                log_returns_list.append(log_return)
            #print(f"Length of log returns list is {len(log_returns_list)}")
            contract_dict[contract] = log_returns_list
        log_returns_dict[key] = contract_dict

    ## Create correlation matrices
    if function == _OUTPUT_CORR_MATRIX:
        keys_list = list(log_returns_dict.keys())
        for i in range(0, len(keys_list)):
            vert_currency = log_returns_dict[keys_list[i]]
            for j in range(i, len(keys_list)):
                horz_currency = log_returns_dict[keys_list[j]]
                df = createCorrelationMatrix(vert_currency, horz_currency, keys_list[i], keys_list[j])
                df.to_csv(f"{directory}/{keys_list[i]}-{keys_list[j]}.csv")

    elif function == _SAVE_DICT:
        currency_names = list(log_returns_dict.keys())
        print(f"Currency names are: {currency_names}")
        ## Contract dates should be the same for each currency, so we can just index into a random one and pull the dict keys from that
        contract_dates = sorted(list(log_returns_dict[currency_names[0]].keys()))

        data = np.empty((len(contract_dates), len(currency_names)), dtype=np.object)
        data_raw_price = np.empty((len(contract_dates), len(currency_names)), dtype=np.object)
        data_sd = np.zeros((len(contract_dates), len(currency_names)))
        for i in range(0, len(currency_names)):
            curr = currency_names[i]

            contract_dict = log_returns_dict[curr]
            contract_dict_raw = price_dict[curr]
            for j in range(0, len(contract_dates)):
                date = contract_dates[j]
                log_returns_list = contract_dict[date]
                price_list = contract_dict_raw[date]

                data[j, i] = log_returns_list
                data_raw_price[j, i] = price_list
                data_sd[j, i] = np.std(log_returns_list)

        df = pd.DataFrame(data, index=contract_dates, columns=currency_names)
        df_raw = pd.DataFrame(data_raw_price, index=contract_dates, columns=currency_names)
        df_sd = pd.DataFrame(data_sd, index=contract_dates, columns=currency_names)

        df.to_csv(f"{directory}/LogReturnsList.csv")
        df_raw.to_csv(f"{directory}/RawPriceList.csv")
        df_sd.to_csv(f"{directory}/SDLogReturnsList.csv")
    else:
        pass

    return log_returns_dict



if __name__ == '__main__':
    parser = ArgumentParser(
        description="corr -d DATA_DIRECTORY")
    parser.add_argument('-d', '--directory', help="Directory with all the data", required=True)
    sysargs = parser.parse_args()
    main(_SAVE_DICT, sysargs.directory)