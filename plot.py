## Want to plot changes in exchange rate against changes in the ratio of interest rates that make up the exchange rate 
## So like, FX_EURUSD vs. IR_EUR/IR_USD

from datetime import datetime
from scipy.stats.stats import pearsonr

## Calculate regression coefficients. If the fit is good enough, 
import csv
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def find_fx_ir_correlations(df, ir_dict):
    columns = df.columns[1:]
    for i in range(0, len(df.index)):
        contract_date = df.iloc[i, 0]
        tup_contract_date = tuple(map(int, contract_date[1:-1].split(', '))) 

        dt = datetime(*tup_contract_date)
        month_year = dt.strftime("%B%Y")
        for col in columns:
            fx_list_raw = df.loc[i,[col]][0]
            fx_price_list = list(map(float, fx_list_raw[1:-1].split(', ')))
            fx_log_returns_list = fx_price_list#[]
            """for p in range(1, len(fx_price_list)):
                log_return = math.log(fx_price_list[p-1]/fx_price_list[p])
                fx_log_returns_list.append(log_return)"""

            ir_key_set = [l for l in set(ir_dict.keys()) if l in col]
            assert(len(ir_key_set) == 1)
            ir_key = ir_key_set[0]

            ir_log_returns_list = ir_dict[ir_key][month_year]
            if len(ir_log_returns_list) != len(fx_log_returns_list):
                cutoff = len(ir_log_returns_list) if len(ir_log_returns_list) < len(fx_log_returns_list) else len(fx_log_returns_list)
                ir_log_returns_list = ir_log_returns_list[:cutoff]
                fx_log_returns_list = fx_log_returns_list[:cutoff]

            corr = round(pearsonr(ir_log_returns_list, fx_log_returns_list)[0], 2)
            print(f"Currency {ir_key} at date {month_year} - correlation is {corr}")
            plt.scatter(ir_log_returns_list, fx_log_returns_list)
            plt.xlabel("Raw normalized interest rate")
            plt.ylabel("FX rate")
            plt.savefig(f"plots_raw/{ir_key}_{month_year}")
            plt.clf()


def transform_ir_list(df):
    columns = [col for col in df.columns[1:] if col != "IR_USD"]
    return_dict = {}

    for i in range(0, len(df.index)):
        contract_date = df.iloc[i, 0]
        tup_contract_date = tuple(map(int, contract_date[1:-1].split(', '))) 

        dt = datetime(*tup_contract_date)
        month_year = dt.strftime("%B%Y")
        for col in columns:
            ir_list_raw = df.loc[i,[col]][0]
            us_ir_list_raw = df.loc[i,['IR_USD']][0]

            ir_array = np.asarray(list(map(float, ir_list_raw[1:-1].split(', '))))
            us_ir_array = np.asarray(list(map(float, us_ir_list_raw[1:-1].split(', '))))

            #print(f"Interest rate for {col}, {month_year} has a corr coefficient with the US rate of {round(pearsonr(ir_array, us_ir_array)[0], 2)}")
            relative_ir_list = ir_array / us_ir_array
            #print(f"Relative IR list: {relative_ir_list}")
            log_returns_list = relative_ir_list #[]
            """for p in range(1, len(relative_ir_list)):
                log_return = math.log(relative_ir_list[p-1]/relative_ir_list[p])
                log_returns_list.append(log_return)"""

            ## Chop off the "IR_" part of the name
            col_key = col[3:]
            if col_key not in return_dict:
                val_dict = {month_year: log_returns_list}
                return_dict[col_key] = val_dict
            else:
                val_dict = return_dict[col_key]
                val_dict[month_year] = log_returns_list
                return_dict[col_key] = val_dict
    return return_dict

def main():
    fp_fx = os.getcwd() + f"/FX/RawPriceList.csv"
    df_fx = pd.read_csv(fp_fx)
    
    fp_ir = os.getcwd() + f"/IR/RawPriceList.csv"
    df_ir = pd.read_csv(fp_ir)

    ir_dict = transform_ir_list(df_ir)
    find_fx_ir_correlations(df_fx, ir_dict)
    #for i in range(0, )
    #for col in df_fx.columns[1:]:


if __name__ == '__main__':
    main()
