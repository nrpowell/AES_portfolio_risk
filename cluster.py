from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime
from itertools import repeat

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import csv
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

_NUM_SIMULATIONS    = 1000
_NUM_DAYS_SIMULATED = 100
_NUM_CLUSTERS       = 80

"""def simulate_one_cluster(random_normal_mtx, cluster_rep, coefs, eigenvectors, eigenvalues, mean_log_returns, std_log_returns, directory):
    n_iterations = _NUM_SIMULATIONS
    cluster_generated_arr = np.empty((eigenvectors.shape[1], n_iterations))
    for i in range(0, n_iterations):
        random_normals = random_normal_mtx[i]
        x = np.multiply(random_normals, np.sqrt(eigenvalues))
        gen_eigenvalues = np.diag(x)
        simulated_evs = np.dot(gen_eigenvalues, eigenvectors)
        simulated_y = np.dot(coefs, simulated_evs)

        cluster_generated_arr[:,i] = simulated_y

    simul_arr = (cluster_generated_arr * std_log_returns) + mean_log_returns
    df_simul_arr = pd.DataFrame(simul_arr)

    if not os.path.isdir(f"{directory}/pcas"):
        os.makedirs(f"{directory}/pcas")
    df_simul_arr.to_csv(f"{directory}/pcas/{cluster_rep}_cluster.csv")"""

def single_simulation(random_normal_vec, eigenvectors, eigenvalues, mean_log_returns, std_log_returns, cluster_rep_names, simulation_output_dict):
    lmbda = np.diag(np.sqrt(eigenvalues))
    eta = np.dot(np.dot(np.transpose(eigenvectors), lmbda), random_normal_vec)
    eta_renormalized = (eta * std_log_returns) + mean_log_returns

    forward_simulation = np.empty((_NUM_DAYS_SIMULATED, eta_renormalized.shape[0]), dtype=float)

    ## The numbers we are calculating here simply need to be multiplied by the day-0 price to get the day-n price; they represent the change on the day-0 price, de-logged
    for d in range(1, _NUM_DAYS_SIMULATED+1):
        date_d_return = np.exp(eta_renormalized * math.sqrt(d))
        forward_simulation[d-1] = np.transpose(date_d_return)

    for n in range(0, len(cluster_rep_names)):
        name = cluster_rep_names[n]
        one_contract_simulation = np.reshape(forward_simulation[:,n], (-1, 1))
        if name in simulation_output_dict:
            running_sim_output = simulation_output_dict[name]
            modified_matrix = np.append(running_sim_output, one_contract_simulation, axis=1)
            simulation_output_dict[name] = modified_matrix
        else:
            simulation_output_dict[name] = one_contract_simulation

    return simulation_output_dict

## TODO: Make this smarter!
def choose_cluster_rep(log_returns_dict, contract_clusters):
    filtered_returns_dict = {}
    for key, value in contract_clusters.items():

        ## ATM, this simply chooses a random cluster representative from the list
        rep = random.choice(value)
        filtered_returns_dict[rep] = log_returns_dict[rep] 

    return filtered_returns_dict

def draw_random_normals(num_columns):
    return np.random.normal(0, 1, (_NUM_SIMULATIONS, num_columns))

def do_pca(log_returns_dict, contract_clusters, mean_log_returns, std_log_returns, directory):
    ## Determine cluster representatives
    filtered_returns_dict = choose_cluster_rep(log_returns_dict, contract_clusters)
    df_returns = pd.DataFrame.from_dict(filtered_returns_dict, orient='columns')

    pca = PCA(n_components = 0.98)
    pca.fit(df_returns)
    #pca_transform = PCA(n_components = 0.98).fit_transform(df_returns)

    simulation_output_dict = {}

    cluster_rep_names = list(df_returns.columns)
    random_normal_mtx = draw_random_normals(len(pca.explained_variance_))

    for i in range(0, _NUM_SIMULATIONS):
        simulation_output_dict = single_simulation(random_normal_mtx[i], pca.components_, pca.explained_variance_, mean_log_returns, std_log_returns, cluster_rep_names, simulation_output_dict)


    for contract, sims_arr in simulation_output_dict.items():
        if not os.path.isdir(f"{directory}/pcas"):
            os.makedirs(f"{directory}/pcas")
        pd.DataFrame(sims_arr).to_csv(f"{directory}/pcas/{contract}_cluster.csv")
    """for r in range(0, pca_transform.shape[0]):
        row_coefs = pca_transform[r]
        simulate_one_cluster(random_normal_mtx, cluster_rep_names[r], row_coefs, pca.components_, pca.explained_variance_, mean_log_returns, std_log_returns, directory)"""

def write_clusters_to_file(directory, contract_clusters):
    if not os.path.isdir(f"{directory}"):
        os.makedirs(f"{directory}")
    fp = os.getcwd() + f"/{directory}/Clusters.csv"
    writer = csv.writer(open(fp,'a'))
    for key, value in contract_clusters.items():
        #print([key] + value)
        writer.writerow(value)

def do_clustering(df, df_2, columns, directory, num_clusters):
    log_returns_dict = OrderedDict()
    price_dict = OrderedDict()

    minimum_list_length = sys.maxsize

    # To normalize to mean 0 and unit variance, we have to keep track of all the data and then perform a normalization at the end
    all_log_returns = []
    all_prices = []

    combined_cluster = "isFX" in df.columns

    for i in range(0, len(df.index)):
        contract_date = df.iloc[i, 0]
        isFX = df.loc[i, ["isFX"]][0]
        tup_contract_date = tuple(map(int, contract_date[1:-1].split(', '))) 

        dt = datetime(*tup_contract_date)
        month_year = dt.strftime("%B%Y")
        for col in columns:
            if isFX and col == "USD":
                continue
                
            log_returns_list_raw = df.loc[i,[col]][0]
            price_list_raw = df_2.loc[i,[col]][0]

            log_returns_list = list(map(float, log_returns_list_raw[1:-1].split(', ')))
            price_list = list(map(float, price_list_raw[1:-1].split(', ')))

            all_log_returns = all_log_returns + log_returns_list
            all_prices = all_prices + price_list

            if len(log_returns_list) < minimum_list_length:
                minimum_list_length = len(log_returns_list)

            key_name = col + "_" + month_year

            if combined_cluster:
                prefix = "FX_" if isFX == 1 else "IR_" 
                key_name = prefix + key_name

            log_returns_dict[key_name] = log_returns_list
            price_dict[key_name] = price_list


    mean_log_returns = np.mean(all_log_returns)
    std_log_returns = np.std(all_log_returns)
    
    mean_prices = np.mean(all_prices)
    std_prices = np.std(all_prices)

    ## Equalize the lengths of the log list and price list (they have the same set of keys)
    for key in log_returns_dict.keys():
        return_list = log_returns_dict[key]
        price_list = price_dict[key]

        log_returns_dict[key] = [((r - mean_log_returns)/std_log_returns) for r in return_list[0:minimum_list_length]]
        price_dict[key] = [((p - mean_prices)/std_prices) for p in price_list[0:minimum_list_length]]

    nparr = np.empty((len(log_returns_dict.keys()), minimum_list_length), dtype=object)
    ordered_contract_list = []
    counter = 0
    for key in log_returns_dict.keys():
        ordered_contract_list.append(key)

        return_list = log_returns_dict[key]
        #price_list = price_dict[key]

        nparr[counter] = return_list #+ price_list
        counter += 1
    df_to_fit = pd.DataFrame(nparr)

    kmeans = KMeans(n_clusters=num_clusters, init='random').fit(df_to_fit)
    
    cluster_labels = kmeans.labels_

    contract_clusters = {}
    for l in range(0, len(cluster_labels)):
        label = cluster_labels[l]
        if label not in contract_clusters:
            contract_clusters[label] = [ordered_contract_list[l]]
        else:
            running_list = contract_clusters[label]
            running_list.append(ordered_contract_list[l])
            contract_clusters[label] = running_list


    ## Calculates a cluster mean for each cluster
    sep = '\n'
    cluster_dict = {}
    write_clusters_to_file(directory, contract_clusters)
    for key, value in contract_clusters.items():
        print(f"Cluster {key} has the following contracts: \n{sep.join(value)}\n\n")
        average_cluster_return_list = []

        counter = 1
        currency_name = str(counter) + "_" + value[0][0:9]
        for i in range(0, minimum_list_length):
            log_returns_for_date = []
            for contract in value:
                log_returns_for_date.append(log_returns_dict[contract][i])
            average_cluster_return_list.append(np.mean(log_returns_for_date))

        while currency_name in cluster_dict:
            currency_name = str(counter + 1) + currency_name[len(str(counter)):]
            counter += 1

        cluster_dict[currency_name] = average_cluster_return_list

    # Cluster correlations
    df_corr = pd.DataFrame.from_dict(cluster_dict)
    corr = df_corr.corr()
    #corr.to_csv("ClusterCorr.csv")
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    #plt.savefig(f"cluster_corr.png")
    do_pca(log_returns_dict, contract_clusters, mean_log_returns, std_log_returns, directory)

def combine_data_frames(df_ir, df_fx, df_2_ir, df_2_fx):
    df_ir.rename(lambda x: x[3:] if x != "Unnamed: 0" else "SettDate", axis='columns', inplace=True)
    df_2_ir.rename(lambda x: x[3:] if x != "Unnamed: 0" else "SettDate", axis='columns', inplace=True)

    df_fx.rename(lambda x: x[6:] if x != "Unnamed: 0" else "SettDate", axis='columns', inplace=True)
    df_2_fx.rename(lambda x: x[6:] if x != "Unnamed: 0" else "SettDate", axis='columns', inplace=True)

    df_fx['USD'] = 0
    df_2_fx['USD'] = 0

    df_ir['isFX'] = 0
    df_2_ir['isFX'] = 0

    df_fx['isFX'] = 1 
    df_2_fx['isFX'] = 1

    frames = [df_fx, df_ir]
    frames_2 = [df_2_fx, df_2_ir]

    concat_log_returns = pd.concat(frames, ignore_index=True, axis=0, sort=False)
    concat_raw_prices = pd.concat(frames_2, ignore_index=True, axis=0, sort=False)

    #concat_log_returns = df_fx.append(df_ir, sort=False)
    #concat_raw_prices = df_2_fx.append(df_2_ir, sort=False)

    return (concat_log_returns, concat_raw_prices)
    #df_2_concat.to_csv("combinedRawPriceList.csv")
        
def set_up_clustering(directory):
    ## Because USDARS doesn't fit neatly into the cluster schematics, we create clusters without it and clusters specifically for it
    if directory == "FX":
        fp = os.getcwd() + f"/{directory}/LogReturnsList.csv"
        df = pd.read_csv(fp)
        fp_2 = os.getcwd() + f"/{directory}/RawPriceList.csv"
        df_2 = pd.read_csv(fp_2)

        columns_main = [col for col in df.columns[1:] if col != "FX_USDARS"]
        do_clustering(df, df_2, columns_main, directory, _NUM_CLUSTERS)

        columns_usdars = [col for col in df.columns[1:] if col == "FX_USDARS"]
        do_clustering(df, df_2, columns_usdars, directory, 10)
    elif directory == "IR":
        fp = os.getcwd() + f"/{directory}/LogReturnsList.csv"
        df = pd.read_csv(fp)
        fp_2 = os.getcwd() + f"/{directory}/RawPriceList.csv"
        df_2 = pd.read_csv(fp_2)
        do_clustering(df, df_2, df.columns[1:], directory, _NUM_CLUSTERS)
    elif directory == "combined":
        fp_ir = os.getcwd() + f"/IR/LogReturnsList.csv"
        df_ir = pd.read_csv(fp_ir)
        fp_fx = os.getcwd() + f"/FX/LogReturnsList.csv"
        df_fx = pd.read_csv(fp_fx)

        fp_2_ir = os.getcwd() + f"/IR/RawPriceList.csv"
        df_2_ir = pd.read_csv(fp_2_ir)
        fp_2_fx = os.getcwd() + f"/FX/RawPriceList.csv"
        df_2_fx = pd.read_csv(fp_2_fx)

        (concat_log_returns, concat_raw_prices) = combine_data_frames(df_ir, df_fx, df_2_ir, df_2_fx)
        do_clustering(concat_log_returns, concat_raw_prices, concat_log_returns.columns[1:-1], directory, _NUM_CLUSTERS)
    else:
        pass


def main(directory):
    set_up_clustering(directory)
    #do_pca(log_returns_dict, contract_clusters, mean_log_returns, std_log_returns, directory)

if __name__ == '__main__':
    parser = ArgumentParser(
        description="cluster -d DATA_DIRECTORY")
    parser.add_argument('-d', '--directory', help="Directory with all the data", required=True)
    sysargs = parser.parse_args()
    main(sysargs.directory)