# AES_portfolio_risk

This was written with Python 3.7.5

How to get the desired outputs:
1. Run 'corr.py -d FX' and 'corr.py -d IR'; price list and log return files will be created in the FX & IR subdirectories.
2. Run 'cluster.py -d combined'; the clusters will be writen to combined/Clusters.csv, and the simulation results to combined/pcas/*
3. Run 'analyze_simulations.py -d combined -t 3'; this calculates the exposures after 1, 5, and 20 holding days of the FX and IR portfolios and saves the histograms to combined/histograms/*.
