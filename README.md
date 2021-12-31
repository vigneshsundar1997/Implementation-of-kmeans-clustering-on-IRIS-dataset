# Implementation-of-kmeans-clustering-on-MNIST-dataset
Implementation of the the kmeans clustering algorithm from scratch using Python on MNIST dataset and analyzing the various hyperparameters


The project folder contains 3 python files: 
1. exploration.py
2. kmeans.py
3. analysis.py

#####
1. exploration.py

This script contains the exploration of the MNIST dataset. It used digits-raw.csv to view each digit as 28*28 grayscale matrix. It uses digits-embedding.csv to visualize 1000 random data points with their cluster labels.

The script outputs 10 images one for each digit and a scatter plot.

Execution : python3 exploration.py

2. kmeans.py

This script contains the k-means clustering of the data points and the calculation of WC-SSD, SC and NMI for the given K. The execution took about 5 to 10 seconds in my system.

It takes in a csv file with embedding and a K value

Execution : python3 kmeans.py dataFileName K

eg: 
python3 kmeans.py digits-embedding.csv 10

3. analysis.py

This script contains the analysis of the k-means algorithm. It performs analysis on the various values of K on the three different datasets.
It also performs seed analysis with the K values and the three different datasets. It also performs clustering for the chosen K value and output the NMI value for the three datasets. It does visualization for 1000 randomnly selected data points with their clusters.

Note: 
1. Once the script is run, it takes some time to calculate. The k-analysis takes about 30 to 40 seconds and the seed analysis takes a considerable amount of time close to 2 to 3 minutes. The NMI calculation and the visualization is done quickly.

Execution : python3 analysis.py option

There are three options with which you can run the script.

Option 1 : The script performs analysis on different K values and outputs the graphs and the values.
Option 2 : The script performs seed analysis for the different K values on the different datasets. It outputs the graphs and the report as well.
Option 3 : The script calculates the NMI values for the three datasets and the performs the visualization on the 1000 random data points.

eg:

Command to run K analysis

python3 analysis.py 1

Command to run seed analysis

python3 analysis.py 2

Command to run computation of NMI and visualization

python3 analysis.py 3
