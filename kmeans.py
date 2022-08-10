"""
This k-means algorithm gathers together countries with similar birthrates and life expectancies on a scatter plot.

The packages which have been imported are matplotlib.pyplot, numpy, csv and random.

First, the program reads the data from a csv file, then it calculates the euclidean distance between two points on the
graph.

Next it calculates the centroid (the average of all the datapoints in the cluster) and then it plots out the
scatter plot using the matplotlib package.

After that, the program calculates the mean birth rates and life expectancies
for each cluster, and prints out the list and number of countries in each cluster.

Lastly the program allows the user to select the data file containing the dataset that they wish to work with, and set
the number of clusters that they want to plot out.
"""


# Importing relevant packages
import matplotlib.pyplot as plt
import numpy as np
import csv
import random


# Defining a function to read the data in the csv file
def read_csv(file):
    """
    This function reads the data from a csv file and adds it to a dictionary
    :param file: the file that the user wishes to use
    :return: a dictionary of the data in the file
    """
    # Creating a dictionary to store the data
    data_dict = {}

    # Using a try-except block to handle file errors
    try:
        with open(file, "r") as csvfile:
            read = csv.reader(csvfile, delimiter=",")
            next(read)

            # Adding the data to the dictionary, with the country names as keys and the data as values
            for row in read:
                country = row[0]
                birthrate = float(row[1])
                life_expectancy = float(row[2])

                data_dict[country] = (birthrate, life_expectancy)

    except FileNotFoundError:
        print('Please enter the correct file name!')

    # Returning the dictionary
    return data_dict


# Defining a function to calculate the Euclidean distance between two points
def euclidean_distance(x1, x2, y1, y2):
    """
    This file calculates the Euclidean distance between two points on the graph
    :param x1: the x co-ordinate of point 1
    :param x2: the x co-ordinate of point 2
    :param y1: the y co-ordinate of point 1
    :param y2: the y co-ordinate of point 2
    :return: the answer of the Euclidean equation (ie the distance between the points)
    """

    # Creating the Euclidean equation that needs to be square-rooted
    euclidean = (x2 - x1) ** 2 + (y2 - y1) ** 2

    # Using the square root function in numpy to calculate the square distance
    eu_sq = np.sqrt(euclidean)

    return eu_sq


# Defining a function to calculate the centroids of a cluster nearest to each data point
def closest_centroid(centroids, datapoint):
    """
    This function finds the nearest centroid to each datapoint
    :param centroids: the list of centroids
    :param datapoint: the datapoints created from the dataset
    :return: the position of the nearest centroid to the datapoint
    """

    # Adding the distances to a list after looping over each set of x and y values
    distances = []
    for point in centroids:
        x1 = point[0]
        y1 = point[1]
        x2 = datapoint[0]
        y2 = datapoint[1]

        # Calling the previously defined function to calculate the Euclidean distance and
        # appending that distance to the list
        distance = euclidean_distance(x1, x2, y1, y2)
        distances.append(distance)

    # Calculating the minimum distance in the array
    min_distance = np.amin(distances)

    # finding the index of that distance in the array
    min_index = (np.where(distances == min_distance))

    # returning that minimum index
    return min_index[0][0]


# Defining a function to initialise the centroids of each cluster
def init_centroids(datapoints, number_clusters):

    """
    This function initialises the centroids of each cluster
    :param datapoints: the list of datapoints from the dataset
    :param number_clusters: the number of clusters in the scatter plot
    :return:
    """

    # Taking a random sample of the data in the dataset
    samples = random.sample(datapoints, number_clusters)

    # Creating a list of centroids
    centroids = [[] for cp in samples]

    # Adding the centroid to the list of centroids and calling the previously defined closest_centroid function
    for point in datapoints:
        centroids[closest_centroid(samples, point)].append(point)

    # Returning the list of centroids
    return centroids


# Defining a function to print out a scatter plot of the data
def cluster_plot(centroids, cmaps):
    """
    This function plots out the graph of the dataset and clusters the datapoints together
    :param centroids: the centroids of the dataset
    :param cmaps: the colours for each cluster on the graph
    :return: the scatter plot of the data
    """

    for index, cluster in enumerate(centroids):
        # Creating a list to store the x and y values of the data
        # The x values come from the birthrates in the data file
        # The y values come from the life expectancy in the file
        x = []
        y = []
        for point in cluster:
            x.append(point[0])
            y.append(point[1])

        # Using the scatter() function to plot the data and also to assign each cluster a different colour
        plt.scatter(x, y, c=cmaps[index])
        plt.title('Life Expectancy vs Birthrate')
        plt.xlabel('Birthrate')
        plt.ylabel('Life Expectancy')
        plt.grid(True)

    # Printing the scatter plot
    plt.show()


# Defining a function to find and print the number of countries in the cluster and the names of the countries
def get_country(cluster, data):
    """
    This function prints out the list of countries in each cluster and the number of countries in each cluster
    :param cluster: the cluster of data
    :param data: the dataset
    :return: the list of countries and the number of countries
    """

    # Adding the countries to a list
    country_list = []
    for country, point, in data.items():
        if point in cluster:
            country_list.append(country)

    # Printing out the list of countries
    print(', '.join(country_list))

    # Printing out the number of countries
    print(f'Number of countries in the cluster: {len(country_list)}')


# Defining a function to find the average birthrate and life expectancy of each cluster
def mean_data(cluster):
    """
    This function calculates the average birthrate and average life expectancy of each cluster
    :param cluster: the cluster of data
    :return: the average life expectancy and average birthrate of that cluster
    """

    # Creating lists to store the birthrate and life expectancies
    x = []
    y = []
    for point in cluster:
        x.append(point[0])
        y.append(point[1])

    # Using Numpy's mean() function to calculate the averages
    average_birthrate = np.mean(x)
    average_life_expectancy = np.mean(y)

    # Printing out the results rounded to two decimal places
    print(f'The mean birthrate for the cluster is {np.round(average_birthrate, decimals=2)}')
    print(f'The mean life expectancy for the cluster is {np.round(average_life_expectancy, decimals=2)}')
    print('\n')


# Defining a function to run the final algorithm
def kmeans_algorithm(datapoints, number_of_clusters, data):
    """
    This function runs the actual k-means algorithm and brings all the previously defined functions together
    :param datapoints: the data points from the dataset
    :param number_of_clusters: the user-defined number of clusters
    :param data: the dataset
    :return: the scatter plot of data and information about each cluster
    """
    # Creating a list of colours to use for the different clusters and randomising the colour used for each cluster
    # using random.sample()
    cmaps = ["Blue", "Green", "Purple", "Pink", "Red", "Orange", "Brown", "Black", "Yellow", "Grey"]
    cmaps = random.sample(cmaps, len(cmaps))

    # Creating an object of the init_centroids function
    centroids = init_centroids(datapoints, number_of_clusters)

    # Calling the function to plot the scatter plot
    cluster_plot(centroids, cmaps)

    # Looping over the centroids to print the information about each cluster
    for index, cluster in enumerate(centroids):
        print(f'Country list for cluster {index + 1} (colour: {cmaps[index]}) :')
        get_country(cluster, data)
        mean_data(cluster)


# Defining a main function to run the program
def main():
    """
    This function asks the user which dataset they wish to analyse and how many clusters they wish to create and calls
    the kmeans_algorithm function
    :return: the output of the kmeans_algorithm() function
    """

    # Asking the user to enter the file name which they wish to analyse
    read = read_csv(input('''Please enter the name of the file containing the dataset you wish to use:
data1953.csv
data2008.csv
dataBoth.csv:\n '''))

    # Asking the user how many clusters of the data they wish to create
    number_cluster = int(input('\nPlease enter the number of clusters you wish to create: '))

    # Creating an object of the datapoints
    datapoints = [point for point in read.values()]

    # Calling the algorithm function
    kmeans_algorithm(datapoints, number_cluster, read)


# Calling the main() function
main()
