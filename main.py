"""
load data set
    choose which data set
    graph data set

===================================================================
clean data set
    how to decide which points are anomolous
    what is defined as needing cleaning, what are we trying to analyse. what is needed for cpd
        unix time x and data points on y for each time
        remove rows with any empty values
    graph clean data

===================================================================
detect change points
    what method of cpd is being used?
        online or offline?
            online select cost, search and eval methods
            offline idk
    run algorithm on data
        how confident are we for each point
        does it look right
        gather data
        plot data with change point

===================================================================
analyse data generated
    compare time, location, variable, etc
    what patterns emerge hhh
    what can be seen on graphs

"""
import pandas as pd
import matplotlib.pyplot as plt


def load_single_dataset(path):
    df = pd.read_csv(path)
    plt.plot(df['ts'].loc[:100], df['E'].loc[:100])
    plt.show()


if __name__ == '__main__':
    path = 'QSR_dataset\\QSR3\\466e\\466e0001-20200220-20200712-60Min-analytics.csv'
    load_single_dataset(path)
