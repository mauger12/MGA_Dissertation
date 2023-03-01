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
    what patterns emerge
    what can be seen on graphs

"""
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt


def load_single_dataset(path):
    df = pd.read_csv(path)
    plt.plot(df['time'], df['value'])
    plt.show()
    return df['value'].values


def offline_cpd(signal):
    # generate signal
    # samples is num of data points, dim is dimensions - each column of power is a dim, sigma is deviation in signal
    # n_samples, dim, sigma = 1000, 1, 2
    # n_bkps = 4  # number of breakpoints
    # signal, bkps = rpt.pw_constant(n_samples, dim, noise_std=sigma)

    # detection
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = algo.predict(pen=3)
    # display
    rpt.display(signal, result)
    plt.show()


if __name__ == '__main__':
    path = 'data1.csv'  # 'QSR_dataset\\QSR3\\466e\\466e0001-20200220-20200712-60Min-analytics.csv'
    data = load_single_dataset(path)
    offline_cpd(data)
