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


def load_signal(path_):
    df = pd.read_csv(path_)
    # plt.plot(df['time'], df['Nile'])
    # plt.show()
    return df['Nile'].values


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


def generate_signal(shift):
    # creation of data
    n, dim = 500, 1  # number of samples, dimension
    n_bkps, sigma = 3, 1  # number of change points, noise standard deviation

    match shift:
        case 'mean':
            signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma, delta=(1, 10))
        case 'correlation':
            signal, bkps = rpt.pw_normal(n, n_bkps)
        case 'linear':
            signal, bkps = rpt.pw_linear(n, dim, n_bkps, noise_std=sigma)
        case 'frequency':
            signal, bkps = rpt.pw_wavy(n, n_bkps, noise_std=sigma)
        case _:
            print('no shift specified')
            quit()

    rpt.display(signal, bkps)
    plt.show()
    return signal, bkps


def pelt(signal, cost):
    # change point detection
    model = "rbf"  # "l2", "rbf"
    algo = rpt.Pelt(model=model, min_size=3, jump=5).fit(signal)
    my_bkps = algo.predict(pen=1)
    return my_bkps


def binary_segmentation(signal):
    return 0


def window_based_detection(signal):
    return 0


if __name__ == '__main__':
    path = 'data1.csv'  # 'QSR_dataset\\QSR3\\466e\\466e0001-20200220-20200712-60Min-analytics.csv'
    data, true_bkps = load_signal('nile.csv'), []
    # offline_cpd(data)
    # data, true_bkps = generate_signal('mean')
    bkps = pelt(data, 'l1')

    # show results
    rpt.display(data, true_bkps, bkps)
    plt.show()
