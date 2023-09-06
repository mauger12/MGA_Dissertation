import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import ruptures as rpt
import random
import warnings
from scipy.signal import find_peaks
from datetime import datetime
from functools import partial
from bayesian_changepoint_detection.generate_data import generate_normal_time_series
from bayesian_changepoint_detection.priors import const_prior
from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection, online_changepoint_detection
from bayesian_changepoint_detection.hazard_functions import constant_hazard
import bayesian_changepoint_detection.offline_likelihoods as offline_ll
import bayesian_changepoint_detection.online_likelihoods as online_ll


# focussed functions
def load_signal(path_, param):
    # path_ is file path, param is param of the data being loaded
    df = pd.read_csv(path_)
    df = df.fillna(0)
    # if param == 'Nile':
    #     loaded_timestamps = pd.to_datetime(df['time'], format="%Y").values
    # else:
    #     loaded_timestamps = np.arange(0, len(df[param].values))

    loaded_timestamps = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M')

    return loaded_timestamps, df[param].values


def graph_signal(timestamps=None, signal=None, label='', bkps=[],
                 path='C:\\Users\\matth\\Documents\\University\\Computing\\Year 3\\Diss\\Graphs\\Dump\\last.png'):
    # ensure all proper values are passed through first
    if signal is None:
        return 0
    if timestamps is None:
        timestamps = np.arange(0, len(signal))

    years = mdates.YearLocator(5)
    days = mdates.DayLocator(interval=14)
    # dfmt = mdates.DateFormatter('%Y')
    dfmt = mdates.DateFormatter('%d %b, %Y')

    fig, ax = plt.subplots(figsize=[24, 10])
    # if no timestamps given, graph signal and breakpoints in hours, else label graph with dates and hours
    if timestamps[0] == 0:
        ax.xaxis.set_major_locator(plt.MaxNLocator(20))
        ax.plot(signal)
        plt.xlabel('Time Step')
        for bkp in bkps:
            ax.axvline(x=bkp, color='red')
            ax.text(bkp - 1, ax.get_ylim()[1] + 200, str(bkp), fontweight='bold',
                    va='top')
    else:
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(dfmt)
        ax.plot(timestamps, signal)
        plt.xlabel('Date')
        for bkp in bkps:
            date = timestamps[bkp - 1].strftime("%d %b %H:%M")
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.axvline(x=timestamps[bkp - 1], color='red')
            ax.text(timestamps[bkp - 1], ax.get_ylim()[0], date, fontweight='bold',
                    va='top', rotation=-45)
    # plt.title(label)
    # plt.ylabel('Power of System')
    # plt.show()
    fig.savefig(path)
    plt.close()


def generate_pwc(num, num_bkps, mean_min=0, mean_max=50, dev=5):
    # have each segment constant length and fill lists with 0s as placeholders
    seglen = int(num / num_bkps)
    means = np.zeros(seglen)
    data = np.zeros(num)

    # set the mean of each segment
    for n in range(0, seglen):
        means[n] = random.uniform(mean_min, mean_max)

    # fill each segment with data with deviation around the mean
    j = 0
    for i in range(0, num):
        data[i] = means[j] + random.uniform(-dev, dev)
        if i % seglen == 0 and i != 0:
            j += 1

    # graph the function
    # fig, ax = plt.subplots(figsize=[24, 10])
    # ax.plot(data)
    # plt.show()

    return data


def generate_pwl(num, num_bkps, dev=10):
    # have each segment constant length and fill lists with 0s as placeholders
    seglen = int(num / num_bkps)
    data = np.zeros(num)

    # pick start values of first segment
    slope = random.uniform(-2, 2)
    data[0] = random.uniform(0, 50)

    # plot each time step based off slope and previous data point
    for i in range(1, num):
        if i % seglen == 0:
            slope = random.uniform(-2, 2)
            # data[i] = random.uniform(data[i-1]-30, data[i-1]+30)

        data[i] = data[i - 1] + slope

    # add deviation to datapoints
    for j in range(0, num):
        data[j] = data[j] + random.uniform(-dev, dev)

    # graph the function
    # fig, ax = plt.subplots(figsize=[24, 10])
    # ax.plot(data)
    # plt.show()

    return data


def generate_chngvar(num, num_bkps, mean=100, bound=100):
    # have each segment constant length, set variance bounds to 0 and fill data with 0s as placeholders
    seglen = int(num / num_bkps)
    var_min = 0
    var_max = 0
    data = np.zeros(num)

    # for each time step generate data point
    for i in range(0, num):
        # if at new segment, change variance
        if i % seglen == 0:
            var_min = random.uniform(-bound, 0)
            var_max = -var_min
        data[i] = mean + random.uniform(var_min, var_max)

    # graph the function
    # fig, ax = plt.subplots(figsize=[24, 10])
    # ax.plot(data)
    # plt.show()

    return data


def save_all_sigs():
    pwc_signal = generate_pwc(500, 6, dev=10)
    pwl_signal = generate_pwl(500, 6, dev=20)
    cv_signal = generate_chngvar(500, 6)
    Nile_timestamps, Nile_signal = load_signal('Nile.csv', 'Nile')

    dict = {'pwc': pwc_signal, 'pwl': pwl_signal, 'cv': cv_signal}
    df = pd.DataFrame(dict)
    df.to_csv('test_signals.csv')


# cpd prediction and evaluation functions
def pelt(data, cost, penalty):
    # change point detection using the pelt algorithm for the optimisation approach
    algo = rpt.Pelt(model=cost).fit(data)
    bkps = algo.predict(pen=penalty)
    # rpt.display(data, bkps)
    # plt.show()
    return bkps


def offline_bayesian(timestamps, data, label, truncate=-10, cutoff=0.15, graph=False,
                     path='C:\\Users\\matth\\Documents\\University\\Computing\\Year 3\\Diss\\Graphs\\Dump\\'):
    # Calculate the probabilities of each point being a true chnagepoint
    prior_function = partial(const_prior, p=1 / (len(data) + 1))
    Q, P, Pcp = offline_changepoint_detection(data, prior_function, offline_ll.StudentT(), truncate=truncate)

    # sum the probabilities of each individual point to create a graph for the whole dataset
    pcp_sum = np.exp(Pcp).sum(0)
    # find the peaks in probabilities of each point being a changepoint
    peaks = find_peaks(pcp_sum, cutoff, distance=120)  # distance is 5days*24 hrs
    peak_pos = timestamps[peaks[0]]

    # graph findings
    if graph:
        # set the time formats for the graphs
        years = mdates.YearLocator(5)
        days = mdates.DayLocator(interval=14)
        # dfmt = mdates.DateFormatter('%Y')
        dfmt = mdates.DateFormatter('%d %b, %Y')

        # plot the data
        fig, ax = plt.subplots(2, figsize=[18, 16])
        ax[0].set_title(label)
        ax[0].xaxis.set_major_locator(years)
        ax[0].xaxis.set_major_formatter(plt.NullFormatter())
        ax[0].plot(timestamps, data[:])
        # ax[0].set_ylabel("Flow [10^8 m^3]")

        # plot the detected changepoints
        for bkp in peak_pos:
            ax[0].axvline(x=bkp, color='red')
            ax[0].text(bkp, ax[0].get_ylim()[0] - 1, bkp.strftime("%d %b %H:%M"),
                       va='top', ha='center', rotation=-45)

        # plot the probabilities on a separate axis
        ax[1].set_title("Probability of a changepoint")
        ax[1].set_ylabel("Probability")
        ax[1].set_xlabel("Timestep")
        ax[1].xaxis.set_major_locator(days)
        ax[1].xaxis.set_major_formatter(dfmt)
        ax[1].plot(timestamps[0:len(timestamps) - 1], pcp_sum)
        plt.show()

        # save plot to file
        fig.savefig(path)
        plt.close()

    return peaks[0]


def online_bayesian(data, timestamps, label, a=0.1, b=0.01, k=1.0, m=0.0, hazard=250, cutoff=0.1, graph=False,
                    path='C:\\Users\\matth\\Documents\\University\\Computing\\Year 3\\Diss\\Graphs\\Dump\\'):
    # define hazard function
    hazard_function = partial(constant_hazard, hazard)

    # Calculate R matrix for all run lengths and timesteps
    R, maxes = online_changepoint_detection(
        data, hazard_function, online_ll.StudentT(alpha=a, beta=b, kappa=k, mu=m)
    )

    # alpha - alpha in gamma distribution prior
    # beta - beta in gamma distribution prior
    # mu - mean from normal distribution
    # kappa - variance from normal distribution

    # define vars and calculate density matrix
    Nw = 10
    epsilon = 1e-7
    sparsity = 1  # only plot every nth data for faster display
    density_matrix = -np.log(R[0:-1:sparsity, 0:-1:sparsity] + epsilon)

    # find the peaks of probabilities of a CP
    peaks = find_peaks(R[Nw, Nw:-1], cutoff, distance=10)

    # each peak is +1 time unit its acc position due to lists starting at index 0
    peak_pos = timestamps[peaks[0]]
    for i in range(0, len(peaks[0])):
        peak_pos[i] = timestamps[peaks[0][i] - 1]

    # generate graphs
    if graph:
        # Plot initial data
        fig, ax = plt.subplots(3, figsize=[18, 16])
        ax[0].set_title(label)
        ax[0].plot(timestamps, data)
        ax[0].set_xlabel("Date")

        # plot lines at the detected CP
        for bkp in peak_pos:
            ax[0].axvline(x=bkp, color='red')
            ax[0].text(bkp, ax[0].get_ylim()[0] - 2, bkp.astype(str)[0:4],
                       va='top', ha='center', )

        # plot colour map for probabilities of each run length
        ax[1].pcolor(np.array(range(0, len(R[:, 0]), sparsity)),
                     np.array(range(0, len(R[:, 0]), sparsity)),
                     density_matrix,
                     cmap=cm.Greys, vmin=0, vmax=density_matrix.max(),
                     shading='auto')
        ax[1].set_ylabel("Run Length")

        # plot Probability of a point being a change point
        ax[2].plot(R[Nw, Nw:-1])
        ax[2].set_ylabel("Probability of Change Point")
        ax[2].set_xlabel("Timestep")

        # save plot to file
        final_path = path + label[-26:-1] + '.png'
        plt.savefig(final_path)
        # plt.show()
        plt.close()

    return peaks[0]


def evaluate(estimated, true, metric, epsilon):
    if metric == 'PR-R' or metric == 'F1':
        # find all estimated CP within epsilon of true CP
        close_est = np.zeros(len(estimated))
        for i in range(0, len(estimated)):
            for j in range(0, len(true)):
                if abs(estimated[i] - true[j]) <= epsilon:
                    close_est[i] = estimated[i]
        close_est = close_est[close_est != 0]

        # calculate precision, recall and f1
        if len(close_est) > 0:
            pr = len(close_est) / len(estimated)
            re = len(close_est) / len(true)
            F1 = 2 * ((pr * re) / (pr + re))
        # if none found return -1
        else:
            if metric == 'PR-R':
                return -1, -1
            else:
                return -1

    match metric:
        # return the Annotation error
        case 'AE':
            return abs(len(estimated) - len(true))
        # return the Precision and recall
        case 'PR-R':
            return pr, re
        # return F1 score
        case 'F1':
            return F1
        case 'MT':
            mt = 0
            # get sum of distances between estimated cp and their closest true cp
            for est in estimated:
                # set smallest dist as some large value so it can narrow down from there
                smallest_dist = 50000
                # find the closest true cp to estimated cp
                for cp in true:
                    dist = abs(est - cp)
                    if dist <= smallest_dist:
                        smallest_dist = dist
                mt += smallest_dist
            # calc final meantime
            if len(estimated) == 0:
                return -1
            else:
                return mt / len(estimated)

    return -1


# parameter testing functions for test data
def test_optimise_pelt(generate_mets=False, graph=False, epsilon=1):
    # define the different lists for the test data
    path = 'C:\\Users\\matth\\Documents\\University\\Computing\\Year 3\\Diss\\Graphs\\Test\\PELT\\'
    costs = ['l1', 'rbf', 'rank']
    metrics = ['AE', 'PR-R', 'F1', 'MT']
    Nile_cp = [28, 100]
    test_cp = [83, 166, 249, 332, 415, 498]
    true_cp = []

    # loop through each data set
    for sig_type in ['Nile']:  # , 'pwc', 'pwl', 'cv'
        if sig_type == 'Nile':
            timestamps, signal = load_signal('Nile.csv', sig_type)
        else:
            timestamps, signal = load_signal('test_signals.csv', sig_type)

        # if specified, generate the metrics for the data
        if generate_mets:
            # define dataframe
            met_vals = pd.DataFrame(columns=['Cost', 'Penalty', 'AE', 'Prec', 'Rec', 'F1', 'MT'])
            i = 0

            # loop through each cost and penalty for pelt algorithm
            for cost in costs:
                for pen in range(1, 100):
                    bkps = pelt(signal, cost, pen)
                    if sig_type == 'Nile':
                        true_cp = Nile_cp
                    else:
                        true_cp = test_cp

                    # calculate metrics for calculated bkps
                    AE = evaluate(bkps, true_cp, 'AE', epsilon)
                    PR, R = evaluate(bkps, true_cp, 'PR-R', epsilon)
                    F1 = evaluate(bkps, true_cp, 'F1', epsilon)
                    MT = evaluate(bkps, true_cp, 'MT', epsilon)
                    # add metrics to dataframe
                    met_vals.loc[i] = [cost, pen, AE, PR, R, F1, MT]
                    i += 1

                    if i % (120) == 0:
                        print('#', end='')
            # save dataframe to csv
            met_vals.to_csv(sig_type + '_PELT_metrics.csv')

        # graph bkps
        if graph:
            # load the datafraame from csv and get only desired rows
            vals = pd.read_csv(sig_type + '_PELT_metrics.csv')
            best_vals = vals.query("Cost == 'l1'").query('F1 == F1.max()')
            best_vals = pd.concat([best_vals, vals.query("Cost == 'rbf'").query('F1 == F1.max()')])
            best_vals = pd.concat([best_vals, vals.query("Cost == 'rank'").query('F1 == F1.max()')])
            best_vals = best_vals.query("AE <= 2").query("MT = MT.min()")
            best_vals = best_vals.reset_index()
            # loop through each row of df and generate the corresponding graph and save it to path
            for index, row in best_vals.iterrows():
                cost = row['Cost']
                pen = row['Penalty']
                bkps = pelt(signal, cost, pen)

                label = 'PELT Analysis of ' + sig_type + ' Data using ' + cost + ' ' + str(pen)
                graph_signal(timestamps, signal, label, bkps=bkps, path=path + label + '.png')

        print('+', end='')


def test_optimise_offline(generate_mets=False, graph=False, epsilon=1):
    warnings.filterwarnings("ignore")
    # define the different lists for the test dat   a
    path = 'C:\\Users\\matth\\Documents\\University\\Computing\\Year 3\\Diss\\Graphs\\Test\\Offline_Bayesian\\'
    metrics = ['AE', 'PR-R', 'F1']
    Nile_cp = [28]
    test_cp = [83, 166, 249, 332, 415, 498]
    true_cp = []
    cutoff = 0.15

    print('[', end='')
    # loop through each data set
    for sig_type in ['Nile']:
        # load the data in
        if sig_type == 'Nile':
            timestamps, signal = load_signal('Nile.csv', sig_type)
        else:
            timestamps, signal = load_signal('test_signals.csv', sig_type)

        # if specified, generate the metrics for the data
        if generate_mets:
            # define dataframe
            met_vals = pd.DataFrame(columns=['Truncation', 'Cutoff', 'AE', 'Prec', 'Rec', 'F1', 'MT'])
            i = 0

            # loop through each cost and penalty for pelt algorithm
            for cutoff in np.arange(0.05, 0.15, 0.025):
                for trunc in range(1, 51):
                    label = 'Bayesian Offline Analysis of ' + sig_type + ' Data truncated at ' + str(-trunc)
                    bkps = offline_bayesian(timestamps, signal, label, -trunc, cutoff, False, path)
                    if sig_type == 'Nile':
                        true_cp = Nile_cp
                    else:
                        true_cp = test_cp

                    # calculate metrics for calculated bkps
                    AE = evaluate(bkps, true_cp, 'AE', epsilon)
                    PR, R = evaluate(bkps, true_cp, 'PR-R', epsilon)
                    F1 = evaluate(bkps, true_cp, 'F1', epsilon)
                    MT = evaluate(bkps, true_cp, 'MT', epsilon)
                    # add metrics to dataframe
                    met_vals.loc[i] = [trunc, cutoff, AE, PR, R, F1, MT]
                    i += 1
                    if i % 20 == 0:
                        print('#', end='')
                # save dataframe to csv
            met_vals.to_csv(sig_type + '_Offline_metrics.csv')

        # graph bkps
        if graph:
            # load the datafraame from csv and get only desired rows
            vals = pd.read_csv(sig_type + '_Offline_metrics.csv')
            # best_vals = vals.query('F1 == F1.max()').query("AE <= AE.min() + 2 & AE !=0")
            # best_vals = best_vals.reset_index()
            # loop through each row of df and generate the corresponding graph and save it to path
            for index, row in vals.iterrows():
                trunc = row['Truncation']
                cutoff = row['Cutoff']
                label = 'Bayesian Offline Analysis of ' + sig_type + ' Data truncated at ' + str(
                    trunc)
                bkps = offline_bayesian(timestamps, signal, label, -trunc, cutoff, True, path)

        print('+', end='')
    print(']')


def test_optimise_online(generate_mets=False, graph=False, epsilon=1):
    # define the different lists for the test dat   a
    path = 'C:\\Users\\matth\\Documents\\University\\Computing\\Year 3\\Diss\\Graphs\\Test\\Online_Bayesian\\'
    metrics = ['AE', 'PR-R', 'F1', 'MT']
    Nile_cp = [28]
    test_cp = [83, 166, 249, 332, 415, 498]
    true_cp = []

    print('[', end='')
    # loop through each data set
    for sig_type in ['Nile']:
        # load the data in
        if sig_type == 'Nile':
            timestamps, signal = load_signal('Nile.csv', sig_type)
        else:
            timestamps, signal = load_signal('test_signals.csv', sig_type)

        # if specified, generate the metrics for the data
        if generate_mets:
            # define dataframe
            met_vals = pd.DataFrame(columns=['a', 'b', 'k', 'm', 'hazard', 'cutoff', 'AE', 'Prec', 'Rec', 'F1', 'MT'])

            # define vars to perform random search for parameters
            i = 0
            iterations = 5000

            # loop through each variable to find the best combo for online detection
            """for a in np.arange(0, 0.5, 0.05):  # 10
                for b in np.arange(0, 10, 0.01):  # 20
                    for k in np.arange(0, 1, 0.01):  # 100
                        for m in np.arange(0, 1, 0.1):  # 10 total 200k
                            label = 'Bayesian Online Analysis of ' + sig_type + ' Data with a,b,k,m values:' + str(a) + ',' + str(b) + ',' + str(k) + ',' + str(m)
                            bkps = online_bayesian(signal, timestamps, label, a, b, k, m, cutoff, False, path)

                            if sig_type == 'Nile':
                                true_cp = Nile_cp
                            else:
                                true_cp = test_cp

                            # calculate metrics for calculated bkps
                            AE = evaluate(bkps, true_cp, 'AE', epsilon)
                            PR, R = evaluate(bkps, true_cp, 'PR-R', epsilon)
                            F1 = evaluate(bkps, true_cp, 'F1', epsilon)
                            # add metrics to dataframe
                            met_vals.loc[i] = [a, b, k, m, cutoff, AE, PR, R, F1]
                            i += 1
                            if i % 4000 == 0:
                                print('#', end='')"""

            # random search for aproximtely the best combo over a range of cutoffs for online detection
            for cutoff in [0.2]:
                search = True
                while search:
                    # sample 50% either side of last best value to try get better values
                    a = round(random.uniform(0, 2), 2)
                    b = round(random.uniform(0, 10), 2)
                    k = round(random.uniform(0, 2), 2)
                    m = round(random.uniform(0, 2), 2)
                    hazard = int(random.uniform(1, 250))

                    label = 'Bayesian Online Analysis of ' + sig_type + ' Data with a,b,k,m values:' + str(
                        a) + ', ' + str(b) + ', ' + str(k) + ', ' + str(m) + '.'
                    bkps = online_bayesian(signal, timestamps, label, a, b, k, m, hazard, cutoff, False, path)

                    if sig_type == 'Nile':
                        true_cp = Nile_cp
                    else:
                        true_cp = test_cp

                    # calculate metrics for calculated bkps
                    AE = evaluate(bkps, true_cp, 'AE', epsilon)
                    PR, R = evaluate(bkps, true_cp, 'PR-R', epsilon)
                    F1 = evaluate(bkps, true_cp, 'F1', epsilon)
                    M1 = evaluate(bkps, true_cp, 'M1', epsilon)

                    # add metrics to dataframe
                    met_vals.loc[i] = [a, b, k, m, hazard, cutoff, AE, PR, R, F1, M1]
                    i += 1

                    if i % (iterations / 10) == 0 or F1 == 1:
                        print('#', end='')
                    if i > iterations:
                        search = False
                        i = 0
            # save dataframe to csv
            met_vals.to_csv(sig_type + '_Online_metrics.csv')

        # graph bkps
        if graph:
            # load the datafraame from csv and get only desired rows
            vals = pd.read_csv(sig_type + '_Online_metrics.csv')
            best_vals = vals.query('F1 == F1.max()').query("AE <= AE.min() & AE !=-1")
            best_vals = best_vals.reset_index()
            # loop through each row of df and generate the corresponding graph and save it to path
            for index, row in best_vals.iterrows():
                a = row['a']
                b = row['b']
                k = row['k']
                m = row['m']
                cutoff = row['cutoff']
                hazard = row['hazard']
                label = 'Bayesian Online Analysis of ' + sig_type + ' Data with a,b,k,m and hazard values:' + str(
                    a) + ', ' + str(
                    b) + ', ' + str(k) + ', ' + str(m) + ', ' + str(hazard)
                bkps = online_bayesian(signal, timestamps, label, a, b, k, m, hazard, cutoff, True, path)

        print('+', end='')
    print(']')


def get_all_bkps(sig_type, label="", graph=True,
                 path='C:\\Users\\matth\\Documents\\University\\Computing\\Year 3\\Diss\\Graphs\\Dump\\'):
    # lod in the signal
    true_cp = []
    if sig_type == 'Nile':
        timestamps, signal = load_signal('Nile.csv', sig_type)
        true_cp = [28, 100]
    else:
        true_cp = [83, 166, 249, 332, 415, 498]
        timestamps, signal = load_signal('test_signals.csv', sig_type)

    # load in the metrics and filter for best params
    opt = pd.read_csv(sig_type + '_PELT_metrics.csv')
    l1 = opt.query('Cost == "l1" & AE !=-1').query('AE == AE.min()').query(
        'F1 == F1.max() & MT == MT.min()').sort_values(by=['F1']).reset_index()
    rbf = opt.query('Cost == "rbf" & AE !=-1').query('AE == AE.min()').query(
        'F1 == F1.max() & MT == MT.min()').sort_values(by=['F1']).reset_index()
    rank = opt.query('Cost == "rank" & AE !=-1').query('AE == AE.min()').query(
        'F1 == F1.max() & MT == MT.min()').sort_values(by=['F1']).reset_index()
    off = pd.read_csv(sig_type + '_Offline_metrics.csv').query('AE != -1').query('AE == AE.min()').query(
        'F1 == F1.max() & MT == MT.min()').sort_values(
        by=['F1']).reset_index()
    on = pd.read_csv(sig_type + '_Online_metrics.csv').query('AE != -1').query('AE == AE.min()').query(
        'F1 == F1.max() & MT == MT.min()').sort_values(
        by=['F1']).reset_index()

    # get the bkps of all the methods
    all_bkps = []
    all_mets = []

    """l1_cp = l1.iloc[-1][['Cost', 'Penalty']].values
    l1_bkp = pelt(signal, l1_cp[0], l1_cp[1])
    all_mets.append(l1_cp)
    all_bkps.append(l1_bkp)

    rbf_cp = rbf.iloc[-1][['Cost', 'Penalty']].values
    rbf_bkp = pelt(signal, rbf_cp[0], rbf_cp[1])
    all_mets.append(rbf_cp)
    all_bkps.append(rbf_bkp)

    rank_cp = rank.iloc[-1][['Cost', 'Penalty']].values
    rank_bkp = pelt(signal, rank_cp[0], rank_cp[1])
    all_mets.append(rank_cp)
    all_bkps.append(rank_bkp)
    """
    off_tc = off.iloc[-1][['Truncation', 'Cutoff', 'AE', 'Prec', 'Rec', 'F1', 'MT']].values
    off_bkp = offline_bayesian(timestamps, signal, "off", -off_tc[0], off_tc[1], False)
    all_mets.append(off_tc)
    all_bkps.append(off_bkp)
    """
    on_ps = on.iloc[-1][['a', 'b', 'k', 'm', 'hazard', 'cutoff', 'AE', 'Prec', 'Rec', 'F1', 'MT']].values
    on_bkp = online_bayesian(signal, timestamps, "on", on_ps[0], on_ps[1], on_ps[2], on_ps[3], on_ps[4], on_ps[5],
                             False)

    all_mets.append(on_ps)
    all_bkps.append(on_bkp)
    """
    # print(all_bkps)

    # graph the signals with the results scattered on
    if graph:
        colours = ['red', 'orange', 'yellow', 'lime', 'violet']
        labels = ['l1', 'rbf', 'rank', 'Bayesian off', 'Bayesian on']
        fig, ax = plt.subplots(figsize=[24, 10])
        h_inc = 10
        h = 0
        c = 0
        ax.xaxis.set_major_locator((plt.MaxNLocator(20)))
        ax.plot(timestamps, signal)
        plt.xlabel('Time Step')
        # label true bkps on graph
        for cp in true_cp:
            ax.axvline(x=timestamps[cp - 1], color='black')
            ax.text(timestamps[cp - 1], 290, str(timestamps[cp - 1])[0:4], va='top', ha='center', weight='bold')

        # plot found bkps on graph
        for cps in all_bkps:
            leglabel = True
            for cp in cps:
                if leglabel:
                    ax.scatter(timestamps[cp - 1], h, s=50, color=colours[c], zorder=2, label=labels[c])
                    leglabel = False
                else:
                    ax.scatter(timestamps[cp - 1], h, s=50, color=colours[c], zorder=2)
            c += 1
            h += h_inc

        ax.legend(loc='upper left')

        # save plot to file
        final_path = path + label + '.png'
        plt.savefig(final_path)
        plt.show()
        plt.close()

    return all_mets, all_bkps


# main prediction function to get results for a QSR
def get_batch_bkps(QSR, QSR_paths, param, method, pen=5, trunc=-10, cutoff=0.2, a=0, b=0, k=0, m=0, h=100, graph=False):
    # param is either P or PF
    # method is rbf, rank, off or on

    out_path = 'C:\\Users\\matth\\Documents\\University\\Computing\\Year 3\\Diss\\Graphs\\QSR' + str(QSR) + '\\' + param

    # Array of arrays containing timestamps and signals for each channel in qsr
    QSR_timestamps = []
    QSR_signals = []

    # get the inividual series for each channel in qsr
    for path in QSR_paths[QSR - 1]:
        timestamps, signal = load_signal(path, param)
        QSR_timestamps.append(timestamps)
        QSR_signals.append(signal)
    all_series = [QSR_timestamps, QSR_signals]

    # find all bkps for all channels with specified method and put into list
    bkps = []

    print('[', end='')
    # PELT
    if method == 'rbf' or method == 'rank':
        label = 'PELT with cost ' + method + ' and penalty ' + str(pen)
        for s in range(0, len(all_series[1])):
            path = out_path + '\\PELT\\' + str(s + 1) + '. ' + label + '.png'
            bkps.append(pelt(all_series[1][s], method, pen))
            if graph:
                graph_signal(all_series[0][s], all_series[1][s], str(s) + '. ' + label, bkps[s], path)
            print('#', end='')
    # Bayesian offline
    elif method == 'off':
        label = 'OffBCD with truncation -600'
        for s in range(0, len(all_series[1])):
            path = out_path + '\\Offline_Bayesian\\' + str(s + 1) + label + '.png'
            bkps.append(offline_bayesian(all_series[0][s], all_series[1][s], '', trunc, cutoff, graph, path))
            print('#', end='')
    # Bayesian online
    elif method == 'on':
        label = 'OnBCD with params: ' + str(a) + ', ' + str(b) + ', ' + str(k) + ', ' + str(m) + ', ' + str(
            h) + ', ' + str(cutoff)
        for s in range(0, len(all_series[1])):
            path = out_path + '\\Online_Bayesian\\' + str(s + 1) + '. ' + label + '.png'
            bkps.append(online_bayesian(all_series[1][s], all_series[0][s], label, a, b, k, m, h, cutoff, graph, path))
            print('#', end='')
    else:
        print('incorrect params specified', end='')
    print(']')

    # each bkp is returned as its index, convert into dates
    bkp_dates = []
    for nbkps in range(0, len(bkps)):
        ts = all_series[0][nbkps]  # timestamps of nth series
        ndates = []
        for bkp in bkps[nbkps]:  # each bkp in nth series
            ndates.append(ts[bkp - 1].strftime("%d %b %H:%M"))  # get date of each bkp in series
        bkp_dates.append(ndates)

    # return list of list of bkps for each signal
    return bkp_dates


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # arrays of file paths of each channel for each QSR
    # where a whitespace is left separates the rooms in the restaurant
    QSR1_paths = ['QSR_dataset\\QSR1\\0ccd\\0ccd0001-20191001-20200531-60Min-analytics.csv',
                  'QSR_dataset\\QSR1\\0ccd\\0ccd0002-20191001-20200531-60Min-analytics.csv',
                  'QSR_dataset\\QSR1\\0ccd\\0ccd0003-20191001-20200531-60Min-analytics-1.csv',
                  'QSR_dataset\\QSR1\\0ccd\\0ccd0003-20191001-20200531-60Min-analytics-2.csv',
                  'QSR_dataset\\QSR1\\0ccd\\0ccd0003-20191001-20200531-60Min-analytics-3.csv',
                  'QSR_dataset\\QSR1\\0ccd\\0ccd0004-20191001-20200531-60Min-analytics.csv',
                  'QSR_dataset\\QSR1\\0ccd\\0ccd0005-20191001-20200531-60Min-analytics.csv',
                  'QSR_dataset\\QSR1\\0ccd\\0ccd0006-20191001-20200531-60Min-analytics.csv',
                  'QSR_dataset\\QSR1\\0ccd\\0ccd0007-20191001-20200531-60Min-analytics.csv',
                  'QSR_dataset\\QSR1\\0ccd\\0ccd0008-20191001-20200531-60Min-analytics-1.csv',
                  'QSR_dataset\\QSR1\\0ccd\\0ccd0008-20191001-20200531-60Min-analytics-2.csv',
                  'QSR_dataset\\QSR1\\0ccd\\0ccd0008-20191001-20200531-60Min-analytics-3.csv',

                  'QSR_dataset\\QSR1\\e00f\\e00f0001-20191001-20200531-60Min-analytics.csv',
                  'QSR_dataset\\QSR1\\e00f\\e00f0002-20191001-20200531-60Min-analytics.csv',
                  'QSR_dataset\\QSR1\\e00f\\e00f0003-20191001-20200531-60Min-analytics.csv',
                  'QSR_dataset\\QSR1\\e00f\\e00f0004-20191001-20200531-60Min-analytics-1.csv',
                  'QSR_dataset\\QSR1\\e00f\\e00f0004-20191001-20200531-60Min-analytics-2.csv',
                  'QSR_dataset\\QSR1\\e00f\\e00f0004-20191001-20200531-60Min-analytics-3.csv',
                  'QSR_dataset\\QSR1\\e00f\\e00f0005-20191001-20200531-60Min-analytics-1.csv',
                  'QSR_dataset\\QSR1\\e00f\\e00f0005-20191001-20200531-60Min-analytics-2.csv',
                  'QSR_dataset\\QSR1\\e00f\\e00f0005-20191001-20200531-60Min-analytics-3.csv',
                  'QSR_dataset\\QSR1\\e00f\\e00f0006-20191001-20200531-60Min-analytics-1.csv',
                  'QSR_dataset\\QSR1\\e00f\\e00f0006-20191001-20200531-60Min-analytics-2.csv',
                  'QSR_dataset\\QSR1\\e00f\\e00f0006-20191001-20200531-60Min-analytics-3.csv',
                  'QSR_dataset\\QSR1\\e00f\\e00f0007-20191001-20200531-60Min-analytics-1.csv',
                  'QSR_dataset\\QSR1\\e00f\\e00f0007-20191001-20200531-60Min-analytics-2.csv',
                  'QSR_dataset\\QSR1\\e00f\\e00f0007-20191001-20200531-60Min-analytics-3.csv',
                  'QSR_dataset\\QSR1\\e00f\\e00f0008-20191001-20200531-60Min-analytics.csv']
    QSR2_paths = ['QSR_dataset\\QSR2\\8d90\\8d900001-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR2\\8d90\\8d900002-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR2\\8d90\\8d900003-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR2\\8d90\\8d900004-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR2\\8d90\\8d900005-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR2\\8d90\\8d900006-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR2\\8d90\\8d900007-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR2\\8d90\\8d900008-20200220-20200712-60Min-analytics.csv',

                  'QSR_dataset\\QSR2\\45ec\\45ec0001-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR2\\45ec\\45ec0002-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR2\\45ec\\45ec0003-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR2\\45ec\\45ec0004-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR2\\45ec\\45ec0005-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR2\\45ec\\45ec0006-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR2\\45ec\\45ec0007-20200220-20200712-60Min-analytics.csv']
    QSR3_paths = ['QSR_dataset\\QSR3\\466e\\466e0001-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR3\\466e\\466e0002-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR3\\466e\\466e0003-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR3\\466e\\466e0004-20200220-20200712-60Min-analytics.csv',

                  'QSR_dataset\\QSR3\\7470\\74700001-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR3\\7470\\74700002-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR3\\7470\\74700003-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR3\\7470\\74700004-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR3\\7470\\74700005-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR3\\7470\\74700006-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR3\\7470\\74700007-20200220-20200712-60Min-analytics.csv',
                  'QSR_dataset\\QSR3\\7470\\74700008-20200220-20200712-60Min-analytics.csv']
    QSR_paths = [QSR1_paths, QSR2_paths, QSR3_paths]

    # load in a specific data set
    # path = 'test_signals.csv'
    # timestamps, signal = load_signal(QSR3_paths[0], 'P')
    # graph_signal(timestamps=timestamps, signal=signal, label="Yearly Flow of Nile Data")

    # generate synthetic signals
    # signal = generate_pwc(500, 6, dev=10)
    # signal = generate_pwl(500, 6, dev=20)
    # signal = generate_chngvar(500, 6)

    # get specific signal from qsr

    # signal = QSR_signals[0]
    # timestamps = QSR_timestamps[0]
    # bkps = []

    # find range of optimal params for each method
    # test_optimise_pelt(True, False, 1)
    # test_optimise_offline(True, False, 1)
    # test_optimise_online(True, False, 1)

    # perform pelt on each signal and create the graph displaying each CP
    # n = 0
    # for signal in QSR_signals:
    #     bkps = pelt(signal, "rbf", 10)
    #     # offline_bayesian(signal)
    #     graph_signal(timestamps=QSR_timestamps[n], signal=signal, label='Pelt Analysis of Power Factor using rbf cost on Channel ' + str(n+1),
    #                  bkps=bkps)
    #     n += 1
    #     print(n)

    # perform different cpd methods with specified parameters
    # bkps = pelt(signal, "l1", 80)
    # print(bkps)
    # online_bayesian(signal, timestamps, "Online Bayesian Detection of Nile Flow Data")

    # label = 'Pelt Analysis of Synthetic Changing Variance Data using rank 2'
    # graph_signal(signal=signal, label=label, bkps=bkps, path='C:\\Users\\matth\\Documents\\University\\Computing\\Year 3\\Diss\\Graphs\\Test\\raw\\' + label + '.png')

    # print(get_all_bkps(sig, 'all cp for ' + sig, False)[0])
    # print(get_all_bkps(sig, 'all cp for ' + sig, False)[1])

    """rbf_p_bkps = get_batch_bkps(1, QSR_paths, 'P', 'rbf', pen=8, graph=True)
    rank_p_bkps = get_batch_bkps(1, QSR_paths, 'P', 'rank', pen=15, graph=True)
    QSR1_P_bkps = pd.DataFrame(columns=['rbf', 'rank'])
    QSR1_P_bkps['rbf'] = rbf_p_bkps
    QSR1_P_bkps['rank'] = rank_p_bkps
    QSR1_P_bkps.to_csv('1P_offline_bkps.csv')"""

    off_pf_bkps = get_batch_bkps(1, QSR_paths, 'PF', 'off', trunc=-600, cutoff=0.15, graph=True)
    # print(off_pf_bkps)
    # offline_bayesian(timestamps, signal, "help", -600, 0.5, True)

    """rbf_pf_bkps = get_batch_bkps(1, QSR_paths, 'PF', 'rbf', pen=8, graph=True)
    rank_pf_bkps = get_batch_bkps(1, QSR_paths, 'PF', 'rank', pen=15, graph=True)
    QSR1_PF_bkps = pd.DataFrame(columns=['rbf', 'rank'])
    QSR1_PF_bkps['rbf'] = rbf_pf_bkps
    QSR1_PF_bkps['rank'] = rank_pf_bkps
    QSR1_PF_bkps.to_csv('1PF_offline_bkps.csv')"""
    # QSR3_PF_bkps['offline'] = off_pf_bkps

    """
    P = pd.read_csv('P_offline_bkps.csv')
    PF = pd.read_csv('PF_offline_bkps.csv')

    for channel in range(0, 4):
        rbf_bkps = P['rbf']
        rank_bkps = P['rank']
        a = rbf_bkps[channel][2:-2].split("', '")
        b = rank_bkps[channel][2:-2].split("', '")
        for i in range(0, len(a)):
            a[i] = a[i][0:6]
        for i in range(0, len(b)):
            b[i] = b[i][0:6]
        print(channel + 1, a)
        print(channel + 1, b)
        print(channel + 1, set(a).intersection(set(b)))
        print()
        """
