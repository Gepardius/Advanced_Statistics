import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson
import scipy.integrate as spi


# 1: BASIC PROBABILITIES AND VISUALIZATIONS (1)
# a) Bernoulli distribution
def bernoulli(p):
    r = 1 - p
    p_l = {"0": r, "1": p}  # Success: vote = "for" = 1 , Failure: vote = "against" = 0
    votes = list(p_l.keys())  # get key values from dictionary
    values = list(p_l.values())  # get values from dictionary
    v = values  # save values for a pie chart
    fig, ax = plt.subplots(figsize=(7, 5))  # bar length, height
    ax.bar(votes, values, 1, edgecolor="gray", linewidth=3)
    plt.xlabel("Vote")
    plt.ylabel("Probability")
    for values in ax.containers:  # show values above bar
        ax.bar_label(values)
    plt.show()

    plt.clf()  # clear previous plot
    plt.pie(v, labels=votes, autopct="%1.1f%%")  # plot a pie chart with percentages
    plt.show()

    expected_value = p
    print(f"Expected value = {expected_value}")


bernoulli(0.03)


# b) Poisson distribution
def poisson_calculation(mu_poisson):
    poisson_probability = []    # list to store probabilities
    x_l = []    # list to store number of meteorites
    for i in range(1000):
        y = poisson.pmf(k=i, mu=mu_poisson)  # mu = 𝜆/expectancy , k = probability that it will happen
        if y > 0.005:
            poisson_probability.append(y)
            x_l.append(i)

    # calculate poisson variance, mean and median
    variance_poisson = poisson.var(mu=mu_poisson, loc=0)
    mean_poisson = poisson.mean(mu=mu_poisson, loc=0)
    median_poisson = poisson.median(mu=mu_poisson, loc=0)

    plt.clf()

    # plot poisson distribution with line plot
    plt.plot(x_l, poisson_probability, color="Green", label="Poisson distribution")
    plt.xlabel("N. of meteorites falling")
    plt.ylabel("Probability")

    # plot medan and variance with scatter plot
    plt.scatter(median_poisson, max(poisson_probability), color="red", label="Median", s=50)
    plt.scatter(variance_poisson, max(poisson_probability), color="orange", label="Variance", marker="x", s=100)

    plt.legend(loc=3)
    plt.show()

    print(f"Median: {median_poisson}")
    print(f"Variance: {variance_poisson}")
    print(f"Mean: {mean_poisson}")

    # table of Probability and N. of meteorites
    meteorites_table = pd.DataFrame(list(zip(poisson_probability, x_l)), columns=["Probability", "N. of meteorites"])
    print(meteorites_table)


poisson_calculation(36)


# c) probability density
# define function and periodic variables
def prob_density(aa, bb):
    aa = (aa % 1) / 99
    bb = (bb % 1) / 99
    return lambda y: aa * np.exp(-0.5 * y) + bb * np.exp(-0.25 * y)


def probability_density_function(a, b):
    minute_probabilities = []
    for i in range(119, 241):
        result_by_min, none = spi.fixed_quad(prob_density(a, b), i / 60, (i + 1) / 60)
        minute_probabilities.append(result_by_min * 100)

    # return probability between 2 and 4 hours
    result_by_h, none = spi.fixed_quad(prob_density(a, b), 2, 4)
    print(result_by_h * 100)

    # mean and variance
    mean = round(np.mean(minute_probabilities), 5)
    variance = np.std(minute_probabilities) ** 2
    variance = round(variance, 10)

    # plot histogram
    plt.clf()
    plt.bar(np.arange(119, 241, 1), minute_probabilities, color="blue", fill=False, width=1, label="prob. histogram")
    plt.plot(np.arange(119, 241, 1), minute_probabilities, color="green", label="probability function")

    plt.scatter(150, minute_probabilities[30], label="Q1", color="red")
    plt.scatter(180, minute_probabilities[60], label="Q2", color="red")
    plt.scatter(210, minute_probabilities[90], label="Q3", color="red")

    plt.text(217, .008, r'$\mu=\ $')
    plt.text(225, .008, mean)
    plt.text(217, .007, r'$\sigma=\ $')
    plt.text(225, .007, variance)

    plt.legend(loc=3)
    plt.xlabel("Minutes")
    plt.ylabel("Probability")
    plt.show()


probability_density_function(0.20, 0.79)


# 2.
# a) Calculate the sample covariance as well as the sample’s expectations (mean) and variances of 𝑋 and 𝑌.
# load data into dataframe and calc. cov, E[X, Y], var[X, Y]
def sample(xy_values):
    x_l = []
    for i in range(0, 40, 2):
        x_value = xy_values.split(",")[i]
        x_value = x_value.replace("(", "")
        x_value = x_value.replace("\n", "")
        x_value = x_value.replace("−", "-")
        x_value = float(x_value)
        x_l.append(x_value)
    # print(x_l)

    y_l = []
    for i in range(1, 41, 2):
        y_value = xy_values.split(",")[i]
        y_value = y_value.replace(")", "")
        y_value = y_value.replace("\n", "")
        y_value = y_value.replace("−", "-")
        y_value = float(y_value)
        y_l.append(y_value)
    # print(y_l)

    df = pd.DataFrame(list(zip(x_l, y_l)), columns=['X', 'Y'])
    df = df.astype(float)
    # print(df)
    x_mean = df["X"].mean()
    y_mean = df["Y"].mean()
    # print(f"x_mean = {x_mean}, y_mean = {y_mean}")

    # covariance = Cov(X,Y) = Σ(Xi – μ)(Yj – ν) / (n)
    # expected value of x and y; E[X, Y] = E(X) = S x P(X = x)
    # Var(X) = E[ (X – m)2 ]; Var(Y) = E[ (Y – m)2 ]
    sum_cov = 0
    x_exp = 0
    y_exp = 0
    for i in range(20):
        xc = df.iloc[i, 0]
        yc = df.iloc[i, 1]
        sum_cov += (xc - x_mean) * (yc - y_mean)    # covariance

        x_exp += xc * (1/20)    # x expected value
        y_exp += yc * (1/20)    # y expected value

    covariance = sum_cov / 20

    var_x = 0
    var_y = 0
    mean_x = df["X"].mean()
    mean_y = df["Y"].mean()
    for i in range(20):
        xc = df.iloc[i, 0]
        yc = df.iloc[i, 1]

        var_x += ((xc - mean_x) ** 2)
        var_y += ((yc - mean_y) ** 2)

    var_x /= 20
    var_y /= 20

    print(f"Covariance = {covariance}")
    print(f"x expected value = {x_exp}")
    print(f"y expected value = {y_exp}")
    print(f"x variance = {var_x}")
    print(f"y variance = {var_y}")
    print(f"Standard deviation of x = {np.sqrt(var_x)}")
    print(f"Standard deviation of y = {np.sqrt(var_y)}")

    # plot scatter plot
    plt.clf()
    plt.scatter(df["X"], df["Y"], color="green")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.show()

    # Y values graph
    plt.clf()
    nl = np.arange(1, 21, 1)
    plt.scatter(nl, df["Y"], color="green")
    plt.xticks(nl)
    plt.xlabel("n")
    plt.ylabel("Y values")
    plt.show()

    # X values graph
    plt.clf()
    nl = np.arange(1, 21, 1)
    plt.scatter(nl, df["X"], color="green")
    plt.xticks(nl)
    plt.xlabel("n")
    plt.ylabel("X values")
    plt.show()


values_c = """(0.906, -1163.46), (-3.445, 296.33), (-7.181, 615.33), (-0.551, 323.82), (-7.569, -381.88), 
(-2.515, -247.44), (1.724, -415), (-4.305, 314.34), (1.352, -279.01), (4.02, -4.64), (14.374, 819.29), (-5.442, -77.98),
 (-0.247, -529.87), (0.023, 80.07), (-7.353, 755.71), (6.906, 91.63), (-10.411, 142.87), (5.779, 364.04), 
 (-8.156, -34.07), (2.249, -887.71)"""

sample(values_c)


# 2. b)
def random_circle():
    xs = []
    ys = []
    hist_bins = np.arange(-1, 1.01, 0.1)
    for n in range(10 ** 5):
        # radius = np.sqrt(np.random.uniform())
        radius = np.random.uniform()
        theta = 2 * np.pi * np.random.uniform()
        x_cord = radius * np.cos(theta)
        y_cord = radius * np.sin(theta)

        xs.append(x_cord)
        ys.append(y_cord)

    plt.figure(1)
    plt.hist(xs, hist_bins)

    plt.figure(2)
    plt.hist(ys, hist_bins)

    plt.figure(3, figsize=(8, 8))
    plt.scatter(xs, ys, s=1)

    # loop
    x_interval = np.arange(-1, 1.001, 0.001)
    pos = np.where(x_interval == 0.0)
    x_interval = np.delete(x_interval, pos)
    print(x_interval)

    y_interval = []
    for x in x_interval:
        y_value = (1/(4 * np.pi))*(np.log(np.sqrt(1-x**2)+1)-np.log(1-np.sqrt(1-x**2)))
        y_interval.append(y_value)

    plt.figure(4)
    plt.plot(x_interval, y_interval, color="#DC143C", lw=2.4)
    # plt.hist(ys, hist_bins)

    plt.show()


random_circle()
