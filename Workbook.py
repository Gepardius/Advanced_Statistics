import statistics
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson
from scipy.stats import norm
from scipy.integrate import quad
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


# bernoulli(0.27)

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


# poisson_calculation(37)

# c) probability density
# 0. 3e−0.5 𝑦 + 0. 6e−0.25 𝑦
# a * np.exp(-0.5 * y) + b * np.exp(-0.25 * y)
# 0.55555555* e^(-0.5 * y) + 0.444444* e^(-0.25 * y)

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

    result_by_h, none = spi.fixed_quad(prob_density(a, b), 2, 4)
    # return probability between 2 and 4 hours
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

    plt.text(210, .006, r'$\mu=\ $')
    plt.text(217, .006, mean)
    plt.text(210, .005, r'$\sigma=\ $')
    plt.text(217, .005, variance)

    plt.legend(loc=3)
    plt.xlabel("Minutes")
    plt.ylabel("Probability")
    plt.show()


# probability_density_function(0.83, 0.16)

# 2.
# a) Calculate the sample covariance as well as the sample’s expectations (mean) and variances of 𝑋 and 𝑌.
# the expected value is the sum of: [(each of the possible outcomes) × (the probability of the outcome occurring)]
values = """(−1.202, 563.024), (2.112, 291.072), (2.827, −893.619), (−0.314, 1321.814),
(−1.477, −91.573), (−6.516, 446.336), (−0.920, −111.487), (3.477, −153.165),
(−7.273, 1076.221), (2.251, 477.931), (−0.713, 909.696), (−0.853, 226.865),
(−3.176, 389.413), (1.913, −47.169), (−1.070, −178.695), (−3.385, 744.486),
(−9.506, 362.670), (−7.004, 364.578), (0.504, 324.975), (2.861, −360.571)"""

x_l = []
for i in range(0, 40, 2):
    x_value = values.split(",")[i]
    x_value = x_value.replace("(", "")
    x_value = x_value.replace("\n", "")
    x_l.append(x_value)
# print(x_l)

y_l = []
for i in range(1, 41, 2):
    y_value = values.split(",")[i]
    y_value = y_value.replace(")", "")
    y_value = y_value.replace("\n", "")
    y_l.append(y_value)
# print(y_l)

df = pd.DataFrame(list(zip(x_l, y_l)), columns =['X', 'Y'])
print(df)
