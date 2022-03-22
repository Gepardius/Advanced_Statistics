import statistics
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson


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
    poisson_probability = []
    x_l = []
    for i in range(1000):
        y = poisson.pmf(k=i, mu=mu_poisson)  # mu = 𝜆/expectancy , k = probability that it will happen
        if y > 0.005:
            poisson_probability.append(y)
            x_l.append(i)

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


poisson_calculation(37)
