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

bernoulli(0.27)

# b) Poisson distribution
# 𝜆 =
poisson_probability = []
x_l = []
for i in range(101):
    y = poisson.pmf(k=i, mu=37)  # mu = average, k = probability that it will happen
    if y > 0.005:
        poisson_probability.append(y)
        x_l.append(i)

plt.clf()
plt.plot(x_l, poisson_probability)
plt.xlabel("N. of meteorites falling")
plt.ylabel("Probability")
plt.show()




