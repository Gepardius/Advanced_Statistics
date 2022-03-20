import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


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

    Expected_value = p
    print(f"Expected value = {Expected_value}")


bernoulli(0.27)

# b) Poisson distribution
# 𝜆 =

