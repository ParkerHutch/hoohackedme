import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def get_data(filename):
    is_csv = filename.split(".")[1]
    assert is_csv == "csv"
    data = pd.read_csv(filename)
    return data


class Plot:

    def __init__(self, filename, save_name):
        self.filename = filename
        self.save_name = save_name
        self.data = get_data(self.filename)

    def plot(self):
        plt.plot(self.data['Guess #'], self.data[' Correct Guesses Total'])
        # plt.scatter(self.data['Guess #'], self.data[' Correct Guesses Total'])
        plt.xlabel("Guess #")
        plt.ylabel("Correct Guesses Total")
        plt.savefig("model-analysis/%s" % self.save_name)


def main():
    filename = sys.argv[1]
    save_name = sys.argv[2]
    p = Plot(filename=filename, save_name=save_name)
    p.plot()


if __name__ == "__main__":
    main()
