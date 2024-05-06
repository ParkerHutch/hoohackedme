import matplotlib.pyplot as plt
import numpy as np


class Plot:

    def __init__(self, filename):
        self.filename = filename
        self.data = self.get_data(self.filename)

    def plot(self):
        pass

    def get_data(self, filename):
        data = {}
        return data


def main():
    p = Plot()
    p.plot()


if __name__ == "__main__":
    main()
