# Observation framework

import os
import csv
import pickle
from collections.abc import Iterable
from matplotlib import pyplot as plt
from typing import List


class Observer:
    def __init__(self, name, path="", suffix="", labels=[]):
        self.name = name
        self.labels = labels
        self.observations = []
        self.set_path(path, suffix)

    def clear(self):
        self.observations = []

    def set_path(self, path="", suffix=""):
        self.suffix = suffix
        self.fullname = os.path.join(
            path, self.name + (f"_{suffix}" if suffix else ""))

    def record(self, variable):
        self.observations.append(variable)

    def write_csv(self, headings=None):
        filename = (f"{self.fullname}.csv")
        if len(self.observations) == 0:
            return
        if headings is None:
            headings = self.labels
        with open(filename, "w") as file_handle:
            csv_writer = csv.writer(file_handle)
            csv_writer.writerow(headings)
            if isinstance(self.observations[0], Iterable):
                csv_writer.writerows(self.observations)
            else:
                csv_writer.writerows([x] for x in self.observations)

    def save_pkl(self):
        filename = (f"{self.fullname}.pkl")
        pickle.dump(self.observations, open(filename, "wb"))

    def plot(self, legend=None):
        if legend is None:
            legend = self.labels
        filename = (f"{self.fullname}.png")
        plt.figure()
        plt.plot(self.observations)
        plt.ylim(ymin=0)
        plt.legend(legend)
        plt.savefig(filename)
        plt.close()

    def avg(self):
        return sum(self.observations) / len(self.observations)
