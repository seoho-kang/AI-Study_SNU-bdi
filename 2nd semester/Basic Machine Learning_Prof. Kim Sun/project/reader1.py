import numpy as np
import csv

class reader(object):
  def __init__(self, data_file = "./data/wdbc.data"):
    self.value = []
    with open(data_file, "rb") as f:
      csv_reader = csv.reader(f, delimiter=",")
      for i, row in enumerate(csv_reader):
        self.value.append(row)
    # print self.value[0]
    # print self.value[0][0]
    # print self.value[0][1]

    self.raw_to_vector(self.value)

    print self.id.shape
    print self.x.shape
    print self.y.shape

  def raw_to_vector(self, value):
    self.id = []
    self.x = []
    self.y = []

    for row in self.value:
        x = np.zeros(30)
        for i in range(30):
            x[i] = float(row[i+2])
        if row[1] == "B":   #Benign = 0, Malignant = 1
            y = 0
        else:
            y = 1
        self.x.append(x)
        self.y.append(y)
        id = int(row[0])
        self.id.append(id)

    self.x, self.y, self.id = np.array(self.x), np.array(self.y), np.array(self.id)
