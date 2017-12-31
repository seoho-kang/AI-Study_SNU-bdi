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
    self.split(num_validation_examples=50)

#     print self.id.shape
    print self.x.shape
    print self.y.shape

    self.num_examples = len(self.x_train)
    self.start_index = 0
    self.shuffle_indices = range(self.num_examples)

    self.num_examples_val = len(self.x_val) # = 50 (validation_data)
    self.start_index_val = 0
    self.shuffle_indices_val = range(self.num_examples_val)
    print self.num_examples_val

  def raw_to_vector(self, value):
    self.id = []
    self.x = []
    self.y = []

    for row in self.value:
        x = np.zeros(30)
        for i in range(30):
            x[i] = float(row[i+2])
        if row[1] == "B":
            y = 0
        else:
            y = 1
        self.x.append(x)
        self.y.append(y)
        id = int(row[0])
        self.id.append(id)

    self.x, self.y, self.id = np.array(self.x), np.array(self.y), np.array(self.id)

  def split(self, num_validation_examples):
    self.x_train = self.x[ num_validation_examples: ]
    self.x_val = self.x[ : num_validation_examples ]

    self.y_train = self.y[ num_validation_examples: ]
    self.y_val = self.y[ : num_validation_examples ]

    self.id_train = self.id[ num_validation_examples: ]
    self.id_val = self.id[ :num_validation_examples ]

  def next_batch(self, batch_size, split="train"):

    if split == "train":
      if self.start_index == 0:
        np.random.shuffle(self.shuffle_indices) # shuffle indices

      end_index = min([self.num_examples, self.start_index + batch_size])
      batch_indices = [ self.shuffle_indices[idx] for idx in range(self.start_index, end_index) ]

      batch_x = self.x_train[ batch_indices ]
      batch_y = self.y_train[ batch_indices ]
      batch_id = self.id_train[ batch_indices ]

      if end_index == self.num_examples:
        self.start_index = 0
      else: self.start_index = end_index

      return batch_x, batch_y, batch_id

    elif split == "val":
      if self.start_index_val == 0:
        np.random.shuffle(self.shuffle_indices_val) # shuffle indices

      end_index = min([self.num_examples_val, self.start_index_val + batch_size])
      batch_indices = [ self.shuffle_indices_val[idx] for idx in range(self.start_index_val, end_index) ]

      batch_x = self.x_val[ batch_indices ]
      batch_y = self.y_val[ batch_indices ]
      batch_id = self.id_val[ batch_indices ]

      if end_index == self.num_examples_val:
        self.start_index_val = 0
      else: self.start_index_val = end_index

      return batch_x, batch_y, batch_id
