import os
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping


DATA_PATH = 'data/data_20180426.csv'
POS_RANGE = {'MIN_X': 0.0, 'MIN_Y': 0.0, 'MIN_Z': 0.0, 'MAX_X':9.0, 'MAX_Y':8.0, 'MAX_Z':3.0}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

def load_data(path, valid_split=0.1):
  lines = []
  with open(path, 'r') as data_file:
    lines = data_file.readlines()
    data_file.close()
  if len(lines) == 0:
    return
  attrs = lines[0].split(', ')
  lines = lines[1:]
  np.random.shuffle(lines)
  raw_data = []
  labels = []
  for line in lines:
    raw_datum = [ float(val) for val in line.split(', ')[:-4] ]
    label = [float(val) for val in line.split(', ')[-4:-1]]
    raw_datum = [raw_datum[i] for i in range(len(raw_datum)) if i % 3 <= 1] # ignore noise level
    if (np.count_nonzero(raw_datum) >= 72):
      raw_data.append(raw_datum)
      labels.append(label)
  raw_data = np.array(raw_data)
  labels = np.array(labels)
  labels[:,0] /= (POS_RANGE['MAX_X'] - POS_RANGE['MIN_X'])
  labels[:,1] /= (POS_RANGE['MAX_Y'] - POS_RANGE['MIN_Y'])
  labels[:,2] /= (POS_RANGE['MAX_Z'] - POS_RANGE['MIN_Z'])
  split_idx = int(np.around(valid_split * (raw_data.shape[0])))
  test_data = raw_data[:split_idx]
  test_labels = labels[:split_idx]
  train_data = raw_data[split_idx:]
  train_labels = labels[split_idx:]
  return test_data, test_labels, train_data, train_labels

if __name__ == '__main__':
  test_data, test_labels, train_data, train_labels = load_data(DATA_PATH)
  print(train_data.shape)

  model = Sequential()
  model.add(Dense(32, activation='relu', input_dim=train_data.shape[1]))
  model.add(BatchNormalization())
  model.add(Dense(3, activation='sigmoid'))
  model.add(BatchNormalization())
  model.compile(optimizer='rmsprop', loss='mse')
  model.summary()

  print('training...')
  model.fit(train_data, train_labels, epochs=10000, batch_size=64, callbacks=[EarlyStopping(monitor='loss', patience=10)])
  model.save_weights('model.h5')

  print('evaluating...')
  predicted_labels = model.predict(test_data)
  
  diff = test_labels - predicted_labels
  diff[:0] *= (POS_RANGE['MAX_X'] - POS_RANGE['MIN_X'])
  diff[:1] *= (POS_RANGE['MAX_Y'] - POS_RANGE['MIN_Y'])
  diff[:2] *= (POS_RANGE['MAX_Z'] - POS_RANGE['MIN_Z'])
  diff = np.sqrt(np.sum(np.square(diff), axis=1))
  print('average distance error: ' + str(np.mean(diff)) + ' m')
