import numpy as np

POS_RANGE = {'MIN_X': 0.0, 'MIN_Y': 0.0, 'MIN_Z': 0.0, 'MAX_X':9.0, 'MAX_Y':8.0, 'MAX_Z':3.0}

def load_data(path, valid_split=0.1):
  lines = []
  with open(path, 'r') as data_file:
    lines = data_file.readlines()
    data_file.close()
  if len(lines) == 0:
    return
  attrs = lines[0].split(', ')
  lines = lines[1:]
  raw_data = []
  labels = []
  label_to_xyz = []
  curr_xyz = np.array([0.0, 0.0, 0.0])
  curr_label = -1
  for line in lines:
    raw_datum = [ float(val) for val in line.split(', ')[:-4] ]
    xyz = [float(val) for val in line.split(', ')[-4:-1]]
    raw_datum = [raw_datum[i] for i in range(len(raw_datum)) if i % 3 <= 1] # ignore noise level
    if (np.count_nonzero(raw_datum) >= 68):
      raw_data.append(raw_datum)
      if (not np.array_equal(curr_xyz, xyz)):
        curr_xyz = xyz
        curr_label += 1
        label_to_xyz.append(xyz)
      labels.append(curr_label)
  raw_data = np.array(raw_data)
  selected_features = np.sort(np.argsort(np.count_nonzero(raw_data, axis=0))[18:])
  print(selected_features)
  raw_data = raw_data[:,selected_features]
  raw_data /= 100
  labels = np.array(labels)
  a = np.arange(raw_data.shape[0])
  np.random.shuffle(a)
  raw_data = raw_data[a]
  labels = labels[a]
  split_idx = int(np.around(valid_split * (raw_data.shape[0])))
  test_data = raw_data[:split_idx]
  test_labels = labels[:split_idx]
  train_data = raw_data[split_idx:]
  train_labels = labels[split_idx:]
  label_to_xyz = np.array(label_to_xyz)
  return test_data, test_labels, train_data, train_labels, label_to_xyz, selected_features

def transform_feat(raw_feat, selected_features):
  feat = raw_feat
  sig_feat_idxs = [i for i in range(raw_feat.shape[1]) if i % 3 <= 1]
  feat = feat[:,sig_feat_idxs]
  qualified_data_idxs = [j for j in range(raw_feat.shape[0]) if np.count_nonzero(feat[j,:]) >= 68]
  feat = feat[qualified_data_idxs, :]
  feat = feat[:,selected_features]
  feat /= 100
  return feat