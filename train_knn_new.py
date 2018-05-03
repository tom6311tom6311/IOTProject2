import numpy as np
import util
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

DATA_PATH = 'data/data_total.csv'
KNN_MODEL_FILE = 'knn_new.model'
SVC_MODEL_FILE = 'svc_new.model'

if __name__ == '__main__':
  test_data, test_labels, train_data, train_labels, label_to_xyz, selected_features = util.load_data(DATA_PATH)
  print(train_data.shape)

  clf = svm.SVC()
  knn = KNeighborsClassifier(n_neighbors=7)
  clf.fit(train_data, train_labels)
  knn.fit(train_data, train_labels)
  
  pickle.dump([clf, label_to_xyz, selected_features], open(SVC_MODEL_FILE, 'wb'))
  [clf, label_to_xyz, selected_features] = pickle.load(open(SVC_MODEL_FILE, 'rb'))
  pickle.dump([knn, label_to_xyz, selected_features], open(KNN_MODEL_FILE, 'wb'))
  [knn, label_to_xyz, selected_features] = pickle.load(open(KNN_MODEL_FILE, 'rb'))

  pred_labels = clf.predict(test_data)
  print(test_labels)
  print(pred_labels)
  print accuracy_score(test_labels, pred_labels)

  pred_labels = knn.predict(test_data)
  print(test_labels)
  print(pred_labels)
  print accuracy_score(test_labels, pred_labels)

  # print('training...')
  # model.fit(train_data, train_labels, epochs=10000, batch_size=64, callbacks=[EarlyStopping(monitor='loss', patience=300)])
  # model.save_weights('model.h5')

  # print('evaluating...')
  # predicted_labels = model.predict(test_data)
  
  # diff = test_labels - predicted_labels
  # diff[:0] *= (POS_RANGE['MAX_X'] - POS_RANGE['MIN_X'])
  # diff[:1] *= (POS_RANGE['MAX_Y'] - POS_RANGE['MIN_Y'])
  # diff[:2] *= (POS_RANGE['MAX_Z'] - POS_RANGE['MIN_Z'])
  # diff = np.sqrt(np.sum(np.square(diff), axis=1))
  # print('average distance error: ' + str(np.mean(diff)) + ' m')