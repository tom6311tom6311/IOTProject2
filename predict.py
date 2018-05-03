import numpy as np
import location_predictor
from subprocess import call
import pickle
import util
import iwlist


MODEL_FILE = 'knn_new.model'
interface_names = ['wlan0','ra0']
picked_essid = ['LBS1', 'LBS2', 'LBS3', 'LBS4', 'NTU@1', 'NTU@2', 'NTU@3', 'NTU@4', 'ntu_peap@1', 'ntu_peap@2', 'ntu_peap@3', 'eduroam@1', 'eduroam@2', 'eduroam@3', 'MD402_1', 'MD402-2', 'NTUMOOC', 'eslab-Dlink652', 'DIRECT-32-HP M377 LaserJet']
picked_mac = ['60:45:CB:12:F5:70', '60:45:CB:12:D9:E8', '60:45:CB:12:E5:98', '60:45:CB:12:DD:78', '00:0B:86:96:63:A2', '00:0B:86:96:86:62', '00:0B:86:96:61:C2', '00:0B:86:96:70:C2', '00:0B:86:96:63:A1', '00:0B:86:96:61:C1', '00:0B:86:96:70:C1', '00:0B:86:96:70:C0', '00:0B:86:96:82:20', '00:0B:86:96:61:C0', '70:70:8B:09:2F:E0', '00:1D:AA:0F:7C:44', 'AC:9E:17:7E:E6:38', 'B8:A3:86:57:17:67', 'EA:9E:B4:2D:81:32']
param_each_ap = ['SIG_Q', 'SIG_L', 'NOISE_L']
repeat_num = 3
repeat_interval = 1


curr_repeat_idx = 0
raw_data = []
while curr_repeat_idx < repeat_num:
  raw_datum = [0.0] * len(interface_names) * len(picked_mac) * len(param_each_ap)
  for ifn_idx, ifn in enumerate(interface_names):
    cells = iwlist.parse(iwlist.scan(interface=ifn))
    for cell in cells:
      if (cell['mac'] in picked_mac):
        idx = picked_mac.index(cell['mac'])
        raw_datum[ifn_idx * len(picked_mac) * len(param_each_ap) + idx * len(param_each_ap)] = float(cell['signal_quality'])
        if (' ' in cell['signal_level_dBm']):
          raw_datum[ifn_idx * len(picked_mac) * len(param_each_ap) + idx * len(param_each_ap) + 1] = float(cell['signal_level_dBm'][:cell['signal_level_dBm'].index(' ')])
          raw_datum[ifn_idx * len(picked_mac) * len(param_each_ap) + idx * len(param_each_ap) + 2] = float(cell['signal_level_dBm'][cell['signal_level_dBm'].index('=')+1:])
        else:
          raw_datum[ifn_idx * len(picked_mac) * len(param_each_ap) + idx * len(param_each_ap) + 1] = float(cell['signal_level_dBm'])
  raw_data.append(raw_datum)
  curr_repeat_idx += 1
  print(raw_datum)
raw_data = np.array(raw_data)
print(raw_data)

print('loading model...')
[knn, label_to_xyz, selected_features] = pickle.load(open(MODEL_FILE, 'rb'))
print('predicting...')
pred_labels = knn.predict(util.transform_feat(raw_data, selected_features))
pred_xyzs = np.array([label_to_xyz[i] for i in pred_labels])
pred_avg_xyz = np.mean(pred_xyzs, axis=0)
print(pred_avg_xyz)

# call(['curl', '-i', '-XPOST', "'http://140.112.18.229:32071/write?db=ntu_iot'", '--data-binary', '"team8,axis=x value=' + str(pos[0]) + '"'])
# call(['curl', '-i', '-XPOST', "'http://140.112.18.229:32071/write?db=ntu_iot'", '--data-binary', '"team8,axis=y value=' + str(pos[1]) + '"'])
# call(['curl', '-i', '-XPOST', "'http://140.112.18.229:32071/write?db=ntu_iot'", '--data-binary', '"team8,axis=z value=' + str(pos[2]) + '"'])
