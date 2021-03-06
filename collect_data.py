import os
import time
from time import gmtime, strftime
import iwlist

interface_names = ['wlan0','ra0']
data_file_name = 'data.csv'
picked_essid = ['LBS1', 'LBS2', 'LBS3', 'LBS4', 'NTU@1', 'NTU@2', 'NTU@3', 'NTU@4', 'ntu_peap@1', 'ntu_peap@2', 'ntu_peap@3', 'eduroam@1', 'eduroam@2', 'eduroam@3', 'MD402_1', 'MD402-2', 'NTUMOOC', 'eslab-Dlink652', 'DIRECT-32-HP M377 LaserJet']
picked_mac = ['60:45:CB:12:F5:70', '60:45:CB:12:D9:E8', '60:45:CB:12:E5:98', '60:45:CB:12:DD:78', '00:0B:86:96:63:A2', '00:0B:86:96:86:62', '00:0B:86:96:61:C2', '00:0B:86:96:70:C2', '00:0B:86:96:63:A1', '00:0B:86:96:61:C1', '00:0B:86:96:70:C1', '00:0B:86:96:70:C0', '00:0B:86:96:82:20', '00:0B:86:96:61:C0', '70:70:8B:09:2F:E0', '00:1D:AA:0F:7C:44', 'AC:9E:17:7E:E6:38', 'B8:A3:86:57:17:67', 'EA:9E:B4:2D:81:32']
param_each_ap = ['SIG_Q', 'SIG_L', 'NOISE_L']
repeat_num = 20
repeat_interval = 1

if not os.path.exists(data_file_name):
  data_file = open(data_file_name, 'a')
  feature_names = []
  for ifn in interface_names:
    for essid in picked_essid:
      for param in param_each_ap:
        feature_names.append(ifn + '_' + essid + '_' + param)
  feature_names.append('X')
  feature_names.append('Y')
  feature_names.append('Z')
  feature_names.append('timestamp')
  data_file.write(', '.join(feature_names) + '\n')
else:
  data_file = open(data_file_name, 'a')

while True:
  x = float(input('Input X: '))
  y = float(input('Input Y: '))
  z = float(input('Input Z: '))
  curr_repeat_idx = 0
  while curr_repeat_idx < repeat_num:
    raw_data = [0.0] * len(interface_names) * len(picked_mac) * len(param_each_ap)
    for ifn_idx, ifn in enumerate(interface_names):
      cells = iwlist.parse(iwlist.scan(interface=ifn))
      for cell in cells:
        if (cell['mac'] in picked_mac):
          idx = picked_mac.index(cell['mac'])
          raw_data[ifn_idx * len(picked_mac) * len(param_each_ap) + idx * len(param_each_ap)] = float(cell['signal_quality'])
          if (' ' in cell['signal_level_dBm']):
            raw_data[ifn_idx * len(picked_mac) * len(param_each_ap) + idx * len(param_each_ap) + 1] = float(cell['signal_level_dBm'][:cell['signal_level_dBm'].index(' ')])
            raw_data[ifn_idx * len(picked_mac) * len(param_each_ap) + idx * len(param_each_ap) + 2] = float(cell['signal_level_dBm'][cell['signal_level_dBm'].index('=')+1:])
          else:
            raw_data[ifn_idx * len(picked_mac) * len(param_each_ap) + idx * len(param_each_ap) + 1] = float(cell['signal_level_dBm'])
    raw_data.append(x)
    raw_data.append(y)
    raw_data.append(z)
    raw_data.append(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print(raw_data)
    data_file.write(', '.join(map(str, raw_data)) + '\n')
    time.sleep(repeat_interval)
    curr_repeat_idx += 1