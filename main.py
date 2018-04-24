import os
from time import gmtime, strftime
import iwlist

data_file_name = 'data.csv'
picked_essid = ['LBS1', 'LBS2', 'LBS3', 'LBS4', 'NTU@1', 'NTU@2', 'NTU@3', 'NTU@4', 'ntu_peap@1', 'ntu_peap@2', 'ntu_peap@3', 'eduroam@1', 'eduroam@2', 'eduroam@3', 'MD402_1', 'MD402-2', 'NTUMOOC', 'eslab-Dlink652', 'DIRECT-32-HP M377 LaserJet']
picked_mac = ['60:45:CB:12:F5:70', '60:45:CB:12:D9:E8', '60:45:CB:12:E5:98', '60:45:CB:12:DD:78', '00:0B:86:96:63:A2', '00:0B:86:96:86:62', '00:0B:86:96:61:C2', '00:0B:86:96:70:C2', '00:0B:86:96:63:A1', '00:0B:86:96:61:C1', '00:0B:86:96:70:C1', '00:0B:86:96:70:C0', '00:0B:86:96:82:20', '00:0B:86:96:61:C0', '70:70:8B:09:2F:E0', '00:1D:AA:0F:7C:44', 'AC:9E:17:7E:E6:38', 'B8:A3:86:57:17:67', 'EA:9E:B4:2D:81:32']
param_each_ap = ['SIG_Q', 'SIG_L', 'NOISE_L']


if not os.path.exists(data_file_name):
  data_file = open(data_file_name, 'a')
  feature_names = []
  for param in param_each_ap:
    for essid in picked_essid:
      feature_names.append(essid + '_' + param)
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

  cells = iwlist.parse(iwlist.scan(interface='ra0'))
  raw_data = [0.0] * (len(picked_mac) * 3)
  for cell in cells:
    if (cell['mac'] in picked_mac):
      idx = picked_mac.index(cell['mac'])
      raw_data[idx] = float(cell['signal_quality'])
      raw_data[idx + len(picked_mac)] = float(cell['signal_level_dBm'][:cell['signal_level_dBm'].index(' ')])
      raw_data[idx + 2 * len(picked_mac)] = float(cell['signal_level_dBm'][cell['signal_level_dBm'].index('=')+1:])
  raw_data.append(x)
  raw_data.append(y)
  raw_data.append(z)
  raw_data.append(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  print(raw_data)
  data_file.write(', '.join(map(str, raw_data)) + '\n')