import iwlist

picked_essid = ['LBS1', 'LBS2', 'LBS3', 'LBS4', 'NTU@1', 'NTU@2', 'NTU@3', 'NTU@4', 'ntu_peap@1', 'ntu_peap@2', 'ntu_peap@3', 'eduroam@1', 'eduroam@2', 'eduroam@3', 'MD402_1', 'MD402-2', 'NTUMOOC', 'eslab-Dlink652', 'DIRECT-32-HP M377 LaserJet']
picked_mac = ['60:45:CB:12:F5:70', '60:45:CB:12:D9:E8', '60:45:CB:12:E5:98', '60:45:CB:12:DD:78', '00:0B:86:96:63:A2', '00:0B:86:96:86:62', '00:0B:86:96:61:C2', '00:0B:86:96:70:C2', '00:0B:86:96:63:A1', '00:0B:86:96:61:C1', '00:0B:86:96:70:C1', '00:0B:86:96:70:C0', '00:0B:86:96:82:20', '00:0B:86:96:61:C0', '70:70:8B:09:2F:E0', '00:1D:AA:0F:7C:44', 'AC:9E:17:7E:E6:38', 'B8:A3:86:57:17:67', 'EA:9E:B4:2D:81:32']

content = iwlist.scan(interface='ra0')
cells = iwlist.parse(content)

raw_data = [0.0] * len(picked_mac) * 2
for cell in cells:
  if (cell['mac'] in picked_mac):
    idx = picked_mac.index(cell['mac'])
    raw_data[idx] = float(cell['signal_quality'])
    raw_data[idx + len(picked_mac)] = float(cell['signal_level_dBm'])

print(raw_data)