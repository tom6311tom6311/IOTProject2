import iwlist

content = iwlist.scan(interface='ra0')
cells = iwlist.parse(content)
print(cells)