import re

line = "this hdr-biz model args= server"
patt = r'server'
pattern = re.compile(patt)
result = pattern.findall(line)
print(pattern)
print(result)


import difflib

cityarea_list = ['市北区', '市南区', '莱州市', '四方区']
a = difflib.get_close_matches('市区',cityarea_list, cutoff=0.7)
print(a)


