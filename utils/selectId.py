import sys


in1 = sys.argv[1]
in2 = sys.argv[2]

list1 = open(in1).readlines()
list2 = open(in2).readlines()

map = dict()
for l1 in list1:
    item = l1.split('\t')
    map[item[0]] = l1

for l2 in list2:
    item = l2.split('\t')
    if item[0] in map:
        l2 = l2.strip('\n')
        print l2

