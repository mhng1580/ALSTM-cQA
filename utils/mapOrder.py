import sys
import os

def getTargetKey(line, lang):
    if lang == 'arab':
        rawKey = line.split('\t')[1]
        q = rawKey.split('_')[0][1:]
        r = rawKey.split('_')[1][1:]
        return (q + '\t' + r, '\t'.join([q, r] + line.split('\t')[2:]))
    elif lang == 'eng':
        return ('\t'.join(line.split('\t')[:2]), line)

orderFile = sys.argv[1]
targetFile = sys.argv[2]
outFile = sys.argv[3]
#filename, ext = os.path.splitext(targetFile)
#outFile = filename + '_keepOrder' + ext

with open(orderFile) as f:
    raw = f.read().splitlines()
    keyList = ['\t'.join(line.split('\t')[:2]) for line in raw]

with open(targetFile) as f:
    raw = f.read().splitlines()
    targetDict = dict([getTargetKey(line, lang='eng') for line in raw])

output = [targetDict[k] for k in keyList]
with open(outFile, 'w') as f:
    f.write('\n'.join(output))
