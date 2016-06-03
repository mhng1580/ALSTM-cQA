import sys

def rmTag(taggedFile, outFile):
    with open(taggedFile) as f:
        taggedLine = f.read().splitlines()
    
    with open(outFile, 'w') as f:
        for l in taggedLine:
            f.write(l.split('\t')[-1] + '\n')


if __name__ == "__main__":
    rmTag(sys.argv[1], sys.argv[2])
