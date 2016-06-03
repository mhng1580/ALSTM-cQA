import sys

def appendTag(prefixFile, targetFile, outFile):
    with open(prefixFile) as f:
        prefix = f.read().splitlines()
    
    with open(targetFile) as f:
        target = f.read().splitlines()
    
    assert(len(prefix) == len(target))
    with open(outFile, 'w') as f:
        for i in range(len(prefix)):
            f.write(prefix[i] + '\t' + target[i] + '\n')

if __name__ == "__main__":
    appendTag(sys.argv[1], sys.argv[2], sys.argv[3])
