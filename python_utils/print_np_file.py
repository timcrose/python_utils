import sys
import numpy as np

def main():
    data = np.load(*sys.argv[1:])
    print(data)
    print(data.shape)
    

main()
