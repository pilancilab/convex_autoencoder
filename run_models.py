#!/usr/bin/env python

import sys
from utils.run_model_helpers import run_dataset


datasets = ['adult', 'connect4', 'dna', 'mushrooms', 'nips', 'ocr', 'rcv1', 'web']
if __name__=='__main__':
    data = datasets[int(sys.argv[1])]
    num_runs = 10
    print('Running Iterations for Dataset ' + data)
    run_dataset(data, num_runs)
    print('\n\n')
