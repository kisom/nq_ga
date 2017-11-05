#!/usr/bin/env python3

import nq
import statistics
import sys
import time

def runner():
    exp = nq.Experiment()
    start = time.process_time()
    exp.run(quiet=True)
    stop = time.process_time()
    
    return (len(exp), round(stop - start, 1))
    
def run():
    exps = []
    for i in range(8):
        sys.stdout.write('Starting experiment {}... '.format(i))
        sys.stdout.flush()
        exps.append(runner())
        print('OK')
        
    return exps
    
def statline(ns, lbl):
   print('{}:', lbl)
   
   mean = statistics.mean(ns)
   med  = statistics.median(ns)
   mn = min(ns)
   mx = max(ns)
   print('\tMean: {}\tMedian: {}\tMin: {}\tMax: {}'.format(mean, med, mn, mx))
    
def show_experiments(exps):
    print('Gens\t\tTime')
    for exp in exps:
        print('{}\t{}'.format(exp[0], exp[1]))
        
    gen_counts, elapsed = zip(*exps)
    statline('Generations', gen_counts)
    statline('Elapsed', gen_counts)
        
def experiment():
    exps = run()
    show_experiments(exps)
    
if __name__ == '__main__':
    experiment()
