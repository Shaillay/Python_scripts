# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 17:13:19 2017

@author: dogrask
"""


import re




from os import path 
import argparse

def main():
    parser=argparse.ArgumentParser(description='Parse stats output from diagnostics of wrongly-connected, wrongly-split droplets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputstats_txt', type=str, 
                      help='A file containing on different rows: \
                      Total droplets: , \
                      Total connected-components: , \
                      Total wrongly-connected-droplets: ,  \
                      Total wrongly-split-droplets: ')
   
    args=parser.parse_args()


    fname = args.inputstats_txt
    lines_list = open(fname).read().splitlines()    ## read in each row as a list    

    results = []
    
    for row in range(len(lines_list)):
        droplets = re.search(r'(Total) (droplets): (\S+)', lines_list[row])
        components = re.search(r'(Total) (connected-components): (\S+)', lines_list[row])
        wcds = re.search(r'(Total) (wrongly-connected-droplets): (\S+)', lines_list[row])
        wsds = re.search(r'(Total) (wrongly-split-droplets): (\S+)', lines_list[row])
        
        try:
            d= droplets.group(3)
            results.append(int(d))
        except:
            droplets = None
            
        try:
            c = components.group(3)
            results.append(int(c))
        except:
            components = None

        try:
            w1 = wcds.group(3)
            results.append(int(w1))
        except:
            wcds = None
        
        try:
            w2 = wsds.group(3)
            results.append(int(w2))
        except:
            wsds = None
            
    
    print results
                     
if __name__ == '__main__':
    main()

