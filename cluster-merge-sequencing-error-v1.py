# -*- coding: utf-8 -*-
"""
input file has these fields:
(not required fields denoted in [...]) 
barcode1	barcode2	[dropletIDs]	edges	[edgetypes]
    
for now, for example, input file is an output generated from:
    make_contamination_v2.py
    make_hyperconnected_contamination_v2.py

1) Read in input file
2) get hold of unique BC sequences, and associated counts
3) run a pair-wise alignment on unique BC sequences to calculate hamming-distances
4) assume the BC sequence with highest count to be the originating BC; merge \
   hamming distance = 1,2,... to this parent sequence; attribute the counts of \
   merged BCs to the originating BC
5) output data thus reduced by clustering and merging
    

Refer:
https://pypi.python.org/pypi/Distance
    
    
Shaillay Dogra
01-May-2017    

"""

import pandas as pd
import argparse
import random
import math
import distance
import time
from datetime import datetime

start_time1 = time.time()
start_time2 = time.clock()
start_time3 = datetime.now()


def combine(df):
    comb_bc=''.join([df['barcode1'],df['barcode2']])
    return comb_bc

def hamming_distance(s1, s2):
    """Return the Hamming distance between equal-length sequences"""
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

def allbc_combined(bc):
    allbcs.extend(bc)
    return




from os import path 
def main():
    parser=argparse.ArgumentParser(description='cluster and merge sequencing errors; \
                                   with some assumotions',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_csv_file', type=str,
                      help='csv_file to import')
    parser.add_argument('--x', type=float, default=2,
                      help='hamming-distance to merge till..., default=2')
    parser.add_argument('--outfile', type=str,
                      help='output file, defaults to input name _merged_till x_hamming-distance.csv')
    args=parser.parse_args()
    
    if args.outfile==None:
        args.outfile=''.join([path.splitext(args.input_csv_file)[0],'_error', \
                              str(args.x), '_contam.csv'])

## this is what the dataframe structure should look like if imported from a simulated file
## df=pd.DataFrame(columns=('barcode1','barcode2','dropletID','edges','edge_type'))

    print "starting to process..."
    df=pd.DataFrame.from_csv(args.input_csv_file, index_col=None)
 #   print df.head(10)
#    print df['barcode1'].head(10)
 
## combine barcodes, associate counts
    df1 = df[['barcode1','barcode2']].apply(combine,axis=1)
    df2 = pd.concat([df1, df['edges']], axis=1)    
#    print df2.head(10)
    print "total combined-barcodes" 
    print len(df2)
#    print df2[0].value_counts()
    print "total unique combined-barcodes"
    print len(df2[0].unique())  # test that this unique function is doing what it is supposed to do
#    print len(df2.drop_duplicates()) # testing it another way

    cutoff = args.x
#    print cutoff
    

#    print df2.head(10)
    df2=df2.sort_values('edges',ascending=False)
#    print df2.head(10)
    total_rows = len(df2)
    checked =[]
    results =[]
    for rows1 in range(total_rows):
        #print "\n"
        if rows1 % 10 == 0:
            print "current row with input query:", 
            print rows1
            #print "\n"
        input = df2.iloc[rows1,0]
        #print rows1, input, "0", df2.iloc[rows1,1], rows1 # "0" is place-holder value of hamming-distance to self
        results.extend([[rows1, input, "0", df2.iloc[rows1,1], rows1]])
        #print input
        for rows2 in range(total_rows):
            if rows2 > rows1:
                if rows2 not in checked:
                    #print "current row with target query:", 
                    #print rows2
                    target = df2.iloc[rows2,0]
                    dist = hamming_distance(input,target)
                    if dist <= cutoff:
                        checked.extend([rows2])
                        #print checked
                        #print rows2, target, dist, df2.iloc[rows2,1], rows1
                        results.extend([[rows2, target, dist, df2.iloc[rows2,1], rows1]])


#    print checked     
#    print results
    outdf = pd.DataFrame(results,columns=['matched','BC', 'distance', 'edges', 'label']) 
    print outdf.head(10)
    df.index_col = ['label']
    df_out = outdf.groupby(df.index_col).agg({'edges':sum})[['edges']]
    print df_out.head(10)
    print len(df_out)
    

##  print "writing to file..."
##  df_out.to_csv(args.outfile, index=False)
    df_out.to_csv("temp_out.csv", index=False)


    #print "--- %s seconds ---" % (time.time() - start_time1)
    #print time.clock() - start_time2, "seconds"
    print('Time-taken: {}'.format(datetime.now()- start_time3))

    
    
if __name__ == '__main__':
    main()





