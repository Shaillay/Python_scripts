# -*- coding: utf-8 -*-
"""

Take an input file
calculate total read counts
user input: desired total reads
fractionate reads at each row accordingly
(so that now the new total read count is same as user input desired total reads)

becausing of using int function when fractionating reads at each row,
it is possible that the new total of fractionated reads 
may not add up exactly to the user desired total reads


"""

import pandas as pd
import argparse





from os import path 
def main():
    parser=argparse.ArgumentParser(description='fractionated reads per row based on desired total reads; \
                                   in order to simulate low read coverage',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter, usage='%(prog)s [input_file.csv --TotalReads]')
    parser.add_argument('input_csv_file', type=str,
                      help='csv_file to import')
    parser.add_argument('--TotalReads', type=int, default=None,
                      help='Desired Total Number of Reads in Million units; for example 10 implies 10 Million total reads')
    parser.add_argument('--outfile', type=str,
                      help='output file, defaults to input name _total_reads lowcoverage.csv')
    args=parser.parse_args()
    
    if args.outfile==None:
        args.outfile=''.join([path.splitext(args.input_csv_file)[0],'total_reads', \
                              str(args.TotalReads), '_lowcoverage.csv'])

# this is what the dataframe structure should look like if imported from a simulated file
 #df=pd.DataFrame(columns=('barcode1','barcode2','dropletID','edges','edge_type'))


    print "starting to process..."
    df=pd.DataFrame.from_csv(args.input_csv_file, index_col=None)
 
    total_reads = df['edges'].sum()
    total_reads_in_millions = float(total_reads)/1000000
    desired_total_reads = args.TotalReads
   
    print "total reads in input file calculated as sum(edges) =", total_reads_in_millions,"M"
    print "user desires the total reads to be =", desired_total_reads, "M"
    
    fraction = float(desired_total_reads)/total_reads_in_millions
    print fraction
    
    df['edges'] = df['edges'].apply(lambda x: int(x*fraction))
    df1=df[df['edges']>=1]

    new_total_reads_in_millions = float(df['edges'].sum())/1000000
    print "new total reads =", new_total_reads_in_millions, "M"

    print 'writing to file...'
    df1.to_csv(args.outfile, index=False)


if __name__ == '__main__':
    main()
