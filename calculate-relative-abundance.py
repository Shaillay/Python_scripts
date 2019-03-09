

#
# from MOCAT output of mOTU counts, calculate relative abundances of bacteria at various taxa level
#
# input file is the data table (mOTU-counts) with taxa annotation
# output file is relative percentages at various taxa levels
#
# SKD - 09.12.2018
#
#


import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from os import path


start_time = datetime.now()


def percentages(df,taxalevel):
    taxadata = df.sort_values(taxalevel)
    taxadatasum = df.groupby(taxalevel, as_index=False).sum()
    temp = taxadatasum[[taxalevel]]
    temp.columns = ['Header']
    temp.insert(0, 'Taxalevel', taxalevel)
    
    fractions = taxadatasum.iloc[:,1:].div(taxadatasum.iloc[:,1:].sum(axis=0), axis=1)
    percentages = fractions * 100

    results = pd.concat([temp,percentages], axis=1)
    formatted = results.append(pd.Series(''), ignore_index = True) 
    return formatted 



def main():
    parser=argparse.ArgumentParser(description='calculate relative abundances', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_csv_file', type=str, help='csv file with counts')
    parser.add_argument('--outfile', type=str, help='output file, defaults to input name _relative_abundances.csv') 
    args=parser.parse_args()

    if args.outfile==None:
        args.outfile=''.join([path.splitext(args.input_csv_file)[0],'_relative_abundances.csv'])

    print ("...")
    print ("...starting to process...")
    print ("...")

    ##readin data table
    df=pd.read_csv(args.input_csv_file)

    phylumdata=percentages(df,'Phylum')
    classdata=percentages(df,'Class')
    orderdata=percentages(df,'Order')
    familydata=percentages(df,'Family')
    genusdata=percentages(df,'Genus')
    speciesdata=percentages(df,'SpeciesCluster')
    
    results = pd.concat([phylumdata,classdata,orderdata,familydata,genusdata,speciesdata],ignore_index=False)

    results.to_csv(args.outfile, index=False, header=True)

    print ("...")
    print ("Done!")
    print('Time-taken: {}'.format(datetime.now()- start_time))
    print ("...")


if __name__ == '__main__':
    main()


