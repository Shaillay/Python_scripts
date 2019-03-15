
#
#
# from MOCAT output of mOTU counts, annotate to various taxa level
#
# input:
# 1) data table (mOTU-counts) with species annotation
# 2) annotation lookup file: mOTU-LG.v1.annotations.txt (installed with MOCAT & also available here: http://vm-lux.embl.de/~kultima/share/mOTU/mOTU.v1.padded.tar.gz
# http://mocat.embl.de/download.html
#
# output file is in a format that can be converted to biom 
#
#


import pandas as pd
import numpy as np
import argparse
from os import path



def main():
    parser=argparse.ArgumentParser(description='annotate taxa levels & format for biom conversion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_tsv_file', type=str, help='tab-separated file with counts')
    parser.add_argument('--outfile', type=str, help='output file, defaults to inputfilename_for_biom.tsv') 
    args=parser.parse_args()

    if args.outfile==None:
        args.outfile=''.join([path.splitext(args.input_tsv_file)[0],'_for_biom.tsv'])

    print ("...")
    print ("...starting to process...")
    print ("...")



    # read taxonomy file
    df=pd.read_csv('mOTU-LG.v1.annotations.txt', sep="\t")

    # reorder columns
    new_order = [7,1,2,3,4,5,6,0]
    df = df[df.columns[new_order]]

    # read abundances file
    df1=pd.read_csv(args.input_tsv_file, sep="\t", skiprows=4)
    df1.columns = df1.columns.str.replace('Unnamed: 0', 'SpeciesCluster')

    # vlookup like function (i.e. add taxonomy to results)
    pdmerge=pd.merge(df1, df, how='left', on='SpeciesCluster')

    # data columns only
    df3=pdmerge.iloc[:,0:(len(pdmerge.columns)-7)] 

    # Coresponding taxonomy in biom style
    fulltax=[str("k__")+pdmerge["Superkingdom"].map(str)+";p__"+pdmerge["Phylum"].map(str)+";c__"+pdmerge["Class"].map(str)+";o__"+pdmerge["Order"].map(str)+";f__"+pdmerge["Family"].map(str)+";g__"+pdmerge["Genus"].map(str)+";s__"+pdmerge["SpeciesCluster"].map(str), pdmerge["SpeciesCluster"]]

    fulltax1=pd.DataFrame(fulltax)
    fulltax2=fulltax1.T

    # Final file
    final=pd.merge(fulltax2,df3, how='inner', on='SpeciesCluster')
    final.drop('SpeciesCluster', axis=1, inplace=True)
    final.columns = final.columns.str.replace('Unnamed 0', 'taxonomy')

    # Add otu number to the column OTU ID
    final['#OTU ID']=range(1,len(final)+1)

    # move first column to end
    cols=final.columns.tolist()
    cols.insert(0,cols.pop(-1))
    final=final[cols]
	
    # Last column to first
    #final[,c(which(colnames(final)=="#OTU ID"),which(colnames(final)!="#OTU ID"))]

    #Make a list of all of the columns in the df
    cols=list(final.columns.values)
    #Remove a given column from list
    cols.pop(cols.index('taxonomy'))
    final=final[cols+['taxonomy']]

    # create a fake dataframe with same column headers and two rows
    fakepd = pd.DataFrame(columns=[list(final.columns.values)], index=['1','2'])
    fakepd.at['1','#OTU ID']="# Constructed from biom file"

    # add extra line with the header values (to hack the pd.concat function for the concatenate)
    fakepd.at['2',:]=list(final.columns.values)

    # header = false is important
    finalformat=pd.concat([fakepd,final])
    finalformat.to_csv('tsv_file_for_biom.txt', index=False, header=False, sep="\t")


    print ("...")
    print ("Done!")
    print ("...")


if __name__ == '__main__':
    main()


