#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Mon May 22 06:20:21 2017

for wrongly-connected droplets:
this script deals with the data
and plots it out using graph-tools

@author: dogrask
"""


import pandas as pd
import graph_tool.all as gt
import matplotlib.pyplot
from os import path 
import argparse



def main():
    parser=argparse.ArgumentParser(description='Plot graph',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_csv_file', type=str, 
                      help='input file.  Must have barcode1, barcode2, dropletIDs, comblocks, biconnected.  Each line is an edge. \
                      Additional edge properties are in additional columns in the file')
    parser.add_argument('integers', metavar='int', type=int, nargs='+', 
                        help='droplet IDs: wrongly split droplet IDs; example input: 200000 50 100')
    args=parser.parse_args()
    

    print "\n \n.... starting... \n \n"
    #print args.integers
    dropletIDs = args.integers
    df=pd.DataFrame.from_csv(args.input_csv_file, index_col=None)
    ######df_filteredbydID=df.filter(lambda x: x[2] in dropletIDs, axis=1)
    ######print df_filteredbydID
    
    
    #print len(df) 
    df=df[['barcode1', 'barcode2', 'dropletIDs', 'edges', 'comblocks', 'biconnected']]
    dfset=df.loc[df['dropletIDs']==99999999]    ### creates an empty dataframe this way as there is no dropletID 99999999
    #print dfset
    filestr='_'
    for id in dropletIDs:
        #print id
        filestr = ''.join([filestr, str(id), '_'])  # to be used later for file name
        dftmp = df.loc[df['dropletIDs']==id]
        #print len(dftmp)
        dfset = pd.concat([dfset,dftmp])

    #print len(filestr)
    if len(filestr)>50:
        filestr=''.join([filestr[:50],'_fname-truncated_'])
        print "\n \n.... filename will be too long.... truncating it... \n "
    #print len(dfset)
    #print dfset.head(20)
    #print dfset.tail(20)
    #print dfset
    
    print filestr
 
    print "\n \n \n...writing out files that can be checked... \n "
    outfile1 = ''.join([path.splitext(args.input_csv_file)[0],filestr, 'dfset.csv'])
    print outfile1
    dfset.to_csv(outfile1, index=False)
    print "\n ... file has been written out... \n \n \n"   

       

    ######dfsetfill = dfset.fillna(99999999)
    dfsetdrop = dfset.dropna()
    #print dfsetdrop.head(20)
    #print dfsetdrop.tail(20)
    
    dfsetdrop['comblocks'] = dfsetdrop['comblocks'].str.replace('u','')
    dfsetdrop['comblocks'] = dfsetdrop['comblocks'].str.replace('d','')
    #print dfsetdrop

    print "\n \n ...writing out files that can be checked... \n "
    outfile2 = ''.join([path.splitext(args.input_csv_file)[0],filestr, 'dfsetdrop.csv'])
    print outfile2
    dfsetdrop.to_csv(outfile2, index=False)
    print "\n ... file has been written out... \n \n \n"   


    checklist = dropletIDs 
    if 200000 in checklist:
        checklist.remove(200000)   # remove the droplet ID = 200000, is the contaminating one, can also belong to different biconnectedIDs, we dont want that case
    #print checklist
 
    bcid=[]
    for dID in checklist:
        dftmp = dfsetdrop.loc[dfsetdrop['dropletIDs']==dID]
        #print dftmp['biconnected']
        id =  dftmp['biconnected'].mean()    # there should be only one biconnectedID for the dropletID here; and it should be same one for different dropletIDs here in this subset of data
        bcid.append(id)

    if min(bcid) == max(bcid):  # this should match since there should be only one biconnectedID corresponding to different dropletIDs in the data subset here
        #print min(bcid)
        #print max(bcid)
        subsetbcid = min(bcid)
    else: print "\n ..... There is a Problem!..... The Biconnected IDs do not match...! \n  ... The script breaks down here.... \n \n \n"
        
    
    dfsetplot = dfsetdrop.loc[dfsetdrop['biconnected']==subsetbcid] # just take the data that belongs to one single biconnected ID
    #print dfsetplot

    print "\n \n ...writing out files that can be checked... \n"
    outfile3 = ''.join([path.splitext(args.input_csv_file)[0],filestr, 'dfsetplot.csv'])
    print outfile3
    dfsetplot.to_csv(outfile3, index=False)
    print "\n ... file has been written out... \n \n \n"   
     
    
    g=gt.Graph(directed=False)
    ndarray=dfsetplot.values.tolist()
    #print ndarray   
    dIDs=g.new_edge_property('int32_t')
    edgethickness=g.new_edge_property('int32_t')
    comblocks=g.new_edge_property('int32_t')
    biconnected=g.new_edge_property('int32_t')
    name=g.add_edge_list(ndarray,hashed=True,string_vals=True, eprops=[dIDs,edgethickness,comblocks,biconnected])

    outfile_dID=''.join([path.splitext(args.input_csv_file)[0],filestr, 'colorby_dropletID', '.png'])
    outfile_cblk=''.join([path.splitext(args.input_csv_file)[0],filestr, 'colorby_combblocks', '.png'])
    outfile_bicon=''.join([path.splitext(args.input_csv_file)[0],filestr, 'colorby_biconnected', '.png'])



    print "\n \n ...writing out graph plots... \n"
    
    #graph_tool.draw.prop_to_size(prop, mi=0, ma=5, log=False, power=0.5)
    #pos = gt.sfdp_layout(g)
    #gt.graph_draw(g, pos=pos, output_size=(5000, 5000), edge_color=dIDs, vertex_size=100, vertex_shape='square', vertex_fill_color='red', vertex_halo=True, vertex_halo_color='green', vertex_halo_size=1.1, vertex_pen_width=20, edge_pen_width=20, output="graph-draw-sfdp1.png")
    #gt.graph_draw(g, pos=pos, output_size=(5000, 5000), edge_color=biconnected, vertex_size=100, vertex_shape='square', vertex_fill_color='red', vertex_halo=True, vertex_halo_color='green', vertex_halo_size=1.1, vertex_pen_width=20, edge_pen_width=20, output="graph-draw-sfdp2.png")
    print outfile_dID
    gt.graph_draw(g, output_size=(5000, 5000), vertex_fill_color='red', edge_color=dIDs, edge_pen_width=50, output=outfile_dID)
    print "\n ... file has been written out... \n \n"   

    #gt.graph_draw(g, output_size=(5000, 5000), edge_color=dIDs, edge_pen_width=edgethickness, output="dropletID-edgecounts.png")
    print outfile_cblk
    gt.graph_draw(g, output_size=(5000, 5000), vertex_fill_color='blue', edge_color=comblocks, edge_pen_width=50, output=outfile_cblk)
    print "\n ... file has been written out... \n \n"   

    print outfile_bicon
    gt.graph_draw(g, output_size=(5000, 5000), vertex_fill_color='green', edge_color=biconnected, edge_pen_width=50, output=outfile_bicon)
    print "\n ... file has been written out... \n \n"   



    print "\n \n..... done .... \n \n"

    #ndarray2=df[['dropletIDs','comblocks']].values.tolist()
    #comblocks2=g.new_vertex_property('int32_t')
    #dIDs2=g.new_vertex_property('int32_t')
    #name=g.add_vertex_list(ndarray2,hashed=True,string_vals=True, vprops=[dIDs2,comblocks2])
    #gt.graph_draw(g, output_size=(5000, 5000), vertex_color=dIDs2, vertex_shape=comblocks2, edge_color=biconnected, edge_pen_width=20, output="all.png")
    #pos = gt.fruchterman_reingold_layout(g, n_iter=1000)
    #gt.graph_draw(g, pos=pos, output="graph-draw-fr.pdf")
    #pos = gt.arf_layout(g, max_iter=0)
    #gt.graph_draw(g, pos=pos, output="graph-draw-arf.pdf")
 
    
       

if __name__ == '__main__':
    main()