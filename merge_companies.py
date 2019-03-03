#!/usr/bin/python
# -*- coding: utf8

import sys
import os
import fnmatch
import pandas as pd

ccs = sys.argv
ccs.pop(0) ## 先頭(script名)を削除

dataset = pd.DataFrame()
for cc in ccs:
    print(cc)
    dir = "./stock_cc_year/"
    filename = 'stocks_%s_1d_*.csv' % (cc)

    ccdataset = pd.DataFrame()
    for file in os.listdir(dir):
        if fnmatch.fnmatch(file, filename):
            print(file)
            readdata = pd.read_csv(dir + file, index_col=0)
            if len(ccdataset) == 0:
                ccdataset = readdata
            else:
                ccdataset = pd.concat([ccdataset, readdata])

    ccdataset = ccdataset.sort_index()
    zzz = pd.Series(ccdataset.index)
    print(zzz[zzz.duplicated()])

    for i in ccdataset.columns:
        ccdataset.rename(columns={i: cc + "_" + i}, inplace=True)

    if(len(dataset) == 0):
        dataset = ccdataset
    else:
        print(ccdataset)
        print(dataset)
        dataset = pd.concat([dataset, ccdataset], axis=1, sort=False)

#print(dataset)