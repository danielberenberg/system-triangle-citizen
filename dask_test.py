#!/usr/bin/env python
# -*- coding: utf-8 -*-
import joblib
from dask.distributed import Client, LocalCluster, wait, fire_and_forget, progress, get_client
from dask_jobqueue import SLURMCluster

from time import sleep
from operator import add

import numpy as np
import dask

@dask.delayed
def slow_increment(x):
    sleep(1)
    return x + 1

@dask.delayed
def slow_rand(x):
    sleep(1)
    return np.random.random()

def step1(N):
    client = get_client()
    rands = []
    for i in range(N):
        rand = client.submit(slow_rand, i)
        rands.append(rand)
    return [r.result() for r in rands]

@dask.delayed
def top_k(it,k):
    return dask.delayed(sorted)(it)

delayed_add = dask.delayed(add)

def step2(top):
    client = get_client()
    elts = []
    for elt in top:
        rand = client.submit(slow_rand, elt)
        chg = client.submit(add, elt, rand)
        elts.append(chg)
    return elts

def peel(x):
    return x[0]

def tiered_slow_increment(N, k):
    client = get_client()
    rands = client.submit(step1, N)
    top = client.submit(top_k, rands, k)
    chg  = client.submit(step2, top)
    top1 = client.submit(top_k, chg, 1)
    final = client.submit(peel, top1)
    return final.result()

if __name__ == '__main__':
    print("Making a cluster")
    cluster = LocalCluster()
    print("Making a worker")
    client = Client(cluster)
    cluster.scale(20)
    
    results = dask.delayed([])
    print("Starting ...")
    for i in range(100):  # for each npz
        futs = dask.delayed([])
        for j in range(150): # make 150 centroid models
            fut = slow_increment(slow_rand(j))
            futs.append(fut)
        top = top_k(futs, 50)
        tfuts= dask.delayed([])
        for t in top: # relax them
            tfut = delayed_add(slow_increment(t), slow_rand(t))
            tfuts.append(tfut)
        top1 = top_k(tfuts, 1)
        results.append(top1)

    print("Done!")

    finals = client.gather(results)
    print(finals)
