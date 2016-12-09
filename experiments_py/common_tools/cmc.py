#########################################################
# function evaluateCMC
# the function calculte the Cumulative Matching Curves (CMC)
#
#  Liming Zhao (zlmzju@gmail.com)
#  input: 
#         gtLabels, N dim int vetor, the groundtruth label of N probes, one test image one label;
#         predictLists, list of int vector, contains the predicted label list for each probe
#  output:
#         cmc, R dim vector (N=numOfGallery, rank 1 to rank R);
#              each element is the recognition probability ([0,1]).
##############################################################
import numpy as np

def evaluateCMC( predictLists, key_probes, key_gallerys):
    N=len(key_probes)
    R=len(predictLists[0])
    histogram=np.zeros(N)
    for testIdx in range(1):
        for rankIdx in range(R):
            histogram[rankIdx]+=1*(key_gallerys[predictLists[testIdx][rankIdx]-1]==key_probes[testIdx])    #1*(true or false)=1 or 0
    cmc=np.cumsum(histogram)
    return cmc/N