'''
This module is for evaluation on link prediction result
'''

import logging
import numpy as np
import torch
from torch.nn import functional as F
from random import sample


log = logging.getLogger()


def ranking_and_hits(model, dev_rank_batcher, batch_size, name, isSilent=False, testSizeForBatchTuple=2, mode="filter"):
    '''
    Evaluate Mean rank and Hits@10 based on data in dev_rank_batcher whose size is batch_size
    :param model: torch module, trained model
    :param dev_rank_batcher: DataStreamer object, including tuples to evaluate
    :param batch_size: int, the size of evaluation batch
    :param name: string, to format the display message
    :param isSilent: boolean, False means to verbose the evaluation result
    :param testSizeForBatchTuple: int; each input (e1, rel) may have multiple positive entities, this parameter is the number
    of positive entities to sample out for evaluation. It's useful for 'raw' evaluation mode.
    :param mode: String, either "filter" or "raw" to represent different evaluation modes
    :return: float, mean rank
    '''
    printDisplayMessage(name)

    # initialize
    hits = []
    ranks = []

    for i, str2var in enumerate(dev_rank_batcher):  # iter over batcher containing multiple batches
        e1, e2, e2_multi1, e2_multi2, rel, rel_reverse = loadInfoFromBatcher(str2var)
        pred1, pred2 = predictProbability(e1, e2, model, rel, rel_reverse)

        # get the data
        pred1, pred2 = pred1.data, pred2.data
        e1, e2 = e1.data, e2.data
        e2_multi1, e2_multi2 = e2_multi1.data, e2_multi2.data

        # sort the prediction tensor along the row
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)

        for j in range(batch_size):  # iterate over batch with the size of 128
            # find the positive entities
            postiveEntities1 = e2_multi1[j][e2_multi1[j] != -1].long().tolist()
            postiveEntities2 = e2_multi2[j][e2_multi2[j] != -1].long().tolist()

            if mode == "raw":
                # sample positive entities to save evaluation time
                if len(postiveEntities1) > testSizeForBatchTuple:
                    testPostiveEntities1 = sample(postiveEntities1, testSizeForBatchTuple)
                if len(postiveEntities2) > testSizeForBatchTuple:
                    testPostiveEntities2 = sample(postiveEntities2, testSizeForBatchTuple)
            elif mode == "filter":
                # save the prediction that is relevant
                target_value1 = pred1[j, e2[j, 0]].item()
                target_value2 = pred2[j, e1[j, 0]].item()
                # make all training tuples to zero: this corresponds to the filtered setting
                pred1[j][postiveEntities1] = 0.0
                pred2[j][postiveEntities2] = 0.0
                # only keep the relevant target values
                pred1[j][e2[j]] = target_value1
                pred2[j][e1[j]] = target_value2
                # remove other values from the test list
                postiveEntities1 = [e2[j, 0]]
                postiveEntities2 = [e1[j, 0]]

            # find the rank of the target entities
            hit1 = 0
            for pos_e1 in postiveEntities1:
                rank1 = (argsort1[j] == pos_e1).nonzero().item()
                if rank1 <= 9:
                    hit1 += 1
                ranks.append(rank1+1)  # rank+1, since the lowest rank is rank 1 not rank 0
            hit1 /= 10

            hit2 = 0
            for pos_e2 in postiveEntities2:
                rank2 = (argsort2[j] == pos_e2).nonzero().item()  # assume only 1 place can be non-zero
                if hit2 <= 9:
                    hit2 += 1
                ranks.append(rank2+1)  # rank+1, since the lowest rank is rank 1 not rank 0
            hit2 /= 10

            hits.extend([hit1, hit2])

    if not isSilent:
        meanRank = np.mean(ranks)
        hits10 = np.mean(hits)
        log.info('Hits @10: %f', hits10)
        log.info('Mean rank: %f', meanRank)
        # log.info('Mean reciprocal rank: %f', np.mean(1. / np.array(ranks)))

    return meanRank


def predictProbability(e1, e2, model, rel, rel_reverse):
    pred1_ = model.forward(e1, rel)
    pred2_ = model.forward(e2, rel_reverse)

    # sigmoid is applied in loss function, here we do it manually to get positive predictions
    pred1 = F.sigmoid(pred1_)
    pred2 = F.sigmoid(pred2_)
    return pred1, pred2


def loadInfoFromBatcher(str2var):
    e1 = str2var['e1']
    e2 = str2var['e2']
    rel = str2var['rel']
    rel_reverse = str2var['rel_eval']
    e2_multi1 = str2var['e2_multi1'].float()
    e2_multi2 = str2var['e2_multi2'].float()
    return e1, e2, e2_multi1, e2_multi2, rel, rel_reverse


def printDisplayMessage(name):
    log.info('')
    log.info('-' * 50)
    log.info(name)
    log.info('-' * 50)
    log.info('')

