#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 04:54:48 2018

@author: admin
"""
import numpy as np
import pandas as pd


### "chances" argument must be an descending sorted list of the number of 
# lottery combinations awarded to each team/seed
chances = [250.0, 199.0, 156.0, 119.0, 88.0, 63.0, 43.0, 28.0, 17.0, 11.0, 
           8.0, 7.0, 6.0, 5.0, 0.0, 0.0] # Old System
chances = [140., 140., 140., 125., 105.,  90.,  75.,  60.,  45.,  30.,  20., 
           15.,  10.,   5.,   0.,   0.] # Current System
chances = [114., 113., 112., 111.,  99.,  89.,  79.,  69.,  59.,  49.,  39., 
           29.,  19.,   9.,   6.,   4.] # Hypothetical System

### "n_rounds" argument must be an integer
n_rounds = 3





def getLotteryProbs(chances, n_rounds):

# Add check if chances sums to 1000
# Add check if chances is a list

    n_seeds = len(chances)
    temp = np.stack((np.arange(1,n_seeds+1), np.array(chances)), axis=-1)
    chart = pd.DataFrame(temp, columns = ['seed', 'chances'])
    #chances_mat = chart.seed.values.reshape(n_seeds,1) 

    seed_permus = {}
    win_chances_xtab = {}  # seed_vs_prev_win_scenarios_pick
    map = {}
    for round in range(n_rounds+1):
    
        rnd = 'rnd {}'.format(round)
        
        if round == 0:
            chart['rnd 1'] = chart.chances/1000
            win_chances_xtab['rnd 1'] = chart.chances.values.reshape(16,1)/1000  
    
        else:

            # create all permutations of the seeds to mimic sequence of pick winners
            temp = tuple(chart['seed'].values for _ in range(round))
            # build a df to hold all the permutations of winners in preceeding rounds 
            # note: star before temp to unpack the passed tuple
            seed_permus[rnd] = np.array(np.meshgrid(*temp)).T.reshape(-1, round)
        
            # delete rows where a seed is repeated
            for col in range(seed_permus[rnd].shape[1]):
                # check if no seed is repeated in a given row, by comparing each
                # col to all others one at a time.
                mask_1 = [pos for pos in range(seed_permus[rnd].shape[1]) if col != pos]
                # apparently only rows can be broadcast, hence the transposition..
                mask_2 = ~np.any((seed_permus[rnd].T[col, :] == seed_permus[rnd].T[mask_1, :]), axis = 0) 
                seed_permus[rnd] = seed_permus[rnd][mask_2, :]

            mask = ['pick {} winner'.format(seed+1) for seed in range(round)]
            seed_permus[rnd] = pd.DataFrame(seed_permus[rnd], columns= mask)
 
            map_seed = chart.set_index('seed').to_dict()


            if round <= (n_rounds-1):
                
                for col in range(round):
                    
                    # map each seeds lottery combos to the table of all 
                    # possible winner sequences
                    mask_1 = 'round {} winner chances'.format(col+1)
                    mask_2 = 'pick {} winner'.format(col+1)
                    seed_permus[rnd][mask_1] = seed_permus[rnd][mask_2].map(map_seed['chances'])
                  
                if round == 1:
                    #column 1 - the only column in seed_permus['rnd 1']
                    mask_1 = 'P(round 1 win | previous winners)'
                    mask_2 = 'pick 1 winner'
                    seed_permus[rnd][mask_1] = seed_permus[rnd][mask_2].map(map_seed['rnd 1'])
                
                
                elif round == 2:
                    
                    #column 1 (i.e. col = 0)
                    mask_1 = 'P(round 1 win | previous winners)'
                    mask_2 = 'pick 1 winner'
                    seed_permus[rnd][mask_1] = seed_permus[rnd][mask_2].map(map_seed['rnd 1'])
                    
                    #final columna (i.e. col = 1)
                    def map_final(group):
                       
                        mask = ['pick {} winner'.format(col+1) for col in range(round-1)]
                        target = group[mask]
                        tgt_indx = target.index
                        # get group key/name in round about way 
                        idx = int(np.setdiff1d(chart.seed.values, target.values.flatten()[:260]))
                        source = seed_permus['rnd {}'.format(round-1)][mask]
                        mask_1 = source.shape[0]
                        mask_2 = pd.concat([target, source]).duplicated(keep= False)[-mask_1:].values
                        return pd.Series(win_chances_xtab[rnd][idx-1 , mask_2], index = tgt_indx) 
                    
                    mask_1 = 'P(round {} win | previous winners)'.format(round)
                    mask_2 = 'pick {} winner'.format(round)
                    temp = seed_permus[rnd].groupby(mask_2).apply(map_final).reset_index(level = mask_2, drop=True)
                    temp.name = mask_1
                    seed_permus[rnd] = seed_permus[rnd].join(temp)
                    
                    
                    # mid columns
                    def map_mid(row):
                        winner_seq = row.values
                        winner_seq = winner_seq[:col+1]
                        return map['rnd {}'.format(col+1)][tuple(winner_seq)]
               
                else:
                    
                    #column 1 (i.e. col = 0)
                    mask_1 = 'P(round 1 win | previous winners)'
                    mask_2 = 'pick 1 winner'
                    seed_permus[rnd][mask_1] = seed_permus[rnd][mask_2].map(map_seed['rnd 1'])
                    
                    
                    
                    
                    for col in range(1, round-1): # to avoid final col and initial col 
                        mask_1 = 'P(round {} win | previous winners)'.format(col+1)
                        # includes more than necessary but cols get cut inside the function 
                        mask_2 = ['pick {} winner'.format(col+1) for col in range(round)] 
                        seed_permus[rnd][mask_1] =  seed_permus[rnd][mask_2].apply(map_mid, axis = 1)
                        
                    #final col (col = round-1)
                    mask_1 = 'P(round {} win | previous winners)'.format(round)
                    mask_2 = 'pick {} winner'.format(round)
                    temp = seed_permus[rnd].groupby(mask_2).apply(map_final).reset_index(level = mask_2, drop=True)
                    temp.name = mask_1
                    seed_permus[rnd] = seed_permus[rnd].join(temp)
                
                    
                    
                    
                # make mapping dicts from the now created permutation tables
                mask = ['pick {} winner'.format(seed+1) for seed in range(round)]
                map[rnd] = seed_permus[rnd].set_index(mask).to_dict()
                map[rnd] = map[rnd]['P(round {} win | previous winners)'.format(round)]
        
                #mat1 - probabilities of each pick sequence if the nth pick is picking third
                mask = ['P(round {} win | previous winners)'.format(seed+1) for seed in range(round)]
                winner_seq_probs = seed_permus[rnd].loc[:, mask].values.prod(axis = 1)
                winner_seq_probs_mat = np.tile(winner_seq_probs, n_seeds).reshape(n_seeds, len(winner_seq_probs))

                # must zero out all probs where the nth seed (seed we are evaluating in a 
                # given row) is in the preceeding winner sequence in any position
                for n in chart.seed.values:
                    mask_1 = ['pick {} winner'.format(seed+1) for seed in range(round)]
                    mask_2 = (seed_permus[rnd][mask_1] == n).any(axis = 1)
                    winner_seq_probs_mat[int(n)-1, mask_2.values] = 0 
                    # seem to only have added 364 zeros... why not 30*16 = 480? Because
                    # of over lap each nonzero line 58 zeros to start, and only 26 are 
                    # added per (non zero) row


                # mat 2  - seed_vs_prev_win_prob of winning scenarios
                # lottery combos invalidated by pick sequence
                mask = ['round {} winner chances'.format(seed+1) for seed in range(round)]
                combos_used = seed_permus[rnd].loc[:, mask].values.sum(axis = 1)
                # lottery combos remaining to be picked
                combos_remaining = 1000 - combos_used
                combos_remaining_mat = np.tile(combos_remaining, n_seeds).reshape(n_seeds, len(combos_remaining))
                # rows = nth seed, cols lott combos remaining for each permutation
                win_chances_xtab['rnd {}'.format(round+1)] = chart.chances.values.reshape(16,1)/combos_remaining_mat

                temp = np.dot(winner_seq_probs_mat, win_chances_xtab['rnd {}'.format(round+1)].T)
                chart['rnd {}'.format(round+1)] = np.diagonal(temp)
            
            else:
                # build conditional probabilities permutation table (all 
                # possible winner sequences) for the final round
               
                # the following assumes that n_rounds >= 3
                
                #column 1 (i.e. col = 0)
                mask_1 = 'P(round 1 win | previous winners)'
                mask_2 = 'pick 1 winner'
                seed_permus[rnd][mask_1] = seed_permus[rnd][mask_2].map(map_seed['rnd 1'])
                
                # mid columns
                for col in range(1, round-1): # to avoid final col and initial col 
                    mask_1 = 'P(round {} win | previous winners)'.format(col+1)
                    # includes more than necessary but cols get cut inside the function 
                    mask_2 = ['pick {} winner'.format(col+1) for col in range(round)] 
                    seed_permus[rnd][mask_1] =  seed_permus[rnd][mask_2].apply(map_mid, axis = 1)
                        
                #final col (col = round-1)
                mask_1 = 'P(round {} win | previous winners)'.format(round)
                mask_2 = 'pick {} winner'.format(round)
                temp = seed_permus[rnd].groupby(mask_2).apply(map_final).reset_index(level = mask_2, drop=True)
                temp.name = mask_1
                seed_permus[rnd] = seed_permus[rnd].join(temp)
                
                

    # create sorted lists of seeds NOT included in each final winner sequence
    # and append to end of df
    name = list(seed_permus.keys())[-1] # get last created seed_permus df
    mask_1 = ['pick {} winner'.format(seed+1) for seed in range(round)]

    def not_picked(row):
        return np.setdiff1d(chart.seed.values, row.values, assume_unique = True).tolist()

    dat = seed_permus[name].loc[:, mask_1].apply(not_picked, axis =1, raw=False)
    mask = ['remain_{}'.format(col+1) for col in range(n_seeds-n_rounds)]
    seed_permus[name][mask] = pd.DataFrame(dat.values.tolist())
    
    # get probability of being 1st, 2nd,...etc NON-lottery pick, put in new df,
    # and concat lottery probs df with non-lottery probs df
    
    remaining = np.zeros((n_seeds, n_seeds-n_rounds))
    def reduce(group):
        mask = ['P(round {} win | previous winners)'.format(col+1) for col in range(n_rounds)]
        return group.loc[:, mask].prod(axis=1).sum(axis=0)

    for n in range(n_seeds-n_rounds):
        remaining[n:n+(n_rounds+1), n] = seed_permus[name].groupby('remain_{}'.format(n+1)).apply(reduce)
 
    mask = ['rnd {}'.format(col+n_rounds+1) for col in range(n_seeds-n_rounds)]
    chart[mask] = pd.DataFrame(remaining)
    
    return chart


###########################

%%time
chart = getLotteryProbs(chances, n_rounds = 3)
# old map probs
# CPU times: user 11.3 s, sys: 203 ms, total: 11.5 s; Wall time: 11.3 s
# looping over apply without passing additional args
# CPU times: user 8.47 s, sys: 108 ms, total: 8.58 s; Wall time: 8.5 s
# streamlined version!!!!
# CPU times: user 565 ms, sys: 8.26 ms, total: 573 ms; Wall time: 571 ms


%%time
chart = getLotteryProbs(chances, n_rounds = 4)
# old map probs
# CPU times: user 3min 13s, sys: 1.69 s, total: 3min 15s: Wall time: 3min 14s
# looping over apply without passing additional args
# CPU times: user 1min 54s, sys: 543 ms, total: 1min 55s; Wall time: 1min 55s
# streamlined version!!!!
# CPU times: user 5.03 s, sys: 82.3 ms, total: 5.12 s; Wall time: 5.11 s


chances = [114., 113., 112., 111.,  99.,  89.,  79.,  69.,  59.,  49.,  39., 
           29.,  19.,   9.,   6.,   4.] # Hypothetical System
%%time
chart = getLotteryProbs(chances, n_rounds = 5)
#CPU times: user 1min 8s, sys: 1.69 s, total: 1min 10s; Wall time: 1min 10s





#############


# Save steps
chart.to_csv('/Users/admin/Desktop/Data Projects/BLA Assesment/Hypothetical System Cond. Prob. Table.csv')
np.savetxt('/Users/admin/Desktop/Data Projects/BLA Assesment/xtab_2_auto.csv', win_chances_xtab['rnd 2'] , delimiter = ',')
np.savetxt('/Users/admin/Desktop/Data Projects/BLA Assesment/xtab_3_auto.csv', win_chances_xtab['rnd 3'] , delimiter = ',')
np.savetxt('/Users/admin/Desktop/Data Projects/BLA Assesment/seq_probs_3_auto.csv', winner_seq_probs_mat, delimiter = ',')
