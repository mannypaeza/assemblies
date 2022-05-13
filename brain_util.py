import brain
import numpy as np
import random
import copy
import pickle

from collections import OrderedDict

# Save obj (could be Brain object, list of saved winners, etc) as file_name
def sim_save(file_name, obj):
	with open(file_name,'wb') as f:
		pickle.dump(obj, f)

def sim_load(file_name):
	with open(file_name,'rb') as f:
		return pickle.load(f)

# Compute item overlap between two lists viewed as sets.
def overlap(a,b):
	return len(set(a) & set(b))

# Compute item overlap between multiple lists viewed as sets.
def overlap_multiple_lists(*args):
	a = set(args[0])
	for i in range(1, len(args)):
		a = a & set(args[i])
	return len(a)

# Compute overlap of each list of winners in winners_list 
# with respect to a specific winners set, namely winners_list[base]
def get_overlaps(winners_list,base,percentage=False):
	overlaps = []
	base_winners = winners_list[base]
	k = len(base_winners)
	for i in range(0,len(winners_list)):
		o = overlap(winners_list[i],base_winners)
		if percentage:
			overlaps.append(float(o)/float(k))
		else:
			overlaps.append(o)
	return overlaps

