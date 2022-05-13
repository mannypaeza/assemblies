import numpy as np
import overlap_sim

import brain
import brain_util as bu
import numpy as np
import copy
import simulations

import sys 
import pandas as pd
import random

n=100000
k=317
p=0.05
project_iter=10 
beta_list = [0.3,0.2,0.1,0.075,0.05,0.03,0.01,0.007,0.005] 
n_areas_list = [2,3,4,5,6,7,8] # number of areas not including the target area (i.e., the area of association)
n_firings_in_assoc_list = [0,1,3,5]
n_runs_per_experiment = 1


def firing_neurons_multiple_areas_associated_separate(n,k,p,beta,project_iter,n_areas,n_firing_areas,n_firings_in_assoc,df):

    b = brain.Brain(p,save_winners=True)

    # area in which we will associate the assemblies
    target_area = str(chr(64+(n_areas+1)))

	# assembly we care about (the person we want to remember)
    important_area = str(chr(64+(n_areas)))
    important_stim_name = "stim" + important_area

    total_stim_dict = {}
    total_area_dict = {}

    # add stimuli and areas in brain b
    for i in range(1, n_areas+1):	
        area_name = str(chr(64+i))
        stim_name = "stim" + area_name
        b.add_stimulus(stim_name,k)
        b.add_area(area_name,n,k,beta)
        total_stim_dict[stim_name] = [area_name]
        total_area_dict[area_name] = [area_name]
    b.add_area(target_area,n,k,beta)
	
    b.project(total_stim_dict,{}) # stimuli projection
    # Create assemblies in each area to stability
    for i in range(0,9):
        b.project(total_stim_dict, total_area_dict)

    # Add target area in lists of area_dict
    for key, value in total_area_dict.items():
        total_area_dict[key].append(target_area)

    # Project the assembly of each area to the target area
    for i in range(1, n_areas+1):
        area_name = str(chr(64+i))
        stim_name = "stim" + area_name
        stim_dict = {stim_name:[area_name]}
        area_dict = {}
        area_dict[area_name] = total_area_dict[area_name]
        b.project(stim_dict,area_dict)
        area_dict[target_area] = [target_area]
        for j in range(0,9):
            b.project(stim_dict,area_dict)
    
    # associate the assemblies in the target area
    # winners_from_interesting_area = the projection of the assembly we are interested in in the target area after the association
    overlap, winners_from_interesting_area = simulations.association_grand_sim_multiple_areas_separate(b,n,k,p,beta,10,20,n_areas,n_firings_in_assoc,df)

    # fire the assemblies that represent the attributes
    final_stim_dict = {}
    final_area_dict = {}
    if n_firing_areas == n_areas-1:
        for i in range(1, n_areas):
            area_name = str(chr(64+i))
            stim_name = "stim" + area_name
            final_stim_dict[stim_name] = [area_name]
            final_area_dict[area_name] = [area_name, target_area]
    else:
        firing_areas = random.sample(range(1, n_areas), n_firing_areas)
        for i in firing_areas:
            area_name = str(chr(64+i))
            stim_name = "stim" + area_name
            final_stim_dict[stim_name] = [area_name]
            final_area_dict[area_name] = [area_name, target_area]
    b.project(final_stim_dict, final_area_dict)

    total_overlap = {}
    winners = b.areas[target_area].winners
    overlap_with_assembly_of_interest = bu.overlap(winners, winners_from_interesting_area)
    total_overlap[0] = float(overlap_with_assembly_of_interest)/float(k)

    # keep projecting the target area to itself
    i = 1
    b.project({},{target_area: [target_area]})
    winners = b.areas[target_area].winners
    overlap_with_assembly_of_interest = bu.overlap(winners, winners_from_interesting_area)
    total_overlap[i] = float(overlap_with_assembly_of_interest)/float(k)
    while True:
        b.project({},{target_area: [target_area]})
        winners = b.areas[target_area].winners
        overlap_with_assembly_of_interest = bu.overlap(winners, winners_from_interesting_area)
        i += 1
        total_overlap[i] = float(overlap_with_assembly_of_interest)/float(k)
        if abs( (total_overlap[i-1] - total_overlap[i]) / float(total_overlap[i-1]) ) <= 0.0005:
            break
    
    df['overlap_with_ass_interest_after_1_firing'] += total_overlap[1]
    df['#firings_till_convergence'] += i
    df['overlap_with_ass_interest_upon_convergence'] += total_overlap[i]
    all_overlaps = total_overlap.values()
    max_overlap = max(all_overlaps)
    df['max_overlap_with_ass_interest'] += max_overlap
    return df

if __name__ == "__main__":
    df = pd.DataFrame(columns=['beta', '#areas', '#firing_areas', '#firings_in_assoc',
                                'avg_pairwise_assoc_overlap', 'total_assoc_overlap', 
                                'overlap_with_ass_interest_after_1_firing', '#firings_till_convergence', 
                                'overlap_with_ass_interest_upon_convergence','max_overlap_with_ass_interest'])
    df.to_csv (r'separate.csv', index = False, header=True)
    for beta in beta_list:
        for n_areas in n_areas_list:
            for n_firing_areas in range(1, n_areas):
                for n_firings_in_assoc in n_firings_in_assoc_list:
                    df2 = {}
                    df2['#areas'] = n_areas
                    df2['beta'] = beta
                    df2['#firing_areas'] = n_firing_areas
                    df2['#firings_in_assoc'] = n_firings_in_assoc
                    df2['avg_pairwise_assoc_overlap'] = 0
                    df2['total_assoc_overlap'] = 0
                    df2['overlap_with_ass_interest_after_1_firing'] = 0
                    df2['#firings_till_convergence'] = 0
                    df2['overlap_with_ass_interest_upon_convergence'] = 0
                    df2['max_overlap_with_ass_interest'] = 0

                    for i in range(n_runs_per_experiment):
                        df2 = firing_neurons_multiple_areas_associated_separate(n,k,p,beta,project_iter,n_areas,n_firing_areas,n_firings_in_assoc,df2)
        
                    df2['avg_pairwise_assoc_overlap'] = float(df2['avg_pairwise_assoc_overlap'])/float(n_runs_per_experiment)
                    df2['total_assoc_overlap'] = float(df2['total_assoc_overlap'])/float(n_runs_per_experiment)
                    df2['overlap_with_ass_interest_after_1_firing'] = float(df2['overlap_with_ass_interest_after_1_firing'])/float(n_runs_per_experiment)
                    df2['#firings_till_convergence'] = float(df2['#firings_till_convergence'])/float(n_runs_per_experiment)
                    df2['overlap_with_ass_interest_upon_convergence'] = float(df2['overlap_with_ass_interest_upon_convergence'])/float(n_runs_per_experiment)
                    df2['max_overlap_with_ass_interest'] = float(df2['max_overlap_with_ass_interest'])/float(n_runs_per_experiment)
        
                    df = pd.DataFrame(columns=['beta', '#areas', '#firing_areas', '#firings_in_assoc',
                                                'avg_pairwise_assoc_overlap', 'total_assoc_overlap', 
                                                'overlap_with_ass_interest_after_1_firing', '#firings_till_convergence', 
                                                'overlap_with_ass_interest_upon_convergence','max_overlap_with_ass_interest'])
                
                    df = df.append(df2, ignore_index = True)
                    df.to_csv (r'separate.csv', mode='a', index=False, header=False)

