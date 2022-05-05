# Library of simulations for preservation overlap in assemblies.
# Assembly operations preserve overlap; the larger the overlap
# between two assemblies x,y in area A, the larger we expect the overlap of 
# proj(x,B) and proj(y,B) to be.

# In our efficiently computable sampling-trick simulation, it is not obvious
# how to create the original assemblies x,y with some particular % overlap
# because we do not model the entire brain area (i.e. sampling-trick: we just model
# the amount needed to simulate projections).
# Hence, we use another property of assemblies: association. If assembly x in A and 
# assembly y in B are fired together into C, proj(x,A) and proj(y,C) will have
# more overlap the more x,y were fired together.
# Hence we exhibit overlap preservation in the following procedure: 
# Create assembly x in A
# Create assembly y in B
# proj(x,C) (until stability)
# proj(y,C) (until stability)
# at this point proj(x,C) and proj(y,C) have as much overlap as random chance (i.e. p)
# associate(x,y,C)
# Now proj(x,C) and proj(y,C) have some percentage overlap
# Now with 4th area D: proj(proj(x,C),D) and proj(proj(y,C),D) for 1 time step

import brain
import brain_util as bu
import numpy as np
import copy

def overlap_sim(n=100000,k=317,p=0.05,beta=0.1,project_iter=10):
	b = brain.Brain(p,save_winners=True)
	b.add_stimulus("stimA",k)
	b.add_area("A",n,k,beta)
	b.add_stimulus("stimB",k)
	b.add_area("B",n,k,beta)
	b.add_area("C",n,k,beta)
	b.add_area("D",n,k,0.0) # final project test area
	b.project({"stimA":["A"],"stimB":["B"]},{})
	# Create assemblies A and B to stability
	for i in range(0, 9):
		b.project({"stimA":["A"],"stimB":["B"]},
			{"A":["A"],"B":["B"]})
	b.project({"stimA":["A"]},{"A":["A","C"]})
	# Project A->C
	for i in range(0, 9):
		b.project({"stimA":["A"]},
			{"A":["A","C"],"C":["C"]})
	# Project B->C
	b.project({"stimB":["B"]},{"B":["B","C"]})
	for i in range(0, 9):
		b.project({"stimB":["B"]},
			{"B":["B","C"],"C":["C"]})
	# Project both A,B to C
	b.project({"stimA":["A"],"stimB":["B"]},
		{"A":["A","C"],"B":["B","C"]})
	for i in range(0, project_iter):
		b.project({"stimA":["A"],"stimB":["B"]},
				{"A":["A","C"],"B":["B","C"],"C":["C"]})
	# Project just B
	b.project({"stimB":["B"]},{"B":["B","C"]})
	# compute overlap
	intersection = bu.overlap(b.areas["C"].saved_winners[-1],b.areas["C"].saved_winners[9])
	assembly_overlap = float(intersection)/float(k)

	b.project({},{"C":["D"]})
	# Project just A
	b.project({"stimA":["A"]},{"A":["A","C"]})
	b.project({},{"C":["D"]})
	D_saved_winners = b.areas["D"].saved_winners
	proj_intersection = bu.overlap(D_saved_winners[0], D_saved_winners[1])
	proj_overlap = float(proj_intersection)/float(k)

	return assembly_overlap, proj_overlap

def overlap_grand_sim(n=100000,k=317,p=0.01,beta=0.05,min_iter=10,max_iter=30):
	b = brain.Brain(p,save_winners=True)
	b.add_stimulus("stimA",k)
	b.add_area("A",n,k,beta)
	b.add_stimulus("stimB",k)
	b.add_area("B",n,k,beta)
	b.add_area("C",n,k,beta)
	b.add_area("D",n,k,0)

	b.project({"stimA":["A"],"stimB":["B"]},{})
	# Create assemblies A and B to stability
	for i in range(0, 10):
		b.project({"stimA":["A"],"stimB":["B"]},
			{"A":["A"],"B":["B"]})
	b.project({"stimA":["A"]},{"A":["A","C"]})
	# Project A->C
	for i in range(0, 10):
		b.project({"stimA":["A"]},
			{"A":["A","C"],"C":["C"]})
	# Project B->C
	b.project({"stimB":["B"]},{"B":["B","C"]})
	for i in range(0, 10):
		b.project({"stimB":["B"]},
			{"B":["B","C"],"C":["C"]})
	# Project both A,B to C
	b.project({"stimA":["A"],"stimB":["B"]},
		{"A":["A","C"],"B":["B","C"]})
	for i in range(0, min_iter-2):
		b.project({"stimA":["A"],"stimB":["B"]},
				{"A":["A","C"],"B":["B","C"],"C":["C"]})
	results = {}
	for i in range(0, min_iter,max_iter+1):
		b.project({"stimA":["A"],"stimB":["B"]},
				{"A":["A","C"],"B":["B","C"],"C":["C"]})
		b_copy1 = copy.deepcopy(b)
		b_copy2 = copy.deepcopy(b)
		# in copy 1, project just A
		b_copy1.project({"stimA":["A"]},{})
		b_copy1.project({},{"A":["C"]})
		# in copy 2, project just B
		b_copy2.project({"stimB":["B"]},{})
		b_copy2.project({},{"B":["C"]})
		intersection = bu.overlap(b_copy1.areas["C"].winners, b_copy2.areas["C"].winners)
		assembly_overlap = float(intersection)/float(k)

		# projecting into D
		b_copy1.project({},{"C":["D"]})
		b_copy1.project({"stimB":["B"]},{})
		b_copy1.project({},{"B":["C"]})
		b_copy1.project({},{"C":["D"]})
		D_saved_winners = b_copy1.areas["D"].saved_winners
		proj_intersection = bu.overlap(D_saved_winners[0], D_saved_winners[1])
		proj_overlap = float(proj_intersection)/float(k)

		print("t=" + str(i) + " : " + str(assembly_overlap) + " -> " + str(proj_overlap) + "\n")
		results[assembly_overlap] = proj_overlap
	return results

def overlap_sim_multi(n=100000,k=317,p=0.05,beta=0.1,project_iter=10):
	b = brain.Brain(p,save_winners=True)
	b.add_stimulus("stimA",k)
	b.add_area("A",n,k,beta)
	b.add_stimulus("stimB",k)
	b.add_area("B",n,k,beta)
	b.add_stimulus("stimC",k)
	b.add_area("C",n,k,beta)
	b.add_area("D",n,k,beta)
	b.add_area("E",n,k,0.0) # final project test area
	b.project({"stimA":["A"],"stimB":["B"],"stimC":["C"]},{}) #edit project function
	# Create assemblies A and B and C to stability
	for i in range(0, 9):
		b.project({"stimA":["A"],"stimB":["B"], "stimC":["C"]},
			{"A":["A"],"B":["B"],"C":["C"]})

	b.project({"stimA":["A"]},{"A":["A","D"]})
	# Project A->D
	for i in range(0, 9):
		b.project({"stimA":["A"]},
			{"A":["A","D"],"D":["D"]})
	# Project B->D
	b.project({"stimB":["B"]},{"B":["B","D"]})
	for i in range(0, 9):
		b.project({"stimB":["B"]},
			{"B":["B","D"],"D":["D"]})
	# Project C->D
	b.project({"stim":["C"]},{"C":["C","D"]})
	for i in range(0, 9):
		b.project({"stimC":["C"]},
			{"B":["B","D"],"D":["D"]})
	# Project both A,B,C to D
	b.project({"stimA":["A"],"stimB":["B"], "stimC":["C"]},
		{"A":["A","D"],"B":["B","D"], "C":["C","D"]})
	for i in range(0, project_iter):
		b.project({"stimA":["A"],"stimB":["B"],"stimC":["C"]},
				{"A":["A","D"],"B":["B","D"],"C":["C","D"],"D":["D"]})

	# last part needs work 

	# Project just B
	b.project({"stimB":["B"]},{"B":["B","D"]})
	# compute overlap 
	intersection = bu.overlap(b.areas["D"].saved_winners[-1],b.areas["D"].saved_winners[9])
	assembly_overlap = float(intersection)/float(k)

	b.project({},{"D":["E"]})
	# Project just A
	b.project({"stimA":["A"]},{"A":["A","D"]})
	b.project({},{"D":["E"]})
	E_saved_winners = b.areas["D"].saved_winners
	proj_intersection = bu.overlap(E_saved_winners[0], E_saved_winners[1])
	proj_overlap = float(proj_intersection)/float(k)

	return assembly_overlap, proj_overlap

def overlap_sim_multiple_areas(n=100000,k=317,p=0.05,beta=0.1,project_iter=10,areas=2):
	b = brain.Brain(p,save_winners=True)
	stim_dict = {} #stimuluation 
	area_dict = {} #areas
	#initiailizing stimulus and areas
	for i in range(1, areas+1):	
		area_name = str(chr(64+i))
		stim_name = "stim" + area_name 
		b.add_stimulus(stim_name,k)	
		b.add_area(area_name,n,k,beta)
		stim_dict[stim_name] = [area_name] 
		area_dict[area_name] = [area_name]

	target_area = str(chr(64+(areas+1)))
	b.add_area(target_area,n,k,beta)
	target_area_2 = str(chr(64+(areas+2)))
	b.add_area(target_area_2,n,k,0.0)
	b.project(stim_dict,{})
	# Create assemblies in each area to stability
	for i in range(0,9):
		b.project(stim_dict, area_dict)

	# Create assemblies A and B to stability
	for i in range(0, 9):
		b.project({"stimA":["A"],"stimB":["B"]},
			{"A":["A"],"B":["B"]})
	b.project({"stimA":["A"]},{"A":["A","C"]})
	# Project A->C
	for i in range(0, 9):
		b.project({"stimA":["A"]},
			{"A":["A","C"],"C":["C"]})
	# Project B->C
	b.project({"stimB":["B"]},{"B":["B","C"]})
	for i in range(0, 9):
		b.project({"stimB":["B"]},
			{"B":["B","C"],"C":["C"]})
	# Project both A,B to C
	b.project({"stimA":["A"],"stimB":["B"]},
		{"A":["A","C"],"B":["B","C"]})
	for i in range(0, project_iter):
		b.project({"stimA":["A"],"stimB":["B"]},
				{"A":["A","C"],"B":["B","C"],"C":["C"]})
	# Project just B
	b.project({"stimB":["B"]},{"B":["B","C"]})
	# compute overlap
	intersection = bu.overlap(b.areas["C"].saved_winners[-1],b.areas["C"].saved_winners[9])
	assembly_overlap = float(intersection)/float(k)

	b.project({},{"C":["D"]})
	# Project just A
	b.project({"stimA":["A"]},{"A":["A","C"]})
	b.project({},{"C":["D"]})
	D_saved_winners = b.areas["D"].saved_winners
	proj_intersection = bu.overlap(D_saved_winners[0], D_saved_winners[1])
	proj_overlap = float(proj_intersection)/float(k)

	return assembly_overlap, proj_overlap	