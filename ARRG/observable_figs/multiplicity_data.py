import pgun_a_b
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
sns.set_style("ticks")
sns.set_context("paper", font_scale = 3.0)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 25

def multiplicity_data(N=int(1e5), E_partons=50.0, mode=1):
	"""
	Generating training dataset for BINN

	N: Total number of sample events to train on
	E_partons: Energy of initial outgoing partons
	mode: event topology, only mode = 1 is supported i.e. q-qbar 
	vec_len: length of vectors which the data will compiled and sorted in

	returns N/vec_length vectors 
	"""
	# Set up grid of (a,b) values
	na, nb = (3, 3)
	a_vals = np.linspace(1, 2, na)
	b_vals = np.linspace(1, 2, nb)

	av, bv = np.meshgrid(a_vals, b_vals)

	# Iterate over each value
	for a,b in zip(av.flatten(),bv.flatten()):
		print(a,b)
		gen = pgun_a_b.ParticleGun(a = a, b = b)
		mult_i = []
		n=0
		# Generate events and collect first emission data
		while n!=N:
			gen.next(mode, E_partons)
			gen.list()
			if gen.strings.endC[1].e() == gen.strings.endA[2].e():
				n+=1
				pz_pT_i.append(np.array([gen.strings.hads[1].pz(), gen.strings.hads[1].pT()]))

		pz_pT.append(np.array(pz_pT_i))
	
	# Convert to numpy array
	pz_pT = np.array(pz_pT)

	#return 

if __name__ == '__main__':
	multiplicity_data(N=1, E_partons=50.0, mode=1)

	print(pz_pT[0])