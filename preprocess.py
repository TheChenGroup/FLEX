from .flex import FLEX
from numpy import load, savez_compressed, zeros

def interpolateMat(ref_npz, n_trunc_old, which_mat='self_energy', return_it=False):
	'''
	generate init `self_energy`/`green_matrix` by interpolating Matsubara frequency in results of higher temperature
	then save it to ./init.npz or return it
	
	work only in the current dir
	'''
	
	with load('./'+ref_npz) as data:
		try:
			G=data[which_mat]
		except:
			G=FLEX.greenMatExtractor(data['green_matrix_extended'], n_trunc_old)
	flex_instance=FLEX('.')
	new_mat = zeros([flex_instance.omega_count] + [flex_instance.dim_el]*3, dtype=complex)
	new_mat[0]=G[0]
	new_mat[flex_instance.n_trunc]=G[n_trunc_old]
	for i in range(1, flex_instance.n_trunc):
		left = int(i / flex_instance.n_trunc * n_trunc_old)
		new_mat[i] = G[left] + (G[left+1]-G[left]) * (i / flex_instance.n_trunc * n_trunc_old - left)
	new_mat[range(-1, -flex_instance.n_trunc-2, -1)] = new_mat[0:flex_instance.n_trunc+1].conj()
	if return_it:
		return new_mat
	else:
		savez_compressed('init', **{which_mat:new_mat})
