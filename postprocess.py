from os.path import dirname
from pathlib import Path
from numpy import load, array, arange, set_printoptions
from sys import maxsize
from re import sub
from .energyband import energyBand
from .env import decimal2OtherBaseDigits
from .flex import FLEX

def selfEnergyImSpecialK(mu, qe_output_dir=False, t_hopping=1):
	'''
	select self_energy at k where energy is nearest and farthest to fermi energy
	then print its imaginary part at different Matsubara frequency to selfEnergyIm-omegaN.txt
	'''
	
	npz_file_name_candidates=['self_energy','tmp']
	for i in npz_file_name_candidates:
		# file name extension is default to .npz
		npz_file_path=i+'.npz'
		if Path(npz_file_path).is_file():
			with load(npz_file_path) as data:
				self_energy=data['self_energy']
			break
	# the below two lines are equivalent
	#if 'self_energy' in locals()
	if 'self_energy' not in vars():
		raise OSError("can't find .npz file containing self_energy")
	
	xi = energyBand(self_energy.shape[1], qe_output_dir, t_hopping) - mu
	k_minimize_xi = decimal2OtherBaseDigits(abs(xi).argmin(), self_energy.shape[1]) # k nearest to fermi surface
	k_maximize_xi = decimal2OtherBaseDigits(abs(xi).argmax(), self_energy.shape[1]) # k farthest to fermi surface
	
	n_trunc=int(self_energy.shape[0]/2-1)
	set_printoptions(threshold=maxsize, linewidth=maxsize)
	with open('selfEnergyIm-omegaN.txt', 'w+') as f:
		_=f.write('# n of omega_n = (2*n + 1) * pi * T | imag(self_energy) at k minimizing abs of xi[k]=energy_band[k]-mu | imag(self_energy) at k maximizing abs(xi[k])\n')
		_=f.write('# xi[k] = {0} {1}\n# n | k={2} | k={3}\n'.format(
			xi[k_minimize_xi[0], k_minimize_xi[1], k_minimize_xi[2]],
			xi[k_maximize_xi[0], k_maximize_xi[1], k_maximize_xi[2]],
			k_minimize_xi,
			k_maximize_xi
		))
		_=f.write(
			sub(r'[\[ ]\[|\]', '', str(
				array([
					arange(-n_trunc-1, n_trunc+1),
					self_energy[:, k_minimize_xi[0], k_minimize_xi[1], k_minimize_xi[2]][arange(-n_trunc-1, n_trunc+1)].imag,
					self_energy[:, k_maximize_xi[0], k_maximize_xi[1], k_maximize_xi[2]][arange(-n_trunc-1, n_trunc+1)].imag
				]).transpose()
			))
			+'\n'
		)

def write2Txt(npz_path, data_tag='G'):
	'''
	write real and imaginary part of green matrix or self energy (`data_tag`='G' or 'S') to .txt
	'''
	
	# read data from npz_path
	if Path(npz_path).is_file():
		with load(npz_path) as data:
			if data_tag=='G':
				try:
					green_matrix=data['green_matrix']
				except:
					green_matrix_extended=data['green_matrix_extended']
			elif data_tag=='S':
				self_energy=data['self_energy']
	else:
		raise OSError(2, '', npz_path)
	
	# format data
	dir_path=dirname(npz_path)
	if data_tag=='G':
		txt_path=dir_path+'/green_matrix.txt'
		txt_head='# green matrix'
		if 'green_matrix_extended' in vars():
			green_matrix=FLEX.greenMatExtractor(green_matrix_extended, FLEX(dir_path).n_trunc)
		data_flat=green_matrix.flatten()
	else:
		txt_path=dir_path+'/self_energy.txt'
		txt_head='# self energy'
		data_flat=self_energy.flatten()
	
	# write data
	common_head='# real imag'
	with open(txt_path, 'wt') as f:
		_=f.write(txt_head+'\n'+common_head+'\n')
		for i in data_flat:
			_=f.write('{0} {1}\n'.format(i.real, i.imag))
