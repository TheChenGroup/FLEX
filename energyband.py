from math import pi, cos
from numpy import array, empty, dot
from spglib import get_symmetry_from_database
from re import fullmatch, split, sub, findall

def readEnergyBands(dim, qe_output_dir):
	'''
	read multiple energy bands in irreducible k space from external output of QE, pw.x with `calculation='bands'`
	then extend the irreducible energy bands to the whole 1st Brillouin zone
	'''
	
	qe_pw_bands=open(qe_output_dir+'/pw_bands.out', 'rt')
	error_message='QE output file is improper. There may be some error in QE calculation of energy bands.'
	num_of_bands=0
	flag=-1
	for i in qe_pw_bands:
		if fullmatch(r'     number of Kohn-Sham states= *(\d+)\n', i):
			num_of_bands=int(sub(r'     number of Kohn-Sham states= *(\d+)\n', r'\1', i))
		elif fullmatch(r'     End of band structure calculation\n', i):
			flag=0
			break
	if num_of_bands==0 or flag==-1:
		raise RuntimeError(error_message)
	else:
		energy_bands=empty((dim, dim, dim, num_of_bands), dtype=float)
		symm_ops=get_symmetry_from_database(517)['rotations']
		for i in qe_pw_bands:
			if fullmatch(r'\n', i) and (flag==0 or flag==2):
				flag=(flag+1)%3
			elif fullmatch(r'          k = .*? \( *\d+ *PWs\)   bands \(ev\):\n', i) and (flag==1):
				# extract k_vec from this line in the output of QE
				k_vec=findall(r'(?: |-)0\..*?(?= |-)', i)
				k_posi=array(list(map(lambda x:round(float(x)*dim), k_vec))) # convert string variable k_vec to integer
				bands_count=0 # record how many bands at this k_posi have been stored to energy_bands
				flag+=1
			elif flag==0:
				part_bands=array(list(map(lambda x:float(x), split(r' +|\n',i)[1:-1])))
				for op in symm_ops:
					# enumerate all point symmetry operations, and all equivalent k_posi's, i.e. k_posi_tmp
					# k_posi_tmp may occur multiple times when enumerate all point symmetry operations
					# so the same part_bands may be assigned to the same k_posi_tmp multiple times, which do no harm
					# avoiding multiple assignments by comparing k_posi_tmp's is inefficient
					k_posi_tmp=op.dot(k_posi)
					energy_bands[k_posi_tmp[0], k_posi_tmp[1], k_posi_tmp[2], bands_count:bands_count+part_bands.size]=part_bands
				bands_count+=part_bands.size
			elif flag==1:
				break
			else:
				raise RuntimeError(error_message)
		return energy_bands

def singleEnergyBand(dim, qe_output_dir):
	'''
	extract only one energy band (nearest to fermi energy) from QE output
	'''
	
	qe_pw_scf=open(qe_output_dir+'/pw_scf.out', 'rt')
	error_message='QE output file is improper. There may be some error in QE calculation of energy bands.'
	flag=False
	for i in qe_pw_scf:
		if fullmatch(r'     the Fermi energy is .* ev\n', i):
			fermi_energy=float(sub(r'     the Fermi energy is (.*) ev\n', r'\1', i))
			flag=True
			break
	if not flag:
		raise RuntimeError(error_message)
	else:
		energy_bands=readEnergyBands(dim, qe_output_dir)
		energy_band=empty((dim, dim, dim), dtype=float)
		for i in range(dim):
			for j in range(dim):
				for k in range(dim):
					energy_band[i,j,k] = energy_bands[i, j, k, abs(energy_bands[i,j,k]-fermi_energy).argmin()]
		return energy_band

def energy(kx, ky, kz, t_hopping=1):
	'''
	return tight-binding energy.
	
	unit of returned value is the same as that of t_hopping
	'''
	
	return -2 * t_hopping * ( cos(kx) + cos(ky) + cos(kz) )

def irrEnergyBand(dim, t_hopping=1):
	'''
	calculate tight-binding energy band in irreducible k space.
	
	only the part where kz >= ky >= kx >= 0 is unique on energy
	all points of other parts can be mapped into the unique part via symmetry operation, under which energy(kx, ky, kz) is remained
	'''
	
	minimal_dim=int(dim/2)+1
	delta_k=2*pi/dim
	irreducible_energy_band=empty((minimal_dim, minimal_dim, minimal_dim))
	for z in range(0, minimal_dim):
		for y in range(0, z+1):
			for x in range(0, y+1):
				irreducible_energy_band[x, y, z]=energy(x*delta_k, y*delta_k, z*delta_k, t_hopping)
	return irreducible_energy_band

def energyBand(dim, qe_output_dir=False, t_hopping=1):
	'''
	map irrEnergyBand(dim) into the whole 1st Brillouin zone
	'''
	
	if qe_output_dir:
		return singleEnergyBand(dim, qe_output_dir)
	else:
		index_1=-int(dim/2)+1
		index_2=int(dim/2)+1
		delta_k=2*pi/dim
		energy_band=empty((dim, dim, dim))
		
		# from now on, energy of (x*delta_k, y*delta_k, z*delta_k) will be stored in energy_band[x, y, z], where x, y and z can be negative
		energy_band[:index_2, :index_2, :index_2]=irrEnergyBand(dim, t_hopping)
		
		for z in range(index_1, index_2):
			for y in range(index_1, index_2):
				for x in range(index_1, index_2):
					if z>=y and y>=x and x>=0:
						continue
					sorted_coord=[abs(x), abs(y), abs(z)]
					sorted_coord.sort()
					energy_band[x, y, z]=energy_band[sorted_coord[0], sorted_coord[1], sorted_coord[2]]
		return energy_band
