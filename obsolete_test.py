# unit of param T, energy and miu: K
# returned value will be always recognized as G_0 * k_Bolzmann, so with unit: 1/K
def greenFunc0(n, T, miu, energy, boost=0):
	omega=(2*n+1)*pi*T
	xi=energy-miu
	if boost==1:
		return -xi/(xi**2+omega**2)
	elif boost==0:
		return 1/(complex(-xi, omega))
	else:
		raise Exception('arg boost=%s is invalid'%boost)

# only the part where kz >= ky >= kx >= 0 is unique on energy
# all points of other parts can be mapped into the unique part via symmetry operation, under which energy(kx, ky, kz) is remained
def irrGreenMatrix0(n, T, miu, dim, boost=0):
	minimal_dim=int(dim/2)+1
	delta_k=2*pi/dim
	if boost==1:
		data_type='float64'
	elif boost==0:
		data_type='complex'
	else:
		raise Exception('arg boost=%s is invalid'%boost)
	irreducible_green_matrix_0=empty((minimal_dim, minimal_dim, minimal_dim), dtype=data_type)
	for z in range(0, minimal_dim):
		for y in range(0, z+1):
			for x in range(0, y+1):
				irreducible_green_matrix_0[x, y, z]=greenFunc0(n, T, miu, energy(x*delta_k, y*delta_k, z*delta_k), boost)
	return irreducible_green_matrix_0

# map irrGreenMatrix0 into the whole 1st Brillouin zone
def greenMatrix0(n, T, miu, dim, boost=0):
	index_1=-int(dim/2)+1
	index_2=int(dim/2)+1
	delta_k=2*pi/dim
	if boost==1:
		data_type='float64'
	elif boost==0:
		data_type='complex'
	else:
		raise Exception('arg boost=%s is invalid'%boost)
	green_matrix_0=empty((dim, dim, dim), dtype=data_type)
	
	# from now on, G_0 of (x*delta_k, y*delta_k, z*delta_k) will be stored in green_matrix_0[x, y, z], where x, y and z can be negative
	green_matrix_0[:index_2, :index_2, :index_2]=irrGreenMatrix0(n, T, miu, dim, boost)
	
	for z in range(index_1, index_2):
		for y in range(index_1, index_2):
			for x in range(index_1, index_2):
				if z>=y and y>=x and x>=0:
					continue
				sorted_coord=[abs(x), abs(y), abs(z)]
				sorted_coord.sort()
				green_matrix_0[x, y, z]=green_matrix_0[sorted_coord[0], sorted_coord[1], sorted_coord[2]]
	return green_matrix_0

# unit of omega_max: K
# WARNING: returned value is a complex number
def nOccupation0(T, miu, dim, omega_max, boost=0):
	n_trunc=int(round((omega_max/pi/T-1)/2))
	green_matrix_0_sum=0
	
	# omega_n/pi/T is from -2*n_trunc-1 to 2*n_trunc+1
	for n in range(-n_trunc-1, n_trunc+1):
		green_matrix_0_sum+=greenMatrix0(n, T, miu, dim, boost).sum()
	n_occup=0.5 + green_matrix_0_sum*T/dim**3
	return n_occup

# wrapper of nOccupation0(), with order of args changed, ready for scipy.optimize.fsolve()
#nOccup0=lambda miu, T, dim, omega_max=100*t_hopping, boost=0: nOccupation0(T, miu, dim, omega_max, boost)

# unit of T and omega_0: K
def greenFuncPhonon(n, T, omega_0):
	omega_n=2*n*pi*T
	return 2*omega_0/(-omega_n**2-omega_0**2)

## miu(T) is actually miu(T, <n>=const)
#n_occup_fixed=5e-2
#for i in range(10):
#	curr_T=(10-i)*T
#	print(curr_T, brentq(lambda miu: nOccup0(miu, curr_T, dim, boost=1)-n_occup_fixed, -6, 2, full_output=1))

## compute <n> for different miu's under the same T
#for i in range(60):
#	#miu=-6*t_hopping*(1+1/i)
#	miu=-i/10*t_hopping
#	n_occup=nOccup0(miu, T, dim, boost=1)
#	print(miu, n_occup)

## test speed of generating energy_band
#dim=8
#energy_band=energyBand(dim)
#for z in range(-int(dim/2)+1, int(dim/2)+1):
#	for x in range(-int(dim/2)+1, int(dim/2)+1):
#		for y in range(-int(dim/2)+1, int(dim/2)+1):
#			print(x, y, z, energy_band[x, y, z])
