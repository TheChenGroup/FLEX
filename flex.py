from math import pi, exp, floor
from numpy import array, arange, empty, zeros, ones, full, dot, flipud, einsum, abs, load, savez_compressed, fromiter, unique
from numpy.linalg import inv
from numexpr import evaluate
from scipy.optimize import brentq
from scipy.fft import fftn, ifftn
from scipy.sparse import identity
from spglib import get_symmetry_from_database
from time import time
from re import match, split, sub
import sys
from os import devnull
from pathlib import Path
from mpi4py.MPI import COMM_WORLD, IN_PLACE, SUM
from .env import delimiter_top_level, delimiter_iteration, cpuTime, decimal2OtherBaseDigits
from .energyband import energyBand
from .maxeigsolver import powerMethodGeneral, powerMethodPao

class FLEX:
	'''
	run FLEX self-consistency and solve eigenvalue to detect transition temperature Tc.
	
	Parameter
	==================================================
	
	check_vs_bak31Ph10/flex.in is used to check new version of this module, so it's always up to date.
	see it for detail.
	
	Method
	==================================================
	
	closeFileHandles(self):
		close file handles
	
	__init__(self, working_dir, silent_init=True, monitor=False, monitor_MPI=False):
		initialize from working_dir/flex.in
	
	preprocessMixing(self, for_who):
	    guarantee that mixing_list and mixing_group_size_list are always available in vars(self)
	
	transMatSlices(self, trans_k):
	    generate index slices according to tuple trans_k for translation of elements in an array
	
	invMatSlice(self):
	    generate index slices for inversion of elements in an array
	
	vp(self, m):
		defined in Pao's eq.(8)
	
	vpString(self, m):
		defined in Pao's eq.(8), but only returns the expression string
	
	vDMatrix(self, m):
	    defined in Pao's eq.(4)
	
	vSMatrix(self, m):
	    defined in Pao's eq.(6)
	
	vTMatrix(self, m):
	    defined in Pao's eq.(7)
	
	fluctuationSlice(self):
		defined in Pao's eq.(14-15), but with m fixed
	
	effInteractionSlice(self):
		defined in Pao's eq.(10-12), but with m fixed
	
	selfEnergyPart(self):
		defined in Pao's eq.(9), but with n-n'=m fixed
	
	selfEnergyThisProcess(self):
		defined in Pao's eq.(9), but with n-n' belong to {m1, m2, ...}
	
	selfEnergySimple(self, mu):
	    defined in Pao's eq.(16), but inverse
	
	greenMatrix(self, mu=None):
	    defined in Pao's eq.(16)
	
	greenMatFFT(self):
	    calculate Discrete Fourier transform (DFT) of green_matrix and its conjugate for use in convolution theorem, which will accelerate summation in fluctuationSlice() and selfEnergyPart()
	
	greenMatExtractor(green_matrix_extended, n_trunc):
	    extract green_matrix from green_matrix_extended
	
	pickMixing(mixing_list, mixing_group_size_list, step):
	    pick up a mixing from mixing_list according to mixing_group_size_list and step count
	    p.s. new and old thing will be mixed with ratio (1-mixing):mixing
	
	FLEXMix(self, step):
	    pick up a mixing from flex_mixing_list via pickMixing()
	
	MigdalSCMix(self, step):
	    pick up a mixing from MigdalSC_mixing_list via pickMixing()
	
	nOccupation(self, green_matrix):
	    calculate occupation number per lattice site from green_matrix
	
	nDeltaFromFixedVal(self, mu):
	    calculate occupation number from mu, then return its deviation from n_occup_fixed
	
	muSolver(self):
	    solve mu according to nDeltaFromFixedVal(self, mu)=0
	
	saveData(self, file_name):
	    save data to file_name, which will be overwritten during every iteration of selfConsistency()
	
	selfConsistency(self):
	    the meaning of being of this whole package
	
	VPairSC(self):
	    revised VPairSCPao() by Luo
	
	VPairSCPao(self):
		defined in Pao's eq.(19), but with (m+m_trunc) % size == rank
	
	VPairCDW(self):
	    Luo's pairing potential for CDW
	
	maxEig(self, state):
	    calculate eigenvalue with max norm according to Pao's eq.(18)
	
	susceptibilityMigdalCDW(self):
	    defined in Dee's eq.(15)
	
	susceptibilityMigdalSC(self):
	    defined in Dee's eq.(16)
	
	Note
	==================================================
	
	array representation of physics quantity
	------------------------------
	
	in this whole package, I use array A to store physical quantity alpha such that alpha(i)=A[i],
	where i belongs to {-N/2, -N/2+1, ..., -1, 0, 1, 2, ..., N/2-1} if i is index of k in reciprocal space,
	or {-N/2-1, -N/2+1, ..., -1, 0, 1, 2, ..., N/2} if i is index of fermion Matsubara frequency,
	or {-N/2, -N/2+1, ..., -1, 0, 1, 2, ..., N/2} if i is index of boson Matsubara frequency.
	
	if alpha has period N, i.e. alpha(i+N)=alpha(i), then it can be proved that
	
	           / A[j], where j belongs to {0, 1, 2, ..., N/2-1}
	alpha(j) = |
	           \ alpha(j-N) = A[j-N] = A[j], where j belongs to {N/2, N/2+1, ..., N-1}
	
	         = A[j], where j belongs to {0, 1, 2, ..., N-1}
	
	how mixing work
	------------------------------
	
	mix new and old self_energy/green_matrix / gamma with ratio (1-mixing):mixing
	
	mixing_num_of_group:
	1. all iterations are split into mixing_num_of_group groups
	2. there are mixing_num_of_group different mixings with equal interval from mixing_min (inclusive) to mixing_max (exclusive)
	
	mixing_group_size:
	every mixing_group_size iterations use the same mixing
	
	limit of tot # of iterations = mixing_group_size * mixing_num_of_group
	
	default value:
	mixing_min = 0
	mixing_max = 1
	mixing_num_of_group = 10
	mixing_group_size = 10
	
	the following is applied when the above four default values are used:
	
	the 0-th group iterations are from 0 to mixing_group_size-1,
	  and use mixing = 0
	the 1st group iterations are from mixing_group_size to 2 * mixing_group_size - 1,
	  and use mixing = 1 / mixing_num_of_group
	...
	the n-th group iterations are from
	    n * mixing_group_size
	  to
	    (n+1) * mixing_group_size - 1
	  and use mixing = n / mixing_num_of_group
	...
	the (mixing_num_of_group-1)-th group iterations are from
	    (mixing_num_of_group-1) * mixing_group_size
	  to
	    mixing_num_of_group * mixing_group_size - 1 = tot # of iterations - 1
	  and use mixing = (mixing_num_of_group-1) / mixing_num_of_group = 1 - 1 / mixing_num_of_group
	
	generally, mixing changes as the following:
	
	mixing_min
	mixing_min + (mixing_max - mixing_min) * 1 / mixing_num_of_group
	...
	mixing_min + (mixing_max - mixing_min) * n / mixing_num_of_group
	...
	mixing_min + (mixing_max - mixing_min) * (1 - 1 / mixing_num_of_group) = mixing_max - (mixing_max - mixing_min) / mixing_num_of_group
	
	i.e.
	```
	mixing = self.mixing_min + (self.mixing_max - self.mixing_min) * floor(step/self.mixing_group_size) / self.mixing_num_of_group
	```
	
	BUT mixing_list and mixing_group_size_list are more general and have higher priority
	num of mixing_group_size_list[i] iterations will use mixing_list[i] as mixing
	
	mixing_list: replacement of mixing_min, mixing_max and mixing_num_of_group
	mixing_group_size_list: replacement of mixing_group_size
	limit of tot # of iterations = sum(mixing_group_size_list)
	
	`_extended` as a suffix for variable
	------------------------------
	
	OBSOLETE
	
	variables with this suffix were once used and only used for summation in convolution,
	which is now solved by convolution theorem, using FFT, much faster and neater.
	'''
	
	# k_Bolzmann=1.38064852e-23
	# unit of t_hopping: K=J/k_Bolzmann
	#t_hopping=1e0 # 1e4K ~ 1eV/k_Bolzmann # defined in input_file_path
	#T=1e0 # defined in input_file_path
	#U=0 # defined in input_file_path
	#g=.5**.5 # defined in input_file_path
	#U_p=1 # defined in input_file_path
	#Omega_p=1 # defined in input_file_path
	#flex_delta_ratio_criteria=1e-3 # defined in input_file_path
	#flex_delta_abs_criteria=1e-6 # defined in input_file_path
	#MigdalSC_delta_ratio_criteria=1e-3 # defined in input_file_path
	#MigdalSC_delta_abs_criteria=1e-6 # defined in input_file_path
	
	self_consistency_for_who='flex', 'MigdalSC'
	size = COMM_WORLD.size
	rank = COMM_WORLD.rank
	
	def closeFileHandles(self):
		'''
		close file handles
		
		+ `print_redirect` redirects all `print()` and `cpuTime()` except loop in `selfEnergyThisProcess()`
		+ `print_redirect_MPI` only redirects `cpuTime()` in loop of `selfEnergyThisProcess()`
		'''
		if self.rank!=0:
			self.print_redirect.close()
		if self.monitor_MPI:
			self.print_redirect_MPI.close()
	
	def __init__(self, working_dir, silent_init=True, monitor=False, monitor_MPI=False):
		'''
		initialize from working_dir/flex.in
		'''
		
		# WARN! `sys.stdout` must be recovered at the end of initialization under the same condition
		# `silent_init` only controls `print()` in `__init__()`
		if silent_init or self.rank!=0:
			stdout_old=sys.stdout
			stdout_null=open(devnull, 'w')
			sys.stdout=stdout_null
		
		# `print_redirect` redirects all `print()` and `cpuTime()` except loop in `selfEnergyThisProcess()`
		if self.rank==0:
			self.print_redirect=None
		else:
			self.print_redirect=open(devnull, 'w')
		
		# `monitor_MPI` only controls `cpuTime()` in loop of `selfEnergyThisProcess()`, which will be redirected to `print_redirect_MPI`
		self.monitor_MPI=monitor_MPI
		if monitor_MPI:
			Path(working_dir+'/tmp').mkdir(parents=True, exist_ok=True)
			self.print_redirect_MPI=open(f'{working_dir}/tmp/rank{self.rank}', 'a')
		else:
			self.print_redirect_MPI=None
		
		self.monitor = monitor # controls all `cpuTime()` except loop in `selfEnergyThisProcess()`
		self.working_dir = working_dir
		
		# the default is that don't use external QE output
		# this behavior can be controlled via use_qe_output in input_file_path
		self.use_qe_output=0 # defined in input_file_path. the default value is for compatibility with old flex.in files, in which there is no use_qe_output defined.
		self.qe_output_dir=False
		
		self.init_npz_dir=False # defined in input_file_path. the default value is for compatibility with old flex.in files, in which there is no init_npz_dir defined.
		
		input_file_path=working_dir+'/flex.in'
		input_dict={}
		with open(input_file_path,'rt') as input_file:
			for i in input_file:
				if not match('^\s*#.*|^\s*$',i):
					line_list=split('=|#',i.strip())
					try:
						input_dict[line_list[0]]=float(line_list[1])
					except:
						input_dict[line_list[0]]=line_list[1]
		print('initializing variables from %s\n%s'%(input_file_path,sub("\{|\}|'",'',str(input_dict).replace(", '",'\n'))))
		vars(self).update(input_dict) # vars(obj) is equivalent to obj.__dict__
		
		print('\nderived variables')
		
		if 'g' in vars(self): # the check for existence is for compatibility with old flex.in files
			self.U_p=2*self.g**2/self.Omega_p
			print('U_p: %f'%self.U_p)
		
		self.dim_el=int(self.dim_el) # defined in input_file_path, only use EVEN integer to be compatible with translation pi in k-space
		self.dim_ph=self.dim_el # MUST be EQUAL to dim_el according to eq (9) and (14, 15)
		print(f'dim_ph: {self.dim_ph}')
		
		# k-space's for electron and phonon are:
		# {dim_el_min, dim_el_min+1, ..., dim_el_max-1} x {...} x {...}
		# {dim_ph_min, dim_ph_min+1, ..., dim_ph_max-1} x {...} x {...}
		self.dim_el_min=-int(self.dim_el/2)
		self.dim_el_max=int(self.dim_el/2)
		self.dim_ph_min=-int(self.dim_ph/2)
		self.dim_ph_max=int(self.dim_ph/2)
		print(f'dim_el_min: {self.dim_el_min}\ndim_el_max: {self.dim_el_max}\ndim_ph_min: {self.dim_ph_min}\ndim_ph_max: {self.dim_ph_max}')
		
		self.omega_max=self.omega_max_over_t*self.t_hopping # defined in input_file_path
		self.n_trunc=int(round((self.omega_max/pi/self.T-1)/2)) # fermion Matsubara frequencies will be looped through via `n` in range(-n_trunc-1, n_trunc+1)
		self.omega_count=2*(self.n_trunc+1)
		print('n_trunc: %d\nomega_count: %d'%(self.n_trunc, self.omega_count))
		
		self.nu_max=self.nu_max_over_t*self.t_hopping # defined in input_file_path
		self.m_trunc=int(round(self.nu_max/pi/self.T/2)) # boson Matsubara frequencies will be looped through via `m` in range(-m_trunc,m_trunc+1)
		self.nu_count=2*self.m_trunc+1
		print('m_trunc: %d\nnu_count: %d'%(self.m_trunc, self.nu_count))
		
		# OBSOLETE
		# only for green_matrix_extended, which is only for verbose summation in fluctuationSlice() and selfEnergyPart()
		#self.n_trunc_extended = self.n_trunc + self.m_trunc
		#self.omega_count_extended = 2*(self.n_trunc_extended+1)
		#print('n_trunc_extended: %d\nomega_count_extended: %d'%(self.n_trunc_extended, self.omega_count_extended))
		
		# before solving equation f(mu)=0, try #mu_sample_num points in [mu_min, mu_max] to find interval [mu_min_new, mu_max_new] such that f(mu_min_new) * f(mu_max_new) < 0
		self.mu_sample_num=max(int(self.mu_sample_num), 2) if 'mu_sample_num' in input_dict else 2 # defined in input_file_path. the check for existence of mu_sample_num is for compatibility with old flex.in files, where there is no mu_sample_num defined.
		self.mu_samples=list(map(lambda i: self.mu_min+i*(self.mu_max-self.mu_min)/(self.mu_sample_num-1), range(self.mu_sample_num)))
		
		for i in self.self_consistency_for_who:
			self.preprocessMixing(i)
			setattr(self, i+'_tot_num_of_iter', sum(getattr(self, i+'_mixing_group_size_list'))) # limit tot # of iterations in self-consistency
			print('%s_tot_num_of_iter: %d'%(i, getattr(self, i+'_tot_num_of_iter')))
		
		# use QE output if use_qe_output > 0
		# use the 1st command line argument as the output directory of QE
		# in other words, you need to copy outputs of QE to there before running this script
		if self.use_qe_output > 0:
			self.qe_output_dir = working_dir
		
		# initialize green_matrix from init.npz only if init_npz_dir > 0
		if self.init_npz_dir > 0:
			self.init_npz_dir = working_dir
		else:
			self.init_npz_dir = False
		
		# initialize mu or fixed occupancy
		# initialize the range [mu_min, mu_max] in which mu is solved from n_occup_fixed
		# n_occup_fixed>0: always solve mu from n_occup_fixed and ignore the initial value of mu
		# n_occup_fixed<=0: always use initial value of mu and ignore n_occup_fixed
		#mu=-5 # defined in input_file_path
		#n_occup_fixed=.4 # defined in input_file_path
		#mu_min=-6 # defined in input_file_path
		#mu_max=2 # defined in input_file_path
		self.n_occup_flag=self.n_occup_fixed>0
		
		# only when (eigen_val_norm - max_eigen_val_norm)/abs(max_eigen_val_norm) > power_method_seq_update_threshold,
		# will max eigen value and vector be updated in power method sequence
		#power_method_seq_update_threshold=1e-6 # defined in input_file_path
		#
		# enumerate the first power_method_seq orthogonal initial eigen vectors sequentially in power method
		# except that
		# 1. power_method_seq < 0
		#    which is treated specially by `powerMethodSeq()` and may be filtered by specific powerMethod*, e.g. `powerMethodGeneral()` or `powerMethodPao()`
		# 2. power_method_seq = 0
		#    or
		#    power_method_seq > x = dimension of eigen vector = omega_count*dim_el**3
		#    which will be renormalized to x by specific powerMethod*, e.g. `powerMethodGeneral()` or `powerMethodPao()`
		# the larger power_method_seq is, the easier to overcome orthogonality between initial and destination eigen vector, which will invalidate power method
		#power_method_seq_sc=1 # defined in input_file_path
		#power_method_seq_cdw=1 # defined in input_file_path
		self.power_method_seq_sc = int(self.power_method_seq_sc) if 'power_method_seq_sc' in input_dict else 1
		self.power_method_seq_cdw = int(self.power_method_seq_cdw) if 'power_method_seq_cdw' in input_dict else 1
		
		# solve max eigen value after each FLEX iteration if solve_eig_during_FLEX > 0
		# defined in input_file_path
		self.solve_eig_during_FLEX = 'solve_eig_during_FLEX' in vars(self) and self.solve_eig_during_FLEX > 0
		
		# WARN! operations mixing scipy.sparse.spmatrix and numpy.ndarray will return numpy.matrix, which retains its 2-D nature through operations
		self.I_omega_count=identity(self.omega_count)
		self.inv_matrix_slice=self.invMatSlice()
		
		# left only for reference
		#self.v_M = empty( ( self.omega_count, self.omega_count) )
		#for i in range(-self.n_trunc-1, self.n_trunc+1):
		#	for j in range(-self.n_trunc-1, self.n_trunc+1):
		#		self.v_M[i,j]=-self.vp(i-j)-self.U
		#
		# right but tricky, refer to docstring at the beginning of this module
		self.n_range=arange(-self.n_trunc-1, self.n_trunc+1, dtype=float)[arange(-self.n_trunc-1, self.n_trunc+1)]
		self.n_range_column=self.n_range[:,None]
		self.v_M = v_M = self.n_range_column-self.n_range # equivalent to self.v_M[i,j]=i-j for all i's and j's
		U=self.U
		U_p=self.U_p
		Omega_p=self.Omega_p
		T=self.T
		_=evaluate('-{}-U'.format(self.vpString('v_M')), out=self.v_M)
		
		if silent_init or self.rank!=0:
			sys.stdout=stdout_old
			stdout_null.close()
	
	def preprocessMixing(self, for_who):
		'''
		guarantee that mixing_list and mixing_group_size_list are always available in vars(self)
		'''
		
		print_redirect=self.print_redirect
		
		if for_who not in self.self_consistency_for_who:
			raise ValueError(f'for_who={for_who} is not in {self.self_consistency_for_who}')
		
		mixing_list, mixing_group_size_list, mixing_min, mixing_max, mixing_num_of_group, mixing_group_size = list(map(lambda x: for_who+'_'+x, ['mixing_list', 'mixing_group_size_list', 'mixing_min', 'mixing_max', 'mixing_num_of_group', 'mixing_group_size']))
		
		# defined in input_file_path. the check for existence is for compatibility with old flex.in files
		if mixing_list in vars(self):
			vars(self)[mixing_list] = list(map(float, getattr(self, mixing_list).split(',')))
		elif mixing_min in vars(self) and mixing_max in vars(self) and mixing_num_of_group in vars(self):
			setattr(self, mixing_list, (
				getattr(self, mixing_min) + (getattr(self, mixing_max) - getattr(self, mixing_min)) / getattr(self, mixing_num_of_group) * arange(getattr(self, mixing_num_of_group))
				).tolist()
		   )
		else:
			# default value:
			# mixing_min=0
			# mixing_max=1
			# mixing_num_of_group=10
			setattr(self, mixing_list, (arange(10)/10).tolist())
		print('%s: %s'%(mixing_list, getattr(self, mixing_list)), file=print_redirect)
		
		# defined in input_file_path. the check for existence is for compatibility with old flex.in files
		if mixing_group_size_list in vars(self):
			setattr(self, mixing_group_size_list, list(map(int, getattr(self, mixing_group_size_list).split(','))))
		elif mixing_group_size in vars(self):
			setattr(self, mixing_group_size_list, full(len(getattr(self, mixing_list)), getattr(self, mixing_group_size)).tolist())
		else:
			# default mixing_group_size is 10
			setattr(self, mixing_group_size_list, full(len(getattr(self, mixing_list)), 10).tolist())
		print('%s: %s'%(mixing_group_size_list, getattr(self, mixing_group_size_list)), file=print_redirect)
	
	def transMatSlices(self, trans_k):
		'''
		generate index slices according to tuple trans_k for translation of elements in an array
		
		trans_k is a tuple, list or array consisting of 3 integers
		'''
		
		dim_el=self.dim_el
		
		trans_matrix_slices=list(map(lambda x: arange(-(x%dim_el),dim_el-(x%dim_el)), trans_k))
		return trans_matrix_slices
	
	def invMatSlice(self):
		'''
		generate index slices for inversion of elements in an array
		'''
		
		dim_el=self.dim_el
		
		inv_matrix_slice=arange(dim_el,0,-1)%dim_el
		return inv_matrix_slice
	
	def vp(self, m):
		'''
		defined in Pao's eq.(8)
		'''
		
		return -self.U_p * self.Omega_p**2 / ( self.Omega_p**2 + (2*m*pi*self.T)**2 )
	
	def vpString(self, m):
		'''
		defined in Pao's eq.(8), but only returns the expression string
		'''
		
		return f'(-U_p * Omega_p**2 / ( Omega_p**2 + (2*({m})*pi*T)**2 ))'
	
	def vDMatrix(self, m):
		'''
		defined in Pao's eq.(4)
		'''
		
		return 2*self.vp(m) + 2*self.U + self.v_M
	
	def vSMatrix(self, m):
		'''
		defined in Pao's eq.(6)
		'''
		
		U=self.U
		n_range=self.n_range
		n_range_column=self.n_range_column
		vpString=self.vpString
		U_p=self.U_p
		Omega_p=self.Omega_p
		T=self.T
		
		# left only for reference
		#v_S=empty((omega_count,omega_count))
		#for i in range(-n_trunc-1, n_trunc+1):
		#	for j in range(-n_trunc-1, n_trunc+1):
		#		v_S[i,j]=.5*(vp(i-j)+vp(i+j+1-m))+U
		v_S=evaluate('.5 * ( {} + {} ) + U'.format(
						vpString('n_range_column-n_range'),
						vpString('n_range_column+n_range+1-m')
						)
					)
		return v_S
	
	def vTMatrix(self, m):
		'''
		defined in Pao's eq.(7)
		'''
		
		n_range=self.n_range
		n_range_column=self.n_range_column
		vpString=self.vpString
		U_p=self.U_p
		Omega_p=self.Omega_p
		T=self.T
		
		# left only for reference
		#v_T=empty((omega_count,omega_count))
		#for i in range(-n_trunc-1, n_trunc+1):
		#	for j in range(-n_trunc-1, n_trunc+1):
		#		v_T[i,j]=.5*(vp(i-j)-vp(i+j+1-m))
		v_T=evaluate('.5 * ( {} - {} )'.format(
						vpString('n_range_column-n_range'),
						vpString('n_range_column+n_range+1-m')
						)
					)
		return v_T
	
	def fluctuationSlice(self,m):
		'''
		defined in Pao's eq.(14-15), but with m fixed
		
		calculate 4-order tensor chi_ph_slice & chi_pp_slice
		'''
		
		omega_count=self.omega_count
		nu_count=self.nu_count
		dim_ph=self.dim_ph
		dim_el=self.dim_el
		n_trunc=self.n_trunc
		m_trunc=self.m_trunc
		T=self.T
		green_matrix_fft=self.green_matrix_fft
		green_matrix_conj_fft=self.green_matrix_conj_fft
		
		chi_ph_slice=zeros((omega_count,dim_ph,dim_ph,dim_ph), complex)
		chi_pp_slice=zeros((omega_count,dim_ph,dim_ph,dim_ph), complex)
		
		for n in set(range(-n_trunc-1, n_trunc+1)) & set(range(-n_trunc-1-m, n_trunc+1-m)):
			chi_ph_slice[n] = -T/dim_el**3 * ifftn(green_matrix_fft[n+m] * green_matrix_conj_fft[n].conj())
			chi_pp_slice[n] = T/dim_el**3 * ifftn(green_matrix_fft[n+m] * green_matrix_fft[-n-1])
		
		self.chi_ph_slice=chi_ph_slice
		self.chi_pp_slice=chi_pp_slice
	
	def effInteractionSlice(self,m):
		'''
		defined in Pao's eq.(10-12), but with m fixed
		
		calculate 4-order tensor V_2_slice, V_ph_slice and V_pp_slice
		'''
		
		omega_count=self.omega_count
		dim_ph=self.dim_ph
		dim_ph_min=self.dim_ph_min
		vDMatrix=self.vDMatrix
		vSMatrix=self.vSMatrix
		vTMatrix=self.vTMatrix
		vp=self.vp
		vpString=self.vpString
		U_p=self.U_p
		Omega_p=self.Omega_p
		T=self.T
		n_range=self.n_range
		n_range_column=self.n_range_column
		v_M=self.v_M
		U=self.U
		chi_ph_slice=self.chi_ph_slice
		chi_pp_slice=self.chi_pp_slice
		I_omega_count=self.I_omega_count
		
		V_2_slice=empty((omega_count,dim_ph,dim_ph,dim_ph), complex)
		V_ph_slice=empty((omega_count,dim_ph,dim_ph,dim_ph), complex)
		V_pp_slice=empty((omega_count,dim_ph,dim_ph,dim_ph), complex)
		
		v_D=vDMatrix(m)
		v_S=vSMatrix(m)
		v_T=vTMatrix(m)
		v_p_element=vp(m)
		
		# left only for reference
		#v_p_matrix=empty((omega_count,omega_count),complex)
		#for i in range(-n_trunc-1, n_trunc+1):
		#	for j in range(-n_trunc-1, n_trunc+1):
		#		v_p_matrix[i,j]=vp(i-j-m)
		v_p_matrix=n_range_column-n_range
		_=evaluate(vpString('v_p_matrix-m'), out=v_p_matrix)
		
		for i in range(dim_ph_min,1):
			for j in range(i,1):
				for k in range(j,1):
					
					# the symmetry of cubic lattice is Pm-3m, the 517-th item in `Table A1.4.2.7` of "International Tables for Crystallography - Volume B: Reciprocal Space"
					# the space-group number of Pm-3m is 221
					symm_ops=get_symmetry_from_database(517)['rotations']
					
					# `equiv_idx` has 48 columns before being `unique`d, all of which are equivalent positions in k-space
					# `transpose` will only return a view, see `numpy.transpose` and `__array_interface__` for reference
					equiv_idx=unique(symm_ops.dot(array([i,j,k])), axis=0).transpose()
					
					# temporarily,
					# R_matrix_whole = R, where R = D, M, S or T
					# these will be updated to other things immediately
					D_matrix_whole=v_D*chi_ph_slice[:,i,j,k]
					M_matrix_whole=v_M*chi_ph_slice[:,i,j,k]
					S_matrix_whole=v_S*chi_pp_slice[:,i,j,k]
					T_matrix_whole=v_T*chi_pp_slice[:,i,j,k]
					
					# update R_matrix_whole to new thing to save memory
					# R_matrix_whole = R * inv(1+R) - R = 1 - inv(1+R) - R, where R = D, M, S or T
					D_matrix_whole=(I_omega_count-inv(I_omega_count+D_matrix_whole)-D_matrix_whole).A
					M_matrix_whole=(I_omega_count-inv(I_omega_count+M_matrix_whole)-M_matrix_whole).A
					S_matrix_whole=(I_omega_count-inv(I_omega_count+S_matrix_whole)-S_matrix_whole).A
					T_matrix_whole=(I_omega_count-inv(I_omega_count+T_matrix_whole)-T_matrix_whole).A
					
					V_2_slice[:,equiv_idx[0],equiv_idx[1],equiv_idx[2]]=(
							-v_p_element
							+ (v_p_element+U) * dot(2*v_p_element-v_p_matrix+U, chi_ph_slice[:,i,j,k])
						).reshape(-1,1)
					
					V_ph_slice[:,equiv_idx[0],equiv_idx[1],equiv_idx[2]]=(
							.5 * (D_matrix_whole*v_D.transpose()).sum(1)
							+1.5 * (M_matrix_whole*v_M.transpose()).sum(1)
						).reshape(-1,1)
					
					V_pp_slice[:,equiv_idx[0],equiv_idx[1],equiv_idx[2]]=(
							-(S_matrix_whole*v_S.transpose()).sum(1)
							-3 * (T_matrix_whole*v_T.transpose()).sum(1)
						).reshape(-1,1)
		
		self.V_2_slice=V_2_slice
		self.V_ph_slice=V_ph_slice
		self.V_pp_slice=V_pp_slice
	
	def selfEnergyPart(self,m):
		'''
		defined in Pao's eq.(9), but with n-n'=m fixed
		'''
		
		omega_count=self.omega_count
		dim_el=self.dim_el
		n_trunc=self.n_trunc
		T=self.T
		green_matrix_fft=self.green_matrix_fft
		green_matrix_conj_fft=self.green_matrix_conj_fft
		V_2_slice=self.V_2_slice
		V_ph_slice=self.V_ph_slice
		V_pp_slice=self.V_pp_slice
		
		self_energy_part=zeros((omega_count,dim_el,dim_el,dim_el),complex)
		
		for n in set(range(-n_trunc-1, n_trunc+1)) & set(range(-n_trunc-1+m, n_trunc+1+m)):
			self_energy_part[n]+=T/dim_el**3*(
								ifftn(fftn(V_2_slice[n])*green_matrix_fft[n-m])
								+ ifftn(fftn(V_ph_slice[n])*green_matrix_fft[n-m])
								+ ifftn(fftn(V_pp_slice[n])*green_matrix_conj_fft[n-m])
							)
		
		return self_energy_part
	
	def selfEnergyThisProcess(self, last_iteration_time, num_of_iteration):
		'''
		defined in Pao's eq.(9), but with n-n' belong to {m1, m2, ...}
		
		sum over contribution to self_energy from m's owned by the current process,
		i.e. sum over only some, instead of all, n_prime's in Pao's eq.(9)
		'''
		
		omega_count=self.omega_count
		dim_el=self.dim_el
		m_trunc=self.m_trunc
		fluctuationSlice=self.fluctuationSlice
		effInteractionSlice=self.effInteractionSlice
		selfEnergyPart=self.selfEnergyPart
		monitor=self.monitor
		size=self.size
		rank=self.rank
		print_redirect=self.print_redirect
		monitor_MPI=self.monitor_MPI
		print_redirect_MPI=self.print_redirect_MPI
		
		dur_fluctuation=0
		dur_effInteraction=0
		dur_selfEnergy=0
		self_energy=zeros((omega_count,dim_el,dim_el,dim_el),complex)
		for m in range(-m_trunc,m_trunc+1):
			if (m+m_trunc) % size == rank:
				
				# particle-particle and particle-hole fluctuation: chi_ph_slice & chi_pp_slice
				last_monitor_time=time()
				fluctuationSlice(m)
				dur_fluctuation += time()-last_monitor_time
				cpuTime(monitor_MPI, f'the {num_of_iteration}-th iteration, m={m}]\n[chi_ph_slice & chi_pp_slice', last_monitor_time, last_iteration_time, print_redirect_MPI)
				
				# effective interaction potentials: V_2_slice, V_ph_slice & V_pp_slice
				last_monitor_time=time()
				effInteractionSlice(m)
				dur_effInteraction += time()-last_monitor_time
				cpuTime(monitor_MPI, 'V_2_slice, V_ph_slice and V_pp_slice', last_monitor_time, last_iteration_time, print_redirect_MPI)
				
				last_monitor_time=time()
				self_energy += selfEnergyPart(m)
				dur_selfEnergy += time()-last_monitor_time
				cpuTime(monitor_MPI, 'self_energy_part', last_monitor_time, last_iteration_time, print_redirect_MPI)
		
		cpuTime(monitor, 'chi_ph & chi_pp', print_redirect=print_redirect, curr_time=dur_fluctuation)
		cpuTime(monitor, 'V_2, V_ph and V_pp', print_redirect=print_redirect, curr_time=dur_effInteraction)
		cpuTime(monitor, 'self_energy', print_redirect=print_redirect, curr_time=dur_selfEnergy)
		return self_energy
	
	def selfEnergySimple(self, mu):
		'''
		defined in Pao's eq.(16), but inverse
		
		inverse of function greenMatrix
		green matrix --> self energy
		
		before FLEX converges, the last iteration is:
		G --> self_energy --> G_new --> |G_new - G| is negligible
		if G and self_energy are saved into a .npz file, then function selfEnergySimple won't be used
		if only G is saved, like some old version of this module does, then self_energy can be generated approximately by this function
		this eq is established in the last iteration:
		1 / (i*omega_n - xi - self_energy) = G_new ~ G
		so:
		self_energy ~ i*omega_n - xi - 1/G
		            = i*omega_n - (energy_band - mu) - 1/G
		where mu is solved and printed to flex.out in the last iteration
		'''
		
		dim_el=self.dim_el
		omega_count=self.omega_count
		t_hopping=self.t_hopping
		qe_output_dir=self.qe_output_dir
		n_trunc=self.n_trunc
		T=self.T
		if 'green_matrix' not in vars(self):
			green_matrix = self.greenMatExtractor(self.green_matrix_extended, n_trunc)
		else:
			green_matrix = self.green_matrix
		
		omega_n = pi*T*(
					1
					+2 * arange(-n_trunc-1, n_trunc+1)[arange(-n_trunc-1, n_trunc+1)]
				  ).reshape(omega_count,1,1,1)
		
		xi=energyBand(dim_el, qe_output_dir, t_hopping)-mu
		return 1j*omega_n - xi - 1/green_matrix
	
	def greenMatrix(self, mu=None):
		'''
		defined in Pao's eq.(16).
		
		calculate green matrix from self energy
		'''
		
		if mu==None:
			mu=self.mu
		dim_el=self.dim_el
		t_hopping=self.t_hopping
		qe_output_dir=self.qe_output_dir
		n_range=self.n_range
		T=self.T
		self_energy=self.self_energy
		
		# left only for reference
		#green_matrix=empty((omega_count,dim_el,dim_el,dim_el),complex)
		#for n in range(n_trunc+1):
		#	omega_n=(2*n+1)*pi*T
		#	green_matrix[n]=1/(1j*omega_n-xi-self_energy[n])
		#	green_matrix[-n-1]=green_matrix[n].conj() # forced time reversal symmetry: green_matrix[omega] = green_matrix[-omega].conj()
		#
		# time reversal symmetry constraint,
		# i.e. green_matrix[omega] = green_matrix[-omega].conj(),
		# i.e. green_matrix[-n-1] = green_matrix[n].conj(),
		# is relaxed and only checked by the calling function after `green_matrix` returned
		omega_n=(2*n_range[:,None,None,None]+1)*pi*T
		xi=energyBand(dim_el, qe_output_dir, t_hopping)-mu
		green_matrix=evaluate('1/(1j*omega_n-xi-self_energy)')
		return green_matrix
	
	def greenMatFFT(self):
		'''
		calculate Discrete Fourier transform (DFT) of green_matrix and its conjugate for use in convolution theorem, which will accelerate summation in fluctuationSlice() and selfEnergyPart()
		'''
		
		omega_count=self.omega_count
		dim_el=self.dim_el
		n_trunc=self.n_trunc
		
		# no need to initialize green_matrix_conj_fft
		#self.green_matrix_conj_fft=empty((omega_count,dim_el,dim_el,dim_el),complex)
		self.green_matrix_fft=empty((omega_count,dim_el,dim_el,dim_el),complex)
		for n in range(-n_trunc-1, n_trunc+1):
			self.green_matrix_fft[n]=fftn(self.green_matrix[n])
			#self.green_matrix_conj_fft[n]=fftn(self.green_matrix[n].conj())
		self.green_matrix_conj_fft=self.green_matrix_fft[::-1] # =flipud(self.green_matrix_fft)
	
	@staticmethod
	def greenMatExtractor(green_matrix_extended, n_trunc):
		'''
		extract green_matrix from green_matrix_extended via removing redundant 0 items
		'''
		
		# obsolete because of too low efficiency, left here only for reference
		#green_matrix=empty((omega_count,dim_el,dim_el,dim_el),complex)
		#green_matrix[0:n_trunc+1]=green_matrix_extended[0:n_trunc+1]
		#green_matrix[-n_trunc-1:]=green_matrix_extended[-n_trunc-1:]
		#
		# right but tricky
		# the 1st pair of brackets are used to extract
		# the 2nd ... sort
		return green_matrix_extended[arange(-n_trunc-1,n_trunc+1)][arange(-n_trunc-1,n_trunc+1)]
	
	@staticmethod
	def pickMixing(mixing_list, mixing_group_size_list, step):
		'''
		pick up a mixing from mixing_list according to mixing_group_size_list and step count
		p.s. new and old thing will be mixed with ratio (1-mixing):mixing
		
		num of mixing_group_size_list[i] iterations will use mixing_list[i] as mixing
		'''
		
		hit=step
		for i in range(len(mixing_group_size_list)):
			hit-=mixing_group_size_list[i]
			if hit<0:
				mixing=mixing_list[i]
				break
		if hit>=0:
			raise ValueError(f'step={step} >= tot_num_of_iter')
		else:
			return mixing
	
	def FLEXMix(self, step):
		'''
		pick up a mixing from flex_mixing_list via pickMixing()
		'''
		
		return self.pickMixing(self.flex_mixing_list, self.flex_mixing_group_size_list, step)
	
	def MigdalSCMix(self, step):
		'''
		pick up a mixing from MigdalSC_mixing_list via pickMixing()
		'''
		
		return self.pickMixing(self.MigdalSC_mixing_list, self.MigdalSC_mixing_group_size_list, step)
	
	def nOccupation(self, green_matrix):
		'''
		calculate occupation number per lattice site from green_matrix
		
		不乘2给出的是均格点均自旋占据数
		*2给出的是均格点占据数，此时半满对应n=1
		'''
		
		T=self.T
		dim_el=self.dim_el
		
		return 2*(.5+green_matrix.real.sum()*T/pow(dim_el,3))
	
	def nDeltaFromFixedVal(self, mu):
		'''
		calculate occupation number from mu, then return its deviation from n_occup_fixed
		'''
		
		nOccupation=self.nOccupation
		greenMatrix=self.greenMatrix
		n_occup_fixed=self.n_occup_fixed
		
		return nOccupation(greenMatrix(mu))-n_occup_fixed
	
	def muSolver(self):
		'''
		solve mu according to nDeltaFromFixedVal(self, mu)=0
		'''
		
		mu_sample_num=self.mu_sample_num
		mu_samples=self.mu_samples
		nDeltaFromFixedVal=self.nDeltaFromFixedVal
		print_redirect=self.print_redirect
		
		mu_func_samples=list(map(nDeltaFromFixedVal, mu_samples))
		print("(mu_sample, mu_func_sample)'s:", list(map(tuple, array([mu_samples,mu_func_samples]).transpose())), file=print_redirect)
		mu_min_new=None
		mu_max_new=None
		for i in range(mu_sample_num):
			if mu_func_samples[i] < 0 and mu_min_new == None:
				mu_min_new=mu_samples[i]
			elif mu_func_samples[i] > 0 and mu_max_new == None:
				mu_max_new=mu_samples[i]
			elif mu_func_samples[i] == 0:
				print('luckily hit solution of mu:', mu_samples[i], file=print_redirect)
				return mu_samples[i]
			if mu_min_new != None and mu_max_new != None:
				print('using new interval to solve mu: [{:.3f}, {:.3f}]'.format(mu_min_new, mu_max_new), file=print_redirect)
				mu_detail=brentq(nDeltaFromFixedVal, mu_min_new, mu_max_new, full_output=True)
				print('solution of mu:',mu_detail, file=print_redirect)
				mu=mu_detail[0]
				return mu
		return 'F'
	
	def saveData(self, file_name):
		'''
		save data to file_name, which will be overwritten during every iteration of selfConsistency()
		'''
		
		if self.rank==0:
			savez_compressed(self.working_dir+'/'+file_name, self_energy=self.self_energy, green_matrix=self.green_matrix)
	
	def selfConsistency(self):
		'''
		the meaning of being of this whole package
		
		self-consistency procedure of FLEX
		
		self-consistency equation is Σ=f(Σ) and below is algorithm flowchart
		in this algorithm flowchart:
		+ Σ_init = self_energy
		+ Σ_new = self_energy_new
		+ Σ_init' = self_energy
		+ Σ_init_lastStep = self_energy_init, backed up from Σ_init before generating Σ_init' in the last iteration
		+ Σ_new_lastStep = self_energy_new
		
		 Σ_init = re-mix(Σ_init_lastStep, Σ_new_lastStep)
		--------<-------<-------
		|                      |(N, can't solve)
		|                      |
		Σ_init --> solve mu (Y)--> G --> Σ_new=f(Σ_init) --> |Σ_new-Σ_init|==0 (Y)--> DONE and output Σ_init
		                                                                        |
		                                                                        |(N)
		                     Σ_init' = mix(Σ_init, Σ_new)                       |
		-----------<--------------------------------------------------<----------
		|
		|
		Σ_init' ...
		'''
		
		omega_count=self.omega_count
		dim_el=self.dim_el
		dim_el_min=self.dim_el_min
		dim_el_max=self.dim_el_max
		init_npz_dir=self.init_npz_dir
		n_occup_flag=self.n_occup_flag
		n_occup_fixed=self.n_occup_fixed
		muSolver=self.muSolver
		mu=self.mu
		greenMatrix=self.greenMatrix
		flex_tot_num_of_iter=self.flex_tot_num_of_iter
		selfEnergyThisProcess=self.selfEnergyThisProcess
		n_trunc=self.n_trunc
		FLEXMix=self.FLEXMix
		flex_delta_ratio_criteria=self.flex_delta_ratio_criteria
		flex_delta_abs_criteria=self.flex_delta_abs_criteria
		nOccupation=self.nOccupation
		solve_eig_during_FLEX=self.solve_eig_during_FLEX
		saveData=self.saveData
		greenMatExtractor=self.greenMatExtractor
		greenMatFFT=self.greenMatFFT
		monitor=self.monitor
		print_redirect=self.print_redirect
		
		# initialize self energy
		if init_npz_dir:
			with load(init_npz_dir+'/init.npz') as npz:
				try:
					self_energy=npz['self_energy']
				except:
					raise RuntimeError("can't find `self_energy` in init.npz")
		else:
			self_energy=zeros((omega_count,dim_el,dim_el,dim_el),complex)
		
		# self-consistent procedure
		print('\n%s\nFLEX self-consistent procedure starting\n'%delimiter_top_level, file=print_redirect)
		criteria=False
		num_of_iteration=0
		num_of_remixing=0
		start_time=time()
		last_iteration_time=start_time
		while not criteria and num_of_iteration < flex_tot_num_of_iter:
			
			print('%s\nthe %d-th FLEX self-consistent iteration'%(delimiter_iteration,num_of_iteration), file=print_redirect)
			
			# generate green function, i.e. green_matrix
			# if mu can't be solved, then try re-mixing with larger ratio of self_energy_init
			# mu is already solved from green_matrix(= self_energy_init in the beginning of current iteration) in the last iteration,
			# so mu will be solved if green_matrix approaches self_energy_init
			self.self_energy=self_energy # only for greenMatrix
			last_monitor_time=time()
			if n_occup_flag:
				print('', file=print_redirect)
				mu=muSolver()
				print('', file=print_redirect)
				if mu == 'F':
					if num_of_iteration == 0:
						raise RuntimeError("can't solve mu in the current interval")
					else:
						mixing=(1+mixing)/2 # will approach 1 from left after infinite iterations given initial mixing < 1
						self_energy = (1-mixing) * self_energy_new + mixing * self_energy_init
						print('the %d-th re-mixing\n%s\n'%(num_of_remixing, delimiter_iteration), file=print_redirect)
						num_of_remixing+=1
						continue
				else:
					num_of_remixing=0
			self.green_matrix=greenMatrix(mu) # only for greenMatFFT
			cpuTime(monitor, 'green_matrix', last_monitor_time, last_iteration_time, print_redirect)
			if not n_occup_flag:
				print(f'n_occup={nOccupation(self.green_matrix):.9e}', file=print_redirect) # occupancy of electron
			
			# save essential data for the current iteration
			saveData('tmp.npz')
			
			# check how well time reversal symmetry, i.e.
			# green_matrix[-1:-n_trunc-2:-1]=green_matrix[0:n_trunc+1].conj()
			# self_energy[-1:-n_trunc-2:-1]=self_energy[0:n_trunc+1].conj()
			# is satisfied
			print('\ncheck time reversal symmetry', file=print_redirect)
			print(f'{abs(self.self_energy.real[0:n_trunc+1]-self.self_energy.real[-1:-n_trunc-2:-1]).max()=}\n{abs(self.self_energy.imag[0:n_trunc+1]+self.self_energy.imag[-1:-n_trunc-2:-1]).max()=}', file=print_redirect)
			print(f'{abs(self.green_matrix.real[0:n_trunc+1]-self.green_matrix.real[-1:-n_trunc-2:-1]).max()=}\n{abs(self.green_matrix.imag[0:n_trunc+1]+self.green_matrix.imag[-1:-n_trunc-2:-1]).max()=}\n', file=print_redirect)
			
			# generate new self energy, i.e. self_energy_new, for convergence test
			#
			last_monitor_time=time()
			greenMatFFT()
			cpuTime(monitor, 'green_matrix_fft & green_matrix_conj_fft', last_monitor_time, last_iteration_time, print_redirect)
			#
			# green_matrix is calculated in the last step
			# self_energy calculated in THIS step is derived from green_matrix
			self_energy_new=selfEnergyThisProcess(last_iteration_time, num_of_iteration)
			#
			# sum over `self_energy` across all processes to get self energy
			print('\ncollecting self_energy from all processes...', file=print_redirect)
			last_monitor_time=time()
			COMM_WORLD.Allreduce(IN_PLACE, self_energy_new, SUM)
			cpuTime(monitor, 'MPI collects all self_energy', last_monitor_time, last_iteration_time, print_redirect)
			
			# convergence test
			# |Σ_new-Σ_init|/|Σ_init| and |Σ_new-Σ_init|
			# Σ_new=self_energy_new is the self energy generated before mixing in the current iteration, i.e. just now
			# Σ_init=self_energy is the self energy initially used in the current iteration, i.e. the self energy generated after last iteration
			# update the 1st one when neither of Σ_new or Σ_init equals to zero
			# otherwise update the 2nd one
			# both should be negligible
			print('', file=print_redirect)
			self_energy_delta_ratio=-1
			self_energy_delta_abs=-1
			last_monitor_time=time()
			for n in range(-n_trunc-1, n_trunc+1):
				for i in range(dim_el_min,dim_el_max):
					for j in range(dim_el_min,dim_el_max):
						for k in range(dim_el_min,dim_el_max):
							if self_energy[n,i,j,k]!=0 and self_energy_new[n,i,j,k]!=0:
								self_energy_delta_ratio=max(self_energy_delta_ratio,abs(self_energy_new[n,i,j,k]-self_energy[n,i,j,k])/abs(self_energy[n,i,j,k]))
							else:
								self_energy_delta_abs=max(self_energy_delta_abs,abs(self_energy_new[n,i,j,k]-self_energy[n,i,j,k]))
			cpuTime(monitor, 'convergence criteria of FLEX', last_monitor_time, last_iteration_time, print_redirect)
			criteria=self_energy_delta_ratio<flex_delta_ratio_criteria and self_energy_delta_abs<flex_delta_abs_criteria
			print('self_energy_delta_ratio=%.9e\nself_energy_delta_abs=%.9e\nconverged=%s'%(self_energy_delta_ratio, self_energy_delta_abs, criteria), file=print_redirect)
			
			if not criteria:
				# mix self_energy_new and self_energy for use in the next iteration
				self_energy_init=self_energy # back up for possible re-mixing in the next iteration
				mixing=FLEXMix(num_of_iteration)
				print(f'\nusing {mixing=} to generate self_energy for next iteration', file=print_redirect)
				self_energy = (1-mixing) * self_energy_new + mixing * self_energy
				
				# solve max eigen value after each iteration
				if solve_eig_during_FLEX:
					last_monitor_time=time()
					for state in ['SC', 'CDW']:
						_=self.maxEig(state)
					print('', file=print_redirect)
					cpuTime(monitor, 'max eigen value', last_monitor_time, last_iteration_time, print_redirect)
			
			num_of_iteration+=1
			curr_time=time()
			print('\ncpu time of the current iteration: %.1fs\ncpu time of FLEX self-consistent procedure (tot, till now): %.1fs\n%s\n'%(curr_time-last_iteration_time, curr_time-start_time, delimiter_iteration), file=print_redirect)
			last_iteration_time=curr_time
		
		if not criteria:
			raise RuntimeError("can't converge in %d iterations"%flex_tot_num_of_iter)
		print('FLEX self-consistent procedure DONE\n%s\n'%delimiter_top_level, file=print_redirect)
	
	def VPairSC(self):
		'''
		revised VPairSCPao() by Luo
		
		calculate pair potential for superconduct
		'''
		
		v_M=self.v_M
		omega_count=self.omega_count
		dim_el=self.dim_el
		dim_el_min=self.dim_el_min
		dim_el_max=self.dim_el_max
		m_trunc=self.m_trunc
		vp=self.vp
		vDMatrix=self.vDMatrix
		dim_ph=self.dim_ph
		dim_ph_min=self.dim_ph_min
		dim_ph_max=self.dim_ph_max
		I_omega_count=self.I_omega_count
		n_trunc=self.n_trunc
		chi_ph=self.chi_ph
		
		v_M_modified=flipud(v_M.transpose())
		
		# initialize superconduct pair potential needed to be filled up via D_matrix and M_matrix
		# the 8 dimensions of V_pair_sc are: n, k_vec, n', k_vec'
		V_pair_sc=zeros((omega_count,dim_el,dim_el,dim_el,omega_count,dim_el,dim_el,dim_el), complex)
		
		# m in this for-loop = n of (omega_n of k - omega_n' of k'), i.e. n-n'
		for m in range(-m_trunc,m_trunc+1):
			V_pair_sc_part1=.5*vp(m)
			v_D=vDMatrix(m)
			v_D_modified=flipud(v_D.transpose())
			for i in range(dim_ph_min,dim_ph_max):
				for j in range(dim_ph_min,dim_ph_max):
					for k in range(dim_ph_min,dim_ph_max):
						
						# R_matrix_inv = R * inv(1+R) = 1 - inv(1+R), where R = D, M, S or T
						D_matrix_inv=(I_omega_count-inv(I_omega_count+v_D*chi_ph[:,m,i,j,k])).A
						M_matrix_inv=(I_omega_count-inv(I_omega_count+v_M*chi_ph[:,m,i,j,k])).A
						
						# range of n' in V_pair(k,k'), where k=(n,i,j,k) and k'=(n',i',j',k')
						n_range_sorted=array(sorted(set(range(-n_trunc-1, n_trunc+1)) & set(range(-n_trunc-1-m, n_trunc-m+1))))
						
						# obsolete for low efficiency
						#for n in n_range_sorted:
						#	# transpose v_D and flip columns according to omega_n
						#	# it's NOT offsetted diag because n could be negative
						#	V_pair_sc_part2=-.5*dot(D_matrix_inv[m+n,:],v_D_modified[n,:])
						#	V_pair_sc_part3=1.5*dot(M_matrix_inv[m+n,:],v_M_modified[n,:])
						#	for a in range(dim_el_min,dim_el_max):
						#		for b in range(dim_el_min,dim_el_max):
						#			for c in range(dim_el_min,dim_el_max):
						#				V_pair_sc[m+n,i+a,j+b,k+c,n,a,b,c] = V_pair_sc_part1 + V_pair_sc_part2 + V_pair_sc_part3
						#
						V_pair_sc[
							(m+n_range_sorted).reshape(n_range_sorted.shape[0],1,1,1),
							(i+arange(dim_el_min,dim_el_max)).reshape(1,dim_el,1,1),
							(j+arange(dim_el_min,dim_el_max)).reshape(1,1,dim_el,1),
							(k+arange(dim_el_min,dim_el_max)).reshape(1,1,1,dim_el),
							n_range_sorted.reshape(n_range_sorted.shape[0],1,1,1),
							arange(dim_el_min,dim_el_max).reshape(1,dim_el,1,1),
							arange(dim_el_min,dim_el_max).reshape(1,1,dim_el,1),
							arange(dim_el_min,dim_el_max).reshape(1,1,1,dim_el)
						] = V_pair_sc_part1 + \
								-.5 * einsum('ij,ij -> i', D_matrix_inv[m+n_range_sorted], v_D_modified[n_range_sorted]).reshape(n_range_sorted.shape[0],1,1,1) + \
								1.5 * einsum('ji,ji -> j', M_matrix_inv[m+n_range_sorted], v_M_modified[n_range_sorted]).reshape(n_range_sorted.shape[0],1,1,1) # 'ij,ij -> i' and 'ji,ji -> j' in einsum do the same thing
		
		# m in this for-loop = n of (omega_n of k + omega_n' of k'), i.e. n+n'+1
		for m in range(-m_trunc,m_trunc+1):
			n_range_sorted=sorted(set(range(-n_trunc-1, n_trunc+1)) & set(range(m-1-n_trunc, m+n_trunc+1)))
			for n in n_range_sorted:
				V_pair_sc[m-1-n,:,:,:,n,...] += .5*vp(m)
		
		return V_pair_sc
	
	def VPairSCPao(self):
		'''
		defined in Pao's eq.(19), but stored as 5-order tensor [n,m,qx,qy,qz] and with (m+m_trunc) % size == rank
		
		Pao's eq.(19) defines 8-order tensor V_pair(k,k'), but it's a 5-order tensor in effect
		and so is stored as V_pair_sc[n,m,qx,qy,qz], where
		n belongs to k and [m,qx,qy,qz] = k-k'
		'''
		
		v_M=self.v_M
		omega_count=self.omega_count
		nu_count=self.nu_count
		dim_el=self.dim_el
		m_trunc=self.m_trunc
		vp=self.vp
		vDMatrix=self.vDMatrix
		dim_ph=self.dim_ph
		dim_ph_min=self.dim_ph_min
		dim_ph_max=self.dim_ph_max
		I_omega_count=self.I_omega_count
		n_trunc=self.n_trunc
		T=self.T
		green_matrix_fft=self.green_matrix_fft
		green_matrix_conj_fft=self.green_matrix_conj_fft
		size=self.size
		rank=self.rank
		
		v_M_modified=flipud(v_M.transpose())
		
		# 5-order tensor V_pair for SC is stored across all processes, each owning part of it
		# `V_pair_sc` for m is stored in `V_pair_sc[:,(m+m_trunc)//self.size]` instead of `V_pair_sc[:,m]`
		V_pair_sc=empty((
							omega_count,
							nu_count//size + ( rank < nu_count%size ),
							dim_ph,dim_ph,dim_ph
						),
						complex)
		
		# m in this for-loop = n of (omega_n of k - omega_n' of k'), i.e. n-n'
		for m in range(-m_trunc, m_trunc+1):
			if (m+m_trunc) % size == rank:
				v_D=vDMatrix(m)
				v_D_modified=flipud(v_D.transpose())
				
				# chi_ph for the current m, the same as `chi_ph_slice` in `fluctuationSlice()`
				chi_ph_slice=zeros((omega_count,dim_ph,dim_ph,dim_ph), complex)
				for n in set(range(-n_trunc-1, n_trunc+1)) & set(range(-n_trunc-1-m, n_trunc+1-m)):
					chi_ph_slice[n] = -T/dim_el**3 * ifftn(green_matrix_fft[n+m] * green_matrix_conj_fft[n].conj())
				
				# WARN! assign meaningful value to empty array `V_pair_sc` before `+=` operation
				V_pair_sc[:,(m+m_trunc)//size] = .5*vp(m)
				for i in range(dim_ph_min,1):
					for j in range(i,1):
						for k in range(j,1):
							
							# the symmetry of cubic lattice is Pm-3m, the 517-th item in `Table A1.4.2.7` of "International Tables for Crystallography - Volume B: Reciprocal Space"
							# the space-group number of Pm-3m is 221
							symm_ops=get_symmetry_from_database(517)['rotations']
							
							# `equiv_idx` has 48 columns before being `unique`d, all of which are equivalent positions in k-space
							# `transpose` will only return a view, see `numpy.transpose` and `__array_interface__` for reference
							equiv_idx=unique(symm_ops.dot(array([i,j,k])), axis=0).transpose()
							
							# R_matrix_inv = R * inv(1+R) = 1 - inv(1+R), where R = D, M, S or T
							D_matrix_inv=(I_omega_count-inv(I_omega_count+v_D*chi_ph_slice[:,i,j,k])).A
							M_matrix_inv=(I_omega_count-inv(I_omega_count+v_M*chi_ph_slice[:,i,j,k])).A
							
							# range of n' in V_pair(k,k'), where k=(n,i,j,k) and k'=(n',i',j',k')
							n_range=array(list(set(range(-n_trunc-1, n_trunc+1)) & set(range(-n_trunc-1-m, n_trunc-m+1)))).reshape(-1,1)
							
							V_pair_sc[
									m+n_range,
									(m+m_trunc)//size,
									equiv_idx[0],equiv_idx[1],equiv_idx[2]
									] += -.5 * D_matrix_inv[m+n_range, n_range] * v_D_modified[m+n_range, n_range] \
										+ 1.5 * M_matrix_inv[m+n_range, n_range] * v_M_modified[m+n_range, n_range]
		
		return V_pair_sc
	
	def VPairCDW(self):
		'''
		Luo's pairing potential for CDW
		'''
		
		omega_count=self.omega_count
		dim_el=self.dim_el
		dim_el_min=self.dim_el_min
		dim_el_max=self.dim_el_max
		U_p=self.U_p
		m_trunc=self.m_trunc
		vp=self.vp
		vDMatrix=self.vDMatrix
		dim_ph=self.dim_ph
		dim_ph_min=self.dim_ph_min
		dim_ph_max=self.dim_ph_max
		v_M=self.v_M
		I_omega_count=self.I_omega_count
		n_trunc=self.n_trunc
		vSMatrix=self.vSMatrix
		vTMatrix=self.vTMatrix
		chi_ph=self.chi_ph
		chi_pp=self.chi_pp
		
		#v_M_modified=flipud(v_M.transpose())
		V_pair_cdw=full((omega_count,dim_el,dim_el,dim_el,omega_count,dim_el,dim_el,dim_el), 2*U_p, complex)
		
		for m in range(-m_trunc,m_trunc+1):
			
			# part of V_pair_cdw
			V_pair_cdw_part1=-vp(m)
			
			#v_D_modified=flipud(v_D.transpose())
			#v_S_modified=flipud(v_S.transpose())
			#v_T_modified=flipud(v_T.transpose())
			v_D=vDMatrix(m)
			v_S=vSMatrix(m)
			v_T=vTMatrix(m)
			
			for i in range(dim_ph_min,dim_ph_max):
				for j in range(dim_ph_min,dim_ph_max):
					for k in range(dim_ph_min,dim_ph_max):
						
						# part of V_pair_cdw
						V_pair_cdw_part2=2*vp(m)**2*chi_ph[:,m,i,j,k].sum()
						
						D_matrix=v_D*chi_ph[:,m,i,j,k]
						M_matrix=v_M*chi_ph[:,m,i,j,k]
						S_matrix=v_S*chi_pp[:, m, i+dim_ph_max, j+dim_ph_max, k+dim_ph_max]
						T_matrix=v_T*chi_pp[:, m, i+dim_ph_max, j+dim_ph_max, k+dim_ph_max]
						
						# R_matrix_inv = R * inv(1+R) = 1 - inv(1+R), where R = D, M, S or T
						D_matrix_inv=(I_omega_count-inv(I_omega_count+D_matrix)).A
						M_matrix_inv=(I_omega_count-inv(I_omega_count+M_matrix)).A
						S_matrix_inv=(I_omega_count-inv(I_omega_count+S_matrix)).A
						T_matrix_inv=(I_omega_count-inv(I_omega_count+T_matrix)).A
						
						n_range_sorted_1=array(sorted(set(range(-n_trunc-1, n_trunc+1)) & set(range(-n_trunc-1-m, n_trunc-m+1))))
						n_range_sorted_2=array(sorted(set(range(-n_trunc-1, n_trunc+1)) & set(range(m-1-n_trunc, m+n_trunc+1))))
						
						# m in this part = n of (omega_n of k - omega_n' of k'), i.e. n-n'
						#
						# vp(array) is also an array
						# vp(array)[i,j] = vp(array[i,j])
						#
						# obsolete for low efficiency
						#for n in n_range_sorted_1:
						#	V_pair_cdw_part3=-vp(m)*dot(chi_ph[:,m,i,j,k], array(list(map(lambda n_double_prime: vp(n-n_double_prime), range(-n_trunc-1, n_trunc+1)))))
						#	V_pair_cdw_part4=.5*dot(D_matrix_inv[m+n,:], v_D[:,m+n])
						#	V_pair_cdw_part5=1.5*dot(M_matrix_inv[m+n,:], v_M[:,m+n])
						#	for a in range(dim_el_min,dim_el_max):
						#		for b in range(dim_el_min,dim_el_max):
						#			for c in range(dim_el_min,dim_el_max):
						#				V_pair_cdw[m+n,i+a,j+b,k+c,n,a,b,c] -= V_pair_cdw_part1 + V_pair_cdw_part2 + V_pair_cdw_part3 + V_pair_cdw_part4 + V_pair_cdw_part5
						#
						V_pair_cdw[
							(m+n_range_sorted_1).reshape(n_range_sorted_1.shape[0],1,1,1),
							(i+arange(dim_el_min,dim_el_max)).reshape(1,dim_el,1,1),
							(j+arange(dim_el_min,dim_el_max)).reshape(1,1,dim_el,1),
							(k+arange(dim_el_min,dim_el_max)).reshape(1,1,1,dim_el),
							n_range_sorted_1.reshape(n_range_sorted_1.shape[0],1,1,1),
							arange(dim_el_min,dim_el_max).reshape(1,dim_el,1,1),
							arange(dim_el_min,dim_el_max).reshape(1,1,dim_el,1),
							arange(dim_el_min,dim_el_max).reshape(1,1,1,dim_el)
						] -= V_pair_cdw_part1 + V_pair_cdw_part2 + \
								-vp(m) * (
									chi_ph[:,m,i,j,k] @
									vp(n_range_sorted_1 - arange(-n_trunc-1, n_trunc+1).reshape(omega_count, 1))
								).reshape(n_range_sorted_1.shape[0],1,1,1) + \
								.5 * einsum('ij,ji -> i', (D_matrix_inv-D_matrix)[m+n_range_sorted_1], v_D[:,m+n_range_sorted_1]).reshape(n_range_sorted_1.shape[0],1,1,1) + \
								1.5 * einsum('ij,ji -> i', (M_matrix_inv-M_matrix)[m+n_range_sorted_1], v_M[:,m+n_range_sorted_1]).reshape(n_range_sorted_1.shape[0],1,1,1)
						
						# m in this part = n of (omega_n of k + omega_n' of k'), i.e. n+n'+1
						#
						# obsolete for low efficiency
						#for n in n_range_sorted_2:
						#	V_pair_cdw_part_1=-dot(S_matrix_inv[m-1-n,:],v_S[:,m-1-n])
						#	V_pair_cdw_part_2=-3*dot(T_matrix_inv[m-1-n,:],v_T[:,m-1-n])
						#	for a in range(dim_el_min,dim_el_max):
						#		for b in range(dim_el_min,dim_el_max):
						#			for c in range(dim_el_min,dim_el_max):
						#				V_pair_cdw[m-1-n,i-a,j-b,k-c,n,a,b,c] -= V_pair_cdw_part_1 + V_pair_cdw_part_2
						#
						V_pair_cdw[
							(m-1-n_range_sorted_2).reshape(n_range_sorted_2.shape[0],1,1,1),
							(i-arange(dim_el_min,dim_el_max)).reshape(1,dim_el,1,1),
							(j-arange(dim_el_min,dim_el_max)).reshape(1,1,dim_el,1),
							(k-arange(dim_el_min,dim_el_max)).reshape(1,1,1,dim_el),
							n_range_sorted_2.reshape(n_range_sorted_2.shape[0],1,1,1),
							arange(dim_el_min,dim_el_max).reshape(1,dim_el,1,1),
							arange(dim_el_min,dim_el_max).reshape(1,1,dim_el,1),
							arange(dim_el_min,dim_el_max).reshape(1,1,1,dim_el)
						] -= -einsum('ij,ji -> i', (S_matrix_inv-S_matrix)[m-1-n_range_sorted_2],v_S[:,m-1-n_range_sorted_2]).reshape(n_range_sorted_2.shape[0],1,1,1) + \
								-3 * einsum('ij,ji -> i', (T_matrix_inv-T_matrix)[m-1-n_range_sorted_2],v_T[:,m-1-n_range_sorted_2]).reshape(n_range_sorted_2.shape[0],1,1,1)
		
		# S and T term in V_pair_cdw can be calculated together with D and M term, no need to seperate them into different for-loop
		## m in this for-loop = n of (omega_n of k + omega_n' of k'), i.e. n+n'+1
		#for m in range(-m_trunc,m_trunc+1):
		#	v_S=vSMatrix(m)
		#	v_T=vTMatrix(m)
		#	#v_S_modified=flipud(v_S.transpose())
		#	#v_T_modified=flipud(v_T.transpose())
		#	for i in range(dim_ph_min,dim_ph_max):
		#		for j in range(dim_ph_min,dim_ph_max):
		#			for k in range(dim_ph_min,dim_ph_max):
		#				S_matrix=v_S*chi_pp[:, m, i+dim_ph_max, j+dim_ph_max, k+dim_ph_max]
		#				T_matrix=v_T*chi_pp[:, m, i+dim_ph_max, j+dim_ph_max, k+dim_ph_max]
		#				
		#				# R_matrix_inv = R * inv(1+R) = 1 - inv(1+R), where R = D, M, S or T
		#				S_matrix_inv=I_omega_count-inv(I_omega_count+S_matrix)-S_matrix
		#				T_matrix_inv=I_omega_count-inv(I_omega_count+T_matrix)-T_matrix
		#				
		#				n_range_sorted=array(sorted(set(range(-n_trunc-1, n_trunc+1)) & set(range(m-1-n_trunc, m+n_trunc+1))))
		#				
		#				# obsolete for low efficiency
		#				#for n in n_range_sorted:
		#				#	V_pair_cdw_part_1=-dot(S_matrix_inv[m-1-n,:],v_S[:,m-1-n])
		#				#	V_pair_cdw_part_2=-3*dot(T_matrix_inv[m-1-n,:],v_T[:,m-1-n])
		#				#	for a in range(dim_el_min,dim_el_max):
		#				#		for b in range(dim_el_min,dim_el_max):
		#				#			for c in range(dim_el_min,dim_el_max):
		#				#				V_pair_cdw[m-1-n,i-a,j-b,k-c,n,a,b,c] -= V_pair_cdw_part_1 + V_pair_cdw_part_2
		#				#
		#				V_pair_cdw[
		#					(m-1-n_range_sorted).reshape(n_range_sorted.shape[0],1,1,1),
		#					(i-arange(dim_el_min,dim_el_max)).reshape(1,dim_el,1,1),
		#					(j-arange(dim_el_min,dim_el_max)).reshape(1,1,dim_el,1),
		#					(k-arange(dim_el_min,dim_el_max)).reshape(1,1,1,dim_el),
		#					n_range_sorted.reshape(n_range_sorted.shape[0],1,1,1),
		#					arange(dim_el_min,dim_el_max).reshape(1,dim_el,1,1),
		#					arange(dim_el_min,dim_el_max).reshape(1,1,dim_el,1),
		#					arange(dim_el_min,dim_el_max).reshape(1,1,1,dim_el)
		#				] -= -einsum('ij,ji -> i', S_matrix_inv[m-1-n_range_sorted],v_S[:,m-1-n_range_sorted]).reshape(n_range_sorted.shape[0],1,1,1) + \
		#						-3 * einsum('ij,ji -> i', T_matrix_inv[m-1-n_range_sorted],v_T[:,m-1-n_range_sorted]).reshape(n_range_sorted.shape[0],1,1,1)
		
		return V_pair_cdw
	
	def maxEig(self, state):
		'''
		calculate eigenvalue with max norm according to Pao's eq.(18)
		'''
		
		T=self.T
		dim_el=self.dim_el
		green_matrix=self.green_matrix
		inv_matrix_slice=self.inv_matrix_slice
		monitor=self.monitor
		transMatSlices=self.transMatSlices
		VPairSC=self.VPairSC
		VPairSCPao=self.VPairSCPao
		VPairCDW=self.VPairCDW
		omega_count=self.omega_count
		m_trunc=self.m_trunc
		eigen_vec_diff_criteria=self.eigen_vec_diff_criteria
		eigen_val_norm_criteria=self.eigen_val_norm_criteria
		power_method_seq_update_threshold=self.power_method_seq_update_threshold
		print_redirect=self.print_redirect
		
		print('%s\nsolving the max-norm eigen value lambda(T) for %s\n'%(delimiter_top_level, state), file=print_redirect)
		if state == 'SC':
			# obsolete because of too low efficiency, left here only for reference
			#green_matrix_auxiliary_sc=array(list(map(shapeInvMatrix, flipud(green_matrix))))
			# obsolete, faster but not enough
			#green_matrix_auxiliary_sc=green_matrix[::-1,::-1,::-1,::-1][:,inv_matrix_indices[0], inv_matrix_indices[1], inv_matrix_indices[2]]
			last_monitor_time=time()
			green_matrix_auxiliary=green_matrix[::-1, inv_matrix_slice, ...][..., inv_matrix_slice, :][..., inv_matrix_slice]
			cpuTime(monitor, 'green_matrix_auxiliary', last_monitor_time, print_redirect=print_redirect)
			last_monitor_time=time()
			V_pair_G_G = (-T/pow(dim_el,3) * VPairSC() * green_matrix * green_matrix_auxiliary).reshape(omega_count*pow(dim_el,3), -1)
			cpuTime(monitor, 'V_pair_G_G', last_monitor_time, print_redirect=print_redirect)
			print('', file=print_redirect)
			eig_result = powerMethodGeneral(V_pair_G_G, eigen_vec_diff_criteria, eigen_val_norm_criteria, self.power_method_seq_sc, power_method_seq_update_threshold, monitor, print_redirect)
			print('the max eigen value lambda(T) is SOLVED for %s\n%s\n'%(state, delimiter_top_level), file=print_redirect)
			return eig_result
		elif state == 'SCPao':
			last_monitor_time=time()
			green_matrix_auxiliary=green_matrix[::-1, inv_matrix_slice, ...][..., inv_matrix_slice, :][..., inv_matrix_slice]
			cpuTime(monitor, 'green_matrix_auxiliary', last_monitor_time, print_redirect=print_redirect)
			last_monitor_time=time()
			V_pair=VPairSCPao()
			cpuTime(monitor, 'V_pair', last_monitor_time, print_redirect=print_redirect)
			print('', file=print_redirect)
			eig_result = powerMethodPao(V_pair, m_trunc, -T/dim_el**3 * green_matrix * green_matrix_auxiliary, eigen_vec_diff_criteria, eigen_val_norm_criteria, self.power_method_seq_sc, power_method_seq_update_threshold, monitor, print_redirect)
			print('the max eigen value lambda(T) is SOLVED for %s\n%s\n'%(state, delimiter_top_level), file=print_redirect)
			return eig_result
		elif state =='CDW':
			# obsolete because of too low efficiency, left here only for reference
			#green_matrix_auxiliary_cdw=array(list(map(lambda x: transMatrix(x, [-dim_el/2]*3), green_matrix)))
			# obsolete, faster but not enough
			#trans_matrix_indices=transMatInd([int(-dim_el/2)]*3)
			#green_matrix_auxiliary_cdw=green_matrix[:,trans_matrix_indices[0], trans_matrix_indices[1], trans_matrix_indices[2]]
			last_monitor_time=time()
			trans_matrix_slices=transMatSlices([int(-dim_el/2)]*3)
			green_matrix_auxiliary=green_matrix[:,trans_matrix_slices[0],...][..., trans_matrix_slices[1],:][..., trans_matrix_slices[2]]
			cpuTime(monitor, 'green_matrix_auxiliary', last_monitor_time, print_redirect=print_redirect)
			last_monitor_time=time()
			V_pair_G_G = (-T/pow(dim_el,3) * VPairCDW() * green_matrix * green_matrix_auxiliary).reshape(omega_count*pow(dim_el,3), -1)
			cpuTime(monitor, 'V_pair_G_G', last_monitor_time, print_redirect=print_redirect)
			print('', file=print_redirect)
			eig_result = powerMethodGeneral(V_pair_G_G, eigen_vec_diff_criteria, eigen_val_norm_criteria, self.power_method_seq_cdw, power_method_seq_update_threshold, monitor, print_redirect)
			print('the max eigen value lambda(T) is SOLVED for %s\n%s\n'%(state, delimiter_top_level), file=print_redirect)
			return eig_result
		else:
			raise ValueError('`state` is not recognized')
	
	def susceptibilityMigdalCDW(self):
		'''
		defined in Dee's eq.(15)
		'''
		
		nu_count=self.nu_count
		dim_ph=self.dim_ph
		dim_el=self.dim_el
		dim_el_max=self.dim_el_max
		n_trunc=self.n_trunc
		m_trunc=self.m_trunc
		T=self.T
		green_matrix_fft=self.green_matrix_fft
		green_matrix_conj_fft=self.green_matrix_conj_fft
		U_p=self.U_p
		size=self.size
		rank=self.rank
		print_redirect=self.print_redirect
		
		print('%s\ncalculating Migdal susceptibility(T) for CDW\n'%delimiter_top_level, file=print_redirect)
		
		# keep the whole 4-order tensor chi0 instead of the only relevant 3-order tensor chi0[0]
		# to verify Dee's declaration:
		# The CDW susceptibility is largest at zero frequency
		chi0=zeros((nu_count,dim_ph,dim_ph,dim_ph), complex)
		
		# sum over 2*chi_ph[n] for n's owned by the current process, where chi_ph is defined as Pao's eq.(14)
		# then sum over all of them across all processes to get chi0, defined as Dee's eq.(10)
		for n in range(-n_trunc-1, n_trunc+1):
			if (n+n_trunc+1) % size == rank:
				for m in set(range(-m_trunc,m_trunc+1)) & set(range(-n_trunc-1-n, n_trunc+1-n)):
					chi0[m] += -2*T/dim_el**3 * ifftn(green_matrix_fft[n+m] * green_matrix_conj_fft[n].conj())
		if rank == 0:
			COMM_WORLD.Reduce(IN_PLACE, chi0, SUM, 0)
			
			chi0_argmax=chi0.real.argmax()
			chi0_argmax_pos=decimal2OtherBaseDigits(chi0_argmax,dim_el)
			chi0_argmax_k=(fromiter(map(lambda x: x if x<dim_el_max else x-dim_el, chi0_argmax_pos), int)/dim_el*2).tolist()
			
			chi_cdw=chi0[0]/(1-U_p*chi0[0])
			chi_cdw_argmax=chi_cdw.real.argmax()
			chi_cdw_argmax_pos=decimal2OtherBaseDigits(chi_cdw_argmax,dim_el)
			chi_cdw_argmax_k=(fromiter(map(lambda x: x if x<dim_el_max else x-dim_el, chi_cdw_argmax_pos), int)/dim_el*2).tolist()
			
			print('chi0 with max real part * U_p (closer to 1, larger susceptibility): %s\nchi0.real.argmax(): %s\ncorresponding k: %s, i.e. %s*pi\n'%(chi0.flat[chi0_argmax]*U_p, chi0_argmax, chi0_argmax_pos, chi0_argmax_k), file=print_redirect)
			print('CDW susceptibility with max real part: %s\nchi_cdw.real.argmax(): %s\ncorresponding k: %s, i.e. %s*pi\n'%(chi_cdw.flat[chi_cdw_argmax], chi_cdw_argmax, chi_cdw_argmax_pos, chi_cdw_argmax_k), file=print_redirect)
			print('Migdal susceptibility(T) for CDW is SOLVED\n%s\n'%delimiter_top_level, file=print_redirect)
			return {'chi0_MigdalCDW':chi0, 'chi_cdw_MigdalCDW':chi_cdw}
			
		else:
			COMM_WORLD.Reduce(chi0, chi0, SUM, 0)
	
	def susceptibilityMigdalSC(self):
		'''
		defined in Dee's eq.(16)
		'''
		
		print_redirect=self.print_redirect
		
		print('%s\ncalculating Migdal susceptibility(T) for SC\n'%delimiter_top_level, file=print_redirect)
		self_energy_phonon=-self.U_p*self.Omega_p*self.chi_ph.sum(axis=0)
		green_matrix_phonon = -1/(
					(
						(2*pi*self.T*arange(-self.m_trunc, self.m_trunc+1)[range(-self.m_trunc-1, self.m_trunc)].reshape(self.nu_count, 1, 1, 1))**2
						+ self.Omega_p**2
					) / 2 / self.Omega_p
					+ self_energy_phonon
				)
		gamma=ones((self.omega_count, self.dim_el, self.dim_el, self.dim_el), complex)
		converge=False
		iter_count=0
		start_time=time()
		last_iteration_time=start_time
		while not converge and iter_count<self.gamma_tot_num_of_iter:
			print('%s\nthe %d-th iteration'%(delimiter_iteration,iter_count), file=print_redirect)
			gamma_new=ones((self.omega_count, self.dim_el, self.dim_el, self.dim_el), complex)
			for n in range(-self.n_trunc-1, self.n_trunc+1):
				for n_prime in set(range(-self.n_trunc-1, self.n_trunc+1)) & set(range(n-self.m_trunc, n+self.m_trunc+1)):
					# calculate sum(convolution) in Dee's eq.17
					# via
					# dft(convolution(f,g)) = dft(f) * dft(g)
					# where g is periodic and dft is discrete Fourier transform
					gamma_new[n] -= self.T/self.dim_el**3 * self.U_p*self.Omega_p/2 * ifftn(
								fftn(
									self.green_matrix[n_prime]
									* self.green_matrix[-n_prime-1][self.inv_matrix_slice][:,self.inv_matrix_slice,:][...,self.inv_matrix_slice]
									* gamma[n_prime]
								)
								* fftn(green_matrix_phonon[n-n_prime])
							)
			nonzero_idx=gamma.nonzero()
			zero_idx=(gamma==0).nonzero()
			gamma_delta_ratio=abs((gamma_new[nonzero_idx] - gamma[nonzero_idx])/gamma[nonzero_idx]).max()
			gamma_delta_abs=-1 if zero_idx[0].shape[0]==0 else abs(gamma_new[zero_idx]).max()
			converge=gamma_delta_ratio < self.MigdalSC_delta_ratio_criteria and gamma_delta_abs < self.MigdalSC_delta_abs_criteria
			print('gamma_delta_ratio=%.9e\ngamma_delta_abs=%.9e\nconverged=%s\n'%(gamma_delta_ratio,gamma_delta_abs,converge), file=print_redirect)
			
			if converge:
				gamma=gamma_new
			else:
				gamma = gamma_new*(1-self.MigdalSCMix(iter_count)) + gamma*self.MigdalSCMix(iter_count)
			iter_count+=1
			curr_time=time()
			print('cpu time of the current iteration: %.1fs\ncpu time of self-consistent procedure (tot, till now): %.1fs\n%s\n'%(curr_time-last_iteration_time, curr_time-start_time, delimiter_iteration), file=print_redirect)
			last_iteration_time=curr_time
		
		if not converge:
			raise RuntimeError("can't converge in %d iterations"%self.gamma_tot_num_of_iter)
		chi_sc=self.T/self.dim_el**3*(self.green_matrix*self.green_matrix[::-1, self.inv_matrix_slice, ...][..., self.inv_matrix_slice, :][..., self.inv_matrix_slice]*gamma).sum()
		print('SC susceptibility: %s\n'%chi_sc, file=print_redirect)
		print('Migdal susceptibility(T) for SC is SOLVED\n%s\n'%delimiter_top_level, file=print_redirect)
		return {'gamma_MigdalSC':gamma, 'chi_sc_MigdalSC':chi_sc}
