from numpy import array, zeros, ones, vdot, savez_compressed
from numpy.linalg import norm
from scipy.fft import fftn, ifftn
from time import time
from mpi4py.MPI import COMM_WORLD, IN_PLACE, SUM
from .env import delimiter_iteration, cpuTime, decimal2OtherBaseDigits

def powerMethod(updateRule, eigen_vec_diff_criteria=1e-3, eigen_val_norm_criteria=1e-3, monitor=False, print_redirect=None):
	'''
	power method, but updateRule(eigen_vec) is not limited to matrix*vector, instead customizable.
	
	function updateRule here is used to:
	1. initialize eigenvector for the 0-th iteration, when called as `updateRule()`
	2. update eigenvector from n-th to (n+1)-th iteration, when called as `updateRule(eigen_vec)`
	'''
	
	eigen_vec=updateRule()
	eigen_vec_diff=-2
	eigen_val_norm=-1
	criteria_eigen=False
	#eigen_vec_diff_criteria=1e-3 # defined in input_file_path
	#eigen_val_norm_criteria=1e-3 # defined in input_file_path
	
	# solve the max eigen value via power method
	num_of_iteration=0
	start_time=time()
	last_iteration_time=start_time
	while not criteria_eigen:
		print('%s\nthe %d-th iteration of power method'%(delimiter_iteration,num_of_iteration,), file=print_redirect)
		
		eigen_vec_last_step=eigen_vec
		eigen_vec_diff_last_step=eigen_vec_diff
		eigen_val_norm_last_step=eigen_val_norm
		
		# eigen_vec = matrix_array * eigen_vec_last_step ~= eigen value * eigen_vec_last_step = eigen_val_norm * exp(i*theta) * eigen_vec_last_step
		# in the limit of convergence, `~=` will become `=`
		# eigen_vec will be normalized to `~= exp(i*theta) * eigen_vec_last_step` as the following
		# eigen_val_norm and exp(i*theta) will be extracted
		last_monitor_time=time()
		#eigen_vec = matrix_array @ eigen_vec_last_step
		eigen_vec = updateRule(eigen_vec_last_step)
		cpuTime(monitor, 'eigen_vec', last_monitor_time, last_iteration_time, print_redirect)
		
		last_monitor_time=time()
		eigen_val_norm=norm(eigen_vec)
		cpuTime(monitor, 'eigen_val_norm', last_monitor_time, last_iteration_time, print_redirect)
		
		# normalize eigen_vec
		last_monitor_time=time()
		eigen_vec/=eigen_val_norm
		cpuTime(monitor, 'normalized eigen_vec', last_monitor_time, last_iteration_time, print_redirect)
		
		# eigen_vec_diff = eigen_vec_last_step^\{dagger} * eigen_vec ~= eigen_vec_last_step^\{dagger} * exp(i*theta) * eigen_vec_last_step = exp(i*theta)
		# in the limit of convergence, `~=` will become `=`, so
		# eigen value = eigen_val_norm * exp(i*theta) = eigen_val_norm * eigen_vec_diff
		#
		# it can be proved that for any complex vectors |a> and |b>, abs(vdot(a,b)) <= norm(a) * norm(b), i.e. |<a|b>| <= ||a>| * ||b>| = sqrt(<a|a>) * sqrt(<b|b>)
		# so, abs(eigen_vec_diff) <= norm(eigen_vec_last_step) * norm(eigen_vec) = 1
		last_monitor_time=time()
		eigen_vec_diff=vdot(eigen_vec_last_step, eigen_vec)
		cpuTime(monitor, 'eigen_vec_diff', last_monitor_time, last_iteration_time, print_redirect)
		
		# convergence test
		# according to the above conclusion about abs(eigen_vec_diff),
		# abs(1-abs(eigen_vec_diff)) < eigen_vec_diff_criteria
		# can be simplified to
		# 1-abs(eigen_vec_diff) < eigen_vec_diff_criteria
		# but this will face unpredictable numerical uncertainty, like abs(eigen_vec_diff) = 1.00001, which will cause fake True
		eigen_vec_diff_converge_test=abs(1-abs(eigen_vec_diff)) < eigen_vec_diff_criteria and abs(eigen_vec_diff-eigen_vec_diff_last_step) < eigen_vec_diff_criteria
		eigen_val_norm_converge_test=abs((eigen_val_norm-eigen_val_norm_last_step)/eigen_val_norm_last_step) < eigen_val_norm_criteria if eigen_val_norm_last_step!=0 else abs(eigen_val_norm) < eigen_val_norm_criteria
		criteria_eigen=eigen_vec_diff_converge_test and eigen_val_norm_converge_test
		print('\neigen_val_norm=%s\neigen_vec_diff=%s\nconverged=%s\n'%(eigen_val_norm,eigen_vec_diff,criteria_eigen), file=print_redirect)
		
		num_of_iteration+=1
		curr_time=time()
		print('cpu time of the current iteration: %.1fs\ncpu time of power method (tot, till now): %.1fs\n%s\n'%(curr_time-last_iteration_time, curr_time-start_time, delimiter_iteration), file=print_redirect)
		last_iteration_time=curr_time
	
	return eigen_vec, eigen_vec_diff, eigen_val_norm, eigen_vec_last_step, eigen_vec_diff_last_step, eigen_val_norm_last_step

def powerMethodSeq(updateRule, eigen_vec_diff_criteria=1e-3, eigen_val_norm_criteria=1e-3, seq=1, power_method_seq_update_threshold=1e-6, monitor=False, print_redirect=None):
	'''
	apply power method to a sequence of initial eigenvectors.
	
	again, function updateRule(eigen_vec, i) is customizable. it is used to:
	1. initialize the i-th kind of eigenvector
	2. update eigenvector for each kind of initialization in the same way
	
	so function `updateRule` receives two arguments,
	one for variable `eigen_vec_last_step` in function `powerMethod`,
	the other for a fixed index outside `powerMethod` to control initialization in it.
	'''
	
	powerMethodClosure=lambda updateRule: powerMethod(updateRule, eigen_vec_diff_criteria, eigen_val_norm_criteria, monitor, print_redirect)
	eigen_update_log=[]
	seq=int(seq)
	if seq==1:
		max_eigen_vec, max_eigen_vec_diff, max_eigen_val_norm, max_eigen_vec_last_step, max_eigen_vec_diff_last_step, max_eigen_val_norm_last_step = powerMethodClosure(lambda eigen_vec=None: updateRule(eigen_vec, 0))
		eigen_update_log.append((0, max_eigen_vec_diff, max_eigen_val_norm))
	elif seq<0:
		# negative `seq` has special meaning
		max_eigen_vec, max_eigen_vec_diff, max_eigen_val_norm, max_eigen_vec_last_step, max_eigen_vec_diff_last_step, max_eigen_val_norm_last_step = powerMethodClosure(lambda eigen_vec=None: updateRule(eigen_vec, seq))
		eigen_update_log.append((0, max_eigen_vec_diff, max_eigen_val_norm))
	else:
		max_eigen_val_norm=-1
		for i in range(seq):
			print('%s\nusing the %d-th initial eigen vector\n%s\n'%(delimiter_iteration,i,delimiter_iteration), file=print_redirect)
			eigen_vec, eigen_vec_diff, eigen_val_norm, eigen_vec_last_step, eigen_vec_diff_last_step, eigen_val_norm_last_step = powerMethodClosure(lambda eigen_vec=None: updateRule(eigen_vec, i))
			if (eigen_val_norm - max_eigen_val_norm)/abs(max_eigen_val_norm) > power_method_seq_update_threshold:
				# only then will max eigen value and vector be updated in power method sequence
				max_eigen_vec=eigen_vec
				max_eigen_vec_diff=eigen_vec_diff
				max_eigen_val_norm=eigen_val_norm
				max_eigen_vec_last_step=eigen_vec_last_step
				max_eigen_vec_diff_last_step=eigen_vec_diff_last_step
				max_eigen_val_norm_last_step=eigen_val_norm_last_step
				eigen_update_log.append((i, max_eigen_vec_diff, max_eigen_val_norm))
	print('%s\neigen_update_log: %s\n%s\n'%(delimiter_iteration,eigen_update_log,delimiter_iteration), file=print_redirect)
	return {'max_eigen_vec':max_eigen_vec,
			'max_eigen_vec_diff':max_eigen_vec_diff,
			'max_eigen_val_norm':max_eigen_val_norm,
			'max_eigen_vec_last_step':max_eigen_vec_last_step,
			'max_eigen_vec_diff_last_step':max_eigen_vec_diff_last_step,
			'max_eigen_val_norm_last_step':max_eigen_val_norm_last_step,
			'eigen_update_log':eigen_update_log}

def powerMethodGeneral(matrix_array, eigen_vec_diff_criteria=1e-3, eigen_val_norm_criteria=1e-3, seq=1, power_method_seq_update_threshold=1e-6, monitor=False, print_redirect=None):
	'''
	power method for general purpose.
	
	updateRule() is limited to matrix*vector
	'''
	
	# updateRule() specific for this kind of power method
	def updateRule(eigen_vec_last_step=None, one_index=0):
		if type(eigen_vec_last_step) == type(None):
			eigen_vec=zeros(len(matrix_array),complex) # len(ND_array) is equivalent to ND_array.shape[0]
			eigen_vec[one_index]=1
			return eigen_vec
		else:
			return matrix_array @ eigen_vec_last_step
	
	# define how to process overflow of seq
	if seq < 1 or seq > len(matrix_array):
		seq = len(matrix_array)
	
	return powerMethodSeq(updateRule, eigen_vec_diff_criteria, eigen_val_norm_criteria, seq, power_method_seq_update_threshold, monitor, print_redirect)

def powerMethodPao(V_pair, m_trunc, green_green_auxiliary, eigen_vec_diff_criteria=1e-3, eigen_val_norm_criteria=1e-3, seq=1, power_method_seq_update_threshold=1e-6, monitor=False, print_redirect=None):
	'''
	power method for Pao's SC.
	
	green_green_auxiliary = -T/dim_el**3 * green_matrix * green_matrix_auxiliary
	'''
	
	# variables specific for this updateRule()
	green_shape=array(green_green_auxiliary.shape)
	n_trunc=int(green_shape[0]/2-1) # WARN! `green_green_auxiliary` implies `n_trunc`
	
	# updateRule() specific for this kind of power method
	# negative `one_index` has special meaning
	def updateRule(eigen_vec_last_step=None, one_index=0):
		if type(eigen_vec_last_step) == type(None):
			if one_index >= 0:
				eigen_vec=zeros(green_shape.prod())
				eigen_vec[one_index]=1
				eigen_vec=eigen_vec.reshape(green_shape)
			elif one_index == -1:
				eigen_vec=ones(green_shape)
			elif one_index == -2:
				# eigen_vec=1 on kx, ky or kz axis
				eigen_vec=zeros(green_shape)
				eigen_vec[:,:,0,0]=1
				eigen_vec[:,0,:,0]=1
				eigen_vec[:,0,0,:]=1
			else:
				raise ValueError(f'one_index={one_index} is invalid')
			eigen_vec/=norm(eigen_vec)
		else:
			# sum over contribution to eigen vector from m's owned by the current process,
			# i.e. sum over only some, instead of all, n_prime's in Pao's eq.(18),
			# then sum over all of them across all processes to get updated eigen vector
			all_others = green_green_auxiliary * eigen_vec_last_step
			eigen_vec=zeros(green_shape,complex)
			for m in range(-m_trunc, m_trunc+1):
				if (m+m_trunc) % COMM_WORLD.size == COMM_WORLD.rank:
					for n in set(range(-n_trunc-1, n_trunc+1)) & set(range(-n_trunc-1+m, n_trunc+1+m)):
						eigen_vec[n] += ifftn( fftn(all_others[n-m]) * fftn(V_pair[n, (m+m_trunc)//COMM_WORLD.size]) )
			COMM_WORLD.Allreduce(IN_PLACE, eigen_vec, SUM)
		return eigen_vec
	
	# renormalize overflow or zero of seq, and filter special negative value according to the definition of `updateRule()`
	if seq > green_shape.prod() or seq == 0 or seq < -2:
		seq = green_shape.prod()
	
	return powerMethodSeq(updateRule, eigen_vec_diff_criteria, eigen_val_norm_criteria, seq, power_method_seq_update_threshold, monitor, print_redirect)
