from time import time

# delimiter between top-level parts or iterations
delimiter_top_level='='*50
delimiter_iteration='-'*30

def cpuTime(monitor, finished_part, last_monitor_time=0, last_iteration_time=None, print_redirect=None, curr_time=None):
	'''
	monitor(print) elapsed time
	'''
	
	if not curr_time:
		curr_time=time()
	if monitor:
		if last_iteration_time != None:
			print('[%s] cpu time for this part & the current iteration: %.3fs %.3fs'%(finished_part, curr_time-last_monitor_time, curr_time-last_iteration_time), file=print_redirect)
		else:
			print('[%s] cpu time for this part: %.3fs'%(finished_part, curr_time-last_monitor_time), file=print_redirect)

def decimal2OtherBaseDigits(decimal, base=10, length_min=3):
	'''
	split all digits of `decimal` under the specified `base` into a list at least `length_min`-element-long
	pad list at the beginning with 0's for small `decimal`
	
	decimal is a non-negative integer
	
	REDUNDANT if length of `digits` is already known, e.g. len(digits)==3,
	in which case this function can be replaced by numpy.unravel_index(decimal, (base, base, base)) with lower speed
	'''
	
	digits=[]
	flag=False # terminating flag
	while not flag:
		digits.insert(0, decimal%base)
		decimal=decimal//base
		flag = decimal==0 and len(digits) >= length_min
	return tuple(digits)
