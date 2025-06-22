from numpy import savez_compressed
from mpi4py.MPI import COMM_WORLD
from .flex import FLEX

def run(working_dir, silent_init=True, monitor=False, monitor_MPI=False, save_result=False):
	flex_instance=FLEX(working_dir, silent_init, monitor, monitor_MPI)
	flex_instance.selfConsistency()
	cdw_Migdal=flex_instance.susceptibilityMigdalCDW()
	eig_sc=flex_instance.maxEig('SCPao')
	flex_instance.closeFileHandles()
	
	if COMM_WORLD.rank==0:
		if save_result:
			# save results
			# rename keys in dictionary `eig_sc` to avoid confliction with those in `eig_cdw`
			for i in list(eig_sc):
				eig_sc[i+'_sc']=eig_sc.pop(i)
			savez_compressed(working_dir+'/flex.npz',
								green_matrix=flex_instance.green_matrix,
								self_energy=flex_instance.self_energy,
								**eig_sc,
								**cdw_Migdal
							)
		else:
			return {'flex_instance':flex_instance, 'eig_sc':eig_sc, 'cdw_Migdal':cdw_Migdal}
