from sys import argv
from .run import run

run(argv[1], silent_init=False, monitor=True, monitor_MPI=True, save_result=True)
