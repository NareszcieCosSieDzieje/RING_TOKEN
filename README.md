
## RING RELIABILITY COORDINATOR

A Python simulation using mpi4py to implement an algorithm that would
make sure a token is passed in a ring topology, assuming that channels are unreliable
but processes (nodes) are reliable.

# TESTED WITH PYTHON 3.10

mpiexec.exe -machinefile mpi_hosts/localhost python .\main.py
mpiexec.exe -n 4 python .\main.py