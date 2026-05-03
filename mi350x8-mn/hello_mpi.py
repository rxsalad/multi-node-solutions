from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

hostname = MPI.Get_processor_name()

print(f"Hello from rank {rank} out of {size} on {hostname}")

if size >= 2:
    if rank == 0:
        msg = "Hello from rank 0"
        comm.send(msg, dest=1, tag=0)
        print("Rank 0 sent message to rank 1")

    elif rank == 1:
        msg = comm.recv(source=0, tag=0)
        print(f"Rank 1 received message: {msg}")