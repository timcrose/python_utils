from python_utils import file_utils, time_utils

def barrier(comm):
    '''
    comm: MPI.COMM_WORLD from mpi4py
        MPI communicator from mpi4py
    
    Purpose: mpi4py's comm.barrier() doesn't always work. This 
        function creates a file for each rank in the comm and 
        each rank in the comm will wait until every other rank
        has created a file. It will then remove these tmp files.
    Notes: Currently only supports the world communicator.
    '''
    output_fpath_prefix = 'barrier_world_rank_'
    output_fpath_search_str = output_fpath_prefix + '*'
    file_utils.output_from_rank(message_args=('Here'), rank=comm.rank, output_fpath_prefix=output_fpath_prefix)
    while len(file_utils.glob(output_fpath_search_str)) < comm.size:
        time_utils.sleep(0.5)
    time_utils.sleep(1.5)
    if comm.rank == 0:
        file_utils.rm(output_fpath_search_str)
    else:
        time_utils.sleep(0.4)
