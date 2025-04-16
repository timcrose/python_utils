from python_utils import file_utils, time_utils
import socket, hashlib
from type_utils import Int, Scalar, Sequence, Any


def root_print(rank: Int, *print_message: Any) -> None:
   if rank == 0:
      print(print_message)


def rank_print(rank_output_path: str, *print_message: Any) -> None:
    '''
    rank_output_path: str
        Path to the log file for a single MPI rank.

    print_message: anything
        arguments to print()

    Return: None

    Purpose: write to a file that belongs to one MPI rank only
        in order to prevent clashing.
    '''
    with open(rank_output_path, mode='a') as f:
        print(print_message, file=f, flush=True)


def parallel_mkdir(rank: Int, dirpath: str, total_timeout: Scalar=1000, time_frame: Scalar=0.05) -> None:
    '''
    rank: int
        mpi4py comm rank
    dirpath: str
        Path of new directory to make. This will create directories leading up to this one if they DNE.
    total_timeout: number
        Amount of time in seconds to wait for dir to be created by root before giving up and assuming an error occurred
        so abort.
    time_frame: number
        Amount of time in seconds for the frequency of checking if the dir has been created.

    Return: None

    Purpose: Have the root rank mkdir and the rest of the ranks wait to save a little bit of time
        compared to comm.barrier(). Plus, this is cleaner.
    '''
    if rank == 0:
        file_utils.mkdir_if_DNE(dirpath)
    else:
        file_utils.wait_for_file_to_exist_and_written_to(dirpath, total_timeout=total_timeout, time_frame=time_frame)


def file_system_barrier(comm) -> None:
    '''
    comm: MPI.Comm from mpi4py
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

def barrier(comm, tag: Int=0, sleep: Scalar=0.01) -> None:
    size = comm.Get_size()
    rank = comm.Get_rank()
    mask = 1
    while mask < size:
        dst = (rank + mask) % size
        src = (rank - mask + size) % size
        req = comm.isend(None, dst, tag)
        while not comm.Iprobe(src, tag):
            time_utils.sleep(sleep)
        comm.recv(None, src, tag)
        req.Wait()
        mask <<= 1

    '''
    # test the beast!
    import time
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    tic = MPI.Wtime()
    if comm.rank==0:
        time.sleep(10)
    barrier(comm)
    toc = MPI.Wtime()
    print(comm.rank, toc-tic)
    '''


def split_up_list_evenly(lst: Sequence, rank: Int, size: Int, include_master: bool=True) -> Sequence:
    '''
    lst: list or np.array
        Overall list of elements to split up amongst ranks
    rank: int
        MPI rank for communicator of size size
    size:
        Number of total ranks in a communicator (size)
    include_master: bool
        True: Split up data among all ranks
        False: Split up data among size - 1 ranks and give rank 0 None

    Return:
    lst: list or np.array
        List of elements that rank rank should work on.

    Purpose: When doing embarrassingly parallel calculations, you need to split up
        an array of tasks to your available ranks. This function divides up the
        array as evenly as possible. e.g. if size = 3, and lst = [0,1,2,3,4], then
        rank 0 gets [0,1], rank 1 gets [2,3], and rank 2 gets [4].
    '''
    if not include_master:
        if rank == 0:
            return None
        rank -= 1
        size -= 1
    num_tasks = len(lst)
    tasks_per_rank = int(num_tasks / size)
    num_remainder_tasks = num_tasks - tasks_per_rank * size
    if rank < num_remainder_tasks:
        lst = lst[rank * (tasks_per_rank + 1) : (rank + 1) * (tasks_per_rank + 1)]
    else:
        lst = lst[num_remainder_tasks + rank * tasks_per_rank : num_remainder_tasks + (rank + 1) * tasks_per_rank]
    return lst


def split_by_node(comm, key: str='rank'):
    '''
    Create split MPI communicators from a common communicator, one for each
    node (hostname) in the set of hostnames where the ranks in comm reside.

    Parameters
    ----------
    comm: mpi4py communicator
        The communicator to split.
        
    key: int or 'rank'
        This int determines the new rank order in the newly created communicator.
        ranks with a lower key will receive a lower rank in the new communicator.
        'rank' will mean key = comm.rank.

    Returns
    -------
    node_comm: mpi4py communicator
        This communicator only contains ranks on a single node.
    '''
    if key == 'rank':
        key = comm.rank
    hostname = socket.gethostname()
    color = int(hashlib.sha1(hostname.encode('utf-8')).hexdigest(), 16) % (2**31)
    node_comm = comm.Split(color=color, key=key)
    return node_comm