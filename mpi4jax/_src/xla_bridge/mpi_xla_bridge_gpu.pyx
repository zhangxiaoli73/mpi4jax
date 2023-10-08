from cpython.pycapsule cimport PyCapsule_New

from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free

from mpi4py.libmpi cimport (
    MPI_Comm,
    MPI_Comm_rank,
    MPI_Comm_size,
    MPI_Datatype,
    MPI_Op,
    MPI_Status,
    MPI_Type_size,
)

from . cimport mpi_xla_bridge

# Config

cdef bint COPY_TO_HOST = False

cpdef void set_copy_to_host(bint enable):
    global COPY_TO_HOST
    COPY_TO_HOST = enable

#
# GPU XLA targets
#

# Allreduce

cdef struct AllreduceDescriptor:
    int nitems
    MPI_Op op
    MPI_Comm comm
    MPI_Datatype dtype


cpdef bytes build_allreduce_descriptor(int nitems, uintptr_t op_handle,
                                       uintptr_t comm_handle, uintptr_t dtype_handle):
    cdef AllreduceDescriptor desc = AllreduceDescriptor(
        nitems, <MPI_Op> op_handle, <MPI_Comm> comm_handle, <MPI_Datatype> dtype_handle
    )
    return bytes((<char*> &desc)[:sizeof(AllreduceDescriptor)])


cdef void mpi_allreduce_gpu(void* stream, void** buffers,
                            const char* opaque, size_t opaque_len) nogil:
    cdef int ierr, dtype_size
    cdef size_t count

    # decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* in_buf = data
    cdef void* out_buf = out_data

    if opaque_len != sizeof(AllreduceDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef AllreduceDescriptor* desc = <AllreduceDescriptor*>(opaque)
    cdef int nitems = desc.nitems
    cdef MPI_Op op = desc.op
    cdef MPI_Comm comm = desc.comm
    cdef MPI_Datatype dtype = desc.dtype

    mpi_xla_bridge.mpi_allreduce(in_buf, out_buf, nitems, dtype, op, comm)


gpu_custom_call_targets = {}

cdef register_custom_call_target(fn_name, void* fn):
    cdef const char* name = "xla._CUSTOM_CALL_TARGET"
    gpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)

register_custom_call_target(b"mpi_allreduce", <void*>(mpi_allreduce_gpu))
