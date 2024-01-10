from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING, Union, Any

from parla.cpu_impl import cpu
from parla import task_runtime
from parla.device import Device

from .coherence import MemoryOperation, Coherence, CPU_INDEX
from .memory import MultiDeviceBuffer

import threading
import numpy
try:  # if the system has no GPU
    import cupy
    num_gpu = cupy.cuda.runtime.getDeviceCount()
except ImportError:
    # PArray only considers numpy or cupy array
    # work around of checking cupy.ndarray when cupy could not be imported
    cupy = numpy
    num_gpu = 0

if TYPE_CHECKING:
    ndarray = Union[numpy.ndarray, cupy.ndarray]
    SlicesType = Union[slice, int, tuple]


class PArray:
    """Multi-dimensional array on a CPU or CUDA device.

    This class is a wrapper around :class:`numpy.ndarray` and :class:`cupy.ndarray`,
    It is used to support Parla sheduler optimization and automatic data movement.

    Args:
        array: :class:`cupy.ndarray` or :class:`numpy.array` object

    Note: some methods should be called within the current task context
    """
    _array: MultiDeviceBuffer
    _coherence: Coherence
    _slices: List[SlicesType]
    _coherence_cv: Dict[int, threading.Condition]

    def __init__(self, array: ndarray, parent: "PArray" = None, slices=None) -> None:
        if parent is not None:  # create a view (a subarray) of a PArray
            # inherit parent's buffer and coherence states
            # so this PArray will becomes a 'view' of its parents
            self._array = parent._array
            self._coherence = parent._coherence

            # _slices is a list so subarray of subarray works
            self._slices = parent._slices.copy()  # copy parent's slices list
            # add current slices to the end
            self._slices.append(slices)

            # inherit parent's condition variables
            self._coherence_cv = parent._coherence_cv

            # identify the slices
            self._slices_hash = self._array.get_slices_hash(slices)

            # a unique ID for this subarray
            # which is the combine of parent id and slice hash
            self.parent_ID = parent.ID
            self.ID = parent.ID * 31 + self._slices_hash  # use a prime number to avoid collision

            self.nbytes = parent.nbytes          # the bytes used by the complete array
            self.subarray_nbytes = array.nbytes  # the bytes used by this subarray

            # no need to register again since parent already did that
        else:  # initialize a new PArray
            # per device buffer of data
            self._array = MultiDeviceBuffer(num_gpu)
            location = self._array.set_complete_array(array)

            # coherence protocol for managing data among multi device
            self._coherence = Coherence(location, num_gpu)

            # no slices since it is a new array rather than a subarray
            self._slices = []

            # a condition variable to acquire when moving data on the device
            self._coherence_cv = {n:threading.Condition() for n in range(num_gpu)}
            self._coherence_cv[CPU_INDEX] = threading.Condition()

            # there is no slices
            self._slices_hash = None

            # use id() of buffer as unique ID
            self.parent_ID = id(self._array)  # no parent
            self.ID = self.parent_ID

            self.nbytes = array.nbytes
            self.subarray_nbytes = self.nbytes  # no subarray

            # Register the parray with the scheduler
            task_runtime.get_scheduler_context().scheduler._available_resources.track_parray(self)

    # Properties:

    @property
    def array(self) -> ndarray:
        """
        The reference to cupy/numpy array on current device.
        Note: should be called within the current task context
        Note: should call A[valid_slices].array instead of A.array,
            when in a tasks which only subarray is auto-moved.
            `valid_slices` is a slices within the auto-moved subarray.
        """
        if self._slices:  # so this is a sub-parray object
            # index into origin array by saved slices
            ret = self._array.get_by_global_slices(self._current_device_index, self._slices[0])
            for s in self._slices[1:]:
                ret = ret[s]
            return ret
        else:  # this is a complete copy
            ret = self._array.get(self._current_device_index)

            if isinstance(ret, list): # get a subarray instead
                raise IndexError("Current device doesn't have a complete copy of this array")
            return ret

    @property
    def _on_gpu(self) -> bool:
        """
        True if the array is on GPU.
        Note: should be called within the current task context
        """
        return self._current_device_index != CPU_INDEX

    @property
    def _current_device_index(self) -> int:
        """
        -1 if the current device is CPU.
        Otherwise GPU ID.
        If called outside the current task context, return data owner's device_id by default
        """
        device = PArray._get_current_device()
        if device is None:  # not called inside current task
            return self._coherence.owner
        elif device.architecture == cpu:
            return CPU_INDEX
        else:
            # assume GPU here, won't check device.architecture == gpu
            # to avoid import `gpu`, which is slow to setup.
            return device.index

    # Public API:

    def nbytes_at(self, device_id:int):
        """ An estimate of bytes used in `device_id` after data is moved there.
        It is neither lower bound nor upper bound.
        """
        if self._slices:
            if isinstance(self._coherence._local_states[device_id], dict): # there are subarrays no this device
                if self._slices_hash in self._coherence._local_states[device_id].keys():  # this subarray is already there
                    return self._array.nbytes_at(device_id)
                else:  # the subarray will be moved to there
                    return self._array.nbytes_at(device_id) + self.subarray_nbytes  # add the incoming subarray size
            else: # there is a complete copy on this device, no need to prepare subarray
                return self.nbytes
        else:
            return self.nbytes   

    def exists_on_device(self, device_id:int):
        return (self._array._buffer[device_id] is not None)

    def update(self, array) -> None:
        """ Update the copy on current device.
        Previous copy on other device are lost.
        This will replace the internal buffer and coherence object completly.

        Args:
            array: :class:`cupy.ndarray` or :class:`numpy.array` object

        Note: should be called within the current task context
        Note: data should be put in OUT/INOUT fields of spawn
        """
        this_device = self._current_device_index

        # clean up data
        self._array._buffer = {n: None for n in range(num_gpu)}
        self._array._buffer[CPU_INDEX] = None

        # copy new data to buffer
        if isinstance(array, numpy.ndarray):
            if this_device != CPU_INDEX:  # CPU to GPU
                self._array.set(this_device, cupy.asarray(array))
            else: # data already in CPU
                self._array.set(this_device, array)
        else:
            if this_device == CPU_INDEX: # GPU to CPU
                self._array.set(this_device, cupy.asnumpy(array))
            else: # GPU to GPU
                if int(array.device) == this_device: # data already in this device
                    self._array.set(this_device, array)
                else:  # GPU to GPU
                    dst_data = cupy.empty_like(array)
                    dst_data.data.copy_from_device_async(array.data, array.nbytes)
                    self._array.set(this_device, dst_data)

        # update size
        self.nbytes = array.nbytes

        # reset coherence
        self._coherence = Coherence(this_device, num_gpu)

        # update shape
        self._array.shape = array.shape

        cupy.cuda.stream.get_current_stream().synchronize()

    # slicing/indexing

    def __getitem__(self, slices: SlicesType) -> PArray | Any:
        if self._slices:  # resolve saved slices first
            ret = self.array[slices]
        else:
            ret = self._array.get_by_global_slices(self._current_device_index, slices)

        # ndarray.__getitem__() may return a ndarray
        if isinstance(ret, numpy.ndarray):
            return PArray(ret, parent=self, slices=slices)
        elif isinstance(ret, cupy.ndarray):
            if ret.shape == ():
                return ret.item()
            else:
                return PArray(ret, parent=self, slices=slices)
        else:
            return ret

    def __setitem__(self, slices: SlicesType, value: PArray | ndarray | Any) -> None:
        """
        Acceptable Slices: Slice, Int, tuple of (Slice, Int, List of Int)
        Example:
            A[0]  # int
            A[:]  # slice
            A[0,:,10]  # tuple of int slice int
            A[2:10:2, 0, [1, 3, 5]]  # tuple of slice int list of Int

        Note: `:` equals to slice(None, None, None)
        Note: `None` or tuple of `None` is not acceptable (even if `numpy.ndarray` accept `None`)
        # TODO: support `None` and `ndarray` as slices
        """
        if isinstance(value, PArray):
            value = value.array

        if self._slices:  # resolve saved slices first
            self.array.__setitem__(slices, value)
        else:
            self._array.set_by_global_slices(self._current_device_index, slices, value)

    def evict_all(self) -> None:
        """
        Evict all copies from buffer, and clean all related fields

        Note: this object should not be accessed anymore after called this method
        """
        self._array = None
        self._coherence = None

    def evict(self, device_id: int = None, keep_one_copy: bool = True) -> None:
        """
        Evict a device's copy and update coherence states.

        Args:
            device_id: if is this not None, data will be moved to this device,
                    else move to current device
            keep_one_copy: if it is True and this is the last copy, 
                    write it back to CPU before evict.

        Note: if this device has the last copy and `keep_one_copy` is false, 
            the whole protocol state will be INVALID then.
            And the system will lose the copy. Be careful when evict the last copy.
        """
        if device_id is None:
            device_id = self._current_device_index

        with self._coherence_cv[device_id]:
            operations = self._coherence.evict(device_id, keep_one_copy)
            for op in operations:
                self._process_operation(op)


    # Coherence update operations:

    def _coherence_read(self, device_id: int = None, slices: SlicesType = None) -> None:
        """ Tell the coherence protocol a read happened on a device.

        And do data movement based on the operations given by protocol.

        Args:
            device_id: if is this not None, data will be moved to this device,
                    else move to current device
            slices: a slices of the subarray to be manipulated
                    by default equals to None, which means the whole array is manipulated

        Note: should be called within the current task context
        """
        if device_id is None:
            device_id = self._current_device_index

        # update protocol and get operation
        operations = self._coherence.read(device_id, self._slices_hash) # locks involve
        self._process_operations(operations, slices) # condition variable involve

    def _coherence_write(self, device_id: int = None, slices: SlicesType = None) -> None:
        """Tell the coherence protocol a write happened on a device.

        And do data movement based on the operations given by protocol.

        Args:
            device_id: if is this not None, data will be moved to this device,
                    else move to current device
            slices: a slices of the subarray to be manipulated
                    by default equals to None, which means the whole array is manipulated

        Note: should be called within the current task context
        """
        if device_id is None:
            device_id = self._current_device_index

        # update protocol and get operation
        operations = self._coherence.write(device_id, self._slices_hash) # locks involve
        self._process_operations(operations, slices) # condition variable involve

    # Device management methods:

    def _process_operations(self, operations: List[MemoryOperation], slices: SlicesType = None) -> None:
        """
        Process the given memory operations.
        Data will be moved, and protocol states is kept unchanged.
        """
        for op in operations:
            if op.inst == MemoryOperation.NOOP:
                pass  # do nothing
            elif op.inst == MemoryOperation.CHECK_DATA:
                if not self._coherence.data_is_ready(op.src):  # if data is not ready, wait
                    with self._coherence_cv[op.src]:
                        while not self._coherence.data_is_ready(op.src):
                            self._coherence_cv[op.src].wait()
            elif op.inst == MemoryOperation.LOAD:
                with self._coherence_cv[op.dst]:  # hold the CV when moving data

                    # if the flag is set, skip this checking
                    if not MemoryOperation.SKIP_SRC_CHECK in op.flag:
                        with self._coherence_cv[op.src]:  # wait on src until it is ready
                            while not self._coherence.data_is_ready(op.src):
                                self._coherence_cv[op.src].wait()

                    if MemoryOperation.LOAD_SUBARRAY in op.flag:
                        self._array.set_slices_mapping(op.dst, slices)  # build slices mapping first

                    # check flag to see if dst is current device
                    dst_is_current_device = op.flag != MemoryOperation.SWITCH_DEVICE_FLAG

                    # copy data
                    self._array.copy_data_between_device(op.dst, op.src, dst_is_current_device)

                    # sync stream before set it as ready, so asyc call is ensured to be done
                    cupy.cuda.stream.get_current_stream().synchronize()

                    # data is ready now
                    if MemoryOperation.LOAD_SUBARRAY in op.flag:
                        self._coherence.set_data_as_ready(op.dst, self._slices_hash)  # mark it as done
                    else:
                         self._coherence.set_data_as_ready(op.dst)
                    self._coherence_cv[op.dst].notify_all()  # let other threads know the data is ready
            elif op.inst == MemoryOperation.EVICT:
                self._array.clear(op.src)  # decrement the reference counter, relying on GC to free the memory
                self._coherence.set_data_as_ready(op.src, None)  # mark it as done
            elif op.inst == MemoryOperation.ERROR:
                raise RuntimeError("PArray gets an error from coherence protocol")
            else:
                raise RuntimeError(f"PArray gets invalid memory operation from coherence protocol, "
                                   f"detail: opcode {op.inst}, dst {op.dst}, src {op.src}")

    @staticmethod
    def _get_current_device() -> Device | None:
        """
        Get current device from task environment.

        Return None if it is not called within the current task context
        """
        if task_runtime.has_environment():
            return task_runtime.get_current_devices()[0]
        return None

    def _auto_move(self, device_id: int = None, do_write: bool = False) -> None:
        """ Automatically move data to current device.

        Multiple copies on different devices will be made based on coherence protocol.

        Args:
            device_id: current device id. CPU use CPU_INDEX as id
            do_write: True if want make the device MO in coherence protocol
                False if this is only for read only in current task

        Note: should be called within the current task context.
        Note: auto-move of subarray's subarray is not supported.
        """
        # this is the view of current array, only data within this range should be moved
        # currently, only use first slices, which means automove of subarray of subarray is not supported
        # TODO: support auto-move subarray of subarray
        slices = None if not self._slices else self._slices[0]

        if do_write:
            self._coherence_write(device_id, slices)
        else:
            self._coherence_read(device_id, slices)

    def _on_same_device(self, other: "PArray") -> bool:
        """
        Return True if the two PArrays are in the same device.
        Note: other has to be a PArray object.
        """
        this_device = self._current_device_index
        return this_device in other._array

    # NumPy/CuPy methods redirection

    def __getattr__(self, item):
        """
        A proxy method that redirect call to methods in :class:`numpy.ndarray` or :class:`cupy.ndarray`
        """
        return getattr(self.array, item)

    # Comparison operators:

    def __lt__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__lt__(y.array)
        else:
            return x.array.__lt__(y)

    def __le__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__le__(y.array)
        else:
            return x.array.__le__(y)

    def __eq__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__eq__(y.array)
        else:
            return x.array.__eq__(y)

    def __ne__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__ne__(y.array)
        else:
            return x.array.__ne__(y)

    def __gt__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__gt__(y.array)
        else:
            return x.array.__gt__(y)

    def __ge__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__ge__(y.array)
        else:
            return x.array.__ge__(y)


    # Truth value of an array (bool):

    def __nonzero__(self):
        return PArray(self.array.__nonzero__())

    # Unary operations:

    def __neg__(self):
        return PArray(self.array.__neg__())

    def __pos__(self):
        return PArray(self.array.__pos__())

    def __abs__(self):
        return PArray(self.array.__abs__())

    def __invert__(self):
        return PArray(self.array.__invert__())

    # Arithmetic:

    def __add__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array + y.array)
        else:
            return PArray(x.array + y)

    def __sub__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array - y.array)
        else:
            return PArray(x.array - y)

    def __mul__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array * y.array)
        else:
            return PArray(x.array * y)

    def __matmul__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array @ y.array)
        else:
            return PArray(x.array @ y)

    def __div__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array / y.array)
        else:
            return PArray(x.array / y)

    def __truediv__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array / y.array)
        else:
            return PArray(x.array / y)

    def __floordiv__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__floordiv__(y.array))
        else:
            return PArray(x.array.__floordiv__(y))

    def __mod__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__mod__(y.array))
        else:
            return PArray(x.array.__mod__(y))

    def __divmod__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__divmod__(y.array))
        else:
            return PArray(x.array.__divmod__(y))

    def __pow__(x, y, modulo):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__pow__(y.array))
        else:
            return PArray(x.array.__pow__(y))

    def __lshift__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__lshift__(y.array))
        else:
            return PArray(x.array.__lshift__(y))

    def __rshift__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__rshift__(y.array))
        else:
            return PArray(x.array.__rshift__(y))

    def __and__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__and__(y.array))
        else:
            return PArray(x.array.__and__(y))

    def __or__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__or__(y.array))
        else:
            return PArray(x.array.__or__(y))

    def __xor__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__xor__(y.array))
        else:
            return PArray(x.array.__xor__(y))

    # Arithmetic, in-place:
    def __iadd__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__iadd__(other.array)
        else:
            self.array.__iadd__(other)
        return self

    def __isub__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__isub__(other.array)
        else:
            self.array.__isub__(other)
        return self

    def __imul__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__imul__(other.array)
        else:
            self.array.__imul__(other)
        return self

    def __idiv__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__idiv__(other.array)
        else:
            self.array.__idiv__(other)
        return self

    def __itruediv__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__itruediv__(other.array)
        else:
            self.array.__itruediv__(other)
        return self

    def __ifloordiv__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__ifloordiv__(other.array)
        else:
            self.array.__ifloordiv__(other)
        return self

    def __imod__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__imod__(other.array)
        else:
            self.array.__imod__(other)
        return self

    def __ipow__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__ipow__(other.array)
        else:
            self.array.__ipow__(other)
        return self

    def __ilshift__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__ilshift__(other.array)
        else:
            self.array.__ilshift__(other)
        return self

    def __irshift__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__irshift__(other.array)
        else:
            self.array.__irshift__(other)
        return self

    def __iand__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__iand__(other.array)
        else:
            self.array.__iand__(other)
        return self

    def __ior__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__ior__(other.array)
        else:
            self.array.__ior__(other)
        return self

    def __ixor__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__ixor__(other.array)
        else:
            self.array.__ixor__(other)
        return self

    # Container customization:

    def __iter__(self):
        return self.array.__iter__()

    def __len__(self):
        return self.array.__len__()

    # Conversion:

    def __int__(self):
        return int(self.array)

    def __float__(self):
        return float(self.array)

    def __complex__(self):
        return complex(self.array)

    def __oct__(self):
        return oct(self.array)

    def __hex__(self):
        return hex(self.array)

    def __bytes__(self):
        return bytes(self.array)

    # String representations:

    def __repr__(self):
        return repr(self._array)

    def __str__(self):
        return str(self._array)

    def __format__(self, format_spec):
        return self._array.__format__(format_spec)
