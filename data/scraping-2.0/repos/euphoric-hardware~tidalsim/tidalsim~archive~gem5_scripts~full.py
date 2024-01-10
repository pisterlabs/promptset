from gem5.utils.requires import requires
from gem5.components.boards.riscv_board import RiscvBoard
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.cachehierarchies.classic.no_cache import NoCache
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.coherence_protocol import CoherenceProtocol
from gem5.isas import ISA
from gem5.components.processors.cpu_types import CPUTypes
from gem5.resources.resource import Resource
from gem5.simulate.simulator import Simulator
# from gem5.simulate.exit_event import ExitEvent

requires(
    isa_required=ISA.RISCV,
    # coherence_protocol_required=CoherenceProtocol.MESI_TWO_LEVEL
)

memory = SingleChannelDDR3_1600(size="2GiB")

cache_hierarchy = NoCache()
# MESITwoLevelCacheHierarchy(
#     l1d_size="16KiB",
#     l1d_assoc=4,
#     l1i_size="16KiB",
#     l1i_assoc=4,
#     l2_size="512kB",
#     l2_assoc=8,
#     num_l2_banks=1,
# )

processor = SimpleProcessor(
    cpu_type=CPUTypes.TIMING,
    num_cores=1,
    isa=ISA.RISCV
)

board = RiscvBoard(
    clk_freq="1MHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy
)

command = "m5 exit;" \
        + "echo 'This is running on Timing CPU cores.';" \
        + "sleep 1;" \
        + "m5 exit;"

board.set_kernel_disk_workload(
    kernel=Resource("riscv-linux-kernel-5.4.49",),
    disk_image=Resource("riscv-ubuntu-18.04-img"),
    readfile_contents=command,
)

simulator = Simulator(
    board=board,
)
simulator.run()
