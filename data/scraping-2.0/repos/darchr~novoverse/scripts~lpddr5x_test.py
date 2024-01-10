import argparse

from util import novoverse


@novoverse
def run_lpddr5x_test(inputs):
    import m5

    from m5.objects import Root

    from gem5.components.boards.test_board import TestBoard
    from gem5.components.memory.memory import ChanneledMemory

    # from gem5.components.memory.dramsim_3 import SingleChannel
    from gem5.components.processors.linear_generator import LinearGenerator
    from gem5.components.processors.random_generator import RandomGenerator

    from components.memory import LPDDR5X
    from components.cmn import CoherentMeshNetwork
    from components.strided_generator import StridedGenerator

    mode, rd_perc, data_limit, *rest = inputs
    if len(rest) != 0:
        print(f"Don't know what to do with {' '.join(rest)}")

    memory = ChanneledMemory(
        dram_interface_class=LPDDR5X, num_channels=1, interleaving_size=128
    )

    generators = {
        "linear": LinearGenerator(
            num_cores=1,
            duration="1ms",
            rate="64GB/s",
            block_size=64,
            min_addr=0,
            max_addr=memory.get_size(),
            rd_perc=rd_perc,
            data_limit=data_limit,
        ),
        "random": RandomGenerator(
            num_cores=1,
            duration="1ms",
            rate="64GB/s",
            block_size=64,
            min_addr=0,
            max_addr=memory.get_size(),
            rd_perc=rd_perc,
            data_limit=data_limit,
        ),
        "strided": StridedGenerator(
            num_cores=1,
            duration="1ms",
            rate="64GB/s",
            block_size=64,
            superblock_size=256,
            stride_size=2048,
            min_addr=0,
            max_addr=memory.get_size(),
            rd_perc=rd_perc,
            data_limit=data_limit,
        ),
    }

    generator = generators[mode]

    board = TestBoard(
        clk_freq="4GHz",
        generator=generator,
        memory=memory,
        cache_hierarchy=None,
    )

    root = Root(full_system=False, system=board)

    board._pre_instantiate()
    m5.instantiate()

    generator.start_traffic()
    print("Beginning simulation!")

    exit_event = m5.simulate()
    print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}.")


def get_inputs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        type=str,
        choices=["linear", "random", "strided"],
        help="Type of traffic to generate. Use linear for minimum latency and "
        "strided for maximum bank parallelism and random for random access.",
    )

    parser.add_argument(
        "rd_perc",
        type=int,
        help="Percentage of memory references "
        "that are reads. Rest will be writes.",
    )

    parser.add_argument(
        "--data-limit",
        type=int,
        required=False,
        default=0,
        help="Limit on the number of bytes that should be accessed.",
    )

    args = parser.parse_args()
    return [args.mode, args.rd_perc, args.data_limit]


if __name__ == "__m5_main__":
    run_lpddr5x_test(get_inputs())
