import os
import argparse

from util import novoverse


def generate_traces(num_cores, reader_id, slc_id, period, access_size, outdir):
    start_addr = slc_id * 2097152
    end_addr = start_addr + 2097152

    ret = []
    for i in range(num_cores):
        if i != reader_id:
            state_lines = [f"STATE 0 10000000000000 IDLE\n"]
            transition_lines = ["INIT 0\n", "TRANSITION 0 0 1\n"]
        else:
            state_lines = [
                f"STATE 0 0 LINEAR 100 {start_addr} {end_addr} 64 500 500 2097152\n",
                f"STATE 1 100000000 IDLE\n",
                f"STATE 2 0 LINEAR 100 {start_addr} {end_addr} 64 500 500 2097152\n",
                f"STATE 3 100000000 IDLE\n",
                f"STATE 4 0 EXIT\n",
                f"STATE 5 0 LINEAR 100 {start_addr} {end_addr} {access_size} {period} {period} 67108864\n",
                f"STATE 6 0 EXIT\n",
                f"STATE 7 10000000000000 IDLE\n",
            ]
            transition_lines = [
                "INIT 0\n",
                "TRANSITION 0 1 1\n",
                "TRANSITION 1 2 1\n",
                "TRANSITION 2 3 1\n",
                "TRANSITION 3 4 1\n",
                "TRANSITION 4 5 1\n",
                "TRANSITION 5 6 1\n",
                "TRANSITION 6 7 1\n",
                "TRANSITION 7 7 1\n",
            ]
        with open(os.path.join(outdir, f"core_{i}"), "w") as trace:
            trace.writelines(state_lines + transition_lines)
            ret.append(trace.name)
    return ret


@novoverse
def run_l2_bandwidth_test(inputs):
    import m5

    from m5.debug import flags
    from m5.objects import Root, DDR4_2400_8x8

    from gem5.components.boards.test_board import TestBoard
    from gem5.components.memory.memory import ChanneledMemory
    from gem5.components.processors.traffic_generator import TrafficGenerator

    from components.cmn import CoherentMeshNetwork

    num_cores, reader_id, slc_id, period, access_size = inputs

    generator = TrafficGenerator(
        generate_traces(
            num_cores,
            reader_id,
            slc_id,
            period,
            access_size,
            m5.options.outdir,
        )
    )

    memory = ChanneledMemory(DDR4_2400_8x8, 4, 128, size="16GiB")
    cache = CoherentMeshNetwork(slice_interleaving_size="2MiB")
    board = TestBoard(
        clk_freq="4GHz",
        generator=generator,
        cache_hierarchy=cache,
        memory=memory,
    )

    root = Root(full_system=False, system=board)

    board._pre_instantiate()

    m5.instantiate()

    generator.start_traffic()
    print("Beginning simulation!")
    exit_events_countered = 0
    while True:
        exit_event = m5.simulate()
        exit_events_countered += 1
        print(
            f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}."
        )
        print(f"Received {exit_events_countered} exit events.")
        if exit_events_countered == 1:
            print(f"Done warming up slc{slc_id}.")
            m5.stats.reset()
        if exit_events_countered == 2:
            print(f"Core {reader_id} done stressing slc{slc_id}.")
            break
    print("Simulation over.")
    print("Dumping the stats.")


def get_inputs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "num_cores",
        type=int,
        help="Number of cores to put in the"
        " system (use to make the CMN more complex).",
    )
    parser.add_argument(
        "reader_id", type=int, help="ID of the core reading data from slc."
    )
    parser.add_argument(
        "slc_id", type=int, help="ID of the system level cache to test."
    )
    parser.add_argument(
        "period",
        type=int,
        help="Number of ticks between two consecutive accesses.",
    )
    parser.add_argument(
        "access_size", type=int, help="Size of each access to the l1 cache."
    )

    args = parser.parse_args()

    return [
        args.num_cores,
        args.reader_id,
        args.slc_id,
        args.period,
        args.access_size,
    ]


if __name__ == "__m5_main__":
    run_l2_bandwidth_test(get_inputs())
