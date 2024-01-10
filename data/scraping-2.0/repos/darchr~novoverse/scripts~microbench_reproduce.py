import os
import argparse

from util import novoverse


@novoverse
def run_microbench(inputs):
    def get_cache(cache_model):
        from gem5.utils.requires import requires
        from gem5.coherence_protocol import CoherenceProtocol

        if cache_model == "octopi":
            requires(
                coherence_protocol_required=CoherenceProtocol.MESI_THREE_LEVEL
            )
            from components.octopi import OctopiCache

            return OctopiCache(
                l1d_size="64kB",
                l1i_size="64kB",
                l2_size="1024kB",
                l3_size="4MB",
                l1d_assoc=4,
                l1i_assoc=4,
                l2_assoc=8,
                l3_assoc=16,
                num_core_complexes=1,
                is_fullsystem=False,
            )
        if cache_model == "bine":
            from components.bine import BineCache

            return BineCache()
        if cache_model == "cmn":
            requires(coherence_protocol_required=CoherenceProtocol.CHI)
            from components.cmn import CoherentMeshNetwork

            return CoherentMeshNetwork()

    def get_processor(processor_model):
        from components.graceproc import (
            GraceProcessor,
            GraceProcessorPipelined,
            GraceProcessor12W,
            GraceProcessor4W,
            GraceProcessorInf,
        )
        from components.processors import NovoVectorProcessor
        from components.processors import ARM_SVE_Parameters

        processor_model_map = {
            "GraceBase": GraceProcessor(8),
            "GracePipelined": GraceProcessorPipelined(8),
            "Grace4Wide": GraceProcessor4W(8),
            "Grace12Wide": GraceProcessor12W(8),
            "GraceInf": GraceProcessorInf(8),
            "Novoverse": NovoVectorProcessor(
                8, ARM_SVE_Parameters(512, False)
            ),
        }

        return processor_model_map[processor_model]

    def get_workload(benchmark, version, opt_level):
        from microbenchmarks import (
            workloads,
            workloads_opt3,
            workloads_v1,
            workloads_v1_opt3,
        )

        translator = {
            "y": {"zero": workloads, "three": workloads_opt3},
            "z": {"zero": workloads_v1, "three": workloads_v1_opt3},
        }

        return translator[version][opt_level][benchmark]

    import m5

    from m5.objects import DDR4_2400_16x4

    from gem5.simulate.simulator import Simulator
    from gem5.components.memory.memory import ChanneledMemory
    from gem5.components.boards.simple_board import SimpleBoard

    processor_model, cache_model, benchmark, version, opt = inputs

    processor = get_processor(processor_model)
    memory = ChanneledMemory(
        dram_interface_class=DDR4_2400_16x4,
        num_channels=4,
        interleaving_size=2**8,
        size="16GiB",
    )
    cache_hierarchy = get_cache(cache_model)

    system = SimpleBoard(
        "2GHz",
        processor,
        memory,
        cache_hierarchy,
    )
    system.set_workload(get_workload(benchmark, version, opt))

    with open(os.path.join(m5.options.outdir, "benchmark"), "w") as outfile:
        outfile.writelines([f"Benchmark: {benchmark}"])

    simulator = Simulator(system)
    simulator.run()


def get_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "processor_model",
        type=str,
        help="Processor model to use.",
        choices=[
            "GraceBase",
            "GracePipelined",
            "Grace4Wide",
            "Grace12Wide",
            "GraceInf",
            "Novoverse",
        ],
    )
    parser.add_argument(
        "cache_model",
        type=str,
        help="Cache model to use.",
        choices=["octopi", "bine", "cmn"],
    )
    parser.add_argument(
        "benchmark",
        type=str,
        help="Name of benchmark to run.",
    )
    parser.add_argument("version", type=str, choices=["y", "z"])
    parser.add_argument("opt_level", type=str, choices=["zero", "three"])
    args = parser.parse_args()
    return [
        args.processor_model,
        args.cache_model,
        args.benchmark,
        args.version,
        args.opt_level,
    ]


if __name__ == "__m5_main__":
    run_microbench(get_inputs())
