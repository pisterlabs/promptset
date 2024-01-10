#!/usr/bin/env python3
from util import generateDegradation, degrade, loadAll

def process(reference, transformed):
    from coherent_point_drift.align import driftRigid
    from coherent_point_drift.geometry import RMSD, rigidXform
    from itertools import islice, starmap
    from functools import partial

    fits = list(islice(driftRigid(reference, transformed), 200))
    fitteds = starmap(partial(rigidXform, transformed), fits)
    rmsds = list(map(partial(RMSD, reference), fitteds))
    return fits, rmsds

def generate(args):
    from multiprocessing import Pool
    from functools import partial
    from itertools import starmap
    from numpy.random import seed, random, randint
    from numpy import iinfo
    from pickle import dumps
    from sys import stdout

    seed(4)
    reference = random((args.N, args.D))
    stdout.buffer.write(dumps(reference))
    seeds = randint(iinfo('int32').max, size=args.repeats)

    with Pool() as p:
        degradations = p.map(partial(generateDegradation, args), seeds)
        transformeds = p.starmap(partial(degrade, reference), degradations)
        fits = p.map(partial(process, reference), transformeds)
        for repeat in zip(degradations, fits):
            stdout.buffer.write(dumps(repeat))

def plot(args):
    from pickle import load
    from sys import stdin
    import matplotlib.pyplot as plt
    from itertools import starmap
    from numpy.random import seed, random
    from coherent_point_drift.geometry import rigidXform, RMSD
    from coherent_point_drift.util import last
    from math import degrees

    seed(4)
    reference = load(stdin.buffer)

    rmsds = []
    rotations = []
    for degradation, (fit, rmsd) in loadAll(stdin.buffer):
        rmsds.append(rmsd)
        rotations.append(degrees(degradation[0][0]))

        plt.figure(0)
        plt.plot(rmsd, alpha=0.3)

        plt.figure(1)
        color = random(3)
        degraded = degrade(reference, *degradation)
        plt.scatter(degraded[:, 0], degraded[:, 1], marker='o', color=color, alpha=0.2)
        fitted = rigidXform(degraded, *last(fit))
        plt.scatter(fitted[:, 0], fitted[:, 1], marker='+', color=color, alpha=0.4)
    plt.scatter(reference[:, 0], reference[:, 1], marker='v', color='black')

    min_rmsds= map(min, rmsds)
    rotation_rmsds = sorted(zip(rotations, min_rmsds), key=lambda x: x[0])

    plt.figure(2)
    plt.plot(*zip(*rotation_rmsds))
    plt.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    from math import pi

    parser = ArgumentParser("Test random data for 2D and 3D local alignment convergence")
    subparsers = parser.add_subparsers()
    parser_gen = subparsers.add_parser('generate', aliases=['gen'],
                                       help="Generate points and RMSD-sequences")
    parser_gen.set_defaults(func=generate)
    parser_gen.add_argument('N', type=int, help='Number of points')
    parser_gen.add_argument('D', type=int, choices=(2, 3), help='Number of dimensions')
    parser_gen.add_argument('repeats', type=int, help='Number of trials to run')

    parser_gen.add_argument('--drop', nargs=2, type=int, default=(0, 0),
                        help='number of points to exclude from the reference set')
    parser_gen.add_argument('--rotate', nargs=2, type=float, default=(-pi, pi),
                        help='The range of rotations to test')
    parser_gen.add_argument('--translate', nargs=2, type=float, default=(-1.0, 1.0),
                        help='The range of translations to test')
    parser_gen.add_argument('--scale', nargs=2, type=float, default=(0.5, 1.5),
                        help='The range of scales to test')
    parser_gen.add_argument('--noise', nargs=2, type=float, default=(0.01, 0.01),
                            help='The amount of noise to add')
    parser_gen.add_argument('--duplicate', nargs=2, type=int, default=(1, 1),
                            help='The range of multiples for each point in the degraded set')
    parser_gen.add_argument('--seed', type=int, default=4,
                            help='The random seed for generating a degradation')

    parser_plot = subparsers.add_parser('plot', help="Plot the genrated convergence rates")
    parser_plot.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)
