import os
import sys
# ensure project root is on sys.path for sibling imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
from utils.gol_simulator import simulate

def main():
    parser = argparse.ArgumentParser(description='Generate GOL patterns for survive and die')
    parser.add_argument('--data_dir', type=str, default=os.path.dirname(__file__),
                        help='base directory to create survive and die subfolders')
    parser.add_argument('--steps', type=int, default=200, help='simulation steps per pattern')
    parser.add_argument('--max_survive', type=int, default=1000, help='number of survive patterns to generate')
    parser.add_argument('--max_die', type=int, default=1000, help='number of die patterns to generate')
    parser.add_argument('--tries_mult', type=int, default=100, help='multiplier for max tries')
    parser.add_argument('--print_every', type=int, default=100, help='print progress every N tries')
    args = parser.parse_args()

    survive_dir = os.path.join(args.data_dir, 'survive')
    die_dir = os.path.join(args.data_dir, 'die')
    os.makedirs(survive_dir, exist_ok=True)
    os.makedirs(die_dir, exist_ok=True)

    N = 32
    STEPS = args.steps
    max_survive = args.max_survive
    max_die = args.max_die
    max_tries = (max_survive + max_die) * args.tries_mult

    survive_count = 0
    die_count = 0
    tries = 0

    while (survive_count < max_survive or die_count < max_die) and tries < max_tries:
        x0 = (np.random.rand(N, N) < 0.3).astype(np.uint8)
        history = simulate(x0, steps=STEPS)
        # survived full simulation?
        if len(history) == STEPS:
            rep_state = history[-1]
            if rep_state.sum() > 0 and survive_count < max_survive:
                np.save(os.path.join(survive_dir, f'pattern_{survive_count}.npy'), rep_state)
                survive_count += 1
        # died out early?
        elif len(history) < STEPS:
            if die_count < max_die:
                np.save(os.path.join(die_dir, f'pattern_{die_count}.npy'), x0)
                die_count += 1
        tries += 1
        if tries % args.print_every == 0:
            print(f"Tries: {tries}, Survive: {survive_count}/{max_survive}, Die: {die_count}/{max_die}")

    print(f"Finished: Survive {survive_count}/{max_survive}, Die {die_count}/{max_die}, Tries {tries}")

if __name__ == '__main__':
    main()
