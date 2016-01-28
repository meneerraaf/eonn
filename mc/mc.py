import sys
from eonn import eonn
from eonn.genome import Genome
from eonn.organism import Pool
from math import cos, log


# State limits
XMIN = -1.2
XMAX =  0.5
VMIN = -0.07
VMAX =  0.07


def bound(value, lb, ub):
	""" Bounds the value between lower- and upperbound lb and ub. """
	assert lb < ub
	return max(lb, min(ub, value))


def draw(pos, vel, action):
	""" Prints a graphic representation of the current state """
	action = int(round(bound(action, -1, 1)))
	x = int(abs(XMIN - pos) / (abs(XMIN - XMAX) / 72))
	for i in range(72):
		if i == x:
			sys.stdout.write(['0', '>', '<'][action])
		else:
			sys.stdout.write('-')
	print


def update(pos, vel, action):
	""" Updates position and velocity with given action. """
	action = round(bound(action, -1, 1))
	vel = bound(vel + 0.001 * action - 0.0025 * cos(3 * pos), VMIN, VMAX)
	pos = bound(pos + vel, XMIN, XMAX)
	if pos <= XMIN:
		vel = 0.0
	return pos, vel


def mc(policy, max_steps=500, verbose=False):
	""" Mountain Car evaluation function. """
	pos, vel, err = -0.5354, 0.0, 0
	for i in range(max_steps):
		action = policy.propagate([pos, vel])[0]
		if verbose:
			draw(pos, vel, action)
		pos, vel = update(pos, vel, action)
		if pos >= XMAX:
			break
		err += 1
	if verbose:
		print 'total steps:', err
	return 1 / log(err)


def main():
	""" Main function. """
	pool = Pool.spawn(Genome.open('mc.net'), 20, std=5.0)
	# Set evolutionary parameters
	eonn.KEEP = 5
	eonn.MUTATE_PROB = 0.9
	eonn.MUTATE_FRAC = 0.2
	eonn.MUTATE_STD = 8.0
	eonn.MUTATE_REPL = 0.1
	# Evolve population
	pool = eonn.optimize(pool, mc)
	champion = max(pool)
	# Print results
	print '\ntrace:'
	mc(champion.policy, verbose=True)
	print '\ngenome:\n%s' % champion.genome


if __name__ == '__main__':
	main()

