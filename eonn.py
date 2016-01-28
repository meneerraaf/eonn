# Copyright (C) 2010 Rogier Koppejan <rogier.koppejan@gmail.com>
#
# This file is part of eonn.
#
# Eonn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Eonn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this library.  If not, see <http://www.gnu.org/licenses/>.

"""
Main module, implementing a complete generational evolutionary algorithm.
"""

import random
from organism import *


SAMPLESIZE  = 5     # Sample size used for tournament selection
KEEP        = 0     # Nr. of organisms copied to the next generation (elitism)
MUTATE_PROB = 0.75  # Probability that offspring is being mutated
MUTATE_FRAC = 0.2   # Fraction of genes that get mutated
MUTATE_STD  = 1.0   # Std. dev. of mutation distribution (gaussian)
MUTATE_REPL = 0.25  # Probability that a gene gets replaced


def optimize(pool, feval, epochs=100, evals=1, verbose=True):
	""" Evolve supplied population using feval as fitness function.

	Keyword arguments:
	pool   -- population to optimize
	feval  -- fitness function
	epochs -- duration of evoluationary run
	evals  -- samples per individual

	"""
	evaluate(pool, feval, evals)
	# Iteratively reproduce and evaluate
	for i in range(epochs):
		pool = epoch(pool, len(pool))
		evaluate(pool, feval, evals)
		if verbose:
			print i+1, '%s' % max(pool)
	# Return pool
	return pool

def evaluate(pool, feval, evals=1):
	""" Evaluate each individual in the population. """
	for org in pool:
		while len(org.evals) < evals:
			org.update(feval(org.policy))

def epoch(pool, size):
	""" Breed a new generation of organisms."""
	offspring = []
	elite = sorted(pool, reverse=True)[:KEEP]
	for i in range(size - KEEP):
		mom, dad = select(pool), select(pool)
		child = reproduce(mom, dad)
		offspring.append(child)
	return Pool(offspring + elite)

def reproduce(mom, dad):
	""" Recombine two parents into one organism. """
	child = mom.crossover(dad)
	if random.random() < MUTATE_PROB:
		child.mutate(MUTATE_FRAC, MUTATE_STD, MUTATE_REPL)
	return child

def select(pool):
	""" Select one individual using tournament selection. """
	return max(random.sample(pool, SAMPLESIZE))
