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
This module implements an artificial neural network (ANN), which is a circuit
of neurons that are connected via synapses. It represents the phenotype of an
organism and is constructed based on the organism's genotype, or genome.

Classes:
neuron  -- implementation of an artificial neuron
synapse -- outgoing link from a neuron with an associated weight
network -- an ANN, i.e., a circuit of interconnected neurons
"""

from math import exp
from genome import *


def sigmoid(activation, response=4.924273):
	return 1.0 / (1.0 + exp(-activation * response))


class Neuron(object):
	""" A neuron receives zero or more inputs and produces one output. """
	def __init__(self, pos, func, bias):
		self.pos = pos
		self.func = func
		self.bias = bias
		self.activation = 0.0
		self.synapses = []

	def attach(self, synapse):
		""" Attach a synapse coming from another neuron. """
		if self.pos != INPUT:
			self.synapses.append(synapse)

	def activate(self):
		""" Activates the neuron by summing all incoming connections. """
		if self.pos != INPUT:
			self.activation = self.bias
			for synapse in self.synapses:
				self.activation += synapse.output

	@property
	def output(self):
		""" Return the output (depends on activation function). """
		if self.func == SIGMOID:
			return sigmoid(self.activation)
		return self.activation


class Synapse:
	""" A synapse permits a neuron to pass its signal to another neuron. """
	def __init__(self, src, weight):
		self.src = src
		self.weight = weight

	@property
	def output(self):
		return self.src.output * self.weight


class Network(object):
	""" Phenotypic representation of a genome, expressing its behavior. """
	def __init__(self, genome):
		neurons = {}
		# Parse genes (assume genes are in order and that neurons come first)
		for gene in genome:
			if gene.type == NEURON:
				id, pos, func, bias = gene.dna
				neurons[id] = Neuron(pos, func, bias)
			else: # SYNAPSE
				src, dst, weight = gene.dna
				neurons[dst].attach(Synapse(neurons[src], weight))
		# Store neurons by type
		neurons = neurons.values()
		self._input  = [x for x in neurons if x.pos == INPUT]
		self._hidden = [x for x in neurons if x.pos == HIDDEN]
		self._output = [x for x in neurons if x.pos == OUTPUT]

	def _load(self, input):
		""" Copy input values to the input neurons of the network. """
		if len(input) != len(self._input):
			raise ValueError, 'Wrong number of input values'
		for i, neuron in enumerate(self._input):
			neuron.activation = input[i]

	def _activate(self):
		""" Activate each (non-input) neuron in the network once. """
		for neuron in self._hidden + self._output:
			neuron.activate()

	def propagate(self, input, t=None):
		""" Propagate the input through the network and return its output.

		Keyword arguments:
		input -- input values
		t     -- number of times to activate the network

		The network is activated until the output no longer changes, or, if set, t
		times after which the output is returned.
		"""
		self._load(input)
		if t is None:
			backup = None
			while self.output != backup:
				backup = self.output
				self._activate()
		else:
			for i in range(t):
				self._activate()
		return self.output

	@property
	def output(self):
		return [neuron.output for neuron in self._output]
