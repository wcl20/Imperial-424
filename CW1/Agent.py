import numpy as np 
import operator

class Agent:

	def __init__(self, states, actions, epsilon):
		self.states = states
		self.actions = actions
		self.epsilon = epsilon

		# Initilize empty list for all (s,a) pairs
		self.returns = {s: {a: [] for a in actions} for s in states}

		# Initialize Q values to 0 for all (s,a) pairs
		self.q_table = {s: {a: 0 for a in actions} for s in states}

	# Act epsilon greedy
	def act(self, state):
		greedy_action = max(self.q_table[state].items(), key=operator.itemgetter(1))[0]
		random_action = np.random.choice(self.actions)
		return random_action if np.random.random() < self.epsilon else greedy_action

	def train(self, episode, gamma):

		# Calculate Returns
		acc = 0
		returns = []
		for state, action, reward in reversed(episode):
			acc = reward + gamma * acc
			returns.append((state, action, acc))
		returns.reverse()

		# First Visit Monte Carlo
		visited = set({})
		for state, action, G in returns:
			if not (state, action) in visited:
				# Append G to list of returns
				self.returns[state][action].append(G)
				# Calculate average from list of returns
				self.q_table[state][action] = np.mean(self.returns[state][action])
				visited.add((state, action))


	def optimal_value_function(self):
		# Initialize value to 0 for all states
		v_table = {s: 0 for s in self.states}

		# For each state ...
		for state in v_table.keys():
			v_table[state] = max(self.q_table[state].items(), key=operator.itemgetter(1))[1]
		return v_table

	def optimal_policy(self):
		# Initilize policy to None for all states
		policy = {s: None for s in self.states}

		# For each state
		for state in policy.keys():
			# Return action with highest Q value
			policy[state] = max(self.q_table[state].items(), key=operator.itemgetter(1))[0]

		return policy


