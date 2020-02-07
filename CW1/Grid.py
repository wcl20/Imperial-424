import numpy as np 

class Grid:

	def __init__(self, p):

		# Grid layout
		self.width = 4
		self.height = 4

		# Current position
		self.x = 0
		self.y = 0
		self.terminal_states = [(0,1), (3,2)]

		# Probability of moving desired direction
		self.p = p 

		self.actions = {
			(0,0): ['E', 'S'],
			(0,2): ['E', 'W'],
			(0,3): ['S', 'W'],
			(1,0): ['N', 'E'],
			(1,1): ['N', 'S', 'W'],
			(1,3): ['N', 'S'],
			(2,1): ['N', 'E'],
			(2,2): ['E', 'S', 'W'],
			(2,3): ['N', 'W']
		}

		self.rewards = {
			(0,0): -1,
			(0,1): 10,
			(0,2): -1,
			(0,3): -1,
			(1,0): -1,
			(1,1): -1,
			(1,3): -1,
			(2,1): -1,
			(2,2): -1,
			(2,3): -1,
			(3,2): -100
		}	

	def current_state(self):
		return (self.x, self.y)

	def set_state(self, position):
		self.x = position[0]
		self.y = position[1]

	def terminal_state(self):
		return self.current_state() in self.terminal_states

	def all_states(self):
		return [state for state in [*self.rewards.keys()] if state not in self.terminal_states]

	def move(self, action):
		# Perform action with probability p
		actions = ['N', 'E', 'S', 'W']
		probabilities = list(map(lambda a: self.p if a == action else (1-self.p) / 3.0, actions))
		action = np.random.choice(actions, p=probabilities)

		if action in self.actions[(self.x, self.y)]:
			if action == 'N':
				self.x = max(0, self.x - 1)
			elif action == 'E':
				self.y = min(self.width - 1, self.y + 1)
			elif action == 'S':
				self.x = min(self.height - 1, self.x + 1)
			elif action == 'W':
				self.y = max(0, self.y - 1)
		return self.rewards[(self.x, self.y)]

	def show_values(self, values):
		print("Values:")
		for x in range(self.height):
			print("-"*32)
			for y in range(self.width):
				value = values.get((x,y), None)
				if not value is None:
					print("{value: ^7.2f}|".format(value=value), end="")
				else:
					print("{:7}|".format(""), end="")		
			print("")
		print("-"*32)

	def show_policy(self, policy):
		print("Policy:")
		for x in range(self.height):
			print("-"*32)
			for y in range(self.width):
				action = policy.get((x,y), ' ')
				print("{action: ^7}|".format(action=action), end="")
			print("")
		print("-"*32)
