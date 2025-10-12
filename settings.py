
import os

# the number of parts in a body
NUM_BODY_MODULES = 30

# how many bodies to train at a time
NUM_BODY_ACTORS = 2

# how many brains to train at the same time
NUM_BRAIN_ACTORS = 2 # os.cpu_count() // NUM_BODY_ACTORS

# how many brains to use in brain population
BRAIN_POPULATION_SIZE = 6

# arena size in tournament selection
ARENA_SIZE = 5

# how many bodies to use in a population, should be multiple of 6
BODY_POPULATION_SIZE = 6

# how many iterations to train body for
# brains have dynamic training iterations
DEFAULT_BODY_ITERATIONS = 2