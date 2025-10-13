
# hardcoded locations for storing things
CHECKPOINT_LOCATION = "checkpoints/"
STATUS_LOCATION = "training_status.txt"
NDE_LOCATION = "NDE"

# used for black box, DON'T TOUCH
GENOTYPE_SIZE = 64

# the number of parts in a body
NUM_BODY_MODULES = 15

# how many bodies to train at a time
NUM_BODY_ACTORS = 2

# how many brains to train at the same time
NUM_BRAIN_ACTORS = 20

# how many brains to use in brain population
BRAIN_POPULATION_SIZE = 30

# arena size in tournament selection
ARENA_SIZE = 3

# how many bodies to use in a population, should be multiple of 6
BODY_POPULATION_SIZE = 18

# how many iterations to train body for
# brains have dynamic training iterations
DEFAULT_BODY_ITERATIONS = 9999