
# hardcoded locations for storing things
CHECKPOINT_LOCATION = "checkpoints/"
NDE_LOCATION = "NDE"

DEBUG = True

# used for black box, DON'T TOUCH
GENOTYPE_SIZE = 64

# the number of parts in a body
NUM_BODY_MODULES = 15

# how many bodies to train at a time
NUM_BODY_ACTORS = 2

# how many brains to train at the same time
NUM_BRAIN_ACTORS = 30

# how many brains to use in brain population
BRAIN_POPULATION_SIZE = 30

# how many bodies to use in a population, should be multiple of 6
BODY_POPULATION_SIZE = 24

# how many iterations to train body for
# brains have dynamic training iterations
DEFAULT_BODY_ITERATIONS = 9999

if DEBUG:
    NUM_BODY_ACTORS = 2
    NUM_BRAIN_ACTORS = 2
    BRAIN_POPULATION_SIZE = 2
    BODY_POPULATION_SIZE = 2
    DEFAULT_BODY_ITERATIONS = 2
