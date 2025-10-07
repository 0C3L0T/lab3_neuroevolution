from evotorch.algorithms import SNES, CMAES, PGPE, GeneticAlgorithm, Cosyne, SteadyStateGA
from evotorch.logging import StdOutLogger

from sample_problem import get_sample_problem

class HookedSNES(SNES):
    def __init__(self,
                 problem,
                 popsize,
                 stdev_init,
                 stopper # extra stop condition. is passed (current_SNES_iteration, self)
                 ):

        super().__init__(
            problem=problem,
            popsize=popsize,
            stdev_init=stdev_init
        )

        self.stopper = stopper

    def run(self, num_generations: int, *, reset_first_step_datetime: bool = True):
        """
        Run the algorithm for the given number of generations
        (i.e. iterations).

        Args:
            num_generations: Number of generations.
            reset_first_step_datetime: If this argument is given as True,
                then, the datetime of the first search step will be forgotten.
                Forgetting the first step's datetime means that the first step
                taken by this new run will be the new first step datetime.
        """
        if reset_first_step_datetime:
            self.reset_first_step_datetime()

        for i in range(int(num_generations)):
            self.step()
            if self.stopper(i, self):
                return

        if len(self._end_of_run_hook) >= 1:
            self._end_of_run_hook(dict(self.status))


def main():

    sample_problem = get_sample_problem(num_actors=6)

    def stopper(i, searcher):
        print(f'stopper called. best.evals: {searcher.status["best"].evals}')

        if i > 100:
            return True

    searcher = HookedSNES(sample_problem, popsize=1000, stdev_init=0.01, stopper=stopper)
    logger = StdOutLogger(searcher)
    searcher.run(5000)



if __name__ == '__main__':
    main()