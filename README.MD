# Lab 3: Robot Olympics

After cloning, you'll probably have to install the dependencies with UV:

```
cd ariel
uv venv
uv sync
```
and then select the ariel interpreter

maybe also add evotorch as a dependency:
```
cd ariel
uv add evotorch
```

TODO (in no particular order):
- [ ] store fittest individual of every generation
- [ ] store Problem instance for pause/resume functionality
- [ ] render video of best individual of every generation
- [ ] improved abortion logic: just kill everything that doesn't move?
