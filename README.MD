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

not sure what else to put here, maybe a TODO?

TODO (in no particular order):
- [ ] figure out if we should use the Individual class that Ariel provides or our own implementation
- [ ] load/store mechanism for both Individual and Population
- [ ] Individual evaluation mechanism (setting up simulation, checking for abortions, getting fitness)
- [ ] robot body Operators (mutation, Xover)
- [ ] visualization of Status class
