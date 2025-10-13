
from pathlib import Path
import pickle
from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding

from settings import NUM_BODY_MODULES


def store_nde(location: str, nde: NeuralDevelopmentalEncoding) -> None:
    with open(f"{location}.pkl", "wb") as f:
        pickle.dump(nde, f)

def load_nde(location: str) -> NeuralDevelopmentalEncoding | None:
    path = Path(f"{location}.pkl")

    if not path.exists():
        return None
    
    with open(path, "rb") as f:
        return pickle.load(f)

def load_or_init_nde(location: str) -> NeuralDevelopmentalEncoding:
    nde: NeuralDevelopmentalEncoding | None = load_nde(location)
    if nde == None:
        print("CREATING NEW NDE")
        nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_BODY_MODULES)
        store_nde(location, nde)
    return nde