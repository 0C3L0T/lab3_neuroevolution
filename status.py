
from dataclasses import dataclass


@dataclass
class Status:
    '''
    Store the status of the training
    '''
    desired_body_iterations: int
    current_body_iteration: int

def load_training_status(void) -> Status:
    return None

def update_training_status(status: Status) -> None:
    return None

def display_training_status(status: Status) -> None:
    return None