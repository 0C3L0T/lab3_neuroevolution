from dataclasses import dataclass
from pathlib import Path


@dataclass
class Status:
    """
    Store the status of the training.
    """
    desired_body_iterations: int
    current_body_iteration: int
    checkpoint_dir: Path


def load_training_status(location: str) -> Status | None:
    """
    Load plaintext status file at location, or None if it doesn't exist.
    Expected format (key=value per line):
        desired_body_iterations=100
        current_body_iteration=42
        checkpoint_dir=checkpoints/
    """
    path = Path(location)
    if not path.exists():
        return None

    data = {}
    with path.open("r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                data[key] = value

    try:
        return Status(
            desired_body_iterations=int(data["desired_body_iterations"]),
            current_body_iteration=int(data["current_body_iteration"]),
            checkpoint_dir=Path(data["checkpoint_dir"]),
        )
    except KeyError:
        # File is incomplete or invalid
        return None


def store_training_status(status: Status, location: str) -> None:
    """
    Store status as plaintext at location. Overwrites only if values differ.
    """
    path = Path(location)
    old = load_training_status(location)

    # Avoid rewriting identical content
    if old == status:
        return

    with path.open("w") as f:
        f.write(f"desired_body_iterations={status.desired_body_iterations}\n")
        f.write(f"current_body_iteration={status.current_body_iteration}\n")
        f.write(f"checkpoint_dir={status.checkpoint_dir}\n")


def display_training_status(status: Status) -> None:
    """
    Pretty-print the current training status.
    """
    print(f"Training progress: {status.current_body_iteration}/{status.desired_body_iterations}")
    # print(f"Checkpoint directory: {status.checkpoint_dir}")
