from .base import CloudPair, DatasetHandler
from .handlers import DATASET_HANDLERS


def create_dataset(name, root):
    try:
        return DATASET_HANDLERS[str(name).upper()](root)
    except KeyError as error:
        choices = ", ".join(DATASET_HANDLERS)
        raise ValueError(f"Unsupported dataset {name!r}. Available: {choices}") from error
