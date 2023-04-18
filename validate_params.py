"""Module to perform config validation and eventuall raise the proper exceptions"""

import logging
from typing import Dict

from .utils.validation_utils.utils import validate_all
from .utils.style_utils.colours import Colours

logger = logging.getLogger(__name__)


def validate_params(model_params: Dict, preprocessing_params: Dict) -> None:
    """Function to validate configuration parameters dictionary.

    Parameters
    ----------
    model_params : Dict
        Dictionary of model configuration params defined in parameters.yaml file.
    preprocessing_params : Dict
        Dictionary of preprocessing configuration params defined in parameters.yaml file.

    Returns
    -------
    None
        If everything works it just prints a message in the log.
        Otherwise it raises the proper exceptions.

    """
    logger.info(Colours.YELLOW + "Starting validation of params." + Colours.ENDC)
    validate_all(model_params, preprocessing_params)
    logger.info(Colours.GREEN + "Parameters are validated." + Colours.ENDC)
