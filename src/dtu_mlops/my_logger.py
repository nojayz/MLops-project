import sys

from loguru import logger

logger.remove()  # remove default handler
logger.add("logs/vae_mnist.log", rotation="100 MB", level="WARNING") 
logger.debug("Debug message: checking variable state")
logger.info("Info message: training started")
logger.warning("Warning message: learning rate is very high")
logger.error("Error message: failed to load checkpoint")
logger.critical("Critical message: out of memory, stopping training")
