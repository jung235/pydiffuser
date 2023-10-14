import logging

logger = logging.getLogger("pydiffuser")
logger.setLevel(logging.INFO)

logger.propagate = False
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
