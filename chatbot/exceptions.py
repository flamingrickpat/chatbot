class UnsupportedLoggingLevel(Exception):
    """
    Unknown logging level in config file.
    """

class ConfigNotFoundException(Exception):
    """
    Path to config file is invalid.
    """

class CharacterAlreadyExistsException(Exception):
    """Chracter name in db already exists."""