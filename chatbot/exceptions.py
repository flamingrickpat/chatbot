class UnsupportedLoggingLevel(Exception):
    """
    Unknown logging level in config file.
    """

class ConfigNotFoundException(Exception):
    """
    Path to config file is invalid.
    """

class CharacterAlreadyExistsException(Exception):
    """Character name in db already exists."""

class CharacterDoesntExistsException(Exception):
    """Character doesn't exist."""
