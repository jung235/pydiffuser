class PydiffuserException(Exception):
    """A generic, base class for all Pydiffuser's errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidTimeError(PydiffuserException):
    """Raise when invalid time is encountered."""


class InvalidDimensionError(PydiffuserException):
    """Raise when invalid spatial dimension is encountered."""


class ShapeMismatchError(PydiffuserException):
    """Raise when invalid shape of `ArrayType` is encountered."""


class MemoryAllocationError(PydiffuserException):
    """Raise when memory allocation fails."""


class ConfigException(PydiffuserException):
    """Error related to configuration."""


class CLIException(PydiffuserException):
    """Error related to CLI."""
