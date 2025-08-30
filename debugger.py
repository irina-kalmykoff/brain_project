class DebugMixin:
    """
    A mixin class providing debug and logging functionality.
    """
    
    def __init__(self, class_name, debug_mode=False):
        """
        Initialize the debug mixin.
        
        Parameters:
        -----------
        class_name : str
            Name of the class to use in log messages
        debug_mode : bool
            Whether to enable debug mode initially
        """
        self.CLASS_NAME = class_name
        self.DEBUG_MODE = debug_mode
    
    def enable_debug(self):
        """Enable debug mode."""
        self.DEBUG_MODE = True
        self.log("Debug mode enabled")
        return self
    
    def disable_debug(self):
        """Disable debug mode."""
        self.log("Debug mode will be disabled")
        self.DEBUG_MODE = False
        return self
    
    def log(self, message):
        """Print standard operational logs."""
        print(f"{self.CLASS_NAME}: {message}")
    
    def debug(self, message):
        """Print debug logs only when DEBUG_MODE is True."""
        if self.DEBUG_MODE:
            print(f"{self.CLASS_NAME} [DEBUG]: {message}")