
import sys

class Warnings:
    def __init__(self):
        pass
    
    def colored_text(self, text, color):
        """
        Returns colored text for terminal output.
        Parameters:
        ----------------
        text : str : Text to be colored
        color : str : Color name ('red', 'green', 'yellow', 'blue')
        Returns:
        ----------------
        str : Colored text
        """

        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'reset': '\033[0m'
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"
        
    def property_warning(self, property):
        print(property)
        if property is None or property not in ['phase', 'amplitude']:
            print(self.colored_text(f'ERROR: Please specify property: "phase" or "amplitude"', 'red'))
            sys.exit(1)