
import sys
import numpy as np

class Warnings:
    def __init__(self):
        self.ecc_warned = False
        self.mass_ratio_warned = False
        self.mean_anomaly_warned = False
        self.chispin_warned = False
    
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
        # Use red for errors, yellow for warnings, green for success messages, and blue for informational messages
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


    def allowed_eccentricity_warning(self, ecc_ref):
        ecc = np.asarray(ecc_ref, dtype=float)

        if not getattr(self, "ecc_warned", False):
            if np.any(ecc < 0):
                print(self.colored_text(
                    "Eccentricity contains non-physical values (e < 0). "
                    "These will be removed from the dataset.", 'yellow'))

            if np.any(ecc == 0):
                print(self.colored_text(
                    "Eccentricity contains circular case (e = 0). "
                    "This will be removed from the dataset to avoid numerical issues in residual calculations.", 'yellow'))

        ecc = ecc[ecc > 0]

        if ecc.size == 0:
            raise ValueError("No valid eccentricity values remain after filtering.")
        
        self.ecc_warned = True
        return ecc


    def allowed_mass_ratio_warning(self, mass_ratio):
        q = np.asarray(mass_ratio, dtype=float)


        if np.any(q <= 0) and not getattr(self, "mass_ratio_warned", False):
            raise ValueError("Mass ratio must be > 0.")

        if np.any(q < 1):
            q = np.where(q < 1, 1/q, q)
            if not getattr(self, "mass_ratio_warned", False):
                print(self.colored_text(
                    "Mass ratio q < 1 detected. Converting to q >= 1 using q -> 1/q.", 'yellow'))

        self.mass_ratio_warned = True
        return q


    def allowed_mean_anomaly_warning(self, mean_ano):
        l = np.asarray(mean_ano, dtype=float)
        wrapped = l % (2 * np.pi)

        if not np.allclose(l, wrapped):
            if not getattr(self, "mean_anomaly_warned", False):
                print(self.colored_text(
                    "Mean anomaly values wrapped into range [0, 2π).", 'yellow'))
            
        self.mean_anomaly_warned = True
        return wrapped


    def allowed_chispin_warning(self, chi):
        chi = np.asarray(chi, dtype=float)
        clipped = np.clip(chi, -1, 1)

        if not np.allclose(chi, clipped):
            if not getattr(self, "chispin_warned", False):
                print(self.colored_text(
                    "Spin values should be in the range [-1, 1]. "
                    "Values outside this range were clipped.", 'yellow'))
            
        self.chispin_warned = True
        return clipped
    

    
    
class Automated_Settings:

    def __init__(self):
        pass
    
    def resolve_property(self, prop, default):
        """
        Checks if a property is explicitly provided. If it is explicitly stated, use the defined property. 
        Otherwise, the method defaults to the instance's predefined property (self.x).
        """
        return default if prop is None else prop
        
    def automated_settings(self, property):
        """
        Automatically sets settings for generating the residual dataset and plotting based on the specified property.
        Parameters:
        ----------------
        property : str : 'phase' or 'amplitude'
        Returns:
        ----------------
        dict : Dictionary of settings for dataset generation and plotting
        """
        
        settings = {}
        
        if property == 'phase':
            settings['plot_residuals_eccentric_evolv'] = True
            settings['plot_residuals_time_evolv'] = False
            settings['save_fig_eccentric_evolv'] = True
            settings['save_fig_time_evolve'] = False
            settings['show_legend'] = True
            
        elif property == 'amplitude':
            settings['plot_residuals_eccentric_evolv'] = False
            settings['plot_residuals_time_evolv'] = True
            settings['save_fig_eccentric_evolv'] = False
            settings['save_fig_time_evolve'] = True
            settings['show_legend'] = False
            
        return settings