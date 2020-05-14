from enum import Enum, EnumMeta

from metrics.interface import omero as interface

# Creating logging services
import logging
module_logger = logging.getLogger('metrics.devices.devices')


DTYPES = {
    'int': (tuple,),
    'float': (tuple,),
    'bool': (type(None),),
    'str': (int,),
    'tuple': (type(None),),
}

def _call_if_callable(f):
    """Call callables, or return value of non-callables."""
    return f() if callable(f) else f


class _Setting:
    """The code of this class was copied from https://github.com/MicronOxford/microscope
    """
    # Settings classes should be private: devices should use a factory method
    # rather than instantiate settings directly; most already use add_setting
    # for this.
    def __init__(self, name, dtype, get_from_db_func, get_from_conf_func, get_from_name_func, set_func, values=None):
        """Create a setting.
        :param name: the setting's name
        :param dtype: a data type from ('int', 'float', 'bool', 'enum', 'str')
        :param get_from_db_func: a function to get the value from the database interface
        :param get_from_conf_func: a function to get the value from the device configuration
        :param get_from_name_func: a function to get the value from the image name
        :param values: a description of allowed values dependent on dtype,
                       or function that returns a description
        Setters and getters accept or return:
            the setting value for int, float, bool and str;
            the setting index into a list, dict or Enum type for enum.
        """
        self.name = name
        if dtype not in DTYPES:
            raise Exception('Unsupported dtype.')
        elif not (isinstance(values, DTYPES[dtype]) or callable(values)):
            raise Exception("Invalid values type for %s '%s':"
                            " expected function or %s"
                            % (dtype, name, DTYPES[dtype]))
        self.dtype = dtype
        self._get_from_db = get_from_db_func
        self._get_from_conf = get_from_conf_func
        self._get_from_name = get_from_name_func
        self._values = values

    def describe(self):
        """Returns a dictionary describing the setting
        """
        return {'type': self.dtype,
                'values': self.values()}

    def get(self):
        """Get a setting"""
        if self._get_from_db is not None:
            value = self._get_from_db()
            if value is not None:
                return value
        if self._get_from_conf is not None:
            value = self._get_from_conf()
            if value is not None:
                return value
        if self._get_from_name is not None:
            value = self._get_from_name()
            if value is not None:
                return value
        module_logger.warning(f'Device setting could not be retrieved: {self.name}')

    def values(self):
        values = _call_if_callable(self._values)
        if values is not None:
            if self.dtype == 'enum':
                if isinstance(values, dict):
                    return list(values.items())
                else:
                    # self._values is a list or tuple
                    return list(enumerate(values))
            elif self._values is not None:
                return values


class Device:
    """A superclass for all the microscope devices and eventually other type of devices."""
    def __init__(self, device_config):
        self.device_config = device_config
        self._settings = dict()

    def add_setting(self, name, dtype, get_from_db_func, get_from_conf_func, get_from_name_func, set_func, values=None):
        """Add a setting definition.
        :param name: the setting's name
        :param dtype: a data type from ('int', 'float', 'bool', 'enum', 'str')
        :param get_from_db_func: a function to get the value from the interface to the database
        :param get_from_conf_func: a function to get the value from the configuration
        :param get_from_name_func: a function to get the value from the image name
        :param set_func: a function to set the value
        :param values: a description of allowed values dependent on dtype,
                       or function that returns a description
        A client needs some way of knowing a setting name and data type,
        retrieving the current value and, if settable, a way to retrieve
        allowable values, and set the value.
        """
        if dtype not in DTYPES:
            raise Exception('Unsupported dtype.')
        elif not (isinstance(values, DTYPES[dtype]) or callable(values)):
            raise Exception("Invalid values type for %s '%s':"
                            " expected function or %s"
                            % (dtype, name, DTYPES[dtype]))
        else:
            self._settings[name] = _Setting(name, dtype,
                                            get_from_db_func,
                                            get_from_conf_func,
                                            get_from_name_func,
                                            set_func,
                                            values)

    def get_setting(self, name, *args):
        """Tries to get a specified setting from the following sources by order of preference:
        - Image from the database interface
        - The microscope config linked through the image name
        - Directly in the image name"""

        try:
            return self._settings[name].get(*args)
        except Exception as err:
            module_logger.error("in get_setting(%s):", name, exc_info=err)
            raise

    def get_all_settings(self):
        """Return ordered settings as a list of dicts."""
        def catch(f):
            try:
                return f()
            except Exception as err:
                module_logger.error("getting %s: %s", f.__self__.name, err)
                return None
        return {k: catch(v.get) for k, v in self._settings.items()}

    def set_setting(self, name, value):
        """Set a setting."""
        try:
            self._settings[name].set(value)
        except Exception as err:
            module_logger.error("in set_setting(%s):", name, exc_info=err)
            raise

    def describe_setting(self, name):
        """Return ordered setting descriptions as a list of dicts."""
        return self._settings[name].describe()

    def describe_settings(self):
        """Return ordered setting descriptions as a list of dicts."""
        return [(k, v.describe()) for (k, v) in self._settings.items()]

    def settings_to_conf(self):
        pass

    def update_settings(self, incoming, init=False):
        """Update settings based on dict of settings and values."""
        if init:
            # Assume nothing about state: set everything.
            my_keys = set(self._settings.keys())
            their_keys = set(incoming.keys())
            update_keys = my_keys & their_keys
            if update_keys != my_keys:
                missing = ', '.join([k for k in my_keys - their_keys])
                msg = 'update_settings init=True but missing keys: %s.' % missing
                _logger.debug(msg)
                raise Exception(msg)
        else:
            # Only update changed values.
            my_keys = set(self._settings.keys())
            their_keys = set(incoming.keys())
            update_keys = set(key for key in my_keys & their_keys
                              if self.get_setting(key) != incoming[key])
        results = {}
        # Update values.
        for key in update_keys:
            if key not in my_keys or not self._settings[key].set:
                # Setting not recognised or no set function implemented
                results[key] = NotImplemented
                update_keys.remove(key)
                continue
            if _call_if_callable(self._settings[key].readonly):
                continue
            self._settings[key].set(incoming[key])
        # Read back values in second loop.
        for key in update_keys:
            results[key] = self._settings[key].get()
        return results


class Microscope(Device):
    """A superclass for the microscopes. Inherit this class when you create a new type of microscope."""
    def __init__(self, device_config):
        super.__init__(device_config)


class WideFieldMicroscope(Microscope):
    """A Widefield microscope"""
    def __init__(self, device_config):
        super.__init__(device_config)

        self.add_setting('objective_lens_refractive_index', 'float',
                         get_from_db_func=interface.get_refractive_index,
                         get_from_conf_func=,
                         get_from_name_func=,
                         set_func=None,
                         values=(1.0, 2.0))

    def _get_objective_nr(self, image):
        img_name = image.getName()
        obj_nrs = [i for i, token in enumerate(self.device_config.getlist('OBJECTIVES', 'names')) if token in img_name]
        if len(obj_nrs) > 1:
            module_logger.error('More than one reference to an objective lens was found ',
                                'in the image name. Only hte first one will be considered.')
        elif len(obj_nrs) == 0:
            module_logger.error('No references to any objective lens were found in the image name')
            return None
        return obj_nrs[0]

    def _get_conf_objective_setting(self, image, option):
        obj_nr = self._get_objective_nr(image)
        values = eval(self.device_config.get('OBJECTIVES', option, None))
        if values is None:
            return None
        else:
            return values[obj_nr]

    def _get_channel_nrs(self, image):
        img_name = image.getName()
        ch_nrs = [i for i, token in enumerate(self.device_config.getlist('CHANNELS', 'names')) if token in img_name]
        if len(ch_nrs) == 0:
            module_logger.error('No references to any channel was found in the image name')
            return None
        return ch_nrs[0]

    def _get_conf_channel_settings(self, image, option):
        ch_nrs = self._get_channel_nrs(image)
        values = eval(self.device_config.get('CHANNELS', option, None))
        if values is None:
            return None
        else:
            return tuple(values)



# to add to if
def get_refractive_index(image):
    objective_settings = image.getObjectiveSettings()
    return objective_settings.getRefractiveIndex()

    def _get

    def _get_metadata_from_name(self, name, token_left, token_right, metadata_type=None):
        start = name.find(token_left)
        if start == -1:
            return None
        name = name[start + len(token_left):]
        end = name.find(token_right)
        name = metadata_type(name[:end])
        if metadata_type is None:
            return name
        else:
            return metadata_type(name)

    def get_image_data(self.image):
        image_name = image.getName()
        image_id = image.getId()
        raw_img = interface.get_intensities(image)
        # Images from Interface come in zctyx dimensions and locally as zcxy.
        # The easiest for the moment is to remove t
        if raw_img.shape[2] == 1:
            raw_img = np.squeeze(raw_img, 2)  # TODO: Fix this time dimension.
        else:
            raise Exception("Image has a time dimension. Time is not yet implemented for this analysis")
        pixel_size = interface.get_pixel_size(image)
        pixel_size_units = interface.get_pixel_size_units(image)

        try:
            objective_settings = image.getObjectiveSettings()
            refractive_index = objective_settings.getRefractiveIndex()
            objective = objective_settings.getObjective()
            lens_na = objective.getLensNA()
            lens_magnification = objective.getNominalMagnification()
        except AttributeError:
            module_logger.warning(f'Image {image.getName()} does not have a declared objective settings.'
                                  f'Falling back to metadata stored in image name.')
            refractive_index = _get_metadata_from_name(image_name, '_ri-', '_', float)
            lens_na = _get_metadata_from_name(image_name, '_na-', '_', float)
            lens_magnification = _get_metadata_from_name(image_name, '_mag-', '_', float)

        channels = image.getChannels()
        excitation_waves = [ch.getExcitationWave() for ch in channels]
        emission_waves = [ch.getEmissionWave() for ch in channels]
        if excitation_waves[0] is None:
            module_logger.warning(f'Image {image.getName()} does not have a declared channels settings.'
                                  f'Falling back to metadata stored in image name.')
            excitation_waves = [_get_metadata_from_name(image_name, '_ex-', '_',
                                                        float)]  # TODO: make this work with more than one channel
            emission_waves = [_get_metadata_from_name(image_name, '_em-', '_', float)]

        return {'image_data': raw_img,
                'image_name': image_name,
                'image_id': image_id,
                'pixel_size': pixel_size,
                'pixel_size_units': pixel_size_units,
                'refractive_index': refractive_index,
                'lens_na': lens_na,
                'lens_magnification': lens_magnification,
                'excitation_waves': excitation_waves,
                'emission_waves': emission_waves,
                }

