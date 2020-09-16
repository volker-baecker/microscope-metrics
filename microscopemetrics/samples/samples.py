# Main samples module defining the sample superclass

from abc import ABC, abstractmethod
import logging

from ..model import model

# We are defining some global dictionaries to register the different analysis types
IMAGE_ANALYSIS_REGISTRY = {}
DATASET_ANALYSIS_REGISTRY = {}
PROGRESSION_ANALYSIS_REGISTRY = {}

# Decorators to register exposed analysis functions
def register_image_analysis(fn):
    IMAGE_ANALYSIS_REGISTRY[fn.__name__] = fn
    return fn


def register_dataset_analysis(fn):
    DATASET_ANALYSIS_REGISTRY[fn.__name__] = fn
    return fn


def register_progression_analysis(fn):
    PROGRESSION_ANALYSIS_REGISTRY[fn.__name__] = fn
    return fn


# Create a logging service
logger = logging.getLogger(__name__)


class Configurator(ABC):
    """This is a superclass taking care of the configuration of a new sample. Helps generating configuration files and
    defines configuration parameters necessary for the different analyses. You should subclass this when you create a
    new sample.
    """

    # The configuration section has to be defined for every subclass
    CONFIG_SECTION: str = None

    def __init__(self, config):
        self.config = config
        self.metadata_definitions = self.define_metadata()

    @abstractmethod
    def define_metadata(self):
        pass

    @classmethod
    def register_sample_analyzer(cls, sample_class):
        cls.SAMPLE_CLASS = sample_class
        return sample_class


class Analyzer(ABC):
    """This is the superclass defining the interface to a sample object. You should subclass this when you create a
    new sample."""
    def __init__(self, config):
        """Add to the init subclass a dictionary mapping analyses strings to functions
        :type config: analysis_config section
        :param config: analysis_config section specifying sample options
        """
        self.config = config

    @staticmethod
    def crete_input_dataset(data: dict = None, metadata: dict = None):
        input_dataset = model.MetricsDataset(data=data, metadata=metadata)
        return input_dataset

    @abstractmethod
    def describe_input_requirements(self):
        pass

    @classmethod
    def get_module(cls):
        """Returns the module name of the class. Without path and extension.
        :returns a string with the module name
        """
        return cls.__module__.split(sep=".")[-1]

    def validate_dataset(self, dataset):
        """Override this method to integrate all the logic of dataset validation"""
        pass

    def _verify_limits(self, key_values, config, object_ref):
        """Verifies that the numeric values provided in the key_values dictionary are within the ranges found in the analysis_config"""
        limits_passed = {
            "uhl_passed": True,
            "lhl_passed": True,
            "usl_passed": True,
            "lsl_passed": True,
            "limits": list(),
            "sources": list(),
        }
        for option, limit in config.items():
            if option[-4:] in ["_uhl", "_lhl", "_usl", "_lsl"]:
                key = option[:-4]
                try:
                    value = key_values[key]
                except KeyError as e:
                    continue

            # Evaluate upper hard limits
            if option.endswith("_uhl"):
                limit = self._evaluate_limit(key_values, limit)
                if limit is not None and value >= limit:
                    key_values[key + "_uhl_passed"] = "No"
                    limits_passed["uhl_passed"] = False
                    limits_passed["limits"].append(option)
                    limits_passed["sources"].append(object_ref)
                else:
                    key_values[key + "_uhl_passed"] = "Yes"

            # Evaluate upper soft limits
            elif option.endswith("_usl"):
                limit = self._evaluate_limit(key_values, limit)
                if limit is not None and value >= limit:
                    key_values[key + "_usl_passed"] = "No"
                    limits_passed["usl_passed"] = False
                    limits_passed["limits"].append(option)
                    limits_passed["sources"].append(object_ref)
                else:
                    key_values[key + "_usl_passed"] = "Yes"

            # Evaluate lower hard limits
            elif option.endswith("_lhl"):
                limit = self._evaluate_limit(key_values, limit)
                if limit is not None and value <= limit:
                    key_values[key + "_lhl_passed"] = "No"
                    limits_passed["lhl_passed"] = False
                    limits_passed["limits"].append(option)
                    limits_passed["sources"].append(object_ref)
                else:
                    key_values[key + "_lhl_passed"] = "Yes"

            # Evaluate lower soft limits
            elif option.endswith("_lsl"):
                limit = self._evaluate_limit(key_values, limit)
                if limit is not None and value <= limit:
                    key_values[key + "_lsl_passed"] = "No"
                    limits_passed["lsl_passed"] = False
                    limits_passed["limits"].append(option)
                    limits_passed["sources"].append(object_ref)
                else:
                    key_values[key + "_lsl_passed"] = "Yes"

        return key_values, limits_passed

    def _evaluate_limit(self, key_values, expression):
        # We keep the evaluation in a separate function to avoid namespace conflict
        locals().update(key_values)
        try:
            limit = eval(expression)
        except NameError as e:
            limit = None
            raise e(f"Could not evaluate expression {expression} as a limit")
        finally:
            return limit

    def analyze_dataset(self, dataset, analyses, config, verify_limits=True):
        """A passthrough to the different analysis implemented on the sample for a particular dataset.
        :param dataset: a dataset object to be analyzed
        :param analyses: a str of list of str specifying the analysis to be made. string to functions to be run
                         are mapped through self.dataset_analysis_to_func
        :param config: a dictionary with the configuration to analyze the image the configuration
        :param verify_limits: Do a verification of the limits established in teh analysis_config file

        :returns a list of image objects
                 a list of tags
                 a list of dicts
                 a list of bools indicating if dicts should be editable or not
                 a dict containing table_names and tables
                 a list of dicts specifying if soft and hard limits are passed or not. if verify_limits is False, an
                 empty list is returned
        """
        out_images = list()
        out_tags = list()
        out_dicts = list()
        out_editables = list()
        out_tables = dict()
        limits_passed = list()
        if isinstance(analyses, str):
            analyses = [analyses]

        for analysis in analyses:
            (
                images,
                rois,
                tags,
                dicts,
                editables,
                tables,
            ) = self.dataset_analysis_to_func[analysis](dataset, config)
            out_images.extend(images)
            out_tags.extend(tags)
            if verify_limits:
                validated_dicts = []
                for d in dicts:
                    validated_dict, passed = self._verify_limits(
                        d, config, f"Dataset_ID:{dataset.getId()}"
                    )
                    validated_dicts.append(validated_dict)
                    limits_passed.append(passed)
                out_dicts.extend(validated_dicts)
            else:
                out_dicts.extend(dicts)
            out_editables.extend(editables)
            out_tables.update(tables)

        return out_images, out_tags, out_dicts, out_editables, out_tables, limits_passed

    def analyze_image(self, image, analyses, config, verify_limits=True):
        """A passthrough to the different analysis implemented on the sample for a particular image.
        :param image: an image object to be analyzed
        :param analyses: a str of list of str specifying the analysis to be made. string to functions to be run
                         are mapped through self.image_analysis_to_func
        :param config: a dictionary with the configuration to analyze the image the configuration
        :param verify_limits: Do a verification of the limits established in teh analysis_config file

        :returns a list of image objects
                 a list of roi objects
                 a list of tags
                 a list of dicts
                 a dict containing table_names and tables
                 a list of dicts specifying if soft and hard limits are passed or not, which limits are not passed
                 and the images ids. if verify_limits is False, an empty list is returned
        """
        out_images = list()
        out_rois = list()
        out_tags = list()
        out_dicts = list()
        out_tables = dict()
        limits_passed = list()
        if isinstance(analyses, str):
            analyses = [analyses]

        for analysis in analyses:
            images, rois, tags, dicts, tables = self.image_analysis_to_func[analysis](
                image, config
            )
            out_images.extend(images)
            out_rois.extend(rois)
            out_tags.extend(tags)
            if verify_limits:
                validated_dicts = []
                for d in dicts:
                    validated_dict, passed = self._verify_limits(
                        d, config, f'Image_ID:{image["image_id"]}'
                    )
                    validated_dicts.append(validated_dict)
                    limits_passed.append(passed)
                out_dicts.extend(validated_dicts)
            else:
                out_dicts.extend(dicts)
            out_tables.update(tables)

        return out_images, out_rois, out_tags, out_dicts, out_tables, limits_passed

    @staticmethod
    def _create_roi(shapes, name=None, description=None):
        """A helper function to create ROIs"""
        roi = {"name": name, "desc": description, "shapes": shapes}
        return roi

    @staticmethod
    def _create_shape(shape_type, **kwargs):
        """A helper function to create roi shapes"""
        if shape_type in ["point", "line", "rectangle", "ellipse", "polygon"]:
            shape = {"type": shape_type, "args": kwargs}
            return shape
        else:
            raise ValueError("Cannot recognize that type of shape")


class Reporter:
    """This is the superclass taking care of creating reports for a particular type of sample.
    You should subclass this when you create a new sample."""

    def __init__(
        self,
        config,
        image_report_to_func={},
        dataset_report_to_func={},
        microscope_report_to_func={},
    ):
        """Add to the init subclass a dictionary mapping analyses strings to functions
        :type config: analysis_config section
        :param config: analysis_config section specifying sample options
        :type image_report_to_func: dict
        :param image_report_to_func: dictionary mapping image analyses strings to functions
        :type dataset_report_to_func: dict
        :param dataset_report_to_func: dictionary mapping dataset analyses strings to functions
        :type microscope_report_to_func: dict
        :param microscope_report_to_func: dictionary mapping microscope analyses strings to functions
        """
        self.config = config
        self.image_report_to_func = image_report_to_func
        self.dataset_report_to_func = dataset_report_to_func
        self.microscope_report_to_func = microscope_report_to_func

    def produce_image_report(self, image):
        pass

    def produce_dataset_report(self, dataset):
        pass

    def produce_device_report(self, device):
        pass

    # TODO: move this where it belongs
    # # Helper functions
    # def get_tables(self, omero_object, namespace_start='', name_filter=''):
    #     tables_list = list()
    #     resources = omero_object._conn.getSharedResources()
    #     for ann in omero_object.listAnnotations():
    #         if isinstance(ann, gw.FileAnnotationWrapper) and \
    #                 ann.getNs().startswith(namespace_start) and \
    #                 name_filter in ann.getFileName():
    #             table_file = omero_object._conn.getObject("OriginalFile", attributes={'name': ann.getFileName()})
    #             table = resources.openTable(table_file._obj)
    #             tables_list.append(table)
    #
    #     return tables_list
    #
