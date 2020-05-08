# Main samples module defining the sample superclass


class Sample:
    """This is the superclass defining the interface to a sample object. You should subclass this in order to create a
    new sample."""
    def __init__(self, config, analysis_to_func):
        """Add to the init subclass a dictionary mapping analyses strings to functions
        :type config: config section
        :param config: config section specifying sample options
        :type analysis_to_func: dict
        :param analysis_to_func: dictionary mapping analyses strings to functions
        """
        self.config = config
        self.analysis_to_func = analysis_to_func

    def validate_dataset(self):
        """Override this method to integrate all the logic of dataset validation"""
        pass

    def analyze_dataset(self, dataset, config=None):
        """Override this method to integrate all the logic of the analyses of a dataset according the config settings
        :param dataset: a dataset object to be analyzed
        :param config: the configuration with which this dataset has to be analyzed. If None is provided, the configuration of the sample instance will be used

        :returns a list of images
                 a list of tags
                 a list of dicts
                 a dict containing table_names and tables
        """
        pass

    def analyze_image(self, image, analyses, config):
        """A passthrough to the different analysis implemented on the sample.
        :param image: an image object to be analyzed
        :param analyses: a str of list of str specifying the analysis to be made. string to functions to be run
                         are mapped through

        :param config: a dictionary with the configuration to analyze the image the configuration

        :returns a list of image objects
                 a list of roi objects
                 a list of tags
                 a list of dicts
                 a dict containing table_names and tables
        """
        out_images = list()
        out_rois = list()
        out_tags = list()
        out_dicts = list()
        out_tables = dict()
        if isinstance(analyses, str):
            analyses = list(analyses)

        for analysis in analyses:
            images, rois, tags, dicts, tables = self.analysis_to_func[analysis](image, config)
            out_images.extend(images)
            out_rois.extend(rois)
            out_tags.extend(tags)
            out_dicts.extend(dicts)
            out_tables.update(tables)

        return out_images, out_tags, out_dicts, out_tables

    def _create_roi(self, shapes, name, description):
        """A helper function to create ROIs"""
        roi = {'name': name,
               'desc': description,
               'shapes': shapes}

        return roi

    def _create_shape(self, shape_type, **kwargs):
        if shape_type in ['point', 'line', 'rectangle', 'ellipse', 'polygon']:
            shape = {'type': shape_type,
                     'args': kwargs}

            return shape
        else:
            raise TypeError('Cannot recognize that type of shape')
