#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------
  Copyright (C) 2020 CNRS. All rights reserved.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along
  with this program; if not, write to the Free Software Foundation, Inc.,
  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

------------------------------------------------------------------------------

This script runs OMERO metrics on the selected dataset.

@author Julio Mateos Langerak
<a href="mailto:julio.mateos-langerak@igh.cnrs.fr">julio.mateos-langerak@igh.cnrs.fr</a>
@version Alpha0.1
<small>
(<b>Internal version:</b> $Revision: $Date: $)
</small>
@since 3.0-Beta4.3
"""

# import omero dependencies
import omero.scripts as scripts
import omero.gateway as gateway
from omero.rtypes import rlong, rstring

# import metrics
from metrics import analysis

# import configuration parser
from metrics.utils.utils import MetricsConfig

# import logging
import logging
from datetime import datetime

config_file = 'my_microscope_config.ini'

# Creating logging services
logger = logging.getLogger('metrics')
logger.setLevel(logging.DEBUG)

# # create file handler which logs even debug messages
# fh = logging.FileHandler('metrics.log')
# fh.setLevel(logging.ERROR)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
# logger.addHandler(fh)
# logger.addHandler(ch)


def run_script_local():
    from credentials import USER, PASSWORD, GROUP, PORT, HOST
    conn = gateway.BlitzGateway(username=USER,
                                passwd=PASSWORD,
                                group=GROUP,
                                port=PORT,
                                host=HOST)

    script_params = {'IDs': [154],
                     'Configuration file name': 'monthly_config.ini',
                     'Comment': 'This is a test comment'}

    try:
        conn.connect()

        logger.info(f'Metrics started using parameters: \n{script_params}')
        logger.info(f'Start time: {datetime.now()}')

        logger.info(f'Connection successful: {conn.isConnected()}')

        # Getting the configuration file associated with the microscope
        config = MetricsConfig()
        config.read(script_params['Configuration file name'])

        datasets = conn.getObjects('Dataset', script_params['IDs'])  # generator of datasets

        for dataset in datasets:
            analysis.analyze_dataset(connection=conn,
                                     script_params=script_params,
                                     dataset=dataset,
                                     config=config)

    finally:
        logger.info('Closing connection')
        conn.close()


def run_script():

    config = MetricsConfig()

    client = scripts.client(
        'Run_Metrics.py',
        """This is the main script of omero.metrics. It will run the analysis on the selected 
        dataset. For more information check \n
        http://www.mri.cnrs.fr\n
        Copyright: Write here some copyright info""",  # TODO: copyright info

        scripts.String(
            "Data_Type", optional=False, grouping="1",
            description="The data you want to work with.", values=[rstring('Dataset')],
            default="Dataset"),

        scripts.List(
            "IDs", optional=False, grouping="1",
            description="List of Dataset IDs").ofType(rlong(0)),

        scripts.String(  # TODO: make enum list with other option
            'Configuration file name', optional=False, grouping='1', default='monthly_config.ini',
            description='Add here any eventuality that you want to add to the analysis'
        ),

        scripts.String(
            'Comment', optional=True, grouping='2',
            description='Add here any eventuality that you want to add to the analysis'
        ),
    )

    try:
        script_params = {}
        for key in client.getInputKeys():
            if client.getInput(key):
                script_params[key] = client.getInput(key, unwrap=True)

        logger.info(f'Metrics started using parameters: \n{script_params}')
        logger.info(f'Start time: {datetime.now()}')

        conn = gateway.BlitzGateway(client_obj=client)

        # Verify user is part of metrics group by checking current group. If not, abort the script
        if conn.getGroupFromContext().getName() != 'metrics':
            raise PermissionError('You are not authorized to run this script in the current context.')

        logger.info(f'Connection success: {conn.isConnected()}')

        datasets = conn.getObjects('Dataset', script_params['IDs'])  # generator of datasets

        for dataset in datasets:
            # Get the project / microscope
            microscope_prj = dataset.getParent()  # We assume one project per dataset

            for ann in microscope_prj.listAnnotations():
                if type(ann) == gateway.FileAnnotationWrapper:
                    if ann.getFileName() == script_params['Configuration file name']:
                        config.read_string(ann.getFileInChunks().__next__().decode())  # TODO: Fix this for large config files
                        break

            analysis.analyze_dataset(connection=conn,
                                     script_params=script_params,
                                     dataset=dataset,
                                     config=config)
        logger.info(f'End time: {datetime.now()}')

    finally:
        logger.info('Closing connection')
        client.closeSession()


if __name__ == '__main__':
    # run_script()
    run_script_local()

