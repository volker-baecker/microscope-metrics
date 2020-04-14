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

This script deletes from a list of datasets the data produced by OMERO-metrics.

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


def clean_dataset(connection, dataset_id, metrics_tag_id):

    dataset = connection.getObject('Dataset', dataset_id)

    # Clean Dataset annotations
    for ann in dataset.listAnnotations():
        if isinstance(ann, (gateway.MapAnnotationWrapper, gateway.FileAnnotationWrapper)):
            connection.deleteObjects('Annotation', [ann.getId()], wait=True)

    # Clean new images tagged as metrics
    for image in dataset.listChildren():
        for ann in image.listAnnotations():
            if type(ann) == gateway.TagAnnotationWrapper and ann.getId() == metrics_tag_id:
                connection.deleteObjects('Image', [image.getId()], deleteAnns=False, deleteChildren=True, wait=True)

    # Clean File and map annotations on rest of images
    for image in dataset.listChildren():
        for ann in image.listAnnotations():
            if isinstance(ann, (gateway.MapAnnotationWrapper, gateway.FileAnnotationWrapper)):
                connection.deleteObjects('Annotation', [ann.getId()], wait=True)

    # Delete all rois
    roi_service = connection.getRoiService()
    for image in dataset.listChildren():
        rois = roi_service.findByImage(image.getId(), None)
        rois_ids = [r.getId().getValue() for r in rois.rois]
        if len(rois_ids) > 1:
            connection.deleteObjects('Roi', rois_ids, wait=True)

def run_script_local():
    from credentials import USER, PASSWORD, GROUP, PORT, HOST
    conn = gateway.BlitzGateway(username=USER,
                                passwd=PASSWORD,
                                group=GROUP,
                                port=PORT,
                                host=HOST)

    script_params = {'Dataset IDs': [1],
                     'Metrics tag ID': 132,
                     'Confirm deletion': True,
                    }

    try:
        conn.connect()

        for dataset_id in script_params['Dataset IDs']:

            clean_dataset(connection=conn,
                          dataset_id=dataset_id,
                          metrics_tag_id=script_params['Metrics tag ID'])

    finally:
        conn.close()


# def run_script():
#
#     client = scripts.client(
#         'Run_Metrics.py',
#         """This is the main script of omero.metrics. It will run the analysis on the selected
#         dataset. For more information check \n
#         http://www.mri.cnrs.fr\n
#         Copyright: Write here some copyright info""",
#
#         scripts.Long(
#             'Dataset ID', optional=False, grouping='1',
#             description='ID of the dataset to be analyzed'
#         ),
#
#         scripts.String(
#             'Configuration file name', optional=False, grouping='1', default='monthly_config.ini',
#             description='Add here any eventuality that you want to add to the analysis'
#         ),
#
#         scripts.String(
#             'Comment', optional=True, grouping='2',
#             description='Add here any eventuality that you want to add to the analysis'
#         ),
#     )
#
#     try:
#         script_params = {}
#         for key in client.getInputKeys():
#             if client.getInput(key):
#                 script_params[key] = client.getInput(key, unwrap=True)
#
#         logger.info(f'Metrics started using parameters: \n{script_params}')
#
#         conn = gateway.BlitzGateway(client_obj=client)
#         logger.info(f'Connection successful: {conn.isConnected()}')
#
#         dataset = conn.getObject('Dataset', script_params['Dataset ID'])
#
#         # Getting the configuration file associated with the microscope
#         config = MetricsConfig()
#         dataset_parents = dataset.listParents()
#         if len(dataset_parents) != 1:
#             logger.error('This dataset is either associated to more than one or no microscope projects')
#             raise Exception()
#         else:
#             microscope_prj = dataset_parents[0]
#
#         prj_annotations = microscope_prj.listFileAnnotations()
#
#         for ann in prj_annotations:
#             if ann.getFileName() == script_params['Configuration file name']:
#                 config.read_string(anns[0].getFileInChunks().__next__().decode())
#                 break
#
#         analyze_dataset(connection=conn,
#                         dataset=dataset,
#                         config=config)
#
#     finally:
#         client.closeSession()


if __name__ == '__main__':
    # run_script()
    run_script_local()

