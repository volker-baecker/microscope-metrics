import pytest
from os import path
from tests.constants import *

from microscopemetrics.samples import psf_beads
import numpy as np


@pytest.fixture()
def psf_beads_analysis():
    temp_dir = path.abspath(TEST_DATA_DIR)
    file_name = 'psf_beads_EM-488_MAG-40.npy'
    file_url = ''
    try:
        data = np.load(path.join(temp_dir, file_name))
    except FileNotFoundError as e:
        repos = np.DataSource(temp_dir)
        repos.open(file_url)

    analysis = psf_beads.PSFBeadsAnalysis()
    analysis.input.data = {'beads_image': data}
    analysis.set_metadata('theoretical_fwhm_lateral_res', 0.300)
    analysis.set_metadata('theoretical_fwhm_axial_res', 0.800)
    analysis.set_metadata('pixel_size', (.35, .06, .06))

    return analysis


def test_run_psf_beads(psf_beads_analysis):
    assert psf_beads_analysis.run()
    assert psf_beads_analysis.output
