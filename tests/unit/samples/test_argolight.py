import pytest
from os import path


from microscopemetrics.samples import argolight
import numpy as np


@pytest.fixture
def argolight_e():
    try:
        data = np.load('/Users/julio/PycharmProjects/microscope-metrics//tests/data/201702_RI510_Argolight-1-1_010_SIR_ALX.npy')
    except FileNotFoundError as e:
        raise e

    analysis = argolight.ArgolightAnalysis()
    analysis.input.data = {'argolight_e': data}
    analysis.set_metadata('spots_distance', 5)
    analysis.set_metadata('pixel_size', (.125, .39, .39))

    return analysis

def test_run_argolight_e(argolight_e):
    output = argolight_e.analyze_spots()
    assert output


