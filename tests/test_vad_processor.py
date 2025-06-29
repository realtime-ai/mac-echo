from unittest.mock import patch, MagicMock

from src.macecho.vad.vad import VadProcessor


@patch('src.macecho.vad.vad.urllib.request.urlretrieve')
@patch('src.macecho.vad.vad.ort.InferenceSession')
def test_frame_size_calculation(mock_session, mock_urlretrieve):
    mock_urlretrieve.return_value = ("dummy", None)
    mock_session.return_value = MagicMock()
    vad = VadProcessor(per_frame_duration=0.032, sampling_rate=16000)
    assert vad.frame_size == 1024
