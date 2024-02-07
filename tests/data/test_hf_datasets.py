from experiment.src.data.hf_datasets import HellaSwagDataset


def test_get_data():
    hellaswag = HellaSwagDataset()
    context, candidates, labels = hellaswag.get_data()
    assert len(context) > 0
    assert len(candidates) > 0
    assert len(labels) > 0
    assert len(context) == len(candidates) == len(labels)