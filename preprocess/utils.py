# description: various useful helpers

class DataKeys(object):
    """standard names for features
    and transformations of features
    """

    # core data keys
    FEATURES = "features"
    LABELS = "labels"
    PROBABILITIES = "probs"
    LOGITS = "logits"
    ORIG_SEQ = "sequence"
    SEQ_METADATA = "example_metadata"


class MetaKeys(object):
    """standard names for metadata keys
    """
    REGION_ID = "region"
    ACTIVE_ID = "active"
    FEATURES_ID = "features"
