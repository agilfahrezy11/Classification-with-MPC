# MPC-LULC Classification Pipeline
# Top-level package imports

from importlib import import_module as _import_module

# Lazy imports — modules are loaded on first access to keep startup fast
# and avoid hard failures when optional heavy dependencies aren't installed.

def __getattr__(name):
    _public_map = {
        "Pipeline": ("mpc_lulc.pipeline", "Pipeline"),
        "STAC_Client": ("mpc_lulc.data_acquisition", "STAC_Client"),
        "Image_Acquisitor": ("mpc_lulc.data_acquisition", "Image_Acquisitor"),
        "Classification_Scheme": ("mpc_lulc.classification_scheme", "Classification_Scheme"),
        "Sample_Manager": ("mpc_lulc.sample_data", "Sample_Manager"),
        "Sample_Quality_Analyzer": ("mpc_lulc.sample_data_quality", "Sample_Quality_Analyzer"),
        "Feature_Extractor": ("mpc_lulc.predictor", "Feature_Extractor"),
        "Classifier": ("mpc_lulc.classification", "Classifier"),
        "Accuracy_Assessor": ("mpc_lulc.accuracy", "Accuracy_Assessor"),
    }
    if name in _public_map:
        module_path, class_name = _public_map[name]
        mod = _import_module(module_path)
        return getattr(mod, class_name)
    raise AttributeError(f"module 'mpc_lulc' has no attribute {name!r}")


__all__ = [
    "Pipeline",
    "STAC_Client",
    "Image_Acquisitor",
    "Classification_Scheme",
    "Sample_Manager",
    "Sample_Quality_Analyzer",
    "Feature_Extractor",
    "Classifier",
    "Accuracy_Assessor",
]
