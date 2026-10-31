"""Locations used by every script. Override with environment variables.
RUL_WORK   : intermediate artifacts and result files (default: ../work, shipped precomputed)
RUL_FIGS   : output of make_figures.py (default: ../figs)
CMAPSS_DIR : directory that contains train_FD001.txt ... (NASA C-MAPSS)
PRONOSTIA_DIR : directory that contains Learning_set/ and Full_Test_Set/ (IEEE PHM 2012 challenge data)"""
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("RUL_WORK", os.path.join(REPO, "work"))
FIGS = os.environ.get("RUL_FIGS", os.path.join(REPO, "figs"))
CMAPSS_ROOT = os.environ.get("CMAPSS_DIR", os.path.join(REPO, "data", "cmapss")).rstrip("/") + "/"
PRON_ROOT = os.environ.get("PRONOSTIA_DIR", os.path.join(REPO, "data", "pronostia"))
os.makedirs(WORK, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
