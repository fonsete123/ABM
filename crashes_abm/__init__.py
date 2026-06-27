# Cap BLAS/Accelerate to one thread per process BEFORE numpy is imported. The model
# works on tiny arrays, so multithreaded BLAS gives no speed-up; under joblib it only
# causes thread oversubscription (14 workers x 14 BLAS threads) that thrashes the CPU
# and makes the Sobol stage crawl. On macOS, threadpoolctl can't throttle vecLib, so
# this env-var cap (read once at numpy import) is the reliable fix. setdefault keeps a
# user's explicit override intact.
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

from .parameters import Params, cfg
from .agent import Trader
from .model import Market, run
from .metrics import metrics, avg_metrics, acf, hill_tail
from .prospect_theory import pt_value

__all__ = ["Params", "cfg", "Trader", "Market", "run",
           "metrics", "avg_metrics", "acf", "hill_tail", "pt_value"]
__version__ = "1.0.0"
