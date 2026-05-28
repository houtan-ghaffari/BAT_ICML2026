__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

import numpy as np
from collections import defaultdict
from typing import Any, Dict, List

from yaml_config import get_task_config
from engine import run_experiment, seed_everything


def main() -> None:
    """
    runs either a 5-fold cross-validation for ESC-50 or a standard experiment for other downstream tasks.
    """

    seed_everything()
    args = get_task_config()
    print("\n=== Experiment Configuration ===")
    for key, value in sorted(vars(args).items()):
        print(f"{key:>25}: {value}")
    print("================================\n")

    if args.task == 'esc50':
        all_metrics: Dict[str, List[float]] = defaultdict(list)
        all_histories: Dict[str, Any] = {}
        for fold in range(1, 6):
            metrics, hist = run_experiment(args, fold_idx=fold)
            all_histories.update(hist)
            for k, v in metrics.items():
                all_metrics[k].append(v)

        print("\n=== ESC-50 5-Fold Cross Validation Results ===")
        for k, values in all_metrics.items():
            print(f"{k}: Mean={np.mean(values):.2f} | Std={np.std(values):.2f} | Folds={np.round(values, 2).tolist()}")

    else:
        run_experiment(args)


if __name__ == '__main__':
    main()
