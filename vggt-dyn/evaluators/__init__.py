"""evaluators — modular evaluation pipeline for VGGT-Dyn outputs.

Each sub-module corresponds to one benchmark / evaluation purpose:

    evaluators/
        metrics.py   — shared metric functions (depth, point-cloud)
        base.py      — BaseEvaluator abstract class
        bonn.py      — Bonn RGB-D (dynamic indoor depth, AbsRel / RMSE / δ)
        sintel.py    — MPI-Sintel depth (.dpt GT, AbsRel / RMSE / δ)
        kitti.py     — KITTI Eigen split (monocular depth, AbsRel / RMSE / δ)
        scared.py    — SCARED (surgical depth, mm-level Chamfer)
        dtu.py       — DTU (multi-view reconstruction, Chamfer + Sim(3))

Usage (from eval.py dispatcher):
    from evaluators.bonn   import BonnEvaluator
    from evaluators.sintel import SintelEvaluator
    from evaluators.kitti  import KITTIEvaluator
    from evaluators.scared import SCaredEvaluator
    from evaluators.dtu    import DTUEvaluator
"""
