def test_psi_runs():
    from momo_ml.metrics.psi import compute_psi
    assert compute_psi([1,2,3], [1,2,3]) >= 0
