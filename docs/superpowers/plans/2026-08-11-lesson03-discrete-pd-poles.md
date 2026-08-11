# Lesson 03 Discrete PD Closed-Loop Poles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a discrete PD state-feedback experiment that calculates and visualizes closed-loop poles and their gain sensitivity.

**Architecture:** `closed_loop_poles.py` imports `discretize_zoh()` from the existing lesson module and keeps all closed-loop algebra, plots, report generation, and CLI output together. A separate test module verifies the mathematical contracts and generated artifacts.

**Tech Stack:** Python 3.11, NumPy, SciPy, Matplotlib, `unittest`.

## Global Constraints

- Reuse `discretize_demo.discretize_zoh`; do not duplicate continuous model matrices.
- The PD feedback convention is `u[k] = -[[Kp, Kd]] @ x[k]`.
- The closed-loop matrix is exactly `Acl = Ad - Bd @ K`.
- A closed-loop pole set is stable only when every magnitude is strictly below one.
- Use the nominal linear model only; lesson 02 nonidealities remain outside this experiment.

---

### Task 1: Implement closed-loop PD algebra

**Files:**
- Create: `lesson03_discretization/src/closed_loop_poles.py`
- Create: `lesson03_discretization/tests/test_closed_loop_poles.py`

**Interfaces:**
- Consumes: `discretize_demo.discretize_zoh(sample_time_s: float)`.
- Produces: `closed_loop_matrix(sample_time_s: float, kp: float, kd: float) -> np.ndarray`, `closed_loop_poles(sample_time_s: float, kp: float, kd: float) -> np.ndarray`, and `is_stable(poles: np.ndarray) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
def test_closed_loop_matrix_matches_state_feedback_formula(self) -> None:
    ad, bd, _, _, _ = discretize_demo.discretize_zoh(0.01)
    actual = closed_loop_poles.closed_loop_matrix(0.01, 15.0, 2.0)
    expected = ad - bd @ np.array([[15.0, 2.0]])
    np.testing.assert_allclose(actual, expected)

def test_closed_loop_poles_report_discrete_stability(self) -> None:
    poles = closed_loop_poles.closed_loop_poles(0.01, 15.0, 2.0)
    self.assertEqual(poles.shape, (2,))
    self.assertTrue(closed_loop_poles.is_stable(poles))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest lesson03_discretization.tests.test_closed_loop_poles.ClosedLoopAlgebraTest -v`

Expected: FAIL because `closed_loop_poles.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def closed_loop_matrix(sample_time_s: float, kp: float, kd: float) -> np.ndarray:
    ad, bd, _, _, _ = discretize_zoh(sample_time_s)
    return ad - bd @ np.array([[kp, kd]])

def closed_loop_poles(sample_time_s: float, kp: float, kd: float) -> np.ndarray:
    return np.linalg.eigvals(closed_loop_matrix(sample_time_s, kp, kd))

def is_stable(poles: np.ndarray) -> bool:
    return bool(np.all(np.abs(poles) < 1.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest lesson03_discretization.tests.test_closed_loop_poles.ClosedLoopAlgebraTest -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add lesson03_discretization/src/closed_loop_poles.py lesson03_discretization/tests/test_closed_loop_poles.py; git commit -m "feat: add discrete PD closed-loop pole algebra"`

### Task 2: Add case comparison and Kp sweep figures

**Files:**
- Modify: `lesson03_discretization/src/closed_loop_poles.py`
- Modify: `lesson03_discretization/tests/test_closed_loop_poles.py`

**Interfaces:**
- Consumes: `closed_loop_poles(...) -> np.ndarray`.
- Produces: `DEFAULT_CASES`, `sweep_kp(kp_values: np.ndarray, kd: float, sample_time_s: float) -> np.ndarray`, `plot_closed_loop_poles(cases) -> Path`, and `plot_gain_sweep(kp_values, kd, sample_time_s) -> Path`.

- [ ] **Step 1: Write the failing tests**

```python
def test_kp_sweep_has_one_pair_of_poles_per_gain(self) -> None:
    kp_values = np.array([0.0, 10.0, 20.0])
    history = closed_loop_poles.sweep_kp(kp_values, kd=2.0, sample_time_s=0.01)
    self.assertEqual(history.shape, (3, 2))

def test_plot_functions_write_png_files(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.object(closed_loop_poles, "ROOT", Path(temp_dir)):
            case_path = closed_loop_poles.plot_closed_loop_poles(
                closed_loop_poles.DEFAULT_CASES
            )
            sweep_path = closed_loop_poles.plot_gain_sweep(
                np.linspace(0.0, 200.0, 20), kd=2.0, sample_time_s=0.01
            )
        self.assertTrue(case_path.is_file())
        self.assertTrue(sweep_path.is_file())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest lesson03_discretization.tests.test_closed_loop_poles.ClosedLoopArtifactTest -v`

Expected: FAIL because sweep and plotting interfaces are not defined.

- [ ] **Step 3: Write minimal implementation**

```python
DEFAULT_CASES = {
    "weak": (2.0, 0.2),
    "medium": (15.0, 2.0),
    "strong": (80.0, 2.0),
}

def sweep_kp(
    kp_values: np.ndarray,
    kd: float,
    sample_time_s: float,
) -> np.ndarray:
    return np.array([
        np.sort_complex(closed_loop_poles(sample_time_s, kp, kd))
        for kp in kp_values
    ])

def plot_closed_loop_poles(cases: dict[str, tuple[float, float]]) -> Path:
    return _save_z_plane_case_plot(cases, ROOT / "figures" / "closed_loop_poles.png")

def plot_gain_sweep(kp_values: np.ndarray, kd: float, sample_time_s: float) -> Path:
    history = sweep_kp(kp_values, kd, sample_time_s)
    return _save_z_plane_sweep_plot(history, ROOT / "figures" / "gain_sweep.png")
```

The two private plotting helpers draw a dashed unit circle, horizontal and vertical axes, grids, and legends before saving nonempty PNG files.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest lesson03_discretization.tests.test_closed_loop_poles.ClosedLoopArtifactTest -v`

Expected: PASS; both files are nonempty PNGs.

- [ ] **Step 5: Commit**

Run: `git add lesson03_discretization/src/closed_loop_poles.py lesson03_discretization/tests/test_closed_loop_poles.py; git commit -m "feat: visualize closed-loop gain sensitivity"`

### Task 3: Generate the analysis report and command-line experiment

**Files:**
- Modify: `lesson03_discretization/src/closed_loop_poles.py`
- Modify: `lesson03_discretization/tests/test_closed_loop_poles.py`
- Create: `lesson03_discretization/reports/closed_loop_analysis.md`

**Interfaces:**
- Consumes: `DEFAULT_CASES`, `closed_loop_poles(...)`, `is_stable(...)`, and both plot functions.
- Produces: `write_analysis_report() -> Path` and `main() -> None`.

- [ ] **Step 1: Write the failing test**

```python
def test_report_records_closed_loop_boundary(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.object(closed_loop_poles, "ROOT", Path(temp_dir)):
            report_path = closed_loop_poles.write_analysis_report()
        report_text = report_path.read_text(encoding="utf-8")
    self.assertIn("Ad - Bd @ K", report_text)
    self.assertIn("噪声", report_text)
    self.assertIn("100 ms", report_text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest lesson03_discretization.tests.test_closed_loop_poles.ClosedLoopReportTest -v`

Expected: FAIL because `write_analysis_report` is not defined.

- [ ] **Step 3: Write minimal implementation**

```python
def write_analysis_report() -> Path:
    report_path = ROOT / "reports" / "closed_loop_analysis.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_build_report_text(), encoding="utf-8")
    return report_path

def main() -> None:
    for name, (kp, kd) in DEFAULT_CASES.items():
        poles = closed_loop_poles(0.01, kp, kd)
        print(name, "poles =", poles, "stable =", is_stable(poles))
    print(plot_closed_loop_poles(DEFAULT_CASES))
    print(plot_gain_sweep(np.linspace(0.0, 200.0, 300), 2.0, 0.01))
    print(write_analysis_report())
```

`_build_report_text()` writes four titled Markdown sections: open-loop versus closed-loop poles, Kp movement, sample-time sensitivity at 1/10/50/100 ms with `Kp=15`, `Kd=2`, and omitted engineering nonidealities including noise, saturation, delay, and model uncertainty.

- [ ] **Step 4: Run complete verification**

Run: `python -m unittest lesson03_discretization.tests.test_discretize_demo lesson03_discretization.tests.test_closed_loop_poles -v; python .\lesson03_discretization\src\closed_loop_poles.py`

Expected: all tests pass; two figures and the report exist.

- [ ] **Step 5: Commit**

Run: `git add lesson03_discretization/src/closed_loop_poles.py lesson03_discretization/tests/test_closed_loop_poles.py lesson03_discretization/figures/closed_loop_poles.png lesson03_discretization/figures/gain_sweep.png lesson03_discretization/reports/closed_loop_analysis.md; git commit -m "analyze discrete closed-loop poles and gain sensitivity"`

## Plan Self-Review

- Spec coverage: Tasks 1–3 implement every required algebra, case, sweep, artifact, report, and validation item.
- Placeholder scan: no unresolved or generic work markers remain.
- Type consistency: every task uses `sample_time_s`, `kp`, `kd`, `np.ndarray` pole values, and `Path` output locations consistently.
