# Lesson 03 Pole Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a continuous-versus-discrete pole comparison for 1 ms, 10 ms, and 500 ms ZOH sampling periods.

**Architecture:** Keep calculations and plotting in `discretize_demo.py`. `continuous_poles()` uses the existing continuous matrix, `discrete_poles()` uses the existing ZOH matrix, and `plot_pole_comparison()` saves a two-panel s-plane/z-plane figure.

**Tech Stack:** Python 3.11, NumPy, SciPy `cont2discrete`, Matplotlib, `unittest`.

## Global Constraints

- Reuse the existing mass-spring-damper matrices without altering plant parameters.
- Compare exactly `0.001`, `0.01`, and `0.5` seconds.
- Save the figure as `lesson03_discretization/figures/pole_compare.png`.
- Validate the exact ZOH mapping `z = exp(s * Ts)`.

---

### Task 1: Add pole-calculation interfaces

**Files:**
- Modify: `lesson03_discretization/src/discretize_demo.py`
- Modify: `lesson03_discretization/tests/test_discretize_demo.py`

**Interfaces:**
- Consumes: `build_continuous_model()` and `discretize_zoh(sample_time_s: float)`.
- Produces: `continuous_poles() -> np.ndarray` and `discrete_poles(sample_time_s: float) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
class PoleCalculationTest(unittest.TestCase):
    def test_discrete_poles_follow_exponential_mapping(self) -> None:
        continuous_poles = discretize_demo.continuous_poles()
        self.assertTrue(np.all(np.real(continuous_poles) < 0.0))
        for sample_time_s in (0.001, 0.01, 0.5):
            expected = np.sort_complex(np.exp(continuous_poles * sample_time_s))
            actual = np.sort_complex(discretize_demo.discrete_poles(sample_time_s))
            np.testing.assert_allclose(actual, expected)
            self.assertTrue(np.all(np.abs(actual) < 1.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest lesson03_discretization.tests.test_discretize_demo.PoleCalculationTest -v`

Expected: FAIL because `continuous_poles` is not defined.

- [ ] **Step 3: Write minimal implementation**

```python
def continuous_poles() -> np.ndarray:
    a, _, _, _ = build_continuous_model()
    return np.linalg.eigvals(a)


def discrete_poles(sample_time_s: float) -> np.ndarray:
    ad, _, _, _, _ = discretize_zoh(sample_time_s)
    return np.linalg.eigvals(ad)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest lesson03_discretization.tests.test_discretize_demo.PoleCalculationTest -v`

Expected: PASS; all three discrete pole pairs have magnitudes below one.

- [ ] **Step 5: Commit**

Run: `git add lesson03_discretization/src/discretize_demo.py lesson03_discretization/tests/test_discretize_demo.py; git commit -m "feat: calculate lesson03 continuous and discrete poles"`

### Task 2: Generate the unit-circle comparison figure

**Files:**
- Modify: `lesson03_discretization/src/discretize_demo.py`
- Modify: `lesson03_discretization/tests/test_discretize_demo.py`

**Interfaces:**
- Consumes: `continuous_poles() -> np.ndarray` and `discrete_poles(sample_time_s: float) -> np.ndarray`.
- Produces: `plot_pole_comparison(sample_times_s: tuple[float, ...]) -> Path`.

- [ ] **Step 1: Write the failing test**

```python
class PoleFigureTest(unittest.TestCase):
    def test_plot_pole_comparison_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(discretize_demo, "ROOT", Path(temp_dir)):
                figure_path = discretize_demo.plot_pole_comparison((0.001, 0.01, 0.5))
            self.assertEqual(figure_path.name, "pole_compare.png")
            self.assertTrue(figure_path.is_file())
            self.assertGreater(figure_path.stat().st_size, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest lesson03_discretization.tests.test_discretize_demo.PoleFigureTest -v`

Expected: FAIL because `plot_pole_comparison` is not defined.

- [ ] **Step 3: Write minimal implementation**

```python
def plot_pole_comparison(sample_times_s: tuple[float, ...]) -> Path:
    import matplotlib.pyplot as plt

    figure, (s_axes, z_axes) = plt.subplots(1, 2, figsize=(10, 4.5))
    s_poles = continuous_poles()
    s_axes.scatter(np.real(s_poles), np.imag(s_poles), label="continuous")
    s_axes.axhline(0.0, color="black", linewidth=0.8)
    s_axes.axvline(0.0, color="black", linewidth=0.8)
    s_axes.set_title("s-plane")
    s_axes.set_xlabel("Real")
    s_axes.set_ylabel("Imaginary")
    s_axes.grid(True)
    s_axes.legend()

    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    z_axes.plot(np.cos(angle), np.sin(angle), "k--", label="unit circle")
    for sample_time_s in sample_times_s:
        poles = discrete_poles(sample_time_s)
        z_axes.scatter(np.real(poles), np.imag(poles), label=f"Ts={sample_time_s:g}s")
    z_axes.axhline(0.0, color="black", linewidth=0.8)
    z_axes.axvline(0.0, color="black", linewidth=0.8)
    z_axes.set_aspect("equal", adjustable="box")
    z_axes.set_title("z-plane")
    z_axes.set_xlabel("Real")
    z_axes.set_ylabel("Imaginary")
    z_axes.grid(True)
    z_axes.legend()

    figure.tight_layout()
    output_dir = ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "pole_compare.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)
    return figure_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest lesson03_discretization.tests.test_discretize_demo.PoleFigureTest -v`

Expected: PASS; a nonempty `pole_compare.png` is created in the patched root.

- [ ] **Step 5: Commit**

Run: `git add lesson03_discretization/src/discretize_demo.py lesson03_discretization/tests/test_discretize_demo.py; git commit -m "feat: plot lesson03 pole comparison"`

### Task 3: Expose the experiment and document it

**Files:**
- Modify: `lesson03_discretization/src/discretize_demo.py`
- Modify: `lesson03_discretization/tests/test_discretize_demo.py`
- Modify: `lesson03_discretization/README.md`
- Modify: `lesson03_discretization/requirements.txt`

**Interfaces:**
- Consumes: `format_results(sample_times_s: tuple[float, ...]) -> str` and `plot_pole_comparison(sample_times_s: tuple[float, ...]) -> Path`.
- Produces: `main() -> None` output that includes the figure path.

- [ ] **Step 1: Write the failing test**

```python
def test_format_results_includes_all_three_sample_times(self) -> None:
    output_text = discretize_demo.format_results((0.001, 0.01, 0.5))
    self.assertIn("Ts = 0.001 s", output_text)
    self.assertIn("Ts = 0.01 s", output_text)
    self.assertIn("Ts = 0.5 s", output_text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest lesson03_discretization.tests.test_discretize_demo.OutputArtifactTest.test_format_results_includes_all_three_sample_times -v`

Expected: FAIL because the named test does not yet exist.

- [ ] **Step 3: Write minimal implementation and documentation**

```python
def main() -> None:
    sample_times_s = (0.001, 0.01, 0.5)
    output_text = format_results(sample_times_s)
    output_path = write_output(output_text)
    figure_path = plot_pole_comparison(sample_times_s)
    print(output_text, end="")
    print(f"\nOutput saved to: {output_path}")
    print(f"Pole figure saved to: {figure_path}")
```

Add `matplotlib` to lesson requirements. Extend the README with the output path and the rules: negative real parts mean continuous stability, magnitudes below one mean discrete stability, and the 500 ms poles move but remain inside the unit circle under exact ZOH.

- [ ] **Step 4: Run full lesson tests and experiment**

Run: `python -m unittest lesson03_discretization.tests.test_discretize_demo -v; python .\lesson03_discretization\src\discretize_demo.py`

Expected: all tests pass; output includes 0.5 s and `figures/pole_compare.png` exists.

- [ ] **Step 5: Commit**

Run: `git add lesson03_discretization/src/discretize_demo.py lesson03_discretization/tests/test_discretize_demo.py lesson03_discretization/README.md lesson03_discretization/requirements.txt lesson03_discretization/Ad_Bd_output.txt lesson03_discretization/figures/pole_compare.png; git commit -m "docs: complete lesson03 pole comparison experiment"`

## Plan Self-Review

- Spec coverage: Tasks 1 and 2 cover calculation, mapping, stability, and figure requirements; Task 3 covers execution and explanation.
- Placeholder scan: no unresolved work markers or generic implementation instructions remain.
- Type consistency: all tasks use `tuple[float, ...]` sampling periods and the documented `Path` output contract.
