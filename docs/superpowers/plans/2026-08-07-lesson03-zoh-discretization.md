# Lesson 03 ZOH Discretization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build experiment 1 of lesson 03: discretize the lesson 01 mass-spring-damper state-space model with ZOH at 1 ms and 10 ms, then save reproducible matrix output.

**Architecture:** `discretize_demo.py` is a small, standalone module that owns the continuous model, its SciPy ZOH conversion, text formatting, and command-line output. The test module imports it directly and validates numerical contracts without invoking subprocesses.

**Tech Stack:** Python 3.11, NumPy 2.4.6, SciPy 1.17.1, `unittest`.

## Global Constraints

- Use the lesson 01 base model `m=1.0`, `c=0.8`, `k=4.0`.
- Use `scipy.signal.cont2discrete(..., method="zoh")`.
- Compare exactly `Ts=0.001 s` and `Ts=0.01 s`.
- Save output to `lesson03_discretization/Ad_Bd_output.txt`.
- Do not add an extreme sampling time, pole figure, discrete controller, or edits to lessons 01–02.
- Do not push to GitHub during this implementation.

---

### Task 1: Continuous Model and ZOH Conversion

**Files:**
- Create: `lesson03_discretization/src/discretize_demo.py`
- Create: `lesson03_discretization/tests/test_discretize_demo.py`

**Interfaces:**
- Produces: `build_continuous_model() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
- Produces: `discretize_zoh(sample_time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]`

- [ ] **Step 1: Write failing conversion tests**

Create a test module that imports the `discretize_demo` module and requires `discretize_zoh` with `getattr`:

```python
discretize_zoh = getattr(discretize_demo, "discretize_zoh", None)
self.assertIsNotNone(discretize_zoh)

ad_1ms, bd_1ms, cd_1ms, dd_1ms, dt_1ms = discretize_zoh(0.001)
ad_10ms, bd_10ms, cd_10ms, dd_10ms, dt_10ms = discretize_zoh(0.01)

self.assertEqual(ad_1ms.shape, (2, 2))
self.assertEqual(bd_1ms.shape, (2, 1))
self.assertEqual(cd_1ms.shape, (2, 2))
self.assertEqual(dd_1ms.shape, (2, 1))
self.assertAlmostEqual(dt_1ms, 0.001)
self.assertAlmostEqual(dt_10ms, 0.01)
self.assertFalse(np.allclose(ad_1ms, ad_10ms))
self.assertFalse(np.allclose(bd_1ms, bd_10ms))
```

Add a second test asserting `discretize_zoh(0.0)` and `discretize_zoh(-0.001)` each raise `ValueError`.

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
conda run -n robot-control python -m unittest lesson03_discretization.tests.test_discretize_demo -v
```

Expected: assertion failure because `discretize_zoh()` does not yet exist.

- [ ] **Step 3: Implement the minimum continuous and discrete model**

Use this exact continuous model:

```python
def build_continuous_model():
    A = np.array([[0.0, 1.0], [-4.0, -0.8]])
    B = np.array([[0.0], [1.0]])
    C = np.eye(2)
    D = np.zeros((2, 1))
    return A, B, C, D
```

Validate `sample_time_s > 0.0`, call `cont2discrete((A, B, C, D), sample_time_s, method="zoh")`, and return its five values.

- [ ] **Step 4: Run focused and full tests and verify GREEN**

```powershell
conda run -n robot-control python -m unittest lesson03_discretization.tests.test_discretize_demo -v
conda run -n robot-control python -m unittest discover -s .\lesson02_servo_control\tests -p "test_*.py" -v
```

Expected: the two new tests and all existing lesson 02 tests pass.

- [ ] **Step 5: Commit the verified conversion core**

```powershell
git add lesson03_discretization/src/discretize_demo.py lesson03_discretization/tests/test_discretize_demo.py
git commit -m "feat: add lesson03 zoh discretization core"
```

---

### Task 2: Text Output Artifact and Lesson README

**Files:**
- Modify: `lesson03_discretization/src/discretize_demo.py`
- Modify: `lesson03_discretization/tests/test_discretize_demo.py`
- Create: `lesson03_discretization/README.md`
- Create: `lesson03_discretization/requirements.txt`
- Generate: `lesson03_discretization/Ad_Bd_output.txt`
- Modify: `README.md`

**Interfaces:**
- Produces: `format_results(sample_times_s: tuple[float, ...]) -> str`
- Produces: `write_output(output_text: str) -> Path`

- [ ] **Step 1: Write failing output tests**

Add tests requiring `format_results` and `write_output` with `getattr`. Assert that formatted output contains the literal labels `Ts = 0.001 s`, `Ts = 0.01 s`, `Ad =`, and `Bd =`. Patch `discretize_demo.ROOT` to a temporary directory, call `write_output`, and assert the resulting file exists and exactly equals the provided output text.

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
conda run -n robot-control python -m unittest lesson03_discretization.tests.test_discretize_demo -v
```

Expected: assertion failure because the formatting and output functions do not exist.

- [ ] **Step 3: Implement formatting, file output, and script entry point**

`format_results((0.001, 0.01))` must call `discretize_zoh()` for each sample time and format `Ad` and `Bd` using `np.array2string(..., precision=8, suppress_small=True)`. `write_output()` writes UTF-8 text to `ROOT / "Ad_Bd_output.txt"`. `main()` formats these two sample times, writes the file, prints the same text, and prints the artifact path.

- [ ] **Step 4: Write documentation and dependencies**

Write the lesson README in Chinese. It must answer: continuous versus digital control, why a robot needs discretization, ZOH meaning, and why sampling time is part of the model. Include one command to run the script and one command to run the lesson tests. Create `requirements.txt` containing exact NumPy and SciPy versions already pinned in the root requirements. Add a root README bullet for lesson 03 and a run command.

- [ ] **Step 5: Run all verification commands**

```powershell
conda run -n robot-control python .\lesson03_discretization\src\discretize_demo.py
conda run -n robot-control python -m unittest lesson03_discretization.tests.test_discretize_demo -v
conda run -n robot-control python -m unittest discover -s .\lesson02_servo_control\tests -p "test_*.py" -v
git diff --check
```

Expected: output text file exists, its two sample-time blocks contain `Ad` and `Bd`, lesson 03 tests pass, existing lesson 02 tests pass, and there are no whitespace errors.

- [ ] **Step 6: Commit artifacts and documentation**

```powershell
git add README.md lesson03_discretization
git commit -m "docs: complete lesson03 discretization experiment"
```
