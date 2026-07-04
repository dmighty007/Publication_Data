from typing import Callable, Dict, Optional

import numpy as np  # type: ignore
from openmm import unit  # type: ignore
from openmm.app import Simulation  # type: ignore
from tqdm.auto import trange


class PathGennieMD:
    """
    PathGennie implementation based on:
    'PathGennie: A Path-Generation Algorithm for Kinetic Pathways'
    J. Chem. Theory Comput. 2016, 12, 5, 2035–2043
    """

    NM_TO_ANG = 10.0

    def __init__(
        self,
        simulation: Simulation,
        projection_fn: Callable[..., np.ndarray],
        projection_args: Optional[Dict] = None,
        mode: str = "escape",
        target_projection: Optional[np.ndarray] = None,
        convergence_fn: Optional[Callable] = None,
        convergence_args: Optional[Dict] = None,
        hybrid_switch_index: int = 0,
        hybrid_switch_value: float = 0.5,
        escape_direction: str = "auto",
        escape_cv_index: int = 0,
        temperature: float = 300.0,
        sigma: float = 0.5,
    ):
        if mode not in ("escape", "target", "hybrid", "resetting"):
            raise ValueError("mode must be 'escape', 'target', 'hybrid', or 'resetting'")
        if mode == "target" and target_projection is None:
            raise ValueError("target_projection required for target mode")
        if mode == "target" and convergence_fn is None:
            raise ValueError("convergence_fn required for target mode")
        if mode == "escape" and convergence_fn is None:
            raise ValueError("convergence_fn required for escape mode")
        if mode == "hybrid" and target_projection is None:
            raise ValueError("target_projection required for hybrid mode")
        if mode == "hybrid" and convergence_fn is None:
            raise ValueError("convergence_fn required for hybrid mode")
        if mode == "resetting" and target_projection is None:
            raise ValueError("target_projection required for resetting mode")
        if mode == "resetting" and convergence_fn is None:
            raise ValueError("convergence_fn required for resetting mode")

        self.sim = simulation
        self.mode = mode
        self.proj_fn = projection_fn
        self.proj_args = projection_args or {}
        self.target = np.asarray(target_projection) if target_projection is not None else None
        self.converge_fn = convergence_fn
        self.converge_args = convergence_args or {}
        self.hybrid_switch_index = hybrid_switch_index
        self.hybrid_switch_value = hybrid_switch_value
        self.escape_direction = escape_direction
        self.escape_cv_index = escape_cv_index

        if self.escape_direction not in ("auto", "increase", "decrease", "radial"):
            raise ValueError("escape_direction must be 'auto', 'increase', 'decrease', or 'radial'")
        if self.escape_cv_index < 0:
            raise ValueError("escape_cv_index must be non-negative")

        # Temperature for velocity re-randomization
        self.temperature = temperature * unit.kelvin
        self.sigma = sigma

    def run(
        self,
        initial_pos: np.ndarray,
        tau1: int = 200,
        tau2: int = 200,
        max_trial: int = 20,
        max_cycle: int = 5000,
        save_freq: int = 10,
        verbosity: int = 1,
        reset_rate_r: float = 0.0,
        reset_prob_tau1: float = 0.0,
        reset_prob_tau2: float = 0.0,
        reset_randomize_velocities: bool = True,
    ):
        if reset_rate_r < 0.0:
            raise ValueError("reset_rate_r must be non-negative")
        if not (0.0 <= reset_prob_tau1 <= 1.0):
            raise ValueError("reset_prob_tau1 must be between 0 and 1")
        if not (0.0 <= reset_prob_tau2 <= 1.0):
            raise ValueError("reset_prob_tau2 must be between 0 and 1")
        if self.mode == "resetting" and reset_rate_r <= 0.0 and (reset_prob_tau1 <= 0.0 and reset_prob_tau2 <= 0.0):
            raise ValueError("resetting mode requires reset_rate_r > 0 or legacy reset probabilities > 0")

        trajectory = []
        metrics_history = []

        # ---- initialize system ----
        self.sim.context.setPositions(initial_pos)
        self.sim.context.setVelocitiesToTemperature(self.temperature)

        # Initial anchor
        anchor_state = self.sim.context.getState(getPositions=True, getVelocities=True)
        pos = self._pos()
        start_proj = self._proj(pos)
        current_proj = start_proj
        current_phase = self._phase_from_projection(current_proj)

        # Metric definition
        def metric(cv, phase):
            if phase == "escape":
                return self._escape_metric(cv, start_proj)
            else:
                # Progress is closeness to target
                return -np.linalg.norm(cv - self.target)

        current_metric = metric(current_proj, current_phase)
        cycle_iter = trange(max_cycle, desc="PathGennie") if verbosity >= 2 else range(max_cycle)

        for cycle in cycle_iter:
            trial_results = []

            if self.mode == "hybrid":
                current_phase = self._phase_from_projection(current_proj)

            #  Run M trials ----
            for _ in range(max_trial):
                # Restore anchor positions but randomize velocities
                self.sim.context.setState(anchor_state)
                self.sim.context.setVelocitiesToTemperature(self.temperature)

                # τ1: sampler segment
                if self.mode == "resetting":
                    if reset_rate_r > 0.0:
                        self._step_with_poisson_reset(
                            n_steps=tau1,
                            reset_rate_r=reset_rate_r,
                            reset_state=anchor_state,
                            randomize_velocities=reset_randomize_velocities,
                        )
                    else:
                        tau1_eff = self._sample_shrunk_tau(tau1, reset_prob_tau1)
                        self.sim.step(tau1_eff)
                else:
                    self.sim.step(tau1)

                trial_pos = self._pos()
                trial_proj = self._proj(trial_pos)
                trial_metric = metric(trial_proj, current_phase if self.mode == "hybrid" else self.mode)

                # Store the state and metric
                # We need to store the FULL state after tau1 to continue it later
                state_after_tau1 = self.sim.context.getState(getPositions=True, getVelocities=True)
                trial_results.append(
                    {"metric": trial_metric, "state": state_after_tau1, "pos": trial_pos, "proj": trial_proj}
                )

            # ---- SELECTION: Boltzmann-weighted probabilistic pick ----
            # Higher metric is better in both modes.
            # P(i) ∝ exp(metric_i / sigma)
            # raw_metrics = np.array([r["metric"] for r in trial_results])
            # shifted = raw_metrics - raw_metrics.max()  # numerical stability
            # weights = np.exp(shifted / (self.sigma + 1e-12))
            # probs = weights / weights.sum()
            # chosen_idx = np.random.choice(len(trial_results), p=probs)
            # # chosen_idx = np.argmax(raw_metrics)  # greedy
            # best_trial = trial_results[chosen_idx]

            # ---- PHYSICAL SCALING SELECTION ----

            raw_metrics = np.array([r["metric"] for r in trial_results])
            m_min = np.min(raw_metrics)
            m_max = np.max(raw_metrics)

            # Avoid division by zero if all trials are identical
            if (m_max - m_min) < 1e-9:
                probs = np.ones(len(trial_results)) / len(trial_results)
            else:
                # 1. Scale metrics between 0 and 1
                # 0 = worst trial of this batch, 1 = best trial
                scaled_metrics = (raw_metrics - m_min) / (m_max - m_min)

                # 2. Boltzmann Weighting on the scaled interval
                # Shifted by 1.0 so the max weight is always exp(0) = 1
                logits = (scaled_metrics - 1.0) / (self.sigma + 1e-12)
                weights = np.exp(logits)
                probs = weights / np.sum(weights)

            chosen_idx = np.random.choice(len(trial_results), p=probs)
            best_trial = trial_results[chosen_idx]

            # ---- runner segment ----
            self.sim.context.setState(best_trial["state"])

            # Save the intermediate state (after tau1) if save_freq allows
            # if cycle % save_freq == 0:
            #    trajectory.append(best_trial["pos"] * self.NM_TO_ANG)

            if self.mode == "resetting":
                if reset_rate_r > 0.0:
                    self._step_with_poisson_reset(
                        n_steps=tau2,
                        reset_rate_r=reset_rate_r,
                        reset_state=best_trial["state"],
                        randomize_velocities=reset_randomize_velocities,
                    )
                else:
                    tau2_eff = self._sample_shrunk_tau(tau2, reset_prob_tau2)
                    self.sim.step(tau2_eff)
            else:
                self.sim.step(tau2)

            # Update anchor
            anchor_state = self.sim.context.getState(getPositions=True, getVelocities=True)
            pos = self._pos()
            current_proj = self._proj(pos)
            previous_phase = current_phase
            current_phase = self._phase_from_projection(current_proj)
            current_metric = metric(current_proj, current_phase)
            metrics_history.append(current_metric)

            if self.mode == "hybrid" and previous_phase != current_phase and verbosity:
                print(f"\nHybrid switch triggered at cycle {cycle}: escape -> target")

            # ---- SAVE (after tau2) ----
            if cycle % save_freq == 0:
                trajectory.append(pos * self.NM_TO_ANG)

            # ---- CONVERGENCE ----
            converge_fn = self.converge_fn
            if converge_fn is None:
                raise ValueError("convergence_fn is required for run()")

            if current_phase == "escape":
                if converge_fn(pos * self.NM_TO_ANG, **self.converge_args):
                    if verbosity:
                        print(f"\nEscape convergence reached at cycle {cycle}")
                    break
            else:  # mode == "target"
                # For target mode, metric is -norm(cv - target), so norm < tol means metric > -tol
                if converge_fn(pos * self.NM_TO_ANG, **self.converge_args):
                    if verbosity:
                        print(f"\nTarget convergence reached at cycle {cycle}")
                    break

                # elif -current_metric < tol_target:
                #     if verbosity:
                #         print(f"\nTarget reached at cycle {cycle}")
                #     break

            if verbosity >= 2 and cycle % 10 == 0:
                print(f"Cycle {cycle:4d}: metric={current_metric:.4f} (best of {max_trial})")

        if verbosity:
            print("Final metric:", current_metric)

        return np.array(trajectory), np.array(metrics_history)

    def _phase_from_projection(self, proj):
        """Return the active phase for the current run configuration."""
        if self.mode in ("escape", "target"):
            return self.mode
        if self.mode == "resetting":
            return "target"

        proj = np.asarray(proj)
        if proj.ndim == 0:
            raise ValueError("hybrid mode requires a vector projection with at least one component")

        if self.hybrid_switch_index >= proj.shape[0]:
            raise ValueError("hybrid_switch_index is out of bounds for the projection")

        switch_value = proj[self.hybrid_switch_index]
        return "target" if switch_value >= self.hybrid_switch_value else "escape"

    def _sample_shrunk_tau(self, tau, max_shrink_fraction):
        """Sample an effective walker length by shrinking tau by a random fraction."""
        if tau <= 0:
            return 0
        if max_shrink_fraction <= 0.0:
            return tau

        shrink = np.random.random() * max_shrink_fraction
        tau_eff = int(np.ceil(tau * (1.0 - shrink)))
        return max(1, tau_eff)

    def _step_with_poisson_reset(self, n_steps, reset_rate_r, reset_state, randomize_velocities=True):
        """Step one integrator step at a time and reset with Poisson rate r."""
        if n_steps <= 0:
            return 0

        dt = self._get_step_size_ps()
        reset_prob = 1.0 - np.exp(-reset_rate_r * dt)
        n_resets = 0

        for _ in range(n_steps):
            self.sim.step(1)
            if np.random.random() < reset_prob:
                self.sim.context.setState(reset_state)
                if randomize_velocities:
                    self.sim.context.setVelocitiesToTemperature(self.temperature)
                n_resets += 1

        return n_resets

    def _get_step_size_ps(self):
        """Return integrator step size in picoseconds when available."""
        try:
            step_size = self.sim.integrator.getStepSize()
            return step_size.value_in_unit(unit.picosecond)
        except Exception:
            return 1.0

    def _pos(self):
        """Returns positions as raw numpy array in nanometers"""
        p = self.sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        return p.value_in_unit(unit.nanometer)

    def _proj(self, pos):
        """pos is raw numpy array in nm. Returns projection (CV) as numpy array."""
        pos_ang = pos * self.NM_TO_ANG
        return np.asarray(self.proj_fn(pos_ang, **self.proj_args))

    def _escape_metric(self, cv: np.ndarray, start_proj: np.ndarray) -> float:
        cv = np.asarray(cv, dtype=float).reshape(-1)
        start_proj = np.asarray(start_proj, dtype=float).reshape(-1)

        if cv.shape != start_proj.shape:
            raise ValueError("projection shape changed during the run")

        direction = self.escape_direction
        if direction == "auto":
            direction = "increase" if cv.size == 1 else "radial"

        if self.mode == "hybrid":
            direction = "increase" if direction == "radial" else direction
            metric_index = self.hybrid_switch_index
        elif direction == "radial":
            return float(np.linalg.norm(cv - start_proj))
        else:
            metric_index = self.escape_cv_index

        if metric_index >= cv.size:
            raise ValueError("escape CV index is out of bounds for the projection")

        delta = float(cv[metric_index] - start_proj[metric_index])
        return delta if direction == "increase" else -delta
