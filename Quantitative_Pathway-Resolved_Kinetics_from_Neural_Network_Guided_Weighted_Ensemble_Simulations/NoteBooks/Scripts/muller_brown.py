import numpy as np
import openmm as mm
from openmm import app, unit


class MullerBrownPotential:
    """
    Müller-Brown potential energy surface implementation for OpenMM.
    V(x, y) = scale * sum_{i=1}^4 A_i * exp[a_i*(x-x0_i)^2 + b_i*(x-x0_i)*(y-y0_i) + c_i*(y-y0_i)^2]

    The raw potential has barriers of ~36-75 kJ/mol (~15-30 kBT at 300K).
    Use energy_scale < 1 to bring the barriers into a thermally accessible range
    while maintaining the topology of the surface.
    """

    # Standard Müller-Brown parameters (dimensionless)
    A_RAW  = [-200, -100, -170, 15]
    a_RAW  = [-1,   -1,   -6.5, 0.7]
    b_RAW  = [ 0,    0,   11,   0.6]
    c_RAW  = [-10,  -10,  -6.5,  0.7]
    x0_RAW = [ 1,    0,   -0.5, -1]
    y0_RAW = [ 0,    0.5,  1.5,  1]

    # Key stationary points (in nm)
    BASIN_A  = np.array([-0.558, 1.442, 0.0])   # V ≈ -146.7
    BASIN_B  = np.array([ 0.623, 0.028, 0.0])   # V ≈ -108.2
    BASIN_C  = np.array([-0.050, 0.467, 0.0])   # V ≈  -80.8
    SADDLE_1 = np.array([-0.822, 0.624, 0.0])   # V ≈  -72.2  (between A & B)
    SADDLE_2 = np.array([ 0.212, 0.293, 0.0])   # V ≈  -72.2  (between B & C)

    def __init__(
        self,
        energy_scale: float = 0.1,
        temperature: float = 300.0,
        friction: float = 1.0,
        step_size: float = 0.001,
        mass: float = 10.0,
    ):
        """
        Parameters
        ----------
        energy_scale : float
            Multiplicative factor applied to the raw potential.
            0.1 gives barriers of ~3-8 kBT at 300K (good for PathGennie demo).
        temperature : float
            Langevin thermostat temperature in Kelvin.
        friction : float
            Langevin friction coefficient in 1/ps.
        step_size : float
            Integration timestep in ps.
        mass : float
            Particle mass in amu.
        """
        self.energy_scale = energy_scale
        self.A = [a * energy_scale for a in self.A_RAW]

        self.temperature = temperature * unit.kelvin
        self.friction = friction / unit.picosecond
        self.step_size = step_size * unit.picosecond
        self.mass = mass * unit.amu

    def create_system(self):
        system = mm.System()
        system.addParticle(self.mass)

        # Build the Müller-Brown force expression
        terms = []
        for i in range(4):
            terms.append(
                f"A{i} * exp(a{i}*(x-x0_{i})*(x-x0_{i}) "
                f"+ b{i}*(x-x0_{i})*(y-y0_{i}) "
                f"+ c{i}*(y-y0_{i})*(y-y0_{i}))"
            )
        force_str = " + ".join(terms)
        force = mm.CustomExternalForce(force_str)
        force.addParticle(0, [])

        for i in range(4):
            force.addGlobalParameter(f"A{i}",   self.A[i])             # kJ/mol
            force.addGlobalParameter(f"a{i}",   self.a_RAW[i])         # 1/nm^2
            force.addGlobalParameter(f"b{i}",   self.b_RAW[i])         # 1/nm^2
            force.addGlobalParameter(f"c{i}",   self.c_RAW[i])         # 1/nm^2
            force.addGlobalParameter(f"x0_{i}", self.x0_RAW[i])        # nm
            force.addGlobalParameter(f"y0_{i}", self.y0_RAW[i])        # nm

        system.addForce(force)

        # Confine z to zero with a stiff harmonic wall
        z_wall = mm.CustomExternalForce("50000.0 * z * z")
        z_wall.addParticle(0, [])
        system.addForce(z_wall)

        return system

    def create_simulation(self):
        system = self.create_system()
        integrator = mm.LangevinMiddleIntegrator(
            self.temperature, self.friction, self.step_size
        )
        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("MB", chain)
        topology.addAtom("P", app.Element.getBySymbol("H"), residue)

        simulation = app.Simulation(topology, system, integrator)
        return simulation

    def get_energy(self, x, y):
        """Analytical energy (uses the SCALED A values)."""
        V = np.zeros_like(x, dtype=float)
        for i in range(4):
            V += self.A[i] * np.exp(
                self.a_RAW[i] * (x - self.x0_RAW[i]) ** 2
                + self.b_RAW[i] * (x - self.x0_RAW[i]) * (y - self.y0_RAW[i])
                + self.c_RAW[i] * (y - self.y0_RAW[i]) ** 2
            )
        return V
