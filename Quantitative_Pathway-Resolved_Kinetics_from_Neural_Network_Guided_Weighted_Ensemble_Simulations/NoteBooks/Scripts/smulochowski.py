import numpy as np
import openmm as mm
from openmm import app, unit


class SmoluchowskiPotential:
    """
    Smoluchowski-inspired 2D potential in OpenMM.

    V(x,y) = alpha * [exp(L1) + exp(L2) + 0.5 exp(L3)]

    Designed as a controllable rare-event landscape with:
    - Gaussian-like basin (L1)
    - Quartic entropic barrier (L2)
    - Harmonic coupling basin (L3)
    """

    def __init__(
        self,
        c1=50.5,
        c2=49.5,
        c3=1e5,
        c4=51.0,
        c5=49.0,
        energy_scale=1.0,     # analogous to MB energy_scale
        temperature=300.0,
        friction=1.0,
        step_size=0.001,
        mass=10.0,
    ):
        # Shape parameters
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5

        # Scaling (acts like barrier height control)
        self.energy_scale = energy_scale

        # Thermodynamics
        self.temperature = temperature * unit.kelvin
        self.friction = friction / unit.picosecond
        self.step_size = step_size * unit.picosecond
        self.mass = mass * unit.amu

    # =========================
    # SYSTEM
    # =========================
    def create_system(self):
        system = mm.System()
        system.addParticle(self.mass)

        expr = """
        scale * (exp(L1) + exp(L2) + 0.5*exp(L3));

        L1 = -c1*(x-0.25)^2 - c1*(y-0.75)^2 - 2*c2*(x-0.25)*(y-0.75);
        L2 = -c3*(x*x*(1-x)*(1-x))*(y*y*(1-y)*(1-y));
        L3 = -c4*x*x - c4*y*y + 2*c5*x*y;
        """

        force = mm.CustomExternalForce(expr)
        force.addParticle(0, [])

        # Parameters
        force.addGlobalParameter("c1", self.c1)
        force.addGlobalParameter("c2", self.c2)
        force.addGlobalParameter("c3", self.c3)
        force.addGlobalParameter("c4", self.c4)
        force.addGlobalParameter("c5", self.c5)
        force.addGlobalParameter("scale", self.energy_scale)

        system.addForce(force)

        # Constrain z → 0
        z_wall = mm.CustomExternalForce("50000.0 * z * z")
        z_wall.addParticle(0, [])
        system.addForce(z_wall)

        return system

    # =========================
    # SIMULATION
    # =========================
    def create_simulation(self):
        system = self.create_system()

        integrator = mm.LangevinMiddleIntegrator(
            self.temperature,
            self.friction,
            self.step_size
        )

        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("SMOL", chain)
        topology.addAtom("P", app.Element.getBySymbol("H"), residue)

        simulation = app.Simulation(topology, system, integrator)
        return simulation

    # =========================
    # ANALYTICAL ENERGY
    # =========================
    def get_energy(self, x, y):
        L1 = -self.c1*(x-0.25)**2 - self.c1*(y-0.75)**2 - 2*self.c2*(x-0.25)*(y-0.75)
        L2 = -self.c3*(x*x*(1-x)*(1-x))*(y*y*(1-y)*(1-y))
        L3 = -self.c4*x*x - self.c4*y*y + 2*self.c5*x*y

        return self.energy_scale * (
            np.exp(L1) + np.exp(L2) + 0.5*np.exp(L3)
        )
