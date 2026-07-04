from abc import ABC, abstractmethod

import openmm as mm
import openmm.app as app
import openmm.unit as unit


class External2DPotential(ABC):
    """
    Abstract base class for 2D external potentials in OpenMM.
    """

    @abstractmethod
    def create_force(self) -> mm.Force:
        """Return an OpenMM Force object."""
        pass


class SingleParticleSimulation:
    """
    Generic single-particle OpenMM simulation in 2D with a pluggable potential.
    """

    def __init__(
        self,
        potential: External2DPotential,
        temperature=300.0,
        timestep=1.0,
        friction=10.0,
        mass=12.0,
        seed=0,
        device=0,
    ):
        self.temperature = temperature * unit.kelvin
        self.timestep = timestep * unit.femtosecond
        self.friction = friction / unit.picosecond
        self.mass = mass * unit.dalton

        # System
        self.system = mm.System()
        self.system.addParticle(self.mass)

        # Add user-defined potential
        self.system.addForce(potential.create_force())

        # Constrain Z to zero (2D dynamics)
        z_force = mm.CustomExternalForce("kz * z^2")
        z_force.addGlobalParameter("kz", 5000.0)
        z_force.addParticle(0, [])
        self.system.addForce(z_force)

        # Topology
        self.topology = app.Topology()
        chain = self.topology.addChain()
        res = self.topology.addResidue("X", chain)
        self.topology.addAtom("X", app.Element.getBySymbol("C"), res)

        # Integrator
        self.integrator = mm.LangevinMiddleIntegrator(self.temperature, self.friction, self.timestep)
        self.integrator.setRandomNumberSeed(seed)

        # Platform
        self.platform, self.properties = self._select_platform(device)

        # Simulation
        self.simulation = app.Simulation(
            self.topology,
            self.system,
            self.integrator,
            self.platform,
            self.properties,
        )

    def _select_platform(self, device):
        try:
            platform = mm.Platform.getPlatformByName("CUDA")
            props = {"Precision": "mixed", "DeviceIndex": str(device)}
        except Exception:
            platform = mm.Platform.getPlatformByName("CPU")
            props = {}
        return platform, props
