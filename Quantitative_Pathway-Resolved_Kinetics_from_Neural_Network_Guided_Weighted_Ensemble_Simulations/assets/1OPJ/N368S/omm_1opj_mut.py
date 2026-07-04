import copy

from openmm import *
from openmm import LangevinMiddleIntegrator, Platform
from openmm.app import PME, GromacsGroFile, GromacsTopFile, HBonds, Simulation
from openmm.unit import amu, femtoseconds, kelvin, nanometers, picosecond


class OpenMMRunner:
    def __init__(self,device = 0):

        ffdir = '/home/suman/Dibyendu/Soft/GMX24_GPU/share/gromacs/top'
        platform_name = "CUDA"
        self.temperature = 300

        self.nonbondedMethod = PME
        self.nonbondedCutoff = 1.0 * nanometers
        self.ewaldErrorTolerance = 0.0005
        self.constraints = HBonds
        self.rigidWater = True
        self.constraintTolerance = 0.000001
        self.hydrogenMass = 1.5 * amu

        gro_file = GromacsGroFile("/scratch/suman/Dibyendu/New_WEPath/Data/1OPJ/N368S/md.gro")
        self.top = GromacsTopFile("/scratch/suman/Dibyendu/New_WEPath/Data/1OPJ/N368S/topol.top", periodicBoxVectors=gro_file.getPeriodicBoxVectors(), includeDir=ffdir)

        self.integrator = LangevinMiddleIntegrator(self.temperature * kelvin, 1.0 / picosecond, 2.0 * femtoseconds)
        self.integrator.setConstraintTolerance(self.constraintTolerance)

        self.system = self.top.createSystem(
            nonbondedMethod=self.nonbondedMethod,
            nonbondedCutoff=self.nonbondedCutoff,
            constraints=self.constraints,
            rigidWater=self.rigidWater,
            ewaldErrorTolerance=self.ewaldErrorTolerance,
            hydrogenMass=self.hydrogenMass
        )

        if platform_name == 'CUDA':
            self.platform = Platform.getPlatformByName('CUDA')
            self.prop = {'Precision': 'mixed', 'DeviceIndex': str(device)}
            self.simulation = Simulation(self.top.topology, self.system, self.integrator, self.platform, self.prop)
        else:
            self.platform = Platform.getPlatformByName('CPU')
            self.simulation = Simulation(self.top.topology, self.system, self.integrator, self.platform)

    def _create_simulation(self):
        """Creates the main OpenMM Simulation object and sets its initial state."""
        new_integrator = copy.copy(self.integrator)
        new_integrator.setRandomNumberSeed(0)
        self.simulation = Simulation(self.top.topology, self.system, new_integrator, self.platform, self.prop)
