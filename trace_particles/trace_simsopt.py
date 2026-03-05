import numpy as np
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.field.tracing import (
    trace_particles_boozer,
    MaxToroidalFluxStoppingCriterion,
    MinToroidalFluxStoppingCriterion,
)
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec, vmec_compute_geometry
from mpi4py import MPI
from simsopt.mhd import Boozer
from simsopt._core import Optimizable
from simsopt.util import proc0_print


#from src.sample.radial_density import RadialDensity
#from src.utils.constants import *  # TODO: let's redo this, this is bad practice (import all)
#from src.utils.divide_work import *  # TODO: let's redo this, this is bad practice (import all)

PROTON_MASS = 1.67262192369e-27  # kg
NEUTRON_MASS = 1.67492749804e-27  # kg
ONE_EV = 1.602176634e-19  # J
ALPHA_PARTICLE_MASS = 2 * PROTON_MASS + 2 * NEUTRON_MASS
FUSION_ALPHA_PARTICLE_ENERGY = 3.52e6 * ONE_EV  # Ekin
FUSION_ALPHA_SPEED_SQUARED = 2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS
V_MAX = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)
ELEMENTARY_CHARGE = 1.602176634e-19  # C
ALPHA_PARTICLE_CHARGE = 2 * ELEMENTARY_CHARGE

def rescale_device(vmec, target_minor_radius, target_volavgB):
    """Rescale a device to reactor scale.

    Args:
        vmec (Vmec): equilibrium to rescale
        target_minor_radius (float): target minor radius for rescaling the device
        target_volavgB (float): target volume-averaged |B| for setting phiedge

    Returns:
        Vmec: Vmec object with the rescaled configuration
    """
    vmec.boundary.unfix_all()

    # Rescale surface and set toroidal flux
    minor_radius = vmec.boundary.minor_radius()
    factor: float = target_minor_radius / minor_radius
    vmec.boundary.x *= factor

    mpol = vmec.indata.mpol
    ntor = vmec.indata.ntor
    ftol_arrray = vmec.indata.ftol_array
    ns_array = vmec.indata.ns_array
    niter_array = vmec.indata.niter_array
    vmec.indata.mpol = 6
    vmec.indata.ntor = 6
    vmec.indata.ns_array[:3] = [16, 50, 0]
    vmec.indata.niter_array[:3] = [2000, 7000, 0]
    vmec.indata.ftol_array[:3] =[1e-16, 1e-12, 1e-10]
    vmec.run()

    vmec.indata.phiedge = vmec.indata.phiedge * target_volavgB / vmec.wout.volavgB

    vmec.indata.mpol = mpol
    vmec.indata.ntor = ntor
    vmec.indata.ns_array = ns_array
    vmec.indata.niter_array = niter_array
    vmec.indata.ftol_array = ftol_arrray
    vmec.need_to_run_code = True
    return vmec



class TraceBoozer(Optimizable):
    """
    A class to make tracing from a vmec configuration a bit
    more convenient.
    """

    def __init__(
        self,
        vmec,
        tracing_tol: float = 1e-8,
        interpolant_degree: int = 3,
        interpolant_level: int = 8,
        bri_mpol: int = 32,
        bri_ntor: int = 32,
        smin: float = 0.01,
        smax: float = 0.99,
    ) -> None:
        """
        Initialize the TraceBoozer class.

        The configuration is scaled for the minor radius to equal target_minor_radius
        and for vmec.wout.volavgB to equal target_volavgB, approximately.

        Parameters:
        ----------
        vmec (Vmec): a Simsopt Vmec object.
        target_minor_radius : float, default 1.7
            Target minor radius for rescaling the device.
        target_volavgB : float, default 5.0
            Target volume-averaged |B| for setting phiedge.
        tracing_tol : float, default 1e-8
            Tolerance for determining tracing accuracy.
        interpolant_degree : int, default 3
            Degree of polynomial interpolants for field interpolation.
            1: fast but inaccurate, 3: slower but more accurate.
        interpolant_level : int, default 8
            Number of points per direction for Boozer radial interpolant.
            5: fast/inaccurate, 8: medium, 12: slow/accurate.
        bri_mpol, bri_ntor : int, default 32
            Number of poloidal and toroidal modes used in BoozXform.
            Lower values (e.g., 16) are faster.
        smin, smax : float, default 0.02, 1.0
            Minimum and maximum values of the normalized toroidal flux.
        """
        # Initialize VMEC
        # self.mpi: MpiPartition = MpiPartition(1)
        # vmec = Vmec(vmec_input, mpi=self.mpi, keep_all_files=False, verbose=False)

        self.vmec = vmec
        self.mpi = vmec.mpi
        self.surf = vmec.boundary

        # Set other attributes
        self.tracing_tol: float = tracing_tol
        self.interpolant_degree: int = interpolant_degree
        self.interpolant_level: int = interpolant_level
        self.bri_mpol: int = bri_mpol
        self.bri_ntor: int = bri_ntor

        # Initialize placeholders
        self.field = None
        self.bri = None

        # max s
        self.smax = smax
        self.smin = smin

        super().__init__(depends_on=[vmec])


    def sync_seeds(self, sd=None):
        """
        Sync the np.random.seed of the various worker groups.
        The seed is a random number <1e6.
        """
        # only sync across mpi group
        seed = np.zeros(1)
        if self.mpi.proc0_groups:
            if sd is not None:
                seed = sd * np.ones(1)
            else:
                seed = np.random.randint(int(1e6)) * np.ones(1)
        self.mpi.comm_groups.Bcast(seed, root=0)
        np.random.seed(int(seed[0]))
        return int(seed[0])

    def flux_grid(self, ns, ntheta, nzeta, nvpar, s_min=0.01, s_max=1.0, vpar_lb=-V_MAX, vpar_ub=V_MAX):
        """
        Build a 4d grid over the flux coordinates and vpar.
        """
        # use fixed particle locations
        surfaces = np.linspace(s_min, s_max, ns)
        thetas = np.linspace(0, 2 * np.pi, ntheta)
        zetas = np.linspace(0, 2 * np.pi / self.vmec.boundary.nfp, nzeta)
        # vpars = symlog_grid(vpar_lb,vpar_ub,nvpar)
        vpars = np.linspace(vpar_lb, vpar_ub, nvpar)
        # build a mesh
        [surfaces, thetas, zetas, vpars] = np.meshgrid(surfaces, thetas, zetas, vpars)
        stz_inits = np.zeros((ns * ntheta * nzeta * nvpar, 3))
        stz_inits[:, 0] = surfaces.flatten()
        stz_inits[:, 1] = thetas.flatten()
        stz_inits[:, 2] = zetas.flatten()
        vpar_inits = vpars.flatten()
        return stz_inits, vpar_inits

    def surface_grid(self, s_label, ntheta, nzeta, nvpar, vpar_lb=-V_MAX, vpar_ub=V_MAX):
        """
        Builds a grid on a single surface.
        """
        # use fixed particle locations
        # theta is [0,pi] with stellsym
        thetas = np.linspace(0, 2 * np.pi, ntheta)
        zetas = np.linspace(0, 2 * np.pi / self.vmec.boundary.nfp, nzeta)
        # vpars = symlog_grid(vpar_lb,vpar_ub,nvpar)
        vpars = np.linspace(vpar_lb, vpar_ub, nvpar)
        # build a mesh
        [thetas, zetas, vpars] = np.meshgrid(thetas, zetas, vpars)
        stz_inits = np.zeros((ntheta * nzeta * nvpar, 3))
        stz_inits[:, 0] = s_label
        stz_inits[:, 1] = thetas.flatten()
        stz_inits[:, 2] = zetas.flatten()
        vpar_inits = vpars.flatten()
        return stz_inits, vpar_inits

    def poloidal_grid(self, zeta_label, ns, ntheta, nvpar, s_max=0.98):
        """
        Builds a grid on a poloidal cross section
        """
        # bounds
        vpar_lb = np.sqrt(FUSION_ALPHA_SPEED_SQUARED) * (-1)
        vpar_ub = np.sqrt(FUSION_ALPHA_SPEED_SQUARED) * (1)
        # use fixed particle locations
        surfaces = np.linspace(0.01, s_max, ns)
        thetas = np.linspace(0, 2 * np.pi, ntheta)
        vpars = np.linspace(vpar_lb, vpar_ub, nvpar)
        # build a mesh
        [surfaces, thetas, vpars] = np.meshgrid(surfaces, thetas, vpars)
        stz_inits = np.zeros((ns * ntheta * nvpar, 3))
        stz_inits[:, 0] = surfaces.flatten()
        stz_inits[:, 1] = thetas.flatten()
        stz_inits[:, 2] = zeta_label
        vpar_inits = vpars.flatten()
        return stz_inits, vpar_inits

    def sample_surface(self, n_particles, s_label):
        """
        Sample the volume using the radial density sampler
        """
        ## divide the particles
        # comm = MPI.COMM_WORLD
        # size = comm.Get_size()
        # rank = comm.Get_rank()
        # SAA sampling over (theta,zeta,vpar) for a fixed surface
        s_inits = s_label * np.ones(n_particles)
        theta_inits = np.zeros(n_particles)
        zeta_inits = np.zeros(n_particles)
        vpar_inits = np.zeros(n_particles)
        # if rank == 0:
        if self.mpi.proc0_groups:
            # randomly sample theta,zeta,vpar
            # theta is [0,pi] with stellsym
            theta_inits = np.random.uniform(0, 2 * np.pi, n_particles)
            zeta_inits = np.random.uniform(0, 2 * np.pi / self.vmec.boundary.nfp, n_particles)
            vpar_inits = np.random.uniform(-V_MAX, V_MAX, n_particles)
        # broadcast the points
        self.mpi.comm_groups.Bcast(theta_inits, root=0)
        self.mpi.comm_groups.Bcast(zeta_inits, root=0)
        self.mpi.comm_groups.Bcast(vpar_inits, root=0)
        # stack the samples
        stp_inits = np.vstack((s_inits, theta_inits, zeta_inits)).T
        return stp_inits, vpar_inits

    def compute_boozer_field(self):
        # to save on recomputes
        if (self.field is not None):
            return self.field, self.bri

        try:
            self.vmec.run()
        except:
            # VMEC failure!
            return None, None

        # Construct radial interpolant of magnetic field
        try:
            bri = BoozerRadialInterpolant(
                equil=self.vmec, order=self.interpolant_degree, mpol=self.bri_mpol, ntor=self.bri_ntor, enforce_vacuum=True
            )
        except:
            # bri failure!
            return None, None

        # Construct 3D interpolation
        nfp = self.vmec.wout.nfp  # This is raising a false negative
        srange = (0, 1, self.interpolant_level)
        thetarange = (0, np.pi, self.interpolant_level)
        zetarange = (0, 2 * np.pi / nfp, self.interpolant_level)
        field = InterpolatedBoozerField(
            bri,
            degree=self.interpolant_degree,
            srange=srange,
            thetarange=thetarange,
            zetarange=zetarange,
            extrapolate=True,
            nfp=nfp,
            stellsym=True,
        )
        self.field = field
        self.bri = bri
        return field, bri

    def compute_modB(self, field, bri, ns=32, ntheta=32, nphi=32):
        """
        Compute |B| on a grid in Boozer coordinates.
        """
        stz_grid, _ = self.flux_grid(ns, ntheta, nphi, 1)
        field.set_points(stz_grid)
        modB = field.modB().flatten()
        return modB

    def compute_modB_vmec(self, s=1.0):
        """
        Compute |B| on a tensor product grid VMEC coordinates, (s, theta, phi).

        This function provides a faster way of computing |B| than the compute_modB function.
        |B| is directly computed from the VMEC output in VMEC coordinates. Since this function
        does not rely on BoozXForm, it is much faster than the compute_modB function.

        Args:
        -----
        s: float, optional

        Returns
        -------
        np.ndarray
            1d array of evals, length ns*ntheta*nphi
        bool
            success flag, True if VMEC run succeeded, False otherwise
        """
        success = True
        # try to run vmec
        try:
            self.vmec.run()
        except:
            # VMEC failure!
            success = False

        if not success:
            return np.array([]), success

        # run vmec and compute the geometric quantites (VMEC may fail)
        theta1d = self.vmec.boundary.quadpoints_theta
        phi1d = self.vmec.boundary.quadpoints_phi
        data = vmec_compute_geometry(self.vmec, s, theta1d, phi1d)  # 3d array

        # return a 1d array
        modB = data.modB.flatten()
        return modB, success
    
    def mirror_ratio(self, s=1.0):
        """Compute the mirror ratio """
        modB, success = self.compute_modB_vmec(s)
        if not success:
            return np.nan
        return np.max(modB) / np.min(modB)

    def compute_mu(self, field, bri, stz_inits, vpar_inits):
        """
        Compute |B| on a grid in Boozer coordinates.
        """
        field.set_points(stz_inits + np.zeros(np.shape(stz_inits)))
        modB = field.modB().flatten()
        vperp_squared = FUSION_ALPHA_SPEED_SQUARED - vpar_inits**2
        mu = vperp_squared / 2 / modB
        return mu

    def compute_mu_crit(self, field, bri, ns=64, ntheta=64, nphi=64):
        """
        Compute |B| on a grid in Boozer coordinates.
        """
        modB = self.compute_modB(field, bri, ns, ntheta, nphi)
        Bmax = np.max(modB)
        mu_crit = FUSION_ALPHA_SPEED_SQUARED / 2 / Bmax
        return mu_crit

    # set up the objective
    def compute_confinement_times(self, stz_inits, vpar_inits, tmax, field=None, bri=None):
        """
        Trace particles in boozer coordinates according to the vacuum GC
        approximation using simsopt.

        x: a point describing the current vmec boundary
        stz_inits: (n,3) array of (s,theta,zeta) points
        vpar_inits: (n,) array of vpar values
        tmax: max tracing time
        """
        n_particles = len(vpar_inits)

        if field is None:
            field, bri = self.compute_boozer_field()
        if field is None:
            # VMEC failure
            return -np.inf * np.ones(len(stz_inits))

        stopping_criteria = [MaxToroidalFluxStoppingCriterion(0.99), MinToroidalFluxStoppingCriterion(self.smin)]

        # comm = MPI.COMM_WORLD

        # trace
        try:
            res_tys, res_zeta_hits = trace_particles_boozer(
                field,
                stz_inits,
                vpar_inits,
                tmax=tmax,
                mass=ALPHA_PARTICLE_MASS,
                charge=ALPHA_PARTICLE_CHARGE,
                Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
                tol=self.tracing_tol,
                mode="gc",
                comm=self.mpi.comm_groups,
                stopping_criteria=stopping_criteria,
                forget_exact_path=True,
            )
        except:
            # tracing failure
            return -np.inf * np.ones(len(stz_inits))

        exit_times = np.zeros(n_particles)
        for ii, res in enumerate(res_zeta_hits):

            # check if particle hit stopping criteria
            if len(res) > 0:
                if int(res[0, 1]) == -1:
                    # particle hit MaxToroidalFluxCriterion
                    exit_times[ii] = res[0, 0]
                if int(res[0, 1]) == -2:
                    # particle hit MinToroidalFluxCriterion
                    exit_times[ii] = tmax
            else:
                # didnt hit any stopping criteria
                exit_times[ii] = tmax

        return exit_times
    
    def energy_loss(self, stz_inits, vpar_inits, tmax, field=None, bri=None):
        """ Compute the energy loss objective"""
        c_times = self.compute_confinement_times(stz_inits, vpar_inits, tmax, field, bri)
        energy = np.mean(3.5 * np.exp(-2 * c_times / tmax))
        if np.any(np.isinf(c_times)):
            energy = 3.5
        return energy

    # set up the objective
    def compute_trajectories(self, stz_inits, vpar_inits, tmax, field=None, bri=None):
        """
        Trace particles in boozer coordinates according to the vacuum GC
        approximation using simsopt.

        x: a point describing the current vmec boundary
        stz_inits: (n,3) array of (s,theta,zeta) points
        vpar_inits: (n,) array of vpar values
        tmax: max tracing time
        """
        n_particles = len(vpar_inits)

        if field is None:
            field, bri = self.compute_boozer_field()
        if field is None:
            # VMEC failure
            return -np.inf * np.ones(len(stz_inits))

        stopping_criteria = [MaxToroidalFluxStoppingCriterion(self.smax), MinToroidalFluxStoppingCriterion(self.smin)]

        # trace
        # TODO: set up try/except for tracing failure
        res_tys, res_zeta_hits = trace_particles_boozer(
            field,
            stz_inits,
            vpar_inits,
            tmax=tmax,
            mass=ALPHA_PARTICLE_MASS,
            charge=ALPHA_PARTICLE_CHARGE,
            Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
            tol=self.tracing_tol,
            mode="gc",
            comm=self.vmec.mpi.comm_groups,
            stopping_criteria=stopping_criteria,
            forget_exact_path=False,
        )

        exit_times = np.zeros(n_particles)
        for ii, res in enumerate(res_zeta_hits):

            # check if particle hit stopping criteria
            if len(res) > 0:
                if int(res[0, 1]) == -1:
                    # particle hit MaxToroidalFluxCriterion
                    exit_times[ii] = res[0, 0]
                if int(res[0, 1]) == -2:
                    # particle hit MinToroidalFluxCriterion
                    exit_times[ii] = tmax
            else:
                # didnt hit any stopping criteria
                exit_times[ii] = tmax

        return res_tys, exit_times

    def compute_average_drift_velocities(self, res_tys, tmax, stz_inits):
        """For each particle and each t, compute the average radial drift velocity
            V(t) = max(0, (s(t) - s(0))/ tau(t) )
        where tau(t) is the confinement time up to time t.

        Args:
            res_tys (list): list of [t,s,theta,zeta] arrays for each particle.
            tmax (float): tmax
            stz_inits (array): array of initial (stz) values
        Returns:
            drift_velocities (list): list of (n,) arrays of average drift velocities for each particle at each time
        """
        n_particles = len(res_tys)
        drift_velocities = []
        for ii in range(n_particles):
            s_traj = np.minimum(res_tys[ii][:, 1], self.smax) # take min just in case bad end behavior
            t_traj = np.minimum(res_tys[ii][:, 0], tmax) # take min just in case bad end behavior
            delta_s = s_traj - stz_inits[ii, 0]
            v_traj = np.maximum(0, delta_s / t_traj)
            drift_velocities.append(v_traj)

        return drift_velocities
    
    def confinement_time_estimators(self, drift_velocities, tmax, stz_inits):
        """Compute the confinement time estimators based on the average drift velocities,
            T(tmax, t) = min(t, (smax - s0) / V(t))

        Args:
            drift_velocities (list): list of drift velocities, one (n,) array per particle
            tmax (float): tmax
            stz_inits (array): array of initial (stz) values
        """
        n_particles = len(drift_velocities)
        confinement_times = []
        for ii in range(n_particles):
            v_traj = drift_velocities[ii]
            s0 = stz_inits[ii, 0]
            t_estimates = np.minimum((self.smax - s0) / v_traj, tmax)
            confinement_times.append(t_estimates)
        return confinement_times

    def compute_modB_contours(self, nphi=64, ntheta=64, s=1.0):
        # get |B| in boozer coordinates
        booz = Boozer(self.vmec)
        booz.register([s]) # register surface
        booz.run()
        bx = booz.bx
        # discretize boozer angles
        theta1d = np.linspace(0, 2 * np.pi, nphi)
        phi1d = np.linspace(0, 2 * np.pi / bx.nfp, ntheta)
        phi, theta = np.meshgrid(phi1d, theta1d, indexing='ij')
        # index of flux surface
        js = 0
        # reconstruct |B| and sqrtg
        modB = np.zeros(np.shape(phi))
        for jmn in range(len(bx.xm_b)):
            m = bx.xm_b[jmn]
            n = bx.xn_b[jmn]
            angle = m * theta - n * phi
            modB += bx.bmnc_b[jmn, js] * np.cos(angle)
            if bx.asym:
                modB += bx.bmns_b[jmn, js] * np.sin(angle)
        return modB, phi, theta



if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    vmec_input = "./vmec_input_files/input.padidar_A"

    mpi = MpiPartition(1)
    vmec = Vmec(vmec_input, mpi=mpi, keep_all_files=False, verbose=False)
    
    # reduce resolution
    vmec.indata.mpol = 6
    vmec.indata.ntor = 6
    vmec.indata.ns_array[:3] = [16, 50, 0]
    vmec.indata.niter_array[:3] = [2000, 7000, 0]
    vmec.indata.ftol_array[:3] =[1e-16, 1e-12, 1e-10]

    vmec = rescale_device(vmec, target_minor_radius=1.7, target_volavgB=5.86)
    vmec.run()

    # Configure quasisymmetry objective:
    from simsopt.mhd import QuasisymmetryRatioResidual
    s1d = np.linspace(1e-3, 1, 10)
    qh_parallel_loss = QuasisymmetryRatioResidual(vmec,
                                    s1d,  # Radii to target
                                    helicity_m=1, helicity_n=1).total()  # (M, N) you want in |B|
    qh_perp_loss = QuasisymmetryRatioResidual(vmec,
                                    s1d,  # Radii to target
                                    helicity_m=1, helicity_n=-1).total()  # (M, N) you want in |B|
    qa_loss = QuasisymmetryRatioResidual(vmec,
                                    s1d,  # Radii to target
                                    helicity_m=1, helicity_n=0).total()  # (M, N) you want in |B|
    print("QA error", qa_loss)
    print("QH(1,1) error", qh_parallel_loss)
    print("QH(1,-1) error", qh_perp_loss)
    print('minor radius', vmec.boundary.minor_radius())
    print('volavgB', vmec.wout.volavgB)
    print('iotas', vmec.wout.iotas[1:])

    # constraint violation
    tmax = 1e-2
    n_particles = 20
    s_label = 0.25

    tracer = TraceBoozer(
        vmec,
        tracing_tol=1e-7,
        interpolant_degree=3,
        interpolant_level=16,
        bri_mpol=16,
        bri_ntor=16,
    )
    tracer.sync_seeds()

    modB, phi, theta = tracer.compute_modB_contours(nphi=64, ntheta=64, s=0.25)
    import matplotlib.pyplot as plt
    plt.contour(phi, theta, modB, cmap='viridis', levels=15)
    plt.xlabel('phi')
    plt.ylabel('theta')
    plt.title('|B| contours at s=0.25')
    plt.show()

    # tracing points
    stz_inits, vpar_inits = tracer.sample_surface(n_particles, s_label)
    c_times = tracer.compute_confinement_times(stz_inits, vpar_inits, tmax)
    proc0_print("loss frac", np.mean(c_times < tmax))
