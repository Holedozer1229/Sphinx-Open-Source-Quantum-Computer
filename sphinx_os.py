import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svdvals, pinv
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, coo_matrix
import sympy as sp
import time
import logging
from joblib import Parallel, delayed
import psutil
import os
import pickle
import gzip
import pandas as pd
from tqdm import tqdm

# Logging Setup
logging.basicConfig(
    filename='sphinx_os_improved.log',
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SphinxOS")

# Physical Constants
G = 6.67430e-11
c = 2.99792458e8
hbar = 1.0545718e-34
e = 1.60217662e-19
epsilon_0 = 8.854187817e-12
mu_0 = 1 / (epsilon_0 * c**2)
m_e = 9.1093837e-31
m_q = 2.3e-30
m_h = 2.23e-30
m_n = 1.67e-27
g_w = 0.653
g_s = 1.221
v_higgs = 246e9 * e / c**2
l_p = np.sqrt(hbar * G / c**3)
kappa = 1e-8
lambda_higgs = 0.5
observer_coupling = 1e-6
alpha = 1 / 137.0
yukawa_e = 2.9e-6
yukawa_q = 1.2e-5
RS = 2.0 * G * m_n / c**2
m_nugget = m_n
lambda_nugget = 0.1

# Pauli and Gell-Mann Matrices
sigma = [
    np.array([[0, 1], [1, 0]], dtype=np.complex64),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
    np.array([[1, 0], [0, -1]], dtype=np.complex64)
]
lambda_matrices = [
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex64),
    np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=np.complex64),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.complex64),
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.complex64),
    np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=np.complex64),
    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.complex64),
    np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=np.complex64),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / np.sqrt(3, dtype=np.complex64)
]

# SU(2) and SU(3) Structure Constants
f_su2 = np.zeros((3, 3, 3), dtype=np.float64)
for a, b, c in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]: f_su2[a, b, c] = 1
for a, b, c in [(2, 1, 0), (0, 2, 1), (1, 0, 2)]: f_su2[a, b, c] = -1
f_su3 = np.zeros((8, 8, 8), dtype=np.float64)
f_su3[0, 1, 2] = 1; f_su3[0, 2, 1] = -1
f_su3[0, 3, 4] = 0.5; f_su3[0, 4, 3] = -0.5
f_su3[0, 5, 6] = 0.5; f_su3[0, 6, 5] = -0.5
f_su3[1, 3, 5] = 0.5; f_su3[1, 5, 3] = -0.5
f_su3[1, 4, 6] = -0.5; f_su3[1, 6, 4] = 0.5
f_su3[2, 3, 6] = 0.5; f_su3[2, 6, 3] = -0.5
f_su3[2, 4, 5] = 0.5; f_su3[2, 5, 4] = -0.5
f_su3[3, 4, 7] = np.sqrt(3)/2; f_su3[3, 7, 4] = -np.sqrt(3)/2
f_su3[5, 6, 7] = np.sqrt(3)/2; f_su3[5, 7, 6] = -np.sqrt(3)/2

# Configuration Parameters
CONFIG = {
    "grid_size": (3, 3, 3, 3, 2, 2),
    "max_iterations": 10,
    "time_delay_steps": 3,
    "ctc_feedback_factor": (1 + np.sqrt(5)) / 2,  # Golden ratio
    "entanglement_factor": 0.2,
    "charge": e,
    "em_strength": 3.0,
    "dt": 1e-12,
    "dx": l_p * 1e5,
    "dv": l_p * 1e3,
    "du": l_p * 1e3,
    "j4_scaling_factor": 1e-20,
    "alpha_em": alpha,
    "alpha_phi": 1e-3,
    "m_nugget": m_n,
    "m_higgs": m_h,
    "m_electron": m_e,
    "m_quark": m_q,
    "vev_higgs": v_higgs,
    "yukawa_e": yukawa_e,
    "yukawa_q": yukawa_q,
    "g_strong": g_s * 1e-5,
    "g_weak": g_w * 1e-5,
    "omega": 3,
    "a_godel": 1.0,
    "entanglement_coupling": 1e-6,
    "rio_scale": 1e-3,
    "flux_coupling": 1e-3,
    "resonance_frequency": 1e6,
    "resonance_amplitude": 0.1,
    "field_clamp_max": 1e6,
    "rtol": 1e-5,  # Adjusted for stability
    "atol": 1e-8,  # Adjusted for stability
    "dt_min": 1e-15,
    "dt_max": 1e-9,
    "max_steps_per_dt": 1000,
    "log_tensors": True,
    "tau": 1.0,
    "info_coupling": 1e-3,
    "lambda_ctc": 0.5,
    "m_shift": 0.1
}

START_TIME = time.perf_counter_ns() / 1e9

# Helper Functions
def compute_entanglement_entropy(field, grid_size):
    entropy = np.zeros(grid_size[:4], dtype=np.float64)
    for idx in np.ndindex(grid_size[:4]):
        local_state = field[idx].flatten()
        local_state = np.nan_to_num(local_state, nan=0.0)
        norm = np.linalg.norm(local_state)
        if norm > 1e-15:
            local_state /= norm
        psi_matrix = local_state.reshape(2, 2)
        schmidt_coeffs = svdvals(psi_matrix)
        probs = schmidt_coeffs**2
        probs = probs[probs > 1e-15]
        entropy[idx] = -np.sum(probs * np.log(probs)) if probs.size > 0 else 0
    return np.mean(entropy)

def repeating_curve(index):
    return 1 if index % 2 == 0 else 0

def construct_6d_gamma_matrices(metric):
    gamma_flat = [
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.complex64),
        np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]], dtype=np.complex64),
        np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]], dtype=np.complex64),
        np.array([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.complex64),
        np.eye(4, dtype=np.complex64),
        np.eye(4, dtype=np.complex64)
    ]
    e_a_mu = np.diag([np.sqrt(abs(metric[i, i])) + 1e-15 for i in range(6)])
    e_mu_a = np.linalg.inv(e_a_mu + 1e-15 * np.eye(6))
    gamma = [sum(e_mu_a[mu, a] * gamma_flat[a] for a in range(6)) for mu in range(6)]
    return [np.nan_to_num(g, nan=0.0) for g in gamma]

def compute_schumann_frequencies(N=3):
    f0 = 7.83e3
    pythagorean_ratios = [1, 2, 8/3]
    return [f0 * ratio for ratio in pythagorean_ratios[:N]]

# Adaptive Grid Class
class AdaptiveGrid:
    def __init__(self, base_grid_size, max_refinement=2):
        self.base_grid_size = base_grid_size
        self.max_refinement = max_refinement
        self.refinement_levels = np.zeros(base_grid_size, dtype=np.int8)
        self.base_deltas = [CONFIG[f"d{dim}"] for dim in ['t', 'x', 'x', 'x', 'v', 'u']]
        self.deltas = np.array(self.base_deltas)
        self.coordinates = self._generate_coordinates()

    def _generate_coordinates(self):
        dims = [np.linspace(0, self.deltas[i] * self.base_grid_size[i], self.base_grid_size[i])
                for i in range(6)]
        return np.stack(np.meshgrid(*dims, indexing='ij'), axis=-1)

    def refine(self, ricci_scalar):
        threshold = np.percentile(np.abs(ricci_scalar), 90)
        for idx in np.ndindex(self.base_grid_size):
            if np.abs(ricci_scalar[idx]) > threshold and self.refinement_levels[idx] < self.max_refinement:
                self.refinement_levels[idx] += 1
        self.deltas = np.array([d / (2 ** np.max(self.refinement_levels)) for d in self.base_deltas])
        self.coordinates = self._generate_coordinates()
        logger.debug(f"Grid refined: max refinement level = {np.max(self.refinement_levels)}")

# Spin Network Class
class SpinNetwork:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.total_points = np.prod(grid_size)
        self.state = np.ones(self.total_points, dtype=np.complex128) / np.sqrt(self.total_points)
        self.indices = np.arange(self.total_points).reshape(grid_size)
        self.ctc_buffer = []
        self.ctc_steps = CONFIG["time_delay_steps"]
        self.ctc_factor = CONFIG["ctc_feedback_factor"]

    def evolve(self, dt, lambda_field, metric, inverse_metric, deltas, nugget_field, higgs_field, em_fields, electron_field, quark_field):
        H = self._build_sparse_hamiltonian(lambda_field, metric, inverse_metric, deltas, nugget_field, higgs_field, em_fields, electron_field, quark_field)
        state_flat = self.state.flatten()
        max_attempts = 3
        current_dt = dt
        total_steps = 0
        for attempt in range(max_attempts):
            try:
                if len(self.ctc_buffer) >= self.ctc_steps:
                    state_past = self.ctc_buffer[-self.ctc_steps].flatten()
                    state_current = state_flat.copy()
                    for _ in range(3):
                        sol = solve_ivp(lambda t, y: -1j * H.dot(y) / hbar, [0, current_dt], state_current,
                                        method='RK45', rtol=CONFIG["rtol"], atol=CONFIG["atol"], vectorized=True)
                        if not sol.success:
                            raise ValueError(f"solve_ivp failed: {sol.message}")
                        state_evolved = sol.y[:, -1]
                        state_current = (1 - self.ctc_factor) * state_evolved + self.ctc_factor * state_past
                        norm = np.linalg.norm(state_current)
                        if norm > 0:
                            state_current /= norm
                else:
                    sol = solve_ivp(lambda t, y: -1j * H.dot(y) / hbar, [0, current_dt], state_flat,
                                    method='RK45', rtol=CONFIG["rtol"], atol=CONFIG["atol"], vectorized=True)
                    if not sol.success:
                        raise ValueError(f"solve_ivp failed: {sol.message}")
                    state_current = sol.y[:, -1]
                    norm = np.linalg.norm(state_current)
                    if norm > 0:
                        state_current /= norm
                total_steps += len(sol.t) - 1
                break
            except Exception as e:
                logger.warning(f"SpinNetwork evolve failed with dt={current_dt}: {e}, attempt {attempt+1}/{max_attempts}")
                if attempt == max_attempts - 1:
                    raise
                current_dt *= 0.5
        self.state = state_current.reshape(self.grid_size)
        self.state = np.clip(self.state, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
        self.ctc_buffer.append(self.state.copy())
        if len(self.ctc_buffer) > self.ctc_steps:
            self.ctc_buffer.pop(0)
        return total_steps

    def _build_sparse_hamiltonian(self, lambda_field, metric, inverse_metric, deltas, nugget_field, higgs_field, em_fields, electron_field, quark_field):
        state_grid = self.state.reshape(self.grid_size)
        N = self.total_points
        kinetic_term = np.zeros_like(state_grid, dtype=np.complex128)
        for mu in range(6):
            grad_mu = np.gradient(state_grid, deltas[mu], axis=mu)
            laplacian_mu = np.gradient(grad_mu, deltas[mu], axis=mu)
            kinetic_term += inverse_metric[..., mu, mu] * laplacian_mu
        kinetic_energy = -hbar**2 / (2 * m_n) * kinetic_term.flatten()
        potential_energy = np.zeros(N, dtype=np.complex128)
        nugget_norm = np.abs(nugget_field.flatten())**2
        higgs_norm = np.abs(higgs_field.flatten())**2
        em_potential = np.abs(em_fields["A"][..., 0].flatten())
        for i in range(N):
            V_nugget = kappa * nugget_norm[i]
            V_higgs = lambda_higgs * higgs_norm[i]
            V_em = e * em_potential[i]
            potential_energy[i] = V_nugget + V_higgs + V_em
        ricci_scalar = np.einsum('...mn,...mn->...', inverse_metric, 
                                np.einsum('...rsmn,...mn->...rs', self._compute_riemann_tensor(metric, inverse_metric, deltas).toarray().reshape(*self.grid_size, 6, 6, 6, 6), inverse_metric))
        ricci_energy = ricci_scalar.flatten() * CONFIG["entanglement_coupling"]
        connection = self._compute_affine_connection(metric, inverse_metric, deltas)
        spin_gravity = np.zeros(N, dtype=np.complex128)
        for idx in np.ndindex(self.grid_size):
            i = self.indices[idx]
            psi_e = electron_field[idx]
            spin_e = np.einsum('i,ij,j->', psi_e.conj(), sigma[2], psi_e).real
            spin_q = sum(np.einsum('i,ij,j->', quark_field[idx, f, c].conj(), sigma[2], quark_field[idx, f, c]).real
                         for f in range(3) for c in range(3))
            spin_density = (m_e * spin_e + m_q * spin_q) / (hbar * c)
            for mu in range(6):
                spin_gravity[i] += spin_density * connection[idx + (mu, mu, mu)]
        lambda_perturbation = lambda_field.flatten() * 1e-3
        diagonal = kinetic_energy + potential_energy + hbar * c * spin_gravity * 1e-3 + ricci_energy + lambda_perturbation
        rows, cols, data = [], [], []
        for idx in np.ndindex(self.grid_size):
            i = self.indices[idx]
            for mu in range(6):
                idx_plus = list(idx)
                idx_minus = list(idx)
                if idx[mu] < self.grid_size[mu] - 1:
                    idx_plus[mu] += 1
                    j = self.indices[tuple(idx_plus)]
                    coupling = inverse_metric[idx][mu, mu] * hbar * c / deltas[mu]
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([coupling, coupling])
                if idx[mu] > 0:
                    idx_minus[mu] -= 1
                    j = self.indices[tuple(idx_minus)]
                    coupling = inverse_metric[idx][mu, mu] * hbar * c / deltas[mu]
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([coupling, coupling])
        H = csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.complex128)
        H += csr_matrix((diagonal, (np.arange(N), np.arange(N))), dtype=np.complex128)
        return H

    def _compute_affine_connection(self, metric, inverse_metric, deltas):
        connection = np.zeros((*self.grid_size, 6, 6, 6), dtype=np.float64)
        for idx in np.ndindex(self.grid_size):
            if all(0 < i < s - 1 for i, s in zip(idx, self.grid_size)):
                for rho in range(6):
                    for mu in range(6):
                        for nu in range(6):
                            dg_mu = np.gradient(metric[..., mu, nu], deltas[rho], axis=rho)[idx]
                            dg_nu = np.gradient(metric[..., rho, mu], deltas[nu], axis=nu)[idx]
                            dg_rho = np.gradient(metric[..., rho, nu], deltas[mu], axis=mu)[idx]
                            connection[idx + (rho, mu, nu)] = 0.5 * np.einsum('rs,s->r', inverse_metric[idx],
                                                                              dg_mu + dg_nu - dg_rho)
        return np.nan_to_num(connection, nan=0.0)

    def _compute_riemann_tensor(self, metric, inverse_metric, deltas):
        riemann_shape = (*self.grid_size, 6, 6, 6, 6)
        data, rows, cols = [], [], []
        connection = self._compute_affine_connection(metric, inverse_metric, deltas)
        for rho in range(6):
            for sigma in range(6):
                for mu in range(6):
                    for nu in range(6):
                        grad_nu_sigma = np.gradient(connection[..., rho, nu, sigma], deltas[nu], axis=nu)
                        grad_mu_sigma = np.gradient(connection[..., rho, mu, sigma], deltas[mu], axis=mu)
                        term1 = np.einsum('...km,...mn->...kn', connection[..., rho, :, mu],
                                          connection[..., :, nu, sigma])
                        term2 = np.einsum('...kn,...mn->...km', connection[..., rho, :, nu],
                                          connection[..., :, mu, sigma])
                        values = grad_nu_sigma - grad_mu_sigma + term1 - term2
                        for idx in np.ndindex(self.grid_size):
                            flat_idx = np.ravel_multi_index(idx + (rho, sigma, mu, nu), riemann_shape)
                            data.append(values[idx])
                            rows.append(flat_idx)
                            cols.append(flat_idx)
        riemann = coo_matrix((data, (rows, cols)), shape=(np.prod(riemann_shape), np.prod(riemann_shape)))
        max_val = np.max(np.abs(data)) if data else 1.0
        if max_val > 1e5:
            riemann = riemann / max_val
        return riemann

# Tetrahedral Lattice Class
class TetrahedralLattice:
    def __init__(self, adaptive_grid):
        self.grid_size = adaptive_grid.base_grid_size
        self.deltas = adaptive_grid.deltas
        self.coordinates = adaptive_grid.coordinates
        self.tetrahedra = []

    def _define_tetrahedra(self):
        self.tetrahedra = []
        for t_idx in range(self.grid_size[0]):
            for v_idx in range(self.grid_size[4]):
                for u_idx in range(self.grid_size[5]):
                    for x_idx in range(self.grid_size[1] - 1):
                        for y_idx in range(self.grid_size[2] - 1):
                            for z_idx in range(self.grid_size[3] - 1):
                                vertices = [
                                    (t_idx, x_idx + dx, y_idx + dy, z_idx + dz, v_idx, u_idx)
                                    for dx in [0, 1] for dy in [0, 1] for dz in [0, 1]
                                ]
                                tetrahedra = [
                                    (vertices[0], vertices[1], vertices[3], vertices[7]),
                                    (vertices[0], vertices[2], vertices[3], vertices[7]),
                                    (vertices[0], vertices[2], vertices[6], vertices[7]),
                                    (vertices[0], vertices[4], vertices[6], vertices[7]),
                                    (vertices[0], vertices[4], vertices[5], vertices[7])
                                ]
                                self.tetrahedra.extend(tetrahedra)
        logger.debug(f"Defined {len(self.tetrahedra)} tetrahedra")

    def compute_barycentric_coordinates(self, point, tetrahedron):
        v0, v1, v2, v3 = [self.coordinates[vert] for vert in tetrahedron]
        p = point[1:4]
        v0, v1, v2, v3 = v0[1:4], v1[1:4], v2[1:4], v3[1:4]
        T = np.array([v1 - v0, v2 - v0, v3 - v0]).T
        p_minus_v0 = p - v0
        try:
            b = np.linalg.solve(T, p_minus_v0)
            b0 = 1 - np.sum(b)
            bary_coords = np.array([b0, b[0], b[1], b[2]])
        except np.linalg.LinAlgError:
            bary_coords = np.array([0.25, 0.25, 0.25, 0.25])
        bary_coords = np.clip(bary_coords, 0, 1)
        bary_coords /= np.sum(bary_coords) + 1e-15
        return bary_coords

    def interpolate_field(self, field, point):
        for tetrahedron in self.tetrahedra:
            bary_coords = self.compute_barycentric_coordinates(point, tetrahedron)
            if np.all(bary_coords >= -1e-5) and np.all(bary_coords <= 1 + 1e-5):
                field_values = [field[vert] for vert in tetrahedron]
                return sum(bary_coords[i] * field_values[i] for i in range(4))
        idx = tuple(np.clip(np.round(point / self.deltas).astype(int), 0, np.array(self.grid_size) - 1))
        return field[idx]

# Sphinx Operating System Class
class SphinxOS:
    def __init__(self):
        self.grid_size = CONFIG["grid_size"]
        self.total_points = np.prod(self.grid_size)
        self.dt = CONFIG["dt"]
        self.adaptive_grid = AdaptiveGrid(self.grid_size, max_refinement=2)
        self.deltas = self.adaptive_grid.deltas
        self.time = 0.0
        self.spin_network = SpinNetwork(self.grid_size)
        self.lattice = TetrahedralLattice(self.adaptive_grid)
        self.lattice._define_tetrahedra()
        self.wormhole_nodes = self._generate_wormhole_nodes()
        self.qubit_states = np.ones((*self.grid_size, 2), dtype=np.complex128) / np.sqrt(2)  # |0> + |1>
        self.temporal_entanglement = np.zeros(self.grid_size, dtype=np.complex128)
        self.quantum_state = np.ones(self.grid_size, dtype=np.complex128) / np.sqrt(self.total_points)
        self.higgs_field = np.ones(self.grid_size, dtype=np.complex128) * v_higgs
        self.electron_field = np.zeros((*self.grid_size, 4), dtype=np.complex128)
        self.quark_field = np.zeros((*self.grid_size, 3, 3, 4), dtype=np.complex128)
        self.nugget_field = np.zeros(self.grid_size, dtype=np.complex128)
        self.observer_field = np.random.normal(0, 1e-6, self.grid_size).astype(np.complex64)
        self.singularity_field = np.zeros((*self.grid_size, 4), dtype=np.complex64)
        self.lambda_field = np.zeros(self.grid_size, dtype=np.float64)
        self.I_mu_nu = np.zeros((*self.grid_size, 6, 6), dtype=np.float64)
        self.phi_range = np.linspace(-10.0, 10.0, 201, dtype=np.float64)
        self.d_phi = self.phi_range[1] - self.phi_range[0]
        self.phi_wave_functions = np.zeros((*self.grid_size, len(self.phi_range)), dtype=np.complex128)
        for idx in np.ndindex(self.grid_size):
            psi = np.exp(-self.phi_range**2)
            psi /= np.sqrt(np.sum(np.abs(psi)**2 * self.d_phi))
            self.phi_wave_functions[idx] = psi
        self.theta12, self.theta13, self.theta23 = 0.227, 0.0037, 0.041
        self.delta = 1.2
        s12, c12 = np.sin(self.theta12), np.cos(self.theta12)
        s13, c13 = np.sin(self.theta13), np.cos(self.theta13)
        s23, c23 = np.sin(self.theta23), np.cos(self.theta23)
        self.CKM = np.array([
            [c12 * c13, s12 * c13, s13 * np.exp(-1j * self.delta)],
            [-s12 * c23 - c12 * s23 * s13 * np.exp(1j * self.delta),
             c12 * c23 - s12 * s23 * s13 * np.exp(1j * self.delta),
             s23 * c13],
            [s12 * s23 - c12 * c23 * s13 * np.exp(1j * self.delta),
             -c12 * s23 - s12 * c23 * s13 * np.exp(1j * self.delta),
             c23 * c13]
        ], dtype=np.complex128)
        self._initialize_a4_mass_matrix()
        self.setup_symbolic_calculations()
        self.metric, self.inverse_metric = self.compute_quantum_metric()
        self.connection = self._compute_affine_connection()
        self.riemann_tensor = self._compute_riemann_tensor()
        self.ricci_tensor, self.ricci_scalar = self._compute_curvature()
        self.stress_energy = self._compute_stress_energy()
        self.einstein_tensor = self._compute_einstein_tensor()
        self.em_fields = self._initialize_em_fields()
        self.weak_fields = self._initialize_weak_fields()
        self.strong_fields = self._initialize_strong_fields()
        self.history = []
        self.fermion_history = []
        self.nugget_history = []
        self.higgs_norm_history = []
        self.entanglement_history = []
        self.bits_entropy_history = []
        self.geom_entropy_history = []
        self.erasure_entropy_history = []
        self.temp_entropy_history = []
        self.ricci_scalar_history = []
        self.rio_pattern_history = []
        self.spin_density_history = []
        self.lambda_history = []
        self.j4_history = []
        self.flux_history = []
        self.I_mu_nu_history = []
        self.cp_asymmetry_history = []
        self.teleportation_fidelity_history = []
        self.bit_state_fraction_history = []
        self.geodesic_path = None
        self.dV = (CONFIG["dt"] * CONFIG["dx"]**3 * CONFIG["dv"] * CONFIG["du"])
        lp = l_p
        area_factor = (CONFIG["dx"]**3 * CONFIG["dv"] * CONFIG["du"])
        lp_scaling = (lp**4)
        base_scale = area_factor / lp_scaling
        target_bits = 32.0
        target_nats = target_bits * np.log(2)
        relative_entropy_approx = 150.0
        self.geom_entropy_scale = target_nats / (relative_entropy_approx * self.dV) / np.log(2)
        logger.debug(f"Adjusted geom_entropy_scale: {self.geom_entropy_scale:.2e}")

    def _initialize_a4_mass_matrix(self):
        m_u, m_d, m_b = [0.002, 0.005, 4.18]
        epsilon = 0.001
        self.quark_mass_matrix = np.array([
            [m_u, epsilon, epsilon],
            [epsilon, m_d, epsilon],
            [epsilon, epsilon, m_b]
        ], dtype=np.complex128)
        self.quark_mass_matrix = 0.5 * (self.quark_mass_matrix + self.quark_mass_matrix.conj().T)

    def setup_symbolic_calculations(self):
        self.t, self.x, self.y, self.z, self.v, self.u = sp.symbols('t x y z v u')
        self.a, self.c_sym, self.m, self.kappa_sym = sp.symbols('a c m kappa', positive=True)
        self.nugget_sym = sp.Function('nugget')(self.t, self.x, self.y, self.z, self.v, self.u)
        r = sp.sqrt(self.x**2 + self.y**2 + self.z**2 + 1e-10)
        schwarzschild = 1 - RS / r
        scaled_l_p = l_p * 1e30
        time_scale = 1e-16
        spatial_scale = 1.0
        compact_scale = 1e18
        g_diag = [
            sp.sympify(time_scale * (-self.c_sym**2 * (1 + self.kappa_sym * self.nugget_sym) * schwarzschild)),
            sp.sympify(spatial_scale * (self.a**2 * (1 + self.kappa_sym * self.nugget_sym))),
            sp.sympify(spatial_scale * (self.a**2 * (1 + self.kappa_sym * self.nugget_sym))),
            sp.sympify(spatial_scale * (self.a**2 * (1 + self.kappa_sym * self.nugget_sym))),
            sp.sympify(compact_scale * (scaled_l_p**2)),
            sp.sympify(compact_scale * (scaled_l_p**2))
        ]
        self.g = sp.diag(*g_diag)
        self.g_inv = self.g.inv()
        self.metric_scale_factors = np.array([time_scale, spatial_scale, spatial_scale, spatial_scale, compact_scale, compact_scale])

    def _generate_wormhole_nodes(self):
        coords = np.zeros((*self.grid_size, 6), dtype=np.float64)
        dims = [np.linspace(0, CONFIG[f"d{dim}"] * size, size, dtype=np.float64)
                for dim, size in zip(['t', 'x', 'x', 'x', 'v', 'u'], self.grid_size)]
        T, X, Y, Z, V, U = np.meshgrid(*dims, indexing='ij')
        R, r = 1.5 * self.deltas[1], 0.5 * self.deltas[1]
        coords[..., 0] = (R + r * np.cos(CONFIG["omega"] * T)) * np.cos(X / self.deltas[1])
        coords[..., 1] = (R + r * np.cos(CONFIG["omega"] * T)) * np.sin(Y / self.deltas[1])
        coords[..., 2] = r * np.sin(CONFIG["omega"] * Z)
        coords[..., 3] = r * np.cos(CONFIG["omega"] * V)
        coords[..., 4] = r * np.sin(CONFIG["omega"] * U)
        coords[..., 5] = c * T
        return np.nan_to_num(coords, nan=0.0)

    def compute_quantum_metric(self):
        metric = np.zeros((*self.grid_size, 6, 6), dtype=np.float64)
        coords = self.lattice.coordinates
        for idx in np.ndindex(self.grid_size):
            subs_dict = {
                self.t: coords[idx][0], self.x: coords[idx][1], self.y: coords[idx][2],
                self.z: coords[idx][3], self.v: coords[idx][4], self.u: coords[idx][5],
                self.a: CONFIG["a_godel"], self.c_sym: c, self.m: m_n,
                self.kappa_sym: kappa, self.nugget_sym: self.nugget_field[idx].real
            }
            g = np.array(sp.N(self.g.subs(subs_dict)), dtype=np.float64)
            if np.any(np.isnan(g)) or np.any(np.isinf(g)):
                g = np.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6)
            metric[idx] = 0.5 * (g + g.T)
        cond = np.linalg.cond(metric)
        inverse_metric = np.zeros_like(metric)
        for idx in np.ndindex(self.grid_size):
            if cond[idx] > 1e10:
                inverse_metric[idx] = pinv(metric[idx], rcond=1e-10)
            else:
                inverse_metric[idx] = np.linalg.inv(metric[idx] + 1e-4 * np.eye(6))
        scale_matrix = np.diag(1.0 / self.metric_scale_factors)
        for idx in np.ndindex(self.grid_size):
            inverse_metric[idx] = scale_matrix @ inverse_metric[idx] @ scale_matrix
        metric *= (1 + CONFIG["entanglement_coupling"] * self.temporal_entanglement.real[..., np.newaxis, np.newaxis])
        inverse_metric /= (1 + CONFIG["entanglement_coupling"] * self.temporal_entanglement.real[..., np.newaxis, np.newaxis])
        return np.nan_to_num(metric, nan=0.0), np.nan_to_num(inverse_metric, nan=0.0)

    def _compute_affine_connection(self):
        return self.spin_network._compute_affine_connection(self.metric, self.inverse_metric, self.deltas)

    def _compute_riemann_tensor(self):
        return self.spin_network._compute_riemann_tensor(self.metric, self.inverse_metric, self.deltas)

    def _compute_curvature(self):
        ricci_tensor_shape = (*self.grid_size, 6, 6)
        riemann = self.riemann_tensor.toarray().reshape(*self.grid_size, 6, 6, 6, 6)
        ricci_tensor = np.einsum('...rsmn,...mn->...rs', riemann, self.inverse_metric)
        ricci_scalar = np.einsum('...mn,...mn->...', self.inverse_metric, ricci_tensor)
        return np.nan_to_num(ricci_tensor, nan=0.0), np.nan_to_num(ricci_scalar, nan=0.0)

    def _compute_stress_energy(self):
        T = np.zeros((*self.grid_size, 6, 6), dtype=np.complex128)
        F = self.em_fields["F"]
        F_nu_alpha = np.einsum('...nu,...beta,...alpha->...nu alpha', F, self.inverse_metric, self.metric)
        T_em = (np.einsum('...mu alpha,...nu alpha->...mu nu', F, F_nu_alpha) -
                0.25 * self.metric * np.einsum('...ab,...ab->...', F, F)) / (4 * np.pi * epsilon_0)
        T += T_em
        T += self.em_fields["J4"][..., np.newaxis, np.newaxis] * self.metric
        quantum_amplitude = np.abs(self.quantum_state)**2
        dphi = [np.gradient(self.nugget_field, self.deltas[mu], axis=mu) for mu in range(6)]
        V_phi = m_nugget**2 * np.abs(self.nugget_field)**2 + lambda_nugget * np.abs(self.nugget_field)**4
        for mu in range(6):
            for nu in range(6):
                T[..., mu, nu] += (dphi[mu] * dphi[nu] - 0.5 * self.metric[..., mu, nu] *
                                   (sum(dphi[k]**2 for k in range(6)) + V_phi))
        T[..., 0, 0] += quantum_amplitude
        for i in range(1, 6):
            T[..., i, i] += quantum_amplitude / 5
        T += CONFIG["info_coupling"] * self.I_mu_nu
        return np.nan_to_num(T, nan=0.0)

    def _compute_einstein_tensor(self):
        ricci_tensor, self.ricci_scalar = self._compute_curvature()
        einstein_tensor = ricci_tensor - 0.5 * self.metric * self.ricci_scalar[..., np.newaxis, np.newaxis]
        return np.nan_to_num(einstein_tensor, nan=0.0)

    def _initialize_em_fields(self):
        A = np.zeros((*self.grid_size, 6), dtype=np.complex128)
        r = np.linalg.norm(self.wormhole_nodes[..., :3], axis=-1) + 1e-15
        A[..., 0] = CONFIG["charge"] / (4 * np.pi * epsilon_0 * r)
        F = np.zeros((*self.grid_size, 6, 6), dtype=np.complex128)
        J = np.zeros((*self.grid_size, 6), dtype=np.complex128)
        base_J = CONFIG["charge"] * c / (4 * np.pi * r**3)
        omega_res = 2 * np.pi * CONFIG["resonance_frequency"]
        resonance = 1 + CONFIG["resonance_amplitude"] * np.sin(omega_res * self.time)
        J[..., 0] = base_J * resonance
        for mu in range(6):
            for nu in range(6):
                grad_A_nu = np.gradient(A[..., nu], self.deltas[mu], axis=mu)
                grad_A_mu = np.gradient(A[..., mu], self.deltas[nu], axis=nu)
                F[..., mu, nu] = grad_A_nu - grad_A_mu
        J_norm = np.einsum('...m,...m->...', J, J)
        k = 2 * np.pi * 1e21
        x = k * self.lambda_field
        coupling = (-x**2 * np.cos(x) + 2 * x * np.sin(x) + 2 * np.cos(x)) * CONFIG["j4_scaling_factor"]
        J4 = J_norm**2 * coupling
        return {"A": A, "F": F, "J": J, "J4": J4}

    def _initialize_weak_fields(self):
        W = np.random.normal(0, 1e-3, (*self.grid_size, 3, 6)).astype(np.complex128) * hbar * c / self.deltas[1]
        F_W = np.zeros((*self.grid_size, 3, 6, 6), dtype=np.complex128)
        for a in range(3):
            for mu in range(6):
                for nu in range(6):
                    dW_mu = np.gradient(W[..., a, nu], self.deltas[mu], axis=mu)
                    dW_nu = np.gradient(W[..., a, mu], self.deltas[nu], axis=nu)
                    nonlinear = CONFIG["g_weak"] * np.einsum('abc,...b,...c->...a', f_su2, W[..., mu], W[..., nu])[..., a]
                    F_W[..., a, mu, nu] = (dW_mu - dW_nu + nonlinear)
        return {"W": W, "F": F_W}

    def _initialize_strong_fields(self):
        G = np.random.normal(0, 1e-3, (*self.grid_size, 8, 6)).astype(np.complex128) * hbar * c / self.deltas[1]
        F_G = np.zeros((*self.grid_size, 8, 6, 6), dtype=np.complex128)
        for a in range(8):
            for mu in range(6):
                for nu in range(6):
                    dG_mu = np.gradient(G[..., a, nu], self.deltas[mu], axis=mu)
                    dG_nu = np.gradient(G[..., a, mu], self.deltas[nu], axis=nu)
                    nonlinear = CONFIG["g_strong"] * np.einsum('abc,...b,...c->...a', f_su3, G[..., mu], G[..., nu])[..., a]
                    F_G[..., a, mu, nu] = (dG_mu - dG_nu + nonlinear)
        return {"G": G, "F": F_G}

    def evolve_gauge_fields(self):
        for a in range(8):
            for mu in range(6):
                for nu in range(6):
                    dG_mu = np.gradient(self.strong_fields['G'][..., a, nu], self.deltas[mu], axis=mu)
                    dG_nu = np.gradient(self.strong_fields['G'][..., a, mu], self.deltas[nu], axis=nu)
                    nonlinear = CONFIG["g_strong"] * np.einsum('abc,...b,...c->...a', f_su3,
                                                              self.strong_fields['G'][..., mu],
                                                              self.strong_fields['G'][..., nu])[..., a]
                    self.strong_fields['F'][..., a, mu, nu] = dG_mu - dG_nu + nonlinear
        for a in range(3):
            for mu in range(6):
                for nu in range(6):
                    dW_mu = np.gradient(self.weak_fields['W'][..., a, nu], self.deltas[mu], axis=mu)
                    dW_nu = np.gradient(self.weak_fields['W'][..., a, mu], self.deltas[nu], axis=nu)
                    nonlinear = CONFIG["g_weak"] * np.einsum('abc,...b,...c->...a', f_su2,
                                                            self.weak_fields['W'][..., mu],
                                                            self.weak_fields['W'][..., nu])[..., a]
                    self.weak_fields['F'][..., a, mu, nu] = dW_mu - dW_nu + nonlinear

    def evolve_nugget_field(self):
        def nugget_deriv(t, phi_flat):
            phi = phi_flat.reshape(self.grid_size)
            laplacian = sum(np.gradient(np.gradient(phi, self.deltas[i], axis=i), self.deltas[i], axis=i)
                           for i in range(6))
            V_deriv = -m_nugget**2 * phi + 3 * lambda_nugget * np.abs(phi)**2 * phi
            ctc_term = CONFIG["lambda_ctc"] * CONFIG["m_shift"] * phi
            return (-laplacian - V_deriv + ctc_term).flatten()
        phi_flat = self.nugget_field.flatten()
        max_attempts = 3
        current_dt = self.dt
        total_steps = 0
        error_estimate = 0.0
        for attempt in range(max_attempts):
            try:
                sol = solve_ivp(nugget_deriv, [0, current_dt], phi_flat, method='RK45',
                                rtol=CONFIG["rtol"], atol=CONFIG["atol"], vectorized=True)
                if not sol.success:
                    raise ValueError(f"solve_ivp failed: {sol.message}")
                total_steps += len(sol.t) - 1
                error_estimate = np.max(np.abs(sol.y[:, -1] - sol.y[:, -2])) payers
                break
            except Exception as e:
                logger.warning(f"evolve_nugget_field failed with dt={current_dt}: {e}, attempt {attempt+1}/{max_attempts}")
                if attempt == max_attempts - 1:
                    raise
                current_dt *= 0.5
        self.nugget_field = sol.y[:, -1].reshape(self.grid_size)
        self.nugget_field = np.nan_to_num(self.nugget_field, nan=0.0)
        self.nugget_field = np.clip(self.nugget_field, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
        logger.debug(f"Nugget field integration error estimate: {error_estimate:.2e}")
        return total_steps

    def evolve_higgs_field(self):
        def higgs_deriv(t, h_flat):
            h = h_flat.reshape(self.grid_size)
            d2_higgs = sum(np.gradient(np.gradient(h, self.deltas[i], axis=i), self.deltas[i], axis=i)
                           for i in range(6))
            h_norm = np.abs(h)**2
            dV_dH = -m_h * c**2 * h + lambda_higgs * h_norm * h
            return (-d2_higgs + dV_dH).flatten()
        h_flat = self.higgs_field.flatten()
        max_attempts = 3
        current_dt = self.dt
        total_steps = 0
        error_estimate = 0.0
        for attempt in range(max_attempts):
            try:
                sol = solve_ivp(higgs_deriv, [0, current_dt], h_flat, method='RK45',
                                rtol=CONFIG["rtol"], atol=CONFIG["atol"], vectorized=True)
                if not sol.success:
                    raise ValueError(f"solve_ivp failed: {sol.message}")
                total_steps += len(sol.t) - 1
                error_estimate = np.max(np.abs(sol.y[:, -1] - sol.y[:, -2])) if sol.t.size > 1 else 0.0
                break
            except Exception as e:
                logger.warning(f"evolve_higgs_field failed with dt={current_dt}: {e}, attempt {attempt+1}/{max_attempts}")
                if attempt == max_attempts - 1:
                    raise
                current_dt *= 0.5
        self.higgs_field = sol.y[:, -1].reshape(self.grid_size)
        self.higgs_field = np.nan_to_num(self.higgs_field, nan=v_higgs)
        self.higgs_field = np.clip(self.higgs_field, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
        logger.debug(f"Higgs field integration error estimate: {error_estimate:.2e}")
        return total_steps

    def evolve_fermion_fields(self):
        def evolve_fermion_at_idx(idx, quark=False, flavor=None, color=None):
            if quark:
                psi = self.quark_field[idx + (flavor, color)]
                H = self.dirac_hamiltonian(psi, idx, quark=True, flavor=flavor, color=color)
            else:
                psi = self.electron_field[idx]
                H = self.dirac_hamiltonian(psi, idx, quark=False)
            max_attempts = 3
            current_dt = self.dt
            steps = 0
            error_estimate = 0.0
            for attempt in range(max_attempts):
                try:
                    sol = solve_ivp(lambda t, y: -1j * H.dot(y) / hbar, [0, current_dt], psi,
                                    method='RK45', rtol=CONFIG["rtol"], atol=CONFIG["atol"], vectorized=True)
                    if not sol.success:
                        raise ValueError(f"solve_ivp failed: {sol.message}")
                    steps += len(sol.t) - 1
                    error_estimate = np.max(np.abs(sol.y[:, -1] - sol.y[:, -2])) if sol.t.size > 1 else 0.0
                    return sol.y[:, -1], steps, error_estimate
                except Exception as e:
                    logger.warning(f"evolve_fermion_at_idx failed with dt={current_dt}: {e}, attempt {attempt+1}/{max_attempts}")
                    if attempt == max_attempts - 1:
                        raise
                    current_dt *= 0.5
            return psi, steps, error_estimate

        total_steps = 0
        total_error = 0.0
        # Electron field evolution
        results = Parallel(n_jobs=-1)(
            delayed(evolve_fermion_at_idx)(idx, quark=False) for idx in np.ndindex(self.grid_size)
        )
        for idx, (psi, steps, error) in zip(np.ndindex(self.grid_size), results):
            self.electron_field[idx] = psi
            total_steps += steps
            total_error += error

        # Quark field evolution
        quark_tasks = [
            (idx, f, c) for idx in np.ndindex(self.grid_size) for f in range(3) for c in range(3)
        ]
        results = Parallel(n_jobs=-1)(
            delayed(evolve_fermion_at_idx)(idx, quark=True, flavor=f, color=c)
            for idx, f, c in quark_tasks
        )
        for (idx, f, c), (psi, steps, error) in zip(quark_tasks, results):
            self.quark_field[idx + (f, c)] = psi
            total_steps += steps
            total_error += error

        self.electron_field = np.nan_to_num(self.electron_field, nan=0.0)
        self.quark_field = np.nan_to_num(self.quark_field, nan=0.0)
        self.electron_field = np.clip(self.electron_field, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
        self.quark_field = np.clip(self.quark_field, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
        logger.debug(f"Fermion fields integration error estimate: {total_error:.2e}")
        return total_steps

    def dirac_hamiltonian(self, psi, idx, quark=False, flavor=None, color=None):
        gamma_mu = construct_6d_gamma_matrices(self.metric[idx])
        mass = m_q if quark else m_e
        field = self.quark_field[..., flavor, color] if quark else self.electron_field
        D_mu_psi = [np.gradient(field, self.deltas[i], axis=i)[idx] if i < len(psi.shape) else psi for i in range(6)]
        H_psi = -1j * c * sum(gamma_mu[0] @ gamma_mu[i] @ D_mu_psi[i] for i in range(1, 6))
        if quark and flavor is not None:
            mass_term = sum(self.quark_mass_matrix[flavor, f] * self.quark_field[idx, f, color]
                            for f in range(3))
            H_psi += (c**2 / hbar) * gamma_mu[0] @ mass_term
        else:
            H_psi += (mass * c**2 / hbar) * gamma_mu[0] @ psi
        H_psi -= 1j * e * sum(self.em_fields["A"][idx][mu] * gamma_mu[mu] @ psi for mu in range(6))
        if quark and flavor is not None and color is not None:
            T_a = lambda_matrices
            strong_term = sum(CONFIG["g_strong"] * self.strong_fields['G'][idx][a, mu] * T_a[a][color, color] * psi
                              for a in range(8) for mu in range(6))
            H_psi += -1j * strong_term
            weak_term = np.zeros_like(psi, dtype=np.complex128)
            for target_flavor in range(3):
                if target_flavor != flavor:
                    V_ckm = self.CKM[flavor, target_flavor]
                    weak_term += -1j * CONFIG["g_weak"] * V_ckm * sum(
                        self.weak_fields['W'][idx][a, mu] * gamma_mu[mu] @ self.quark_field[idx, target_flavor, color]
                        for a in range(3) for mu in range(6)
                    )
            H_psi += weak_term
        return np.nan_to_num(H_psi, nan=0.0)

    def compute_lambda(self, t, coords, N=3):
        if hasattr(self, '_last_lambda_time') and abs(self._last_lambda_time - t) < 1e-15:
            return self.lambda_field
        frequencies = compute_schumann_frequencies(N)
        omega = [2 * np.pi * f for f in frequencies]
        lambda_field = np.zeros(self.grid_size, dtype=np.float64)
        x = coords[..., 1] / self.deltas[1]
        for n in range(N):
            A_n = 1e-21
            term = (-x**2 * np.cos(omega[n] * t) + 2 * x * np.sin(omega[n] * t) +
                    2 * np.cos(omega[n] * t))
            lambda_field += A_n * term
        self.lambda_field = np.nan_to_num(lambda_field, nan=0.0)
        self._last_lambda_time = t
        return self.lambda_field

    def compute_rio_pattern(self, iteration):
        P = np.abs(self.quantum_state)**2
        F = np.sqrt(np.sum([np.gradient(self.nugget_field, self.deltas[mu], axis=mu)**2 for mu in range(6)], axis=0))
        phi_shifted = np.roll(self.nugget_field, shift=[1, 1, 0, 0, 0, 0], axis=tuple(range(6)))
        M = np.cos(CONFIG["alpha_phi"] * P) * np.cos(CONFIG["alpha_phi"] * phi_shifted)
        return M * F, F

    def compute_quantum_flux(self):
        fine_factor = 2
        fine_grid_size = (self.grid_size[0], self.grid_size[1] * fine_factor, self.grid_size[2] * fine_factor,
                          self.grid_size[3] * fine_factor, self.grid_size[4], self.grid_size[5])
        fine_deltas = [d / fine_factor if i in [1, 2, 3] else d for i, d in enumerate(self.deltas)]
        fine_coords = np.zeros((*fine_grid_size, 6), dtype=np.float64)
        dims = [np.linspace(0, fine_deltas[i] * fine_grid_size[i], fine_grid_size[i])
                for i in range(6)]
        fine_coords[..., 0], fine_coords[..., 1], fine_coords[..., 2], \
        fine_coords[..., 3], fine_coords[..., 4], fine_coords[..., 5] = np.meshgrid(*dims, indexing='ij')
        psi_fine = np.zeros(fine_grid_size, dtype=np.complex128)
        for idx in np.ndindex(fine_grid_size):
            point = fine_coords[idx]
            psi_fine[idx] = self.lattice.interpolate_field(self.quantum_state, point)
        psi_conj_fine = np.conj(psi_fine)
        J_fine = np.zeros((*fine_grid_size, 6), dtype=np.complex128)
        for mu in range(6):
            grad_psi = np.gradient(psi_fine, fine_deltas[mu], axis=mu)
            grad_psi_conj = np.gradient(psi_conj_fine, fine_deltas[mu], axis=mu)
            J_fine[..., mu] = (hbar / (2 * m_n * 1j)) * (psi_conj_fine * grad_psi - psi_fine * grad_psi_conj)
        J = np.zeros((*self.grid_size, 6), dtype=np.complex128)
        for idx in np.ndindex(self.grid_size):
            fine_idx_start = tuple(i * fine_factor for i in idx)
            fine_slice = tuple(slice(fine_idx_start[k], fine_idx_start[k] + fine_factor) if k in [1, 2, 3]
                               else slice(fine_idx_start[k], fine_idx_start[k] + 1) for k in range(6))
            J[idx] = np.mean(J_fine[fine_slice], axis=(1, 2, 3))
        J_mag = np.sqrt(np.sum(np.abs(J)**2, axis=-1))
        return J, J_mag

    def compute_information_tensor(self):
        I_mu_nu = np.zeros((*self.grid_size, 6, 6), dtype=np.float64)
        rho_bdry = np.zeros(self.grid_size, dtype=np.complex128)
        for idx in np.ndindex(self.grid_size):
            if idx[1] == 0 or idx[1] == self.grid_size[1] - 1:
                psi = self.electron_field[idx]
                psi = psi / (np.linalg.norm(psi) + 1e-15)
                psi_matrix = psi.reshape(2, 2)
                rho_bdry[idx] = np.sum(np.abs(psi_matrix)**2)
        rho_0 = 1.0 / 4
        self.relative_entropy = np.zeros(self.grid_size, dtype=np.float64)
        for idx in np.ndindex(self.grid_size):
            if idx[1] == 0 or idx[1] == self.grid_size[1] - 1:
                ratio = rho_bdry[idx].real / rho_0 if rho_bdry[idx].real > 1e-15 and rho_0 > 1e-15 else 1e-15
                self.relative_entropy[idx] = np.log(ratio) if ratio > 0 else 0
        alpha = CONFIG["alpha_em"]
        tau = CONFIG["tau"]
        for idx in np.ndindex(self.grid_size):
            I_mu_nu[idx] = (self.relative_entropy[idx] * self.metric[idx] +
                           alpha * tau * self.einstein_tensor[idx] * self.metric[idx])
        return np.nan_to_num(I_mu_nu, nan=0.0)

    def compute_entanglement_geodesics(self):
        entanglement = np.zeros(self.grid_size, dtype=np.float64)
        for idx in np.ndindex(self.grid_size):
            local_state = self.electron_field[idx].flatten()
            norm = np.linalg.norm(local_state)
            if norm > 1e-15:
                local_state /= norm
            psi_matrix = local_state.reshape(2, 2)
            schmidt_coeffs = svdvals(psi_matrix)
            probs = schmidt_coeffs**2
            probs = probs[probs > 1e-15]
            entanglement[idx] = -np.sum(probs * np.log(probs)) if probs.size > 0 else 0
        J, J_mag = self.compute_quantum_flux()
        J_temporal = J[..., 0].real
        cost = entanglement + CONFIG["flux_coupling"] * np.abs(J_temporal)
        start_idx = (0, 0, 0, 0, 0, 0)
        end_idx = (self.grid_size[0] - 1, self.grid_size[1] - 1, self.grid_size[2] - 1,
                   self.grid_size[3] - 1, self.grid_size[4] - 1, self.grid_size[5] - 1)
        path = [start_idx]
        current = list(start_idx)
        while current != list(end_idx):
            candidates = []
            for dim in [1, 2, 3]:
                if current[dim] < self.grid_size[dim] - 1:
                    next_idx = current.copy()
                    next_idx[dim] += 1
                    candidates.append(tuple(next_idx))
            if not candidates:
                break
            costs = [cost[cand] for cand in candidates]
            next_idx = candidates[np.argmin(costs)]
            path.append(next_idx)
            current = list(next_idx)
        smooth_path = []
        for i in range(len(path) - 1):
            p0, p1 = path[i], path[i + 1]
            for alpha in np.linspace(0, 1, 5):
                point = np.array(p0) + alpha * (np.array(p1) - np.array(p0))
                phys_point = np.array([point[i] * self.deltas[i] for i in range(6)])
                interpolated_cost = self.lattice.interpolate_field(cost, phys_point)
                smooth_path.append((phys_point, interpolated_cost))
        return path, smooth_path

    def compute_bits_entropy(self):
        entropy = 0.0
        patch_size = 2
        num_patches = 0
        qubit_probs = np.abs(self.qubit_states[..., 1])**2  # Probability of |1>
        for t in range(0, self.grid_size[0], patch_size):
            for x in range(0, self.grid_size[1], patch_size):
                for y in range(0, self.grid_size[2], patch_size):
                    for z in range(0, self.grid_size[3], patch_size):
                        for v in range(0, self.grid_size[4], patch_size):
                            for u in range(0, self.grid_size[5], patch_size):
                                patch = qubit_probs[
                                    t:min(t+patch_size, self.grid_size[0]),
                                    x:min(x+patch_size, self.grid_size[1]),
                                    y:min(y+patch_size, self.grid_size[2]),
                                    z:min(z+patch_size, self.grid_size[3]),
                                    v:min(v+patch_size, self.grid_size[4]),
                                    u:min(u+patch_size, self.grid_size[5])
                                ]
                                p1 = np.mean(patch)
                                p0 = 1 - p1
                                if 0 < p1 < 1:
                                    entropy += -patch.size * (p1 * np.log(p1) + p0 * np.log(p0))
                                num_patches += 1
        if num_patches > 0:
            entropy *= (self.total_points / (num_patches * (patch_size**6)))
        return entropy

    def compute_geometric_entropy(self):
        entropy = np.sum(np.abs(self.relative_entropy)) * self.dV * self.geom_entropy_scale
        return entropy

    def compute_temporal_entropy(self):
        probs = np.abs(self.quantum_state)**2
        probs = probs / (np.sum(probs) + 1e-15)
        probs = probs[probs > 1e-15]
        entropy = -np.sum(probs * np.log(probs)) if probs.size > 0 else 0.0
        return entropy

    def compute_cp_asymmetry(self):
        particle_amplitude = np.abs(self.quark_field[..., 2, :, :]).sum()
        cp_factor = 1 + 0.0245
        antiparticle_amplitude = particle_amplitude * cp_factor
        A_CP = (particle_amplitude - antiparticle_amplitude) / (particle_amplitude + antiparticle_amplitude)
        return A_CP

    def run_teleportation(self):
        idx_a = (0, 0, 0, 0, 0, 0)
        idx_b = (self.grid_size[0] - 1, self.grid_size[1] - 1, self.grid_size[2] - 1,
                 self.grid_size[3] - 1, self.grid_size[4] - 1, self.grid_size[5] - 1)
        initial_state = self.qubit_states[idx_a].copy()
        steps = self.spin_network.evolve(self.dt, self.lambda_field, self.metric, self.inverse_metric,
                                         self.deltas, self.nugget_field, self.higgs_field, self.em_fields,
                                         self.electron_field, self.quark_field)
        state_a = self.qubit_states[idx_a]
        state_b = self.qubit_states[idx_b]
        fidelity = np.abs(np.dot(state_a.conj(), state_b))**2
        ricci_effect = np.abs(self.ricci_scalar[idx_a] + self.ricci_scalar[idx_b]) / (2 * np.max(np.abs(self.ricci_scalar)) + 1e-15)
        fidelity *= (1 - 0.1 * ricci_effect)
        return {"fidelity": fidelity, "steps": steps}

    def emulate_on_hardware(self):
        num_qubits = 2
        shots = 1024
        counts = {'00': 0, '11': 0}
        for _ in range(shots):
            state = np.ones(num_qubits, dtype=np.complex128) / np.sqrt(2)
            for i in range(num_qubits):
                idx = (0, i, 0, 0, 0, 0)
                threshold = np.mean(self.temporal_entanglement.real)
                em_magnitude = np.linalg.norm(self.em_fields["F"], axis=(-1, -2))[idx]
                em_influence = CONFIG["em_strength"] * em_magnitude / (np.max(em_magnitude) + 1e-15)
                if self.temporal_entanglement[idx].real > threshold and em_influence > 0.5:
                    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                    state[i] = H @ np.array([state[i], 0])[0]
                    norm = np.abs(state[i])
                    if norm > 0:
                        state[i] /= norm
            probs = np.abs(state)**2
            if np.all(probs < 0.1):
                counts['00'] += 1
            elif np.all(probs > 0.9):
                counts['11'] += 1
        fidelity = counts['00'] / shots
        return {"fidelity": fidelity}

    def adjust_time_step(self, steps_taken):
        target_steps = 10
        if steps_taken > CONFIG["max_steps_per_dt"]:
            self.dt *= 0.5
        elif steps_taken > target_steps * 1.5:
            self.dt *= 0.9
        elif steps_taken < target_steps * 0.5 and steps_taken > 0:
            self.dt *= 1.1
        self.dt = max(CONFIG["dt_min"], min(self.dt, CONFIG["dt_max"]))
        logger.debug(f"Adjusted dt: {self.dt}")

    def quantum_walk(self, iteration):
        t_start = self.time
        t_end = t_start + self.dt
        max_attempts = 3
        current_dt = self.dt
        total_steps = 0
        erasure_entropy = 0.0
        for attempt in range(max_attempts):
            try:
                self.time = t_start
                self.lambda_field = self.compute_lambda(self.time, self.lattice.coordinates)
                steps = self.spin_network.evolve(current_dt, self.lambda_field, self.metric, self.inverse_metric,
                                                self.deltas, self.nugget_field, self.higgs_field, self.em_fields,
                                                self.electron_field, self.quark_field)
                total_steps += steps
                prob = np.abs(self.quantum_state)**2
                nugget_smooth = np.zeros_like(self.nugget_field)
                for idx in np.ndindex(self.grid_size):
                    point = self.lattice.coordinates[idx]
                    nugget_smooth[idx] = self.lattice.interpolate_field(self.nugget_field, point)
                self.quantum_state += CONFIG["lambda_ctc"] * CONFIG["m_shift"] * self.quantum_state
                norm = np.linalg.norm(self.quantum_state)
                if norm > 0:
                    erasure_entropy = np.log(norm) if norm > 1e-15 else 0.0
                    self.quantum_state /= norm
                self.quantum_state = np.clip(self.quantum_state, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
                self.temporal_entanglement = CONFIG["entanglement_factor"] * np.abs(self.quantum_state)**2
                threshold = np.mean(self.temporal_entanglement.real)
                em_magnitude = np.linalg.norm(self.em_fields["F"], axis=(-1, -2))
                em_influence = CONFIG["em_strength"] * em_magnitude / (np.max(em_magnitude) + 1e-15)
                for idx in np.ndindex(self.grid_size):
                    if self.temporal_entanglement[idx].real > threshold and em_influence[idx] > 0.5:
                        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                        self.qubit_states[idx] = H @ self.qubit_states[idx]
                        norm = np.linalg.norm(self.qubit_states[idx])
                        if norm > 0:
                            self.qubit_states[idx] /= norm
                entanglement_entropy = compute_entanglement_entropy(self.electron_field, self.grid_size)
                bits_entropy = self.compute_bits_entropy()
                geom_entropy = self.compute_geometric_entropy()
                temp_entropy = self.compute_temporal_entropy()
                cp_asymmetry = self.compute_cp_asymmetry()
                teleportation_result = self.run_teleportation()
                bit_state_fraction = np.mean(np.abs(self.qubit_states[..., 1])**2)
                self.entanglement_history.append(entanglement_entropy)
                self.bits_entropy_history.append(bits_entropy)
                self.geom_entropy_history.append(geom_entropy)
                self.erasure_entropy_history.append(erasure_entropy)
                self.temp_entropy_history.append(temp_entropy)
                self.cp_asymmetry_history.append(cp_asymmetry)
                self.teleportation_fidelity_history.append(teleportation_result["fidelity"])
                self.bit_state_fraction_history.append(bit_state_fraction)
                self.nugget_history.append(np.mean(np.abs(self.nugget_field)**2))
                total_steps += teleportation_result["steps"]
                total_steps += self.evolve_nugget_field()
                total_steps += self.evolve_higgs_field()
                total_steps += self.evolve_fermion_fields()
                self.evolve_gauge_fields()
                self.metric, self.inverse_metric = self.compute_quantum_metric()
                self.connection = self._compute_affine_connection()
                self.riemann_tensor = self._compute_riemann_tensor()
                self.einstein_tensor = self._compute_einstein_tensor()
                self.I_mu_nu = self.compute_information_tensor()
                self.I_mu_nu_history.append(np.mean(np.linalg.norm(self.I_mu_nu, axis=(-1, -2))))
                self.ricci_scalar_history.append(np.mean(np.abs(self.ricci_scalar)))
                self.adaptive_grid.refine(self.ricci_scalar)
                self.deltas = self.adaptive_grid.deltas
                self.lattice.deltas = self.deltas
                self.lattice.coordinates = self.adaptive_grid.coordinates
                break
            except Exception as e:
                logger.warning(f"quantum_walk failed with dt={current_dt}: {e}, attempt {attempt+1}/{max_attempts}")
                if attempt == max_attempts - 1:
                    raise
                current_dt *= 0.5
                total_steps += 1
        self.adjust_time_step(total_steps)
        self.time = t_end

        def visualize(self, iteration):
        if iteration % 5 != 0 and iteration != CONFIG["max_iterations"] - 1:
            return
        # First figure: Entropies
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.plot(self.entanglement_history, label="Entanglement Entropy", color='blue')
        plt.xlabel("Iteration")
        plt.ylabel("S_ent (nats)")
        plt.title("Entanglement Entropy")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(self.bits_entropy_history, label="Configurational Entropy", color='orange')
        plt.xlabel("Iteration")
        plt.ylabel("S_bits (bits)")
        plt.title("Configurational Entropy")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 3)
        plt.plot(self.geom_entropy_history, label="Geometric Entropy", color='green')
        plt.xlabel("Iteration")
        plt.ylabel("S_geom (bits)")
        plt.title("Geometric Entropy")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 4)
        plt.plot(self.erasure_entropy_history, label="Erasure Entropy", color='red')
        plt.xlabel("Iteration")
        plt.ylabel("S_erasure (nats)")
        plt.title("Erasure Entropy")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 5)
        plt.plot(self.temp_entropy_history, label="Temporal Entropy", color='purple')
        plt.xlabel("Iteration")
        plt.ylabel("S_temp (nats)")
        plt.title("Temporal Entropy")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'entropy_plots_iter_{iteration}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Second figure: Physical Quantities
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.plot(self.nugget_history, label="Nugget Field Norm", color='cyan')
        plt.xlabel("Iteration")
        plt.ylabel("|φ_n|^2")
        plt.title("Nugget Field Norm")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(self.ricci_scalar_history, label="Ricci Scalar", color='magenta')
        plt.xlabel("Iteration")
        plt.ylabel("R")
        plt.title("Ricci Scalar")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 3)
        plt.plot(self.I_mu_nu_history, label="Info Tensor Norm", color='brown')
        plt.xlabel("Iteration")
        plt.ylabel("|I_μν|")
        plt.title("Information Tensor Norm")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 4)
        plt.plot(self.cp_asymmetry_history, label="CP Asymmetry", color='pink')
        plt.xlabel("Iteration")
        plt.ylabel("A_CP")
        plt.title("CP Asymmetry")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 5)
        plt.plot(self.teleportation_fidelity_history, label="Teleportation Fidelity", color='gray')
        plt.xlabel("Iteration")
        plt.ylabel("Fidelity")
        plt.title("Teleportation Fidelity")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 6)
        plt.plot(self.bit_state_fraction_history, label="Bit State Fraction", color='black')
        plt.xlabel("Iteration")
        plt.ylabel("P(|1>)")
        plt.title("Bit State Fraction")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'physical_quantities_iter_{iteration}.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Visualizations saved for iteration {iteration}")

    def save_metrics_to_csv(self, iteration):
        metrics = {
            "Iteration": list(range(len(self.entanglement_history))),
            "Entanglement_Entropy": self.entanglement_history,
            "Bits_Entropy": self.bits_entropy_history,
            "Geometric_Entropy": self.geom_entropy_history,
            "Erasure_Entropy": self.erasure_entropy_history,
            "Temporal_Entropy": self.temp_entropy_history,
            "Nugget_Norm": self.nugget_history,
            "Ricci_Scalar": self.ricci_scalar_history,
            "Info_Tensor_Norm": self.I_mu_nu_history,
            "CP_Asymmetry": self.cp_asymmetry_history,
            "Teleportation_Fidelity": self.teleportation_fidelity_history,
            "Bit_State_Fraction": self.bit_state_fraction_history
        }
        df = pd.DataFrame(metrics)
        csv_path = f'metrics_iter_{iteration}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Metrics saved to {csv_path}")

    def save_checkpoint(self, iteration):
        checkpoint = {
            'iteration': iteration,
            'time': self.time,
            'quantum_state': self.quantum_state,
            'nugget_field': self.nugget_field,
            'higgs_field': self.higgs_field,
            'electron_field': self.electron_field,
            'quark_field': self.quark_field,
            'em_fields': self.em_fields,
            'metric': self.metric,
            'inverse_metric': self.inverse_metric,
            'ricci_scalar': self.ricci_scalar,
            'I_mu_nu': self.I_mu_nu,
            'qubit_states': self.qubit_states,
            'temporal_entanglement': self.temporal_entanglement,
            'grid_size': self.grid_size,
            'deltas': self.deltas
        }
        checkpoint_path = f'checkpoint_iter_{iteration}.pkl.gz'
        with gzip.open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, filepath):
        try:
            with gzip.open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)
            expected_shapes = {
                'quantum_state': self.grid_size,
                'nugget_field': self.grid_size,
                'higgs_field': self.grid_size,
                'electron_field': (*self.grid_size, 4),
                'quark_field': (*self.grid_size, 3, 3, 4),
                'qubit_states': (*self.grid_size, 2),
                'temporal_entanglement': self.grid_size,
                'metric': (*self.grid_size, 6, 6),
                'inverse_metric': (*self.grid_size, 6, 6),
                'ricci_scalar': self.grid_size,
                'I_mu_nu': (*self.grid_size, 6, 6)
            }
            for key, expected_shape in expected_shapes.items():
                if checkpoint[key].shape != expected_shape:
                    raise ValueError(f"Shape mismatch for {key}: expected {expected_shape}, got {checkpoint[key].shape}")
            self.time = checkpoint['time']
            self.quantum_state = checkpoint['quantum_state']
            self.nugget_field = checkpoint['nugget_field']
            self.higgs_field = checkpoint['higgs_field']
            self.electron_field = checkpoint['electron_field']
            self.quark_field = checkpoint['quark_field']
            self.em_fields = checkpoint['em_fields']
            self.metric = checkpoint['metric']
            self.inverse_metric = checkpoint['inverse_metric']
            self.ricci_scalar = checkpoint['ricci_scalar']
            self.I_mu_nu = checkpoint['I_mu_nu']
            self.qubit_states = checkpoint['qubit_states']
            self.temporal_entanglement = checkpoint['temporal_entanglement']
            self.deltas = checkpoint['deltas']
            self.adaptive_grid.deltas = self.deltas
            self.lattice.deltas = self.deltas
            logger.info(f"Checkpoint loaded from {filepath} at iteration {checkpoint['iteration']}")
            return checkpoint['iteration']
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {filepath}: {e}")
            raise

def main():
    os.makedirs('outputs', exist_ok=True)
    sphinx_os = SphinxOS()
    start_iteration = 0
    checkpoint_file = 'checkpoint_latest.pkl.gz'
    if os.path.exists(checkpoint_file):
        try:
            start_iteration = sphinx_os.load_checkpoint(checkpoint_file)
            logger.info(f"Resumed from checkpoint at iteration {start_iteration}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting from iteration 0.")
            start_iteration = 0

    for iteration in tqdm(range(start_iteration, CONFIG["max_iterations"]), desc="Simulation Progress"):
        try:
            sphinx_os.quantum_walk(iteration)
            sphinx_os.save_metrics_to_csv(iteration)
            sphinx_os.visualize(iteration)
            if iteration % 5 == 0 or iteration == CONFIG["max_iterations"] - 1:
                sphinx_os.save_checkpoint(iteration)
                with gzip.open('checkpoint_latest.pkl.gz', 'wb') as f:
                    pickle.dump({
                        'iteration': iteration,
                        'time': sphinx_os.time,
                        'quantum_state': sphinx_os.quantum_state,
                        'nugget_field': sphinx_os.nugget_field,
                        'higgs_field': sphinx_os.higgs_field,
                        'electron_field': sphinx_os.electron_field,
                        'quark_field': sphinx_os.quark_field,
                        'em_fields': sphinx_os.em_fields,
                        'metric': sphinx_os.metric,
                        'inverse_metric': sphinx_os.inverse_metric,
                        'ricci_scalar': sphinx_os.ricci_scalar,
                        'I_mu_nu': sphinx_os.I_mu_nu,
                        'qubit_states': sphinx_os.qubit_states,
                        'temporal_entanglement': sphinx_os.temporal_entanglement,
                        'grid_size': sphinx_os.grid_size,
                        'deltas': sphinx_os.deltas
                    }, f)
            logger.debug(f"Iteration {iteration} completed. Time: {sphinx_os.time:.2e} s")
        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {e}")
            sphinx_os.save_checkpoint(iteration)
            break

    runtime = time.perf_counter_ns() / 1e9 - START_TIME
    memory_usage = psutil.Process().memory_info().rss / 1024**2
    logger.info(f"Simulation completed. Runtime: {runtime:.2f} s, Memory: {memory_usage:.2f} MB")

if __name__ == "__main__":
    main()