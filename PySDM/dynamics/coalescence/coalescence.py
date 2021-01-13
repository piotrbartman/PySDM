"""
Created at 07.06.2019
"""

import numpy as np

from .random_generator_optimizer import RandomGeneratorOptimizer


class Coalescence:

    def __init__(self, kernel, seed=None, croupier=None, adaptive=False, max_substeps=128, optimized_random=False):
        self.core = None
        self.kernel = kernel
        self.rnd_opt = RandomGeneratorOptimizer(optimized_random=optimized_random, max_substeps=max_substeps, seed=seed)
        self.enable = True
        self.__adaptive = adaptive
        self.__n_substep = None
        self.croupier = croupier

        self.temp = None
        self.prob = None
        self.is_first_in_pair = None

        self.actual_length = None

    def register(self, builder):
        self.core = builder.core
        self.temp = self.core.PairwiseStorage.empty(self.core.n_sd, dtype=float)
        self.__n_substep = self.core.Storage.empty(self.core.mesh.n_cell, dtype=int)
        self.__n_substep[:] = 1
        self.rnd_opt.register(builder)
        self.kernel.register(builder)

        Index = lambda length: self.core.Index.empty(length)
        PairIndicator = lambda length: self.core.PairIndicator(length)
        IndexedStorage = lambda idx, shape, dtype: self.core.IndexedStorage.empty(idx, shape, dtype)
        PairwiseStorage = lambda shape, dtype: self.core.PairwiseStorage.empty(shape, dtype)
        Storage = lambda shape, dtype: self.core.Storage.empty(shape, dtype)
        N_SD = self.core.n_sd
        N_CELL = self.core.mesh.n_cell
        N_IA = 0
        N_IE = 1
        # <listing>
        # model state
        idx = Index(N_SD)
        multiplicities = IndexedStorage(idx, N_SD, int)
        intensive = IndexedStorage(idx, (N_IA, N_SD), float)
        extensive = IndexedStorage(idx, (N_IE, N_SD), float)
        cell_id = IndexedStorage(idx, N_SD, int)
        prob = PairwiseStorage(N_SD, float)
        cell_idx = Index(N_CELL)
        cell_start = Storage(N_CELL + 1, int)

        # helper vars (used within time-step)
        is_first_in_pair = PairIndicator(N_SD)
        # </listing>
        self.is_first_in_pair = is_first_in_pair
        self.idx = idx
        self.multiplicities = multiplicities
        self.intensive = intensive
        self.extensive = extensive
        self.cell_id = cell_id
        self.prob = prob
        self.cell_idx = cell_idx
        self.cell_start = cell_start

        self.adaptive_memory = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
        self.subs = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
        self.msub = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))

        if self.croupier is None:
            self.croupier = self.core.backend.default_croupier

        self.collision_rate = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
        self.collision_rate_deficit = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))

    def __call__(self):
        if self.enable:
            self.step(0, self.adaptive_memory)

    def step(self, s, adaptive_memory):
        self.idx = self.core.particles._Particles__idx
        assert self.idx.shape == self.core.particles._Particles__idx.shape
        assert self.idx.dtype == self.core.particles._Particles__idx.dtype
        self.multiplicities.data = self.core.particles['n'].data
        assert self.intensive.shape == self.core.particles.get_intensive_attrs().shape
        assert self.intensive.dtype == self.core.particles.get_intensive_attrs().dtype
        self.intensive = self.core.particles.get_intensive_attrs()
        assert self.extensive.shape == self.core.particles.get_extensive_attrs().shape
        assert self.extensive.dtype == self.core.particles.get_extensive_attrs().dtype
        self.extensive = self.core.particles.get_extensive_attrs()
        self.cell_id.data = self.core.particles["cell id"].data
        self.__step(self.is_first_in_pair,
                    self.prob,
                    self.idx,
                    self.multiplicities,
                    self.intensive,
                    self.extensive,
                    self.cell_id,
                    self.cell_idx,
                    self.cell_start)

    def __step(self, is_first_in_pair, prob, idx, multiplicities, intensive, extensive, cell_id, cell_idx, cell_start):
        coalescence = lambda idx, intensive, extensive, gamma, cell_id, cell_idx: self.core.bck.coalescence(n=multiplicities,
                                  volume=self.core.particles['volume'],
                                  idx=idx,
                                  length=len(idx),
                                  intensive=intensive,
                                  extensive=extensive,
                                  gamma=gamma,
                                  healthy=self.core.particles._Particles__healthy_memory,
                                  adaptive=False,
                                  cell_id=cell_id,
                                  cell_idx=cell_idx,
                                  subs=self.__n_substep,
                                  adaptive_memory=self.adaptive_memory,
                                  collision_rate=self.collision_rate,
                                  collision_rate_deficit=self.collision_rate_deficit
                                  )
        normalize = lambda prob: self.core.normalize(prob, self.temp, self.__n_substep)
        pairs_rand, rand = self.rnd_opt.get_random_arrays(0)
        compute_gamma = lambda prob: self.compute_gamma(prob, rand)
        kernel = self.kernel
        counting_sort = lambda cell_id, cell_idx, cell_start, idx: \
            self.core.particles._Particles__cell_caretaker(cell_id, cell_idx, cell_start, idx, len(idx))
        # <listing>
        idx.remove_if(multiplicities, equal=0)
        counting_sort(cell_id, cell_idx, cell_start, idx)
        idx.shuffle(pairs_rand)
        is_first_in_pair.update(
            cell_start,
            cell_idx,
            cell_id
        )
        kernel(prob, is_first_in_pair)
        prob.times_max(self.core.particles['n'], is_first_in_pair)
        normalize(prob)
        compute_gamma(prob)
        coalescence(idx, intensive, extensive, prob, cell_id, cell_idx)
        self.core.particles.attributes['volume'].mark_updated()
        # </listing>

    def compute_gamma(self, prob, rand):
        self.core.backend.compute_gamma(prob, rand)

