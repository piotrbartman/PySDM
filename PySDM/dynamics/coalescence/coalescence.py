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

        Index = lambda length, _: self.core.Index.empty(length)
        PairIndicator = lambda length, _: self.core.PairIndicator(length)
        IndexedStorage = lambda idx, shape, dtype: self.core.IndexedStorage.empty(idx, shape, dtype)
        PairwiseStorage = lambda shape, dtype: self.core.PairwiseStorage.empty(shape, dtype)
        Storage = lambda shape, dtype: self.core.Storage.empty(shape, dtype)
        N_SD = self.core.n_sd
        N_CELL = self.core.mesh.n_cell
        N_IA = 0
        N_IE = 1
        # <listing>
        # model state
        idx = Index(N_SD, int)
        multiplicities = IndexedStorage(idx, N_SD, int)
        intensive = IndexedStorage(idx, (N_IA, N_SD), float)
        extensive = IndexedStorage(idx, (N_IE, N_SD), float)
        volume_view = extensive[0:1, :]

        cell_id = IndexedStorage(idx, N_SD, int)
        cell_idx = Index(N_CELL, int)
        cell_start = Storage(N_CELL + 1, int)

        prob = PairwiseStorage(N_SD // 2, float)
        pair_flag = PairIndicator(N_SD, bool)

        u01 = Storage(N_SD, float)
        # </listing>
        self.pair_flag = pair_flag
        self.idx = idx
        self.multiplicities = multiplicities
        self.intensive = intensive
        self.extensive = extensive
        self.cell_id = cell_id
        self.prob = prob
        self.cell_idx = cell_idx
        self.cell_start = cell_start
        self.u01 = u01
        self.volume_view = volume_view

        self.adaptive_memory = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
        self.subs = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
        self.msub = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))

        if self.croupier is None:
            self.croupier = self.core.backend.default_croupier

        self.collision_rate = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
        self.collision_rate_deficit = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
        self.rnd = self.core.Random(self.core.n_sd, 0)

    def __call__(self):
        if self.enable:
            self.step(0, self.adaptive_memory)

    def step(self, s, adaptive_memory):
        self.idx = self.core.particles._Particles__idx
        assert self.idx.shape == self.core.particles._Particles__idx.shape
        assert self.idx.dtype == self.core.particles._Particles__idx.dtype
        self.multiplicities = self.core.particles['n']
        assert self.intensive.shape == self.core.particles.get_intensive_attrs().shape
        assert self.intensive.dtype == self.core.particles.get_intensive_attrs().dtype
        self.intensive = self.core.particles.get_intensive_attrs()
        assert self.extensive.shape == self.core.particles.get_extensive_attrs().shape
        assert self.extensive.dtype == self.core.particles.get_extensive_attrs().dtype
        self.extensive = self.core.particles.get_extensive_attrs()
        self.cell_id.data = self.core.particles["cell id"].data
        self.__step(self.pair_flag,
                    self.prob,
                    self.idx,
                    self.multiplicities,
                    self.intensive,
                    self.extensive,
                    self.cell_id,
                    self.cell_idx,
                    self.cell_start,
                    self.u01,
                    self.volume_view)

    def __step(self, pair_flag, prob, idx, multiplicities, intensive, extensive, cell_id, cell_idx, cell_start, u01, volume_view):
        update_attributes = lambda ultiplicities, intensive, extensive, volume_view, idx, prob: self.core.bck.coalescence(
                                  multiplicities=multiplicities,
                                  volume=volume_view, #self.core.particles['volume'],
                                  idx=idx,
                                  length=len(idx),
                                  intensive=self.core.particles.get_intensive_attrs(),
                                  extensive=extensive,
                                  gamma=prob,
                                  healthy=self.core.particles._Particles__healthy_memory,
                                  adaptive=False,
                                  cell_id=cell_id,
                                  cell_idx=cell_idx,
                                  subs=self.__n_substep,
                                  adaptive_memory=self.adaptive_memory,
                                  collision_rate=self.collision_rate,
                                  collision_rate_deficit=self.collision_rate_deficit
                                  )
        normalize = lambda prob, dt, dv, cell_id, cell_idx, cell_start: \
            self.core.backend.normalize(prob, cell_id, cell_idx, cell_start, self.temp, dt, dv, self.__n_substep)
        _pairs_rand, _rand = self.rnd_opt.get_random_arrays(0)
        urand = lambda array, _: array.urand(self.rnd)
        compute_gamma = lambda prob, rand: self.compute_gamma(prob, rand)
        coalescence_kernel = lambda prob, pair_flag, volume_view: self.kernel(prob, pair_flag)

        def __shuffle_per_cell(self, cell_start, idx, cell_idx, cell_id, u01):
            idx.shuffle(u01)
            self.core.particles._Particles__cell_caretaker(cell_id, cell_idx, cell_start, idx, len(idx))

        shuffle_per_cell = lambda cell_start, idx, cell_idx, cell_id, u01: __shuffle_per_cell(self, cell_start, idx, cell_idx, cell_id, u01)
        N_SD = self.core.n_sd
        remove_if_equal_0 = lambda idx, multiplicities: idx.remove_if_equal(multiplicities, value=0)
        dt = self.core.env.dt
        dv = self.core.mesh.dv
        flag_pairs = lambda pair_flag, cell_id, cell_idx, cell_start: \
            pair_flag.update(cell_start=cell_start, cell_idx=cell_idx, cell_id=cell_id)
        times_max = lambda prob, multiplicities, pair_flag: prob.times_max(multiplicities, pair_flag)
        volume_view = self.core.particles["volume"]
        # <listing>
        # step 0: removal of super-droplets with zero multiplicity
        remove_if_equal_0(idx,  # i/o
                          multiplicities)  # in

        # step 1: cell-wise shuffling, pair flagging
        urand(u01, N_SD)
        shuffle_per_cell(cell_start,  # out
                         idx,  # i/o
                         cell_idx, cell_id, u01[:N_SD])  # in

        flag_pairs(pair_flag,  # out
                   cell_id, cell_idx, cell_start)  # in

        # step 2: collision probability evaluation
        coalescence_kernel(prob,  # out
                           pair_flag, volume_view)  # in

        times_max(prob,  # i/o
                  multiplicities, pair_flag)  # in

        normalize(prob,  # i/o
                  dt, dv, cell_id, cell_idx, cell_start)  # in

        # step 3: collision triggering and attribute updates
        urand(u01, N_SD // 2)

        compute_gamma(prob,  # i/o
                      u01[:N_SD // 2])  # in

        update_attributes(multiplicities, intensive, extensive,  # i/o
                          volume_view, idx, prob)  # in

        # </listing>

    def compute_gamma(self, prob, rand):
        self.core.backend.compute_gamma(prob, rand)

