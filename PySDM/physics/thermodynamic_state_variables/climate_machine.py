from PySDM.physics import constants as const
from PySDM.physics.thermodynamic_state_variables._thermodynamic_state_variables import _ThermodynamicStateVariables


class ClimateMachine(_ThermodynamicStateVariables):
    @staticmethod
    def qv(density, energy, moisture, droplet_total_volume, dv):
        raise NotImplementedError()

    @staticmethod
    def T(density, internal_energy, moisture, droplet_total_volume):
        return T_0 + (internal_energy - spec_hum_qv * I_v0) / c_vm

    @staticmethod
    def p(density, energy, moisture, droplet_total_volume):
        raise NotImplementedError()

    @staticmethod
    def pv(p, qv):
        raise NotImplementedError()

    @staticmethod
    def dthd_dt(rhod, thd, T, dqv_dt, lv):
        raise NotImplementedError()

    @staticmethod
    def th_dry(th_std, qv):
        return 0

    @staticmethod
    def density(total_pressure, vapour_mixing_ratio, liquid_mixing_ratio, temperature):
        assert liquid_mixing_ratio == 0
        Rq = const.Rv / (1 / vapour_mixing_ratio + 1) + const.Rd / (1 + vapour_mixing_ratio)
        return total_pressure / Rq / temperature

    @staticmethod
    def rho_of_rhod_qv(rhod, qv):
        raise NotImplementedError()

    @staticmethod
    def rhod_of_pd_T(pd, T):
        raise NotImplementedError()
