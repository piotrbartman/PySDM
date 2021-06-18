class _ThermodynamicStateVariables:
    @staticmethod
    def qv(density_var, energy_var, moisture_var, /, droplet_total_volume, dv):
        raise NotImplementedError()

    @staticmethod
    def T(density_var, energy_var, moisture_var, droplet_total_volume):
        raise NotImplementedError()

    @staticmethod
    def p(density_var, energy_var, moisture_var, droplet_total_volume):
        raise NotImplementedError()

    @staticmethod
    def pv(p, qv):
        raise NotImplementedError()

    @staticmethod
    def dthd_dt(rhod, thd, T, dqv_dt, lv):
        raise NotImplementedError()

    @staticmethod
    def th_dry(th_std, qv):
        raise NotImplementedError()

    @staticmethod
    def rho_d(p, qv, theta_std):
        raise NotImplementedError()

    @staticmethod
    def rho_of_rhod_qv(rhod, qv):
        raise NotImplementedError()

    @staticmethod
    def rhod_of_pd_T(pd, T):
        raise NotImplementedError()
