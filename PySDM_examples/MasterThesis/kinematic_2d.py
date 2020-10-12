"""
Created at 05.10.2020
"""

import matplotlib
import numpy as np
from matplotlib import pylab as plt

from PySDM.products.dynamics.condensation.condensation_timestep import CondensationTimestep
from PySDM.products.state.particle_mean_radius import ParticleMeanRadius
from PySDM.products.state.particles_concentration import AerosolConcentration
from PySDM_examples.ICMW_2012_case_1.demo_plots import _Plot
from PySDM_examples.ICMW_2012_case_1.storage import Storage


class _ImagePlot(_Plot):
    line_args = {'color': 'red', 'alpha': .75, 'linestyle': ':', 'linewidth': 5}

    def __init__(self, grid, size, product):
        super().__init__()
        self.nans = np.full(grid, np.nan)

        self.dx = size[0] / grid[0]
        self.dz = size[1] / grid[1]

        self.xlim = (0, size[0])
        self.zlim = (0, size[1])

        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.zlim)

        self.data = self.nans
        self.cmap = 'YlGnBu'
        self.label = f"{product.description} [{product.unit}]"
        self.scale = product.scale

        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Z [m]')

        self.name = product.name

        self.range = product.range

    def init_plot(self, data):
        self.im = self.ax.imshow(self._transpose(data),
                                 origin='lower',
                                 extent=(*self.xlim, *self.zlim),
                                 cmap=self.cmap,
                                 norm=matplotlib.colors.LogNorm(vmin=self.range[0], vmax=self.range[1])
                                 if self.scale == 'log' else None
                                 )
        plt.colorbar(self.im, ax=self.ax).set_label(self.label)
        if not (self.scale == 'log'):
            self.im.set_clim(vmin=self.range[0], vmax=self.range[1])

    @staticmethod
    def _transpose(data):
        if data is not None:
            return data.T

    def update(self, data, step):
        self.step = step
        data = self._transpose(data)
        if data is not None:
            self.im.set_data(data)
            self.ax.set_title(f"min:{np.amin(data):.4g}    max:{np.amax(data):.4g}    t [min]:{(step - 3600) // 60}")

    def show(self, path=None):
        if path is not None:
            plt.savefig(path + 'plots\\' + f'{(self.step - 3600) // 60}_{self.name}.pdf', format='pdf')
        else:
            plt.show()


def latex_sim(name_1, name_2):
    result = '''
        \\begin{columns}
            \\begin{column}{.5\\textwidth}
'''
    for i in range(21):
        result += '                \\only<' + str(i + 1) + '>{\\pgfimage[width=1.1\\textwidth]{img-local/sim/' + str(
            2 * i) + '_' + name_1 + '}}\n'
    result += \
        '''            \\end{column}
                    \\begin{column}{.5\\textwidth}
        '''
    for i in range(21):
        result += '                \\only<' + str(i + 1) + '>{\\pgfimage[width=1.1\\textwidth]{img-local/sim/' + str(
            2 * i) + '_' + name_2 + '}}\n'

    result += \
        '''         \\end{column}
                \\end{columns}
        '''
    return result


if __name__ == '__main__':
    steps = range(3600, 6001, 120)

    prods = []
    prods.append({'product': AerosolConcentration(None), 'scale': 'lin', 'cmap': 'YlGn', 'range': (0, 150)})
    prods.append({'product': ParticleMeanRadius(), 'scale': 'log', 'cmap': 'YlGnBu', 'range': (0.1, 30)})
    prods.append({'product': CondensationTimestep(), 'scale': 'log', 'cmap': 'magma', 'range': (1 / 32, 1)})

    workdir = 'C:\\Users\\piotr\\PycharmProjects\\PySDM\\PySDM_examples\\ICMW_2012_case_1\\output\\'

    for prod in prods:
        storage = Storage(path=workdir + 'data')
        product = prod['product']
        product.range = prod['range']

        step = 0
        plotter = _ImagePlot((128, 128), (1500, 1500), product)
        plotter.cmap = prod['cmap']
        plotter.scale = prod['scale']
        plotter.init_plot(plotter.nans)

        for step in steps:
            plotter.update(storage.load(step, product.name), step)
            plotter.show(path=workdir)
    # print(latex_sim('dt_cond', 'radius_m1'))
