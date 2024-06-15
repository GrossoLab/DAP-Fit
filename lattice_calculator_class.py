import numpy as np


def match_energy(fit_class, coincidence_func_class, raw_data, fit_energies):
    det_idx_matrix = []
    det_vals_matrix = []
    data_idx_matrix = []
    if len(np.asarray(raw_data)) > 0:
        raw_data = list(raw_data)
        for e in fit_energies:
            test_data = fit_class.set(e)
            det_idx, det_vals, data_idx = coincidence_func_class.val(test_data, raw_data)
            det_idx_matrix.append(np.asarray(det_idx))
            det_vals_matrix.append(np.asarray(det_vals))
            data_idx_matrix.append(np.asarray(data_idx))

    det_mat = reshape_detuning(det_vals_matrix, det_idx_matrix, fit_class)
    data_idx_out = reshape_detuning(data_idx_matrix, det_idx_matrix, fit_class)
    return det_mat, data_idx_out

class DAP_Set:
    def __init__(self, dist_array, dielectric=6.93):
        self.dist_array = dist_array
        self.dielectric = dielectric
        self.dap_set = energy_from_distance(dist_array, 0, dielectric=dielectric)

    def set(self, e):
        test_data = self.dap_set + e
        return sorted(test_data, reverse=True)


class Plus_Match:
    def __init__(self, bounds):
        self.bounds = bounds

    def val(self, energy, data):
        data = sorted(data, reverse=True)
        energy_c = energy.copy()
        test_c = data.copy()
        e_idx_full = [j for j, el in enumerate(energy_c)]
        e_idx_full = [e_idx_full[j] for j, el in enumerate(energy_c) if self.bounds[0] < el < self.bounds[1]]
        energy_c = [el for el in energy_c if self.bounds[0] < el < self.bounds[1]]
        t_idx_full = [j for j, el in enumerate(test_c)]
        e_idx_calc = e_idx_full.copy()
        t_idx_calc = t_idx_full.copy()
        t_idx_res = []
        e_idx_res = []
        det_list = []
        a, b = np.meshgrid(energy_c, test_c)
        resarray = np.abs(a - b)
        rej_thresh = max(np.abs(data[0]-energy[-1]), np.abs(data[-1]-energy[0]))
        if resarray.shape[0] > 1 and resarray.shape[1] > 0:
            while np.min(resarray) < rej_thresh:
                mindex = np.unravel_index(resarray.argmin(), resarray.shape)
                det_list.append(resarray[mindex])
                t_idx_res.append(t_idx_calc[mindex[0]])
                e_idx_res.append(e_idx_calc[mindex[1]])
                resarray[mindex[0], :] = rej_thresh
                resarray[:, mindex[1]] = rej_thresh
        return e_idx_res, det_list, t_idx_res

class Checker_Match:
    def __init__(self, bounds):
        self.bounds = bounds

    def val(self, energy, data):
        data = sorted(data, reverse=True)
        energy_c = energy.copy()
        test_c = data.copy()
        # Here's an attempt to check the bounds of what we're doing here. we can turn it off luckily enough.
        e_idx_full = [j for j, el in enumerate(energy_c)]
        e_idx_full = [e_idx_full[j] for j, el in enumerate(energy_c) if self.bounds[0] < el < self.bounds[1]]
        energy_c = [el for el in energy_c if self.bounds[0] < el < self.bounds[1]]
        t_idx_full = [j for j, el in enumerate(test_c)]
        e_idx_calc = e_idx_full.copy()
        t_idx_calc = t_idx_full.copy()
        t_idx_res = []
        e_idx_res = []
        det_list = []
        a, b = np.meshgrid(energy_c, test_c)
        resarray = np.abs(a - b)
        rej_thresh = max(np.abs(data[0] - energy[-1]), np.abs(data[-1] - energy[0]))
        if resarray.shape[0] > 1 and resarray.shape[1] > 0:
            while np.min(resarray) < rej_thresh:
                mindex = np.unravel_index(resarray.argmin(), resarray.shape)
                det_list.append(resarray[mindex])
                t_idx_res.append(t_idx_calc[mindex[0]])     # Perhaps an unnecessary calculation, but not so bad
                e_idx_res.append(e_idx_calc[mindex[1]])
                resarray[:mindex[0], mindex[1]:] = rej_thresh
                resarray[mindex[0]:, :mindex[1]] = rej_thresh
                resarray[mindex[0], :] = rej_thresh
                resarray[:, mindex[1]] = rej_thresh

        return e_idx_res, det_list, t_idx_res



# Reshaping data into NaN matrix for faster calculation in post
def reshape_detuning(det_vals, det_idx, fit_class):
    resarray = np.full([len(det_vals), len(fit_class.dist_array)], np.nan)
    for i, row in enumerate(det_idx):
        for j, col in enumerate(row):
            resarray[i, col] = det_vals[i][j]
    return resarray


def det_calc(det_mat, thresh, transformations=[], norms=[]):
    det_mat[np.isnan(det_mat)] = 0
    det_mat[np.where((det_mat > thresh))] = 0

    for transformation in transformations:
        det_mat = transformation.apply(det_mat)

    det_sum_arr = np.sum(det_mat, axis=1)
    for norm in norms:
        det_sum_arr = norm.apply(det_sum_arr, det_mat)

    return det_sum_arr


def det_calc_squared(det_mat, thresh, transformations):
    det_mat[np.isnan(det_mat)] = 0
    det_mat[np.where((det_mat > thresh))] = 0
    for transformation in transformations:
        det_mat = transformation.apply(det_mat)

    det_mat2 = det_mat.copy()**2        # array operations for speed here
    n_els_arr = np.zeros(det_mat2.shape)
    n_els_arr[det_mat > 0] = 1
    n_els_arr = np.sum(n_els_arr, axis=1)
    n_els_arr[n_els_arr == 0] = 1   # Cant divide zero by zero here
    result = np.sqrt(np.sum(det_mat2, axis=1) / n_els_arr)
    return result


# Coincidence Calculation on Detuning Matrix
def co_calc(det_mat, co_lim, transformations=[], norms=[]):
    det_mat2 = det_mat.copy()
    det_mat2[np.where((det_mat > co_lim))] = np.NaN
    det_mat2[np.isnan(det_mat2)] = np.inf
    det_mat2[np.where((det_mat2 <= co_lim))] = 1
    det_mat2[np.where((det_mat2 != 1))] = 0
    for transformation in transformations:
        det_mat2 = transformation.apply(det_mat2)
    co_arr = np.sum(det_mat2, axis=1)
    for norm in norms:
        co_arr = norm.apply(co_arr, det_mat)
    return co_arr


# Creates energies in eV from series of distances as well as transition energy
# Great for creating synthetic data for testing or for fitting
def energy_from_distance(dist, en, dielectric=6.93):
    e = 1.60217663 * 10 ** (-19)  # Units C
    pi = 3.13159265353  # Unitless
    eps_0 = 8.8541878128 * 10 ** (-12)  # Units C/V*m
    k = e * 10 ** 13 / (4 * pi * eps_0 * dielectric)  # Only one unit of e to keep in eV
    # Times 10^13 because of saved value of distance in 10^-13m (10^-3 A)
    # energy = en + k / dist
    energy = np.zeros(len(dist))
    for i in range(len(dist)):
        energy[i] = en + k / dist[i]
    return energy


def penalty_func(m, a, b):
    return (1 / 2) * (1 - (2 / np.pi) * np.arctan(a * (m - b)))


class Co_Penalty:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def apply(self, mat):
        cols = np.asarray(range(mat.shape[1]))
        cols = penalty_func(cols, self.a, self.b)
        mult_mat = np.zeros((mat.shape[0], mat.shape[1]))
        for row in range(mult_mat.shape[0]):
            mult_mat[row, :] = cols
        return np.multiply(mat, mult_mat)

class Det_Norm:
    def __init__(self, bound=np.inf):
        self.bound = bound
    def apply(self, co_arr, det_mat):
        normarray = np.zeros(det_mat.shape[0])
        for row in range(det_mat.shape[0]):
            normarray[row] = len([el for el in det_mat[row, :] if self.bound > el > 0])
        normarray += 1 # Need this here so we don't divide by zero
        return np.divide(co_arr, normarray)


class Co_Linear:
    def apply(self, mat):
        cols = np.asarray(range(mat.shape[1]))
        cols = (100 - cols)/100
        mult_mat = np.zeros((mat.shape[0], mat.shape[1]))
        for row in range(mult_mat.shape[0]):
            mult_mat[row, :] = cols
        return np.multiply(mat, mult_mat)


class Co_Step:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def apply(self, mat):
        cols = np.asarray(range(mat.shape[1]))
        mult_mat = np.zeros((mat.shape[0], mat.shape[1]))
        for row in range(mult_mat.shape[0]):
            mult_mat[row, :] = cols
        return np.multiply(mat, mult_mat)


class Co_Smooth:
    class Co_Step:
        def __init__(self, smooth, b):
            self.smooth

        def apply(self, mat):
            cols = np.asarray(range(mat.shape[1]))
            mult_mat = np.zeros((mat.shape[0], mat.shape[1]))
            for row in range(mult_mat.shape[0]):
                mult_mat[row, :] = cols
            return np.multiply(mat, mult_mat)

if __name__ == '__main__':
    print("DAP Code")
