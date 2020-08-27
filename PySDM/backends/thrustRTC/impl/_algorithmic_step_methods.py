"""
Created at 20.03.2020
"""

import ThrustRTC as trtc
from PySDM.backends.thrustRTC.nice_thrust import nice_thrust
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS


class AlgorithmicStepMethods:

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def amax(row, idx):
        perm_in = trtc.DVPermutation(row.data, idx.data)
        index = trtc.Max_Element(perm_in.range(0, len(row)))
        row_idx = idx[index]
        result = row[row_idx]
        return result

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def amin(row, idx):
        perm_in = trtc.DVPermutation(row.data, idx.data)
        index = trtc.Min_Element(perm_in.range(0, len(row)))
        row_idx = idx[index]
        result = row[row_idx]
        return result

    __cell_id_body = trtc.For(['cell_id', 'cell_origin', 'strides', 'n_dims', 'size'], "i", '''
        cell_id[i] = 0;
        for (int j = 0; j < n_dims; j++) 
        {
            cell_id[i] += cell_origin[size * i + j] * strides[j];
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def cell_id(cell_id, cell_origin, strides):
        n_dims = trtc.DVInt64(strides.shape[1])
        size = trtc.DVInt64(cell_origin.shape[0])
        AlgorithmicStepMethods.__cell_id_body.launch_n(cell_id.size(), [cell_id, cell_origin, strides, n_dims, size])

    __distance_pair_body = trtc.For(['data_out', 'data_in', 'is_first_in_pair'], "i", '''
        if (is_first_in_pair[i]) 
        {
            data_out[i] = abs(data_in[i] - data_in[i + 1]);
        } else {
            data_out[i] = 0;
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def distance_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        perm_in = trtc.DVPermutation(data_in, idx)
        if length > 1:
            AlgorithmicStepMethods.__distance_pair_body.launch_n(length - 1, [data_out, perm_in, is_first_in_pair])

    __find_pairs_body = trtc.For(['cell_start', 'perm_cell_id', 'is_first_in_pair'], "i", '''
        is_first_in_pair[i] = (
            perm_cell_id[i] == perm_cell_id[i+1] &&
            (i - cell_start[perm_cell_id[i]]) % 2 == 0
        );
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def find_pairs(cell_start, is_first_in_pair, cell_id, idx, length):
        perm_cell_id = trtc.DVPermutation(cell_id, idx)
        if length > 1:
            AlgorithmicStepMethods.__find_pairs_body.launch_n(length - 1, [cell_start, perm_cell_id, is_first_in_pair])

    __max_pair_body = trtc.For(['data_out', 'perm_in', 'is_first_in_pair'], "i", '''
        if (is_first_in_pair[i]) 
        {
            data_out[i] = max(perm_in[i], perm_in[i + 1]);
        } else {
            data_out[i] = 0;
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def max_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        perm_in = trtc.DVPermutation(data_in, idx)
        if length > 1:
            AlgorithmicStepMethods.__max_pair_body.launch_n(length - 1, [data_out, perm_in, is_first_in_pair])

    __sort_pair_body = trtc.For(['data_out', 'data_in', 'is_first_in_pair'], "i", '''
        if (is_first_in_pair[i]) {
            if (data_in[i] < data_in[i + 1]) {
                data_out[i] = data_in[i + 1];
                data_out[i + 1] = data_in[i];
            } else {
                data_out[i] = data_in[i];
                data_out[i + 1] = data_in[i + 1];
            }
        } else {
            data_out[i] = 0;
        }
        ''')

    __polynomial_pair_body = trtc.For(
        ['data_out', 'perm_in', 'is_first_in_pair', 'len_0', 'len_1', 'coef_0', 'coef_1', 'pow_0', 'pow_1'], "i", '''
        data_out[i] = 0;
        if (is_first_in_pair[i]) 
        {
            for (int j=0; j < len_0; ++j) {
                data_out[i] += coef_0[j] * pow(data_in[idx[i]], pow_0[j]);
            }
            for (int j=0; j < len_1; ++j) {
                data_out[i] += coef_1[j] * pow(data_in[idx[i + 1]], pow_1[j]);
            }
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def polynomial_pair(data_out, data_in, is_first_in_pair, idx, length, coef_0, coef_1, pow_0, pow_1):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        # TODO
        len_0 = trtc.DVInt64(len(coef_0))
        len_1 = trtc.DVInt64(len(coef_1))
        dcoef_0 = trtc.device_vector_from_dvs(coef_0)
        dcoef_1 = trtc.device_vector_from_dvs(coef_1)
        dpow_0 = trtc.device_vector_from_dvs(pow_0)
        dpow_1 = trtc.device_vector_from_dvs(pow_1)
        perm_in = trtc.DVPermutation(data_in, idx)
        if length > 1:
            AlgorithmicStepMethods.__sum_pair_body.launch_n(
                length - 1,
                [data_out, perm_in, is_first_in_pair, len_0, len_1, dcoef_0, dcoef_1, dpow_0, dpow_1])


    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def sort_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        perm_in = trtc.DVPermutation(data_in, idx)
        trtc.Fill(data_out, trtc.DVDouble(0))
        if length > 1:
            AlgorithmicStepMethods.__sort_pair_body.launch_n(length - 1, [data_out, perm_in, is_first_in_pair])

    __sum_pair_body = trtc.For(['data_out', 'perm_in', 'is_first_in_pair'], "i", '''
        if (is_first_in_pair[i]) 
        {
            data_out[i] = perm_in[i] + perm_in[i + 1];
        } else {
            data_out[i] = 0;
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def sum_pair(data_out, data_in, is_first_in_pair, idx, length):
        # note: silently assumes that data_out is not permuted (i.e. not part of state)
        perm_in = trtc.DVPermutation(data_in, idx)
        if length > 1:
            AlgorithmicStepMethods.__sum_pair_body.launch_n(length - 1, [data_out, perm_in, is_first_in_pair])
