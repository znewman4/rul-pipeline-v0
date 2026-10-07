function export_forward_library(fname, crack_sizes, fmc_results, array_el_xc, response_unit)
%EXPORT_FORWARD_LIBRARY Write a BristolFE crack-size sweep in the rul_pipeline schema.
%
%   export_forward_library(fname, crack_sizes, fmc_results, array_el_xc, response_unit)
%
%   crack_sizes   : N-vector of crack sizes in METRES, strictly increasing.
%   fmc_results   : 1xN cell; fmc_results{i} is the BristolFE FMC result struct
%                   for crack_sizes(i), e.g. main.doms{1}.res.fmc, with fields
%                     .time       (n_t x 1)        [s]
%                     .time_data  (n_t x n_pairs)  (scattered field only)
%                     .tx, .rx    (n_pairs x 1)    1-based element indices
%   array_el_xc   : n_el x 2 element centre coordinates [x z] in metres.
%   response_unit : char, e.g. 'displacement (m)'.
%
%   Python side:  lib = rul_pipeline.io.load_forward_library(fname)
%
%   PROPOSED SCHEMA (variables at top level of the .mat file):
%     crack_sizes       (N x 1)              double, metres
%     responses         (N x n_t x n_pairs)  double, crack index FIRST
%     dims              {'crack','time','pair'}
%     time              (n_t x 1)            seconds
%     fs                scalar               Hz
%     tx, rx            (n_pairs x 1)        1-based
%     element_positions (n_el x 2)           metres
%     crack_size_unit   'm'
%     response_unit     char
%   Save with -v7 for files < 2 GB (scipy.io.loadmat) or -v7.3 above that (h5py).

N = numel(crack_sizes);
assert(numel(fmc_results) == N, 'One FMC result per crack size required.');
assert(all(diff(crack_sizes) > 0), 'crack_sizes must be strictly increasing.');

time = fmc_results{1}.time(:);
[n_t, n_pairs] = size(fmc_results{1}.time_data);
responses = zeros(N, n_t, n_pairs);
for i = 1:N
    r = fmc_results{i};
    assert(isequal(size(r.time_data), [n_t, n_pairs]), 'All FMC results must share one time base and pair list.');
    assert(isequal(r.tx(:), fmc_results{1}.tx(:)) && isequal(r.rx(:), fmc_results{1}.rx(:)), 'tx/rx order must match.');
    responses(i, :, :) = real(r.time_data);
end

crack_sizes = crack_sizes(:);
dims = {'crack', 'time', 'pair'};
fs = 1 / mean(diff(time));
tx = fmc_results{1}.tx(:);
rx = fmc_results{1}.rx(:);
element_positions = array_el_xc;
crack_size_unit = 'm';

info = whos('responses');
if info.bytes > 1.9e9
    version = '-v7.3';
else
    version = '-v7';
end
save(fname, 'crack_sizes', 'responses', 'dims', 'time', 'fs', 'tx', 'rx', ...
     'element_positions', 'crack_size_unit', 'response_unit', version);
end
