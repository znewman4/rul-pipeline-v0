"""Forward-library container, loaders and synthetic measurement generation."""

import numpy as np
import pytest
from scipy.io import savemat

from rul_pipeline.io import (
    ForwardLibrary,
    flatten_response,
    load_forward_library,
    pairs_to_fmc_cube,
    save_forward_library_npz,
)
from rul_pipeline.synthetic import (
    add_measurement_noise,
    measurement_from_library,
    noise_sigma,
    toy_fmc_library,
)

A = np.array([1.0, 2.0, 3.0]) * 1e-3


def _lib(**overrides):
    kwargs = dict(crack_sizes=A, responses=np.random.default_rng(0).normal(size=(3, 4, 4, 10)),
                  dims=("crack", "tx", "rx", "time"), time=np.arange(10) / 25e6, fs=25e6)
    kwargs.update(overrides)
    return ForwardLibrary(**kwargs)


# ----------------------------------------------------------- canonical shape checks


def test_canonical_library_shapes():
    lib = _lib()
    assert lib.n_candidates == 3
    assert lib.measurement_shape == (4, 4, 10)
    assert lib.measurement_dims == ("tx", "rx", "time")
    assert lib.flat_responses().shape == (3, 160)


def test_flatten_response_matches_library_row_order():
    lib = _lib()
    np.testing.assert_array_equal(flatten_response(lib.responses[1], lib), lib.flat_responses()[1])


def test_flatten_response_refuses_transposed_measurement():
    lib = _lib()
    with pytest.raises(ValueError, match="does not match library measurement shape"):
        flatten_response(np.moveaxis(lib.responses[0], -1, 0), lib)


# ----------------------------------------------------------- malformed data


def test_wrong_crack_axis_gives_hint():
    r = np.zeros((4, 10, 3))  # crack axis stored last, MATLAB style
    with pytest.raises(ValueError, match="pass crack_axis=2"):
        ForwardLibrary(crack_sizes=A, responses=r, dims=("crack", "time", "x"))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"crack_sizes": A[::-1].copy()}, "strictly increasing"),
        ({"crack_sizes": np.array([0.0, 1e-3, 2e-3])}, "strictly positive"),
        ({"dims": ("crack", "tx", "time")}, "Name every axis"),
        ({"dims": ("tx", "crack", "rx", "time")}, "dims\\[0\\] must be 'crack'"),
        ({"time": np.arange(9) / 25e6}, "time has 9 samples"),
        ({"fs": 50e6}, "disagrees with 1/dt"),
        ({"responses": np.full((3, 4, 4, 10), np.nan)}, "non-finite"),
        ({"tx": np.array([1, 2])}, "given together"),
    ],
)
def test_malformed_library_raises(overrides, message):
    with pytest.raises(ValueError, match=message):
        _lib(**overrides)


def test_crack_sizes_in_mm_trigger_warning():
    with pytest.warns(UserWarning, match="exported in mm"):
        _lib(crack_sizes=np.array([1.0, 2.0, 3.0]))


# ----------------------------------------------------------- file loaders


def test_npz_round_trip(tmp_path):
    lib = _lib()
    loaded = load_forward_library(save_forward_library_npz(lib, tmp_path / "lib.npz"))
    np.testing.assert_array_equal(loaded.responses, lib.responses)
    np.testing.assert_allclose(loaded.crack_sizes, lib.crack_sizes)
    assert loaded.dims == lib.dims and loaded.fs == lib.fs


def test_mat_v5_with_struct_paths_mm_units_and_crack_axis_last(tmp_path):
    rng = np.random.default_rng(1)
    r_matlab = rng.normal(size=(10, 6, 3))  # (time, pair, crack) as a MATLAB user might stack it
    savemat(tmp_path / "lib.mat", {"lib": {"a_mm": np.array([[1.0], [2.0], [3.0]]), "fmc": r_matlab,
                                           "t": np.arange(10) / 25e6, "tx": [1, 1, 2, 2, 3, 3],
                                           "rx": [1, 2, 1, 2, 1, 2]}})
    lib = load_forward_library(
        tmp_path / "lib.mat",
        key_map={"crack_sizes": "lib.a_mm", "responses": "lib.fmc", "time": "lib.t", "tx": "lib.tx", "rx": "lib.rx"},
        dims=("time", "pair", "crack"), crack_axis=-1, crack_size_unit="mm",
    )
    assert lib.dims == ("crack", "time", "pair")
    np.testing.assert_allclose(lib.crack_sizes, A)
    np.testing.assert_array_equal(lib.responses[2], r_matlab[:, :, 2])
    assert any("moved crack axis" in p for p in lib.metadata["provenance"])


def test_mat_v73_hdf5_dims_restored_to_matlab_order(tmp_path):
    h5py = pytest.importorskip("h5py")
    r = np.random.default_rng(2).normal(size=(3, 4, 5))  # MATLAB order (crack, pair, time)
    with h5py.File(tmp_path / "lib73.mat", "w") as f:  # mimic MATLAB -v7.3 (column-major => reversed)
        for name, val in {"crack_sizes": A[None, :], "responses": r}.items():
            f.create_dataset(name, data=val.T).attrs["MATLAB_class"] = np.bytes_("double")
        f.create_dataset("crack_size_unit", data=np.array([[ord("m")]], dtype=np.uint16)).attrs["MATLAB_class"] = np.bytes_("char")
    lib = load_forward_library(tmp_path / "lib73.mat", dims=("crack", "pair", "time"))
    assert lib.metadata["format"] == "mat-v7.3"
    np.testing.assert_array_equal(lib.responses, r)


def test_missing_variables_and_dims_errors(tmp_path):
    savemat(tmp_path / "bad.mat", {"a": A, "R": np.zeros((3, 5))})
    with pytest.raises(KeyError, match="key_map"):
        load_forward_library(tmp_path / "bad.mat")
    with pytest.raises(ValueError, match="no axis names"):
        load_forward_library(tmp_path / "bad.mat", key_map={"crack_sizes": "a", "responses": "R"})
    with pytest.raises(ValueError, match="Unsupported file type"):
        (tmp_path / "x.csv").write_text("1")
        load_forward_library(tmp_path / "x.csv")


def test_pairs_to_fmc_cube_places_columns_explicitly():
    n_t = 7
    tx, rx = np.array([1, 1, 2, 2]), np.array([1, 2, 1, 2])  # 1-based MATLAB indices
    data = np.stack([np.full(n_t, 10 * t + r) for t, r in zip(tx, rx)], axis=1)  # (n_t, n_pairs)
    cube = pairs_to_fmc_cube(data, tx, rx, n_elements=2)
    assert cube.shape == (2, 2, n_t)
    assert cube[1, 0, 0] == 21 and cube[0, 1, 3] == 12


# ----------------------------------------------------------- synthetic measurement


def test_noise_level_specifications_agree():
    y = np.sin(np.linspace(0, 20, 1000))
    rms, peak = np.sqrt(np.mean(y**2)), np.max(np.abs(y))
    assert noise_sigma(y, sigma=0.1) == 0.1
    assert np.isclose(noise_sigma(y, rms_fraction=0.5), 0.5 * rms)
    assert np.isclose(noise_sigma(y, snr_db=20), peak / 10)
    assert np.isclose(noise_sigma(y, snr_db=0, snr_reference="rms"), rms)
    with pytest.raises(ValueError, match="exactly one"):
        noise_sigma(y, sigma=0.1, snr_db=10)


def test_synthetic_measurement_returns_truth_and_noise():
    lib = toy_fmc_library(np.array([1.0, 2.0, 3.0]) * 1e-3, n_elements=8)
    m = measurement_from_library(lib, 1, sigma=0.01, rng=np.random.default_rng(0))
    assert m.y.shape == lib.measurement_shape and m.a_true == pytest.approx(2e-3)
    assert np.std(m.y - m.y_clean) == pytest.approx(0.01, rel=0.05)


def test_bandlimited_noise_is_in_band_and_has_requested_sigma():
    fs, f_c = 25e6, 5e6
    y, s = add_measurement_noise(np.zeros((32, 512)), sigma=0.2, colour="bandlimited", fs=fs, f_c=f_c,
                                 rng=np.random.default_rng(3))
    assert np.isclose(y.std(), 0.2)
    power = np.mean(np.abs(np.fft.rfft(y, axis=-1)) ** 2, axis=0)
    f = np.fft.rfftfreq(512, 1 / fs)
    assert abs(f[np.argmax(power)] - f_c) < 0.5e6
