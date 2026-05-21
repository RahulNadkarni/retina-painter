"""Unit tests for src.geometry.build_retinal_cap_mesh."""
import numpy as np
import pytest

from src.geometry import (
    assign_etdrs_subfield,
    build_layered_mesh,
    build_retinal_cap_mesh,
)

# Small mesh keeps exact counts easy to reason about; defaults exercised too.
N_RADIAL, N_ANGULAR = 8, 12


@pytest.fixture(scope="module")
def mesh():
    return build_retinal_cap_mesh(n_radial=N_RADIAL, n_angular=N_ANGULAR)


def test_vertex_count(mesh):
    # 1 fovea vertex + one vertex per (ring, angle)
    assert len(mesh.vertices) == 1 + N_RADIAL * N_ANGULAR
    # fan (n_angular tris) + (n_radial-1) strips of 2*n_angular tris
    assert len(mesh.faces) == N_ANGULAR + (N_RADIAL - 1) * N_ANGULAR * 2


def test_vertex_count_defaults():
    m = build_retinal_cap_mesh()  # 200 x 200
    assert len(m.vertices) == 1 + 200 * 200 == 40001


def test_central_vertex_at_origin(mesh):
    assert np.allclose(mesh.vertices[0], [0.0, 0.0, 0.0])
    # the fovea is also the unique closest vertex to the origin
    dist = np.linalg.norm(mesh.vertices, axis=1)
    assert np.argmin(dist) == 0
    assert (dist[1:] > 0).all()


def test_manifold(mesh):
    # every edge is shared by at most 2 faces => no non-manifold edges
    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    assert counts.max() <= 2
    # consistent face winding (orientable surface)
    assert mesh.is_winding_consistent
    # an open cap is not watertight; its single boundary is the rim ring
    assert not mesh.is_watertight
    assert int((counts == 1).sum()) == N_ANGULAR
    # topological disk
    assert mesh.euler_number == 1


def test_vertices_lie_on_sphere():
    R = 12.0
    m = build_retinal_cap_mesh(radius_mm=R, n_radial=N_RADIAL, n_angular=N_ANGULAR)
    centre = np.array([0.0, 0.0, R])
    d = np.linalg.norm(m.vertices - centre, axis=1)
    assert np.allclose(d, R)


def test_geodesic_extent_matches_cap_diameter():
    R, cap = 12.0, 6.0
    m = build_retinal_cap_mesh(radius_mm=R, cap_diameter_mm=cap,
                               n_radial=N_RADIAL, n_angular=N_ANGULAR)
    # rim vertices' polar angle * R == cap_diameter/2 (geodesic radius)
    rim = m.vertices[1 + (N_RADIAL - 1) * N_ANGULAR:]
    phi = np.arctan2(np.hypot(rim[:, 0], rim[:, 1]), R - rim[:, 2])
    assert np.allclose(phi * R, cap / 2.0)


def test_invalid_params():
    with pytest.raises(ValueError):
        build_retinal_cap_mesh(n_angular=2)
    with pytest.raises(ValueError):
        build_retinal_cap_mesh(cap_diameter_mm=0.0)


# --- assign_etdrs_subfield laterality ---------------------------------------
# subfield indices: 0 central; 1-4 inner sup/nas/inf/tem; 5-8 outer sup/nas/inf/tem
def test_subfield_probe_points():
    # one point per quadrant in the inner ring (r=1.0 mm), plus the fovea
    pts = np.array([
        [0.0, 0.0, 0.0],    # central
        [0.0, 1.0, 0.0],    # +y superior
        [1.0, 0.0, 0.0],    # +x  -> nasal (OD) / temporal (OS)
        [0.0, -1.0, 0.0],   # -y inferior
        [-1.0, 0.0, 0.0],   # -x  -> temporal (OD) / nasal (OS)
    ])
    od = assign_etdrs_subfield(pts, laterality="OD")
    os_ = assign_etdrs_subfield(pts, laterality="OS")
    assert list(od) == [0, 1, 2, 3, 4]    # central, in-sup, in-nasal, in-inf, in-temporal
    assert list(os_) == [0, 1, 4, 3, 2]   # nasal<->temporal swapped; sup/inf unchanged


def test_os_swaps_nasal_temporal_only():
    # n_angular=70 is not divisible by 8, so no vertex lands exactly on a 45-deg
    # quadrant diagonal (where the boundary tie-break isn't reflection-symmetric).
    mesh = build_retinal_cap_mesh(n_radial=24, n_angular=70)
    v = np.asarray(mesh.vertices)
    od = assign_etdrs_subfield(v, laterality="OD")
    os_ = assign_etdrs_subfield(v, laterality="OS")

    # OS swaps nasal<->temporal in both rings (2<->4, 6<->8); everything else same
    swap = {0: 0, 1: 1, 2: 4, 3: 3, 4: 2, 5: 5, 6: 8, 7: 7, 8: 6, -1: -1}
    expected = np.array([swap[i] for i in od])
    assert np.array_equal(os_, expected)

    # explicit: superior/inferior labels are vertex-for-vertex identical...
    for unchanged in (1, 3, 5, 7):
        assert np.array_equal(od == unchanged, os_ == unchanged)
    # ...while nasal and temporal masks are exchanged
    assert np.array_equal(od == 2, os_ == 4) and np.array_equal(od == 4, os_ == 2)
    assert np.array_equal(od == 6, os_ == 8) and np.array_equal(od == 8, os_ == 6)

    # rigorous invariant (holds for *every* vertex, boundaries included):
    # labelling OS equals labelling the x-mirrored mesh as OD
    vm = v.copy(); vm[:, 0] *= -1
    assert np.array_equal(os_, assign_etdrs_subfield(vm, laterality="OD"))


def test_laterality_default_and_validation():
    pts = np.array([[1.0, 0.0, 0.0]])  # +x
    assert assign_etdrs_subfield(pts)[0] == 2          # default OD -> nasal
    assert assign_etdrs_subfield(pts, laterality="os")[0] == 4  # case-insensitive
    with pytest.raises(ValueError):
        assign_etdrs_subfield(pts, laterality="left")
    with pytest.raises(TypeError):  # laterality is keyword-only
        assign_etdrs_subfield(pts, (0.0, 0.0), (0.5, 1.5, 3.0), "OS")


# --- build_layered_mesh -------------------------------------------------------
def test_build_layered_mesh_offsets():
    nr, na = 8, 12
    base = build_retinal_cap_mesh(n_radial=nr, n_angular=na)
    names = ["RNFL", "GCL+IPL"]
    thick = [np.full((nr, na), 40.0), np.full((nr, na), 80.0)]  # microns
    layers = build_layered_mesh(base, thick, names)

    assert list(layers) == names                      # keys, in order
    bv = np.asarray(base.vertices)
    for name in names:
        m = layers[name]
        assert len(m.vertices) == len(bv) and len(m.faces) == len(base.faces)
        assert np.allclose(m.vertices[:, :2], bv[:, :2])   # x, y unchanged
    # cumulative +Z offset: 40 um, then 40+80 = 120 um (-> mm)
    assert np.allclose(layers["RNFL"].vertices[:, 2], bv[:, 2] + 0.040)
    assert np.allclose(layers["GCL+IPL"].vertices[:, 2], bv[:, 2] + 0.120)


def test_build_layered_mesh_validation():
    base = build_retinal_cap_mesh(n_radial=8, n_angular=12)
    with pytest.raises(ValueError):           # mismatched lengths
        build_layered_mesh(base, [np.full((8, 12), 1.0)], ["a", "b"])
    with pytest.raises(ValueError):           # wrong thickness shape
        build_layered_mesh(base, [np.full((5, 5), 1.0)], ["a"])
