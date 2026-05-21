"""Build a studio render template around the retinal cap mesh.

Run headless:

    blender --background --python setup_scene.py

It wipes the default scene, imports ``retinal_cap.obj``, adds a three-point
light rig (key / fill / rim) and a camera at 30 deg elevation framed on the
mesh, configures Cycles (128 samples, 1920x1080), sets a neutral-grey world,
and saves ``retinal_template.blend`` -- all next to this script in ``blender/``.

Light/camera distances and light power scale with the mesh's bounding sphere, so
the rig works regardless of the mesh's physical size (the cap is only ~6 mm).
"""
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

# Paths resolve next to this script (the repo's blender/ directory).
SCRIPT_DIR = Path(__file__).resolve().parent
OBJ_PATH = SCRIPT_DIR / "retinal_cap.obj"
BLEND_PATH = SCRIPT_DIR / "retinal_template.blend"


def clear_scene() -> None:
    """Remove every object from the default startup scene (data-level, so it
    works in --background where operator context is limited)."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def import_cap(path: Path):
    """Import the OBJ and return the imported mesh object.

    Imports with up=Z/forward=Y so the mesh keeps its authored orientation
    (cap axis along +Z) rather than OBJ's default Y-up conversion.
    """
    if not path.exists():
        raise FileNotFoundError(f"Mesh not found: {path}. Build it first with src.geometry.")
    before = set(bpy.data.objects)
    try:
        bpy.ops.wm.obj_import(filepath=str(path), up_axis="Z", forward_axis="Y")
    except AttributeError:  # very old Blender
        bpy.ops.import_scene.obj(filepath=str(path), axis_up="Z", axis_forward="Y")
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not new:
        raise RuntimeError("OBJ import produced no mesh object.")
    return new[0]


def bounding_sphere(obj):
    """World-space (center, radius) of the object's bounding box."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    center = sum(corners, Vector((0, 0, 0))) / len(corners)
    radius = max((c - center).length for c in corners)
    return center, radius


def add_target(center):
    target = bpy.data.objects.new("CamTarget", None)
    target.location = center
    target.empty_display_size = 0.2
    bpy.context.scene.collection.objects.link(target)
    return target


def add_area_light(name, location, target, energy, size):
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.size = size
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    track = obj.constraints.new("TRACK_TO")  # aim the light at the mesh
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    return obj


def spherical(center, radius, az_deg, el_deg):
    """Point at distance `radius` from `center`, azimuth/elevation in degrees."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    return center + Vector((
        radius * math.cos(el) * math.cos(az),
        radius * math.cos(el) * math.sin(az),
        radius * math.sin(el),
    ))


def add_three_point_lights(center, Rs, target):
    """Classic key / fill / rim rig, scaled to the bounding sphere."""
    d = 2.5 * Rs                       # light distance from the mesh
    four_pi_d2 = 4 * math.pi * d * d   # ~ inverse-square falloff normaliser
    # target irradiances (W/m^2) -> power so exposure is scale-independent
    add_area_light("Key",  spherical(center, d, az_deg=-50, el_deg=35), target,
                   energy=12.0 * four_pi_d2, size=1.5 * Rs)
    add_area_light("Fill", spherical(center, d, az_deg=60,  el_deg=15), target,
                   energy=4.0 * four_pi_d2,  size=2.5 * Rs)  # softer, dimmer
    add_area_light("Rim",  spherical(center, d, az_deg=160, el_deg=45), target,
                   energy=9.0 * four_pi_d2,  size=1.0 * Rs)  # back/edge light


def add_camera(center, Rs, target, elevation_deg=30.0, azimuth_deg=-35.0):
    """Camera at the given elevation, distanced so the mesh is framed."""
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50.0
    cam_data.sensor_width = 36.0

    # vertical FOV is the limiting dimension for a near-square object in 16:9
    res_x, res_y = 1920, 1080
    hfov = 2 * math.atan((cam_data.sensor_width / 2) / cam_data.lens)
    vfov = 2 * math.atan(math.tan(hfov / 2) * res_y / res_x)
    dist = Rs / math.sin(vfov / 2) * 1.2   # fit bounding sphere + margin

    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = spherical(center, dist, azimuth_deg, elevation_deg)
    bpy.context.scene.collection.objects.link(cam)
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    bpy.context.scene.camera = cam
    return cam


def configure_render():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100


def set_grey_world(value=0.2):
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    # Worlds always use nodes in Blender 5+ (use_nodes is deprecated even to read);
    # only enable it on older Blender where node_tree may be absent.
    node_tree = world.node_tree
    if node_tree is None:
        world.use_nodes = True
        node_tree = world.node_tree
    bg = node_tree.nodes.get("Background")
    if bg is None:
        bg = node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (value, value, value, 1.0)
    bg.inputs["Strength"].default_value = 1.0


def main():
    clear_scene()
    mesh = import_cap(OBJ_PATH)
    center, Rs = bounding_sphere(mesh)
    print(f"Imported '{mesh.name}': bounding-sphere center={tuple(round(v,3) for v in center)}, radius={Rs:.3f}")

    target = add_target(center)
    add_three_point_lights(center, Rs, target)
    cam = add_camera(center, Rs, target)
    configure_render()
    set_grey_world()

    BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))

    scene = bpy.context.scene
    print("--- scene summary ---")
    print(f"engine={scene.render.engine}, samples={scene.cycles.samples}, "
          f"resolution={scene.render.resolution_x}x{scene.render.resolution_y}")
    print(f"objects: {[o.name for o in bpy.data.objects]}")
    print(f"active camera: {scene.camera.name} at "
          f"{tuple(round(v,2) for v in cam.location)}")
    print(f"Saved blend -> {BLEND_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # surface errors with a non-zero exit in --background
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
