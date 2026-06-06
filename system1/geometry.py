import math


def bbox_center_to_bearing(
    u: float,
    v: float,
    img_width: int,
    img_height: int,
    azimuth_deg: float,
    elevation_deg: float,
    hfov_deg: float,
    vfov_deg: float,
) -> tuple[float, float, float]:
    """RTSP path: pixel bbox center + camera orientation → ENU unit bearing vector.

    Uses flat-field approximation; valid for hFOV < ~120°.
    """
    dx = (u - img_width / 2.0) / img_width
    dy = -(v - img_height / 2.0) / img_height  # image y down → elevation y up

    alpha = math.radians(azimuth_deg + dx * hfov_deg)
    phi = math.radians(elevation_deg + dy * vfov_deg)

    E = math.cos(phi) * math.sin(alpha)
    N = math.cos(phi) * math.cos(alpha)
    U = math.sin(phi)

    mag = math.sqrt(E * E + N * N + U * U)
    return (E / mag, N / mag, U / mag)


def _quat_rotate(q: list[float], v: list[float]) -> list[float]:
    """Rotate vector v by unit quaternion q = [x, y, z, w] (Hamilton product).

    Returns the rotated vector as [x, y, z].
    """
    qx, qy, qz, qw = q
    vx, vy, vz = v

    # t = 2 * (q.xyz × v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    return [
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    ]


def sim_bearing(
    center_px: list[float],
    K: list[float],
    rot_q: list[float],
) -> tuple[float, float, float]:
    """Simulation path: pixel + Unity K matrix + world quaternion → ENU unit bearing vector.

    Args:
        center_px: [u, v] pixel coordinates (origin top-left, v increases downward)
        K: flattened row-major 3×3 pinhole matrix [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        rot_q: Unity world rotation quaternion [x, y, z, w]

    Returns:
        (E, N, U) unit vector in ENU frame

    Coordinate frames:
        Unity camera local: +X right, +Y up, +Z forward
        Unity world:        +X east, +Y up, +Z north  (left-handed)
        ENU:                +E east, +N north, +U up  (right-handed)
    """
    u, v = center_px
    fx, cx = K[0], K[2]
    fy, cy = K[4], K[5]

    # Back-project to camera-local direction.
    # Image v increases downward; Unity camera +Y is up → negate y.
    d_cam = [
        (u - cx) / fx,
        -(v - cy) / fy,
        1.0,
    ]
    mag = math.sqrt(d_cam[0] ** 2 + d_cam[1] ** 2 + d_cam[2] ** 2)
    d_cam = [c / mag for c in d_cam]

    # Rotate to Unity world space.
    d_world = _quat_rotate(rot_q, d_cam)

    # Unity world → ENU axis remap: Unity +X→E, +Y→U, +Z→N
    E, U, N = d_world[0], d_world[1], d_world[2]

    mag2 = math.sqrt(E * E + N * N + U * U)
    return (E / mag2, N / mag2, U / mag2)
