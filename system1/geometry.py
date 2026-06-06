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
    """Pixel bbox center + camera orientation → ENU unit bearing vector.

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
