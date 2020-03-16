import numpy as np


def closest_approach(p1, p2, p3, p4):
    """Closest approach of two line segments: (p1, p2) and (p3, p4).

    See https://math.stackexchange.com/a/2812513
    """
    R12 = (p2 - p1) @ (p2 - p1)
    R22 = (p4 - p3) @ (p4 - p3)
    D3121 = (p3 - p1) @ (p2 - p1)
    D4121 = (p4 - p1) @ (p2 - p1)
    D4321 = (p4 - p3) @ (p2 - p1)
    D4331 = (p4 - p3) @ (p3 - p1)
    D4332 = (p4 - p3) @ (p3 - p2)

    denom = R12 * R22 + D4321**2
    if denom == 0:
        s = np.inf
        t = np.inf
    else:
        s = (D4321 * D4331 + D3121 * R22) / denom
        t = (D4321 * D3121 - D4331 * R12) / denom

    # print(f's = {s:f}, t={t:f}')
    if 0 <= s <= 1 and 0 <= t <= 1:
        c1 = p1 + s * (p2 - p1)
        c2 = p3 + t * (p4 - p3)
        return np.linalg.norm(c2 - c1)
    # closest point on L1 to p3
    s = np.clip(D3121 / R12, 0, 1)
    c1 = p1 + s * (p2 - p1)
    d1 = np.linalg.norm(c1 - p3)
    # closest point on L1 to p4
    s = np.clip(D4121 / R12, 0, 1)
    c2 = p1 + s * (p2 - p1)
    d2 = np.linalg.norm(c2 - p4)
    # closest point on L2 to p1
    t = np.clip(-D4331 / R22, 0, 1)
    c3 = p3 + t * (p4 - p3)
    d3 = np.linalg.norm(c3 - p1)
    # closest point on L2 to p2
    t = np.clip(-D4332 / R22, 0, 1)
    c4 = p3 + t * (p4 - p3)
    d4 = np.linalg.norm(c4 - p2)

    return min([d1, d2, d3, d4])
