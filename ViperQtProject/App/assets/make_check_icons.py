# -*- coding: utf-8 -*-
# make_check_icons.py — 生成复选框指示器 PNG（未勾=灰框 / 勾选=蓝框+白对勾）
# 纯 Python + zlib 实现，无第三方依赖；4x4 子采样抗锯齿
import os
import struct
import zlib

W = 32
SS = 4  # 子采样


def rounded_inside(x, y, pad, rad):
    x0, y0 = float(pad), float(pad)
    x1, y1 = float(W - pad), float(W - pad)
    if x < x0 or x > x1 or y < y0 or y > y1:
        return False
    cx = min(max(x, x0 + rad), x1 - rad)
    cy = min(max(y, y0 + rad), y1 - rad)
    return (x - cx) ** 2 + (y - cy) ** 2 <= rad * rad


def stroke_inside(x, y, pad, rad, th):
    if rounded_inside(x, y, pad, rad):
        if not rounded_inside(x, y, pad + th, max(0.0, rad - th)):
            return True
    return False


def seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return (wx * wx + wy * wy) ** 0.5
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
    t = c1 / c2
    dx, dy = px - (ax + t * vx), py - (ay + t * vy)
    return (dx * dx + dy * dy) ** 0.5


CHECK_SEGS = (((8.0, 17.0), (14.0, 23.0)), ((14.0, 23.0), (25.0, 10.0)))
CHECK_W = 3.2


def render(checked):
    rows = []
    step = 1.0 / SS
    for py in range(W):
        row = bytearray()
        for px in range(W):
            r = g = b = a = 0
            for sy in range(SS):
                for sx in range(SS):
                    x = px + (sx + 0.5) * step
                    y = py + (sy + 0.5) * step
                    col = None
                    if checked:
                        if rounded_inside(x, y, 2.0, 7.0):
                            col = (42, 109, 168)  # #2a6da8 蓝底
                        for (p1, p2) in CHECK_SEGS:
                            if seg_dist(x, y, p1[0], p1[1], p2[0], p2[1]) <= CHECK_W:
                                col = (255, 255, 255)  # 白对勾
                                break
                    else:
                        if stroke_inside(x, y, 2.0, 7.0, 2.4):
                            col = (122, 130, 140)  # 灰框
                    if col is not None:
                        r += col[0]
                        g += col[1]
                        b += col[2]
                        a += 255
            n = SS * SS
            row += bytes((r // n, g // n, b // n, a // n))
        rows.append(bytes(row))
    return rows


def write_png(path, w, h, rgba_rows):
    def chunk(tag, data):
        out = struct.pack(">I", len(data)) + tag + data
        out += struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return out

    raw = b"".join(b"\x00" + r for r in rgba_rows)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    idat = zlib.compress(raw, 9)
    with open(path, "wb") as f:
        f.write(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    write_png(os.path.join(here, "check_off.png"), W, W, render(False))
    write_png(os.path.join(here, "check_on.png"), W, W, render(True))
    print("icons written:", os.path.join(here, "check_off.png"), os.path.join(here, "check_on.png"))


if __name__ == "__main__":
    main()
