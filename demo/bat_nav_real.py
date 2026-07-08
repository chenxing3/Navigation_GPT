# """
# Bat-inspired GPT navigation animation — driven by REAL model output.

# Reconstructs one homing trajectory from the feather file using the exact
# 5 m-per-azimuth scheme (delta = 5*[sin, cos] of the bearing), colours the
# trail + sizes the uncertainty halo by the REAL per-token entropy, and marks
# the real recalibration SLs (where LOC tokens fire). One SL gets a schematic
# magnifier that shows the echo check as a variable-length wander ending in a
# single '!'. No real map is available, so the backdrop is neutral — overlay
# the real Hula reconstruction with ax.imshow(img, extent=[0,W,0,H]) for the
# canal / landscapes.

# Change ROW to animate a different trajectory. Origin is SE, roost is NW
# (as in the real data).
# """
# import numpy as np, pandas as pd, re, math, os
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from matplotlib.path import Path
# from matplotlib.patches import PathPatch, Circle, Rectangle
# from matplotlib.transforms import Affine2D
# from matplotlib.collections import LineCollection
# from matplotlib.animation import FuncAnimation, PillowWriter

# plt.rcParams["font.family"] = ["Arial"]
# FEATHER = '../dataset/trajectory/full_model/homing_trajectory/trajectories_iterative_with_echo_20260416_153756.feather'
# ROW = 9
# MARKER = "bat"            # 'arrow' or 'bat'
# LN2 = math.log(2)

# # ----------------------------------------------------------- reconstruct real path
# def reconstruct(row):
#     x, y = row["loc1_x"], row["loc1_y"]
#     toks = str(row["full_generated_sequence"]).split()
#     ents = [float(e) for e in str(row["full_sequence_entropies"]).split(",") if e.strip()]
#     mids = list(dict.fromkeys(re.findall(r"LOC_\d+", str(row["middle_locations_texts"]))))
#     n = min(len(toks), len(ents))
#     path = [(x, y)]; sent = []; sl = {}
#     for tok, e in zip(toks[:n], ents[:n]):
#         if tok.startswith("["):
#             m = re.search(r"LOC_\d+", tok)
#             if m and m.group() in mids and m.group() not in sl:
#                 sl[m.group()] = len(path) - 1
#         else:
#             try: az = math.radians(float(tok))
#             except ValueError: continue
#             x += 5.0 * math.sin(az); y += 5.0 * math.cos(az)
#             path.append((x, y)); sent.append(e / LN2)
#     path = np.array(path)
#     pent = np.concatenate([[sent[0]], sent]) if sent else np.zeros(len(path))
#     return path, pent, [sl[m] for m in mids if m in sl], np.array([row["loc2_x"], row["loc2_y"]])

# df = pd.read_feather(FEATHER)
# PATH_ABS, PENT, SL_STEPS, DEST_ABS = reconstruct(df.iloc[ROW])

# # local coords (shift so map starts near 0), portrait
# PAD = 400.0
# x0, y0 = PATH_ABS[:, 0].min() - PAD, PATH_ABS[:, 1].min() - PAD
# PATH = PATH_ABS - [x0, y0]
# DEST = DEST_ABS - [x0, y0]
# ORIG = PATH[0]
# W = np.ptp(PATH_ABS[:, 0]) + 2 * PAD
# H = np.ptp(PATH_ABS[:, 1]) + 2 * PAD

# # downsample to ~190 flight frames, keep SL alignment
# L = len(PATH); STRIDE = max(1, L // 190)
# idx = np.arange(0, L, STRIDE)
# if idx[-1] != L - 1: idx = np.append(idx, L - 1)
# PDS = PATH[idx]
# # smooth the entropy (edge-padded moving average) for both trail colour and halo
# def _smooth(a, w=13):
#     if w < 2 or len(a) < w: return a
#     p = w // 2
#     return np.convolve(np.pad(a, p, mode="edge"), np.ones(w) / w, mode="same")[p:p + len(a)]
# EDS = _smooth(PENT[idx], 13)
# EHS = EDS
# SL_DS = [int(np.argmin(np.abs(idx - s))) for s in SL_STEPS]
# NF = len(PDS)
# ENT_LO, ENT_HI = float(np.percentile(PENT, 5)), float(np.percentile(PENT, 95))

# # ----------------------------------------------------------- glyphs
# def closed(v):
#     v = list(v)
#     return Path(v + [v[0]], [Path.MOVETO] + [Path.LINETO] * (len(v) - 1) + [Path.CLOSEPOLY])
# def bat_path():
#     r = [(0.00, 0.98), (0.10, 0.55), (0.32, 0.62), (0.96, 0.50), (0.58, 0.14),
#          (0.40, 0.00), (0.24, -0.18), (0.12, -0.40), (0.00, -0.24)]
#     return closed(r + [(-a, b) for (a, b) in reversed(r[1:-1])])
# def arrow_path():
#     return closed([(0.0, 1.0), (0.60, -0.35), (0.0, -0.02), (-0.60, -0.35)])
# GLYPH = bat_path() if MARKER == "bat" else arrow_path()
# GSIZE = 0.03 * min(W, H)                      # marker scale relative to map
# def place(patch, xy, heading, ax, scale=1.0):
#     h = heading if np.linalg.norm(heading) > 1e-6 else np.array([0.0, 1.0])
#     th = np.arctan2(h[1], h[0]) - np.pi / 2
#     patch.set_transform(Affine2D().scale(GSIZE * scale).rotate(th).translate(*xy) + ax.transData)

# # ----------------------------------------------------------- inside-SL schematic (variable, one '!')
# def inside_path():
#     rp = np.random.default_rng(21)
#     n = int(rp.integers(6, 10))
#     pos = np.array([rp.uniform(-.5, -.2), rp.uniform(-.5, -.2)]); pts = [pos.copy()]
#     ang = rp.uniform(0, 2 * np.pi)
#     for _ in range(n - 1):
#         ang += rp.uniform(-1.1, 1.1)
#         nxt = pos + rp.uniform(.2, .34) * np.array([np.cos(ang), np.sin(ang)])
#         if np.linalg.norm(nxt) > .66:
#             ang += np.pi; nxt = pos + .26 * np.array([np.cos(ang), np.sin(ang)])
#         pos = nxt; pts.append(pos.copy())
#     return np.array(pts)
# INUV = inside_path(); NST = len(INUV)
# CONF = list(np.linspace(0.16, 0.46, NST - 1)) + [0.72]     # only last >= 0.5 -> single '!'

# # ----------------------------------------------------------- choreography
# PING_FR, HOLD = 7, 3
# R_PING = 0.09 * min(W, H)
# STEP_FR, STEP_HOLD = 6, 5

# def base(f, **kw):
#     hd = PDS[f] - PDS[f - 1] if f > 0 else np.array([0.0, 1.0])
#     s = dict(xy=PDS[f], heading=hd, trail=f, ent=EHS[f],
#              ping_r=0.0, ping_a=0.0, bang=False, fail=False,
#              mag=False, mstep=0, reveal=False, mconf=0.0, mring=0.0, status="en route")
#     s.update(kw); return s

# def ping(f, e0, ok=True):
#     fr = []; hd = PDS[f] - PDS[f - 1]
#     for j in range(PING_FR):
#         t = j / (PING_FR - 1)
#         fr.append(base(f, heading=hd, ping_r=R_PING * t, ping_a=0.55 * (1 - t),
#                        ent=e0, bang=(ok and t > .45), status="recalibrating"))
#     for _ in range(HOLD):
#         fr.append(base(f, heading=hd, ent=e0, bang=ok, status="recalibrating"))
#     return fr

# def magnifier(f):
#     fr = []; hd = PDS[f] - PDS[f - 1]
#     fr += ping(f, EHS[f], ok=True)[:PING_FR]
#     for step in range(NST):
#         for j in range(STEP_FR):
#             t = j / (STEP_FR - 1)
#             fr.append(base(f, heading=hd, ent=EHS[f], mag=True, mstep=step, mconf=CONF[step],
#                            mring=t, reveal=(t > .5), status="Eho check at sensory location"))
#         for _ in range(STEP_HOLD):
#             fr.append(base(f, heading=hd, ent=EHS[f], mag=True, mstep=step, mconf=CONF[step],
#                            reveal=True, status="Echo check at sensory location"))
#     for _ in range(HOLD + 3):
#         fr.append(base(f, heading=hd, ent=EHS[f], mag=True, mstep=NST - 1, mconf=CONF[-1],
#                        reveal=True, bang=True, status="localized"))
#     return fr

# frames = []
# mag_sl = SL_DS[-1] if SL_DS else -1        # last SL gets the magnifier
# for f in range(NF):
#     frames.append(base(f))
#     if f in SL_DS:
#         if f == mag_sl:
#             frames += magnifier(f)
#         else:
#             frames += ping(f, EHS[f], ok=True)
# for _ in range(12):
#     frames.append(base(NF - 1, ent=EHS[-1], status="ARRIVED at roost"))

# # ================================================================ figure
# FW = 5.9; fig, ax = plt.subplots(figsize=(FW, FW * H / W * 1.02), dpi=100)
# fig.patch.set_facecolor("#101216")
# ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
# ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#565952")

# pc = np.random.default_rng(3)
# for _ in range(int(W * H / 1.6e6) + 20):
#     x, y = pc.uniform(0, W * .9), pc.uniform(0, H * .9)
#     w, h = pc.uniform(.10, .22) * W, pc.uniform(.06, .14) * H; g = pc.uniform(.30, .42)
#     ax.add_patch(Rectangle((x, y), w, h, facecolor=(g, g, g - .02),
#                            edgecolor="#3f423d", lw=.6, alpha=.5, zorder=0))
# # rural settlement, below the roost (top-left), fully inside the map
# rs = DEST + np.array([0.0, -H * .08])
# ax.add_patch(Rectangle((rs[0] - W * .10, rs[1] - H * .04), W * .22, H * .08,
#                         facecolor="#6f6a63", edgecolor="#4a4640", lw=.8, alpha=.7, zorder=0))
# # ax.text(rs[0], rs[1] + H * .05, "rural settlement", color="#e9e4da",
# #         fontsize=9, alpha=.85, ha="center")

# ax.text(rs[0], rs[1] + H * .01, "Rural settlement", color="#e9e4da",
#         fontsize=13, alpha=.85, ha="center") # Increased to 13
# # # water canal — to the right (east) of roost, origin and the SLs
# # cx = min(max(ORIG[0], DEST[0], PDS[SL_DS[-1]][0]) + 0.06 * W, 0.94 * W)
# # cy = np.linspace(0.02 * H, 0.985 * H, 120)
# # cxw = cx + 0.018 * W * np.sin(np.linspace(0.4, 3.2, 120))
# # ax.plot(cxw, cy, color="white", lw=5, alpha=.9, solid_capstyle="round", zorder=1)
# # ax.text(cx - 0.035 * W, 0.5 * H, "water canal", color="white", fontsize=9,
# #         alpha=.85, rotation=83, ha="center", va="center")





# # water canal — straight line to SL, arc, then straight north
# start_x = ORIG[0] + 0.03 * W
# start_y = ORIG[1] - 0.05 * H

# # Anchor the bend slightly to the right of the last SL
# bend_idx = SL_DS[-1] if len(SL_DS) > 0 else len(PDS) // 2
# bend_x = PDS[bend_idx][0] + 0.08 * W
# bend_y = PDS[bend_idx][1]

# # End point (straight north from the bend)
# end_x = bend_x
# end_y = DEST[1] + 0.1 * H

# # Create the arc (rounded corner) using a localized Bezier curve
# r = 0.08 * H  # Radius of the arc bend

# # Calculate the vector for the first straight segment (Start -> Bend)
# dx, dy = bend_x - start_x, bend_y - start_y
# dist = np.hypot(dx, dy)
# ux, uy = dx / dist, dy / dist

# # Define where the arc starts (backing up from the bend)
# arc_start_x = bend_x - ux * r
# arc_start_y = bend_y - uy * r

# # Define where the arc ends (moving straight north from the bend)
# arc_end_x = bend_x
# arc_end_y = bend_y + r

# # Generate 40 smooth points for the corner arc
# t = np.linspace(0, 1, 40)
# arc_x = (1 - t)**2 * arc_start_x + 2 * (1 - t) * t * bend_x + t**2 * arc_end_x
# arc_y = (1 - t)**2 * arc_start_y + 2 * (1 - t) * t * bend_y + t**2 * arc_end_y

# # Stitch it all together: Start point -> Arc points -> End point
# cxw = np.concatenate(([start_x], arc_x, [end_x]))
# cy = np.concatenate(([start_y], arc_y, [end_y]))

# # Draw the canal
# ax.plot(cxw, cy, color="white", lw=5, alpha=.9, solid_capstyle="round", zorder=1)

# # Place text dynamically on the straight north segment
# ax.text(end_x - 0.035 * W, bend_y + 0.15 * H, "water canal", color="white", fontsize=14,
#         alpha=.85, rotation=90, ha="center", va="center")

# # --- ADD CROP FIELD LABELS ---
# # Position 1: Near the first part of the flight (Seg 0)
# seg0_idx = SL_DS[0] // 2 if len(SL_DS) > 0 else len(PDS) // 4
# ax.text(PDS[seg0_idx][0] + 0.12 * W, PDS[seg0_idx][1], "crop field", 
#         color="#b0c4a3", fontsize=14, alpha=0.6, rotation=15, ha="center")

# # # Position 2: Near the final stretch heading North (Seg 2)
# # seg2_idx = SL_DS[-1] + (len(PDS) - SL_DS[-1]) // 2 if len(SL_DS) > 0 else 3 * len(PDS) // 4
# # ax.text(PDS[seg2_idx][0] - 0.7 * W, PDS[seg2_idx][1], "crop field", 
# #         color="#b0c4a3", fontsize=14, alpha=0.6, rotation=-10, ha="center")



# def sonar(xy, s=1.0, z=3):
#     for rr, a in [(.9, .9), (1.6, .55), (2.3, .3)]:
#         ax.add_patch(Circle(xy, rr * .012 * min(W, H), fill=False, ec="#5ad1ff",
#                             lw=1.3, alpha=a, zorder=z))
#     ax.add_patch(Circle(xy, .003 * min(W, H), color="#5ad1ff", zorder=z))
# for f in SL_DS:
#     sonar(PDS[f])

# ax.scatter(*ORIG, marker="^", s=210, color="#33cc55", edgecolor="white", lw=1.0, zorder=6)
# ax.text(ORIG[0] - W * .05, ORIG[1], "origin", color="#bff0c6", fontsize=16, va="center", ha="right")
# ax.scatter(*DEST, marker="v", s=220, color="#ff4d4d", edgecolor="white", lw=1.0, zorder=6)
# ax.text(DEST[0] + W * .05, DEST[1], "Roost", color="#ffc2c2", fontsize=16, va="center")
# ax.set_title("Bat-inspired GPT — a real homing flight",
#              color="#eef1f5", fontsize=16, pad=6)

# trail = LineCollection([], cmap="turbo", linewidth=3.0, zorder=4)
# trail.set_clim(ENT_LO, ENT_HI); ax.add_collection(trail)
# halo  = Circle((0, 0), 0, color="#ff8a3d", alpha=0.16, zorder=3); ax.add_patch(halo)
# ring1 = Circle((0, 0), 0, fill=False, ec="#5ad1ff", lw=2.0, zorder=6); ax.add_patch(ring1)
# ring2 = Circle((0, 0), 0, fill=False, ec="#5ad1ff", lw=1.2, zorder=6); ax.add_patch(ring2)
# glyph = PathPatch(GLYPH, facecolor="#0f1116", edgecolor="white", lw=1.0, zorder=8); ax.add_patch(glyph)
# bang  = ax.text(0, 0, "", fontsize=22, fontweight="bold", ha="center", va="center", zorder=9)
# hud   = ax.text(0.5, 0.022, "", transform=ax.transAxes, ha="center", va="bottom",
#                 color="#e7eaef", fontsize=12, family="monospace", linespacing=1.5,
#                 bbox=dict(boxstyle="round,pad=0.35", fc="#1b1e24", ec="none", alpha=.8))
# cb = fig.colorbar(trail, ax=ax, fraction=0.036, pad=0.015)
# cb.set_label("uncertainty (next-token entropy, bits)", color="#aeb6c2", fontsize=16)
# cb.ax.tick_params(colors="#aeb6c2", labelsize=10); cb.outline.set_edgecolor("#2a2f38")

# # magnifier in the open corner farthest from the path centroid
# cen = PDS.mean(0)
# corners = np.array([[W*.74, H*.74], [W*.26, H*.26], [W*.74, H*.26], [W*.26, H*.74]])
# INS_C = corners[np.argmax(np.linalg.norm(corners - cen, axis=1))]
# INS_R = 0.22 * min(W, H)
# def Lc(u, v): return INS_C + np.array([u, v]) * INS_R
# mag_bg = Circle(tuple(INS_C), INS_R, facecolor="#0e1117", edgecolor="#5ad1ff", lw=2.2, alpha=0, zorder=10)
# ax.add_patch(mag_bg)
# conn = [ax.plot([], [], color="#5ad1ff", lw=.9, alpha=0, zorder=9)[0] for _ in range(2)]
# mag_zone = Circle(tuple(Lc(0, 0)), INS_R * .82, fill=False, ec="#9fb2c9", ls="--", lw=1.2,
#                   alpha=0, zorder=11); ax.add_patch(mag_zone)
# INXY = np.array([Lc(u, v) for u, v in INUV])
# inside_line, = ax.plot([], [], color="#cfd7e3", lw=1.4, ls=":", alpha=0, zorder=11)
# mag_ring = Circle((0, 0), 0, fill=False, ec="#5ad1ff", lw=1.6, alpha=0, zorder=12); ax.add_patch(mag_ring)
# mag_glyph = PathPatch(GLYPH, facecolor="#0f1116", edgecolor="white", lw=.8, alpha=0, zorder=13)
# ax.add_patch(mag_glyph)
# verdicts = [ax.text(*INXY[i], "", fontsize=23, fontweight="bold", ha="right",
#                     va="center", zorder=13, alpha=0) for i in range(NST)]
# mag_title = ax.text(INS_C[0], INS_C[1] + INS_R * 1.03, "", color="#5ad1ff",
#                     fontsize=10, fontweight="bold", ha="center", alpha=0, zorder=13)
# mag_txt = ax.text(INS_C[0], INS_C[1] - INS_R * .92, "", color="#e7eaef",
#                   fontsize=12, ha="center", va="top", family="monospace", alpha=0, zorder=13)
# cbar_bg = Rectangle((INS_C[0] - INS_R * .5, INS_C[1] - INS_R * .7), INS_R, INS_R * .1,
#                     facecolor="#2a2f38", alpha=0, zorder=13)
# cbar_fg = Rectangle((INS_C[0] - INS_R * .5, INS_C[1] - INS_R * .7), 0, INS_R * .1,
#                     facecolor="#3ddc84", alpha=0, zorder=14)
# ax.add_patch(cbar_bg); ax.add_patch(cbar_fg)
# MAG = [mag_bg, mag_zone, inside_line, mag_ring, mag_glyph, mag_title, mag_txt,
#        cbar_bg, cbar_fg] + conn + verdicts

# def mag_alpha(a):
#     mag_bg.set_alpha(min(a, .95))
#     for art in [mag_zone, mag_title, mag_txt, cbar_bg]: art.set_alpha(a)
#     for c in conn: c.set_alpha(a * .7)

# def animate(i):
#     s = frames[i]; k = s["trail"]
#     if k > 1:
#         pts = PDS[:k + 1].reshape(-1, 1, 2)
#         trail.set_segments(np.concatenate([pts[:-1], pts[1:]], axis=1))
#         trail.set_array(EDS[:k])
#     place(glyph, s["xy"], s["heading"], ax)
#     halo.center = tuple(s["xy"])
#     halo.set_radius(.02 * min(W, H) + .10 * min(W, H) * (s["ent"] - ENT_LO) / (ENT_HI - ENT_LO + 1e-9))
#     halo.set_color("#ff7a33")
#     ring1.center = ring2.center = tuple(s["xy"])
#     ring1.set_radius(s["ping_r"]); ring1.set_alpha(s["ping_a"])
#     ring2.set_radius(s["ping_r"] * .6); ring2.set_alpha(s["ping_a"])
#     if s["bang"]:
#         bang.set_position((s["xy"][0], s["xy"][1] + .05 * H)); bang.set_text("!"); bang.set_color("#3ddc84")
#     else:
#         bang.set_text("")
#     d = np.linalg.norm(s["xy"] - DEST)
#     hud.set_text(f"{s['status']}\ndist to roost: {d:5.0f} m")

#     if s["mag"]:
#         mag_alpha(.95)
#         base_xy = PDS[mag_sl]
#         for c, ang in zip(conn, [1.05, 1.35]):
#             p = Lc(np.cos(np.pi * ang) * .85, np.sin(np.pi * ang) * .85)
#             c.set_data([base_xy[0], p[0]], [base_xy[1], p[1]])
#         nshow = s["mstep"] + (1 if s["reveal"] else 0)
#         inside_line.set_data(INXY[:max(1, nshow), 0], INXY[:max(1, nshow), 1]); inside_line.set_alpha(.8)
#         bxy = INXY[s["mstep"]]
#         place(mag_glyph, bxy, [0, 1], ax, scale=.85); mag_glyph.set_alpha(1)
#         mag_ring.center = tuple(bxy); mag_ring.set_radius(INS_R * .33 * s["mring"])
#         mag_ring.set_alpha(.55 * (1 - s["mring"]))
#         for i2, vt in enumerate(verdicts):
#             if i2 < nshow:
#                 ok = CONF[i2] >= .5
#                 vt.set_text("!" if ok else "?"); vt.set_color("#3ddc84" if ok else "#ffb03a")
#                 vt.set_alpha(1); vt.set_position((INXY[i2][0] + INS_R * .07, INXY[i2][1] + INS_R * .07))
#             else:
#                 vt.set_alpha(0)
#         mag_title.set_text("Inside the sensory location"); mag_title.set_alpha(1)
#         mag_txt.set_text(f"confidence {s['mconf']:.2f}"); mag_txt.set_alpha(1)
#         cbar_bg.set_alpha(1); cbar_fg.set_alpha(1)
#         cbar_fg.set_width(INS_R * s["mconf"]); cbar_fg.set_color("#3ddc84" if s["mconf"] >= .5 else "#ffb03a")
#     else:
#         mag_alpha(0); mag_glyph.set_alpha(0); mag_ring.set_alpha(0); inside_line.set_alpha(0)
#         cbar_fg.set_alpha(0)
#         for vt in verdicts: vt.set_alpha(0)
#     return [trail, glyph, halo, ring1, ring2, bang, hud] + MAG

# anim = FuncAnimation(fig, animate, frames=len(frames), interval=55, blit=False)

# if os.environ.get("PREVIEW") == "1":
#     print(f"W={W:.0f} H={H:.0f} NF={NF} frames={len(frames)} SL_DS={SL_DS} magSL={mag_sl}")
#     print(f"INS_C={INS_C.round(0)} INS_R={INS_R:.0f} ENT[{ENT_LO:.2f},{ENT_HI:.2f}]")
#     for tag, i in [("fly", NF // 2), ("mag", next(j for j, s in enumerate(frames)
#                                                   if s['mag'] and s['mstep'] == NST - 1 and s['reveal']))]:
#         animate(i); fig.savefig(f"./preview_{tag}.png", dpi=110, facecolor=fig.get_facecolor())
#         print("preview", tag, i)
# else:
#     out = "./bat_nav_real.gif"
#     anim.save(out, writer=PillowWriter(fps=18)); print("frames:", len(frames), "->", out)








"""
Bat-inspired GPT navigation animation — driven by REAL model output.

Reconstructs one homing trajectory from the feather file using the exact
5 m-per-azimuth scheme (delta = 5*[sin, cos] of the bearing), colours the
trail + sizes the uncertainty halo by the REAL per-token entropy, and marks
the real recalibration SLs (where LOC tokens fire). One SL gets a schematic
magnifier that shows the echo check as a variable-length wander ending in a
single '!'. No real map is available, so the backdrop is neutral — overlay
the real Hula reconstruction with ax.imshow(img, extent=[0,W,0,H]) for the
canal / landscapes.

Change ROW to animate a different trajectory. Origin is SE, roost is NW
(as in the real data).
"""
import numpy as np, pandas as pd, re, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle, Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["font.family"] = ["Arial"]
FEATHER = '../dataset/trajectory/full_model/homing_trajectory/trajectories_iterative_with_echo_20260416_153756.feather'
ROW = 9
MARKER = "bat"            # 'arrow' or 'bat'
LN2 = math.log(2)

# ----------------------------------------------------------- reconstruct real path
def reconstruct(row):
    x, y = row["loc1_x"], row["loc1_y"]
    toks = str(row["full_generated_sequence"]).split()
    ents = [float(e) for e in str(row["full_sequence_entropies"]).split(",") if e.strip()]
    mids = list(dict.fromkeys(re.findall(r"LOC_\d+", str(row["middle_locations_texts"]))))
    n = min(len(toks), len(ents))
    path = [(x, y)]; sent = []; sl = {}
    for tok, e in zip(toks[:n], ents[:n]):
        if tok.startswith("["):
            m = re.search(r"LOC_\d+", tok)
            if m and m.group() in mids and m.group() not in sl:
                sl[m.group()] = len(path) - 1
        else:
            try: az = math.radians(float(tok))
            except ValueError: continue
            x += 5.0 * math.sin(az); y += 5.0 * math.cos(az)
            path.append((x, y)); sent.append(e / LN2)
    path = np.array(path)
    pent = np.concatenate([[sent[0]], sent]) if sent else np.zeros(len(path))
    return path, pent, [sl[m] for m in mids if m in sl], np.array([row["loc2_x"], row["loc2_y"]])

df = pd.read_feather(FEATHER)
PATH_ABS, PENT, SL_STEPS, DEST_ABS = reconstruct(df.iloc[ROW])

# local coords (shift so map starts near 0), portrait
PAD = 400.0
x0, y0 = PATH_ABS[:, 0].min() - PAD, PATH_ABS[:, 1].min() - PAD
PATH = PATH_ABS - [x0, y0]
DEST = DEST_ABS - [x0, y0]
ORIG = PATH[0]
W = np.ptp(PATH_ABS[:, 0]) + 2 * PAD
H = np.ptp(PATH_ABS[:, 1]) + 2 * PAD

# downsample to ~190 flight frames, keep SL alignment
L = len(PATH); STRIDE = max(1, L // 190)
idx = np.arange(0, L, STRIDE)
if idx[-1] != L - 1: idx = np.append(idx, L - 1)
PDS = PATH[idx]
# smooth the entropy (edge-padded moving average) for both trail colour and halo
def _smooth(a, w=13):
    if w < 2 or len(a) < w: return a
    p = w // 2
    return np.convolve(np.pad(a, p, mode="edge"), np.ones(w) / w, mode="same")[p:p + len(a)]
# EDS = _smooth(PENT[idx], 13)
# EHS = EDS
# SL_DS = [int(np.argmin(np.abs(idx - s))) for s in SL_STEPS]




EDS = _smooth(PENT[idx], 13)
EHS = EDS

NF = len(PDS)

# --- ADD THIS NEW BLOCK ---
# Calculate smooth banking (flipping) to avoid jitter when flying North/South
raw_headings = np.zeros((NF, 2))
raw_headings[1:] = PDS[1:] - PDS[:-1]
raw_headings[0] = np.array([0.0, 1.0])
# 1.0 if right, -1.0 if left
raw_flips = np.where(raw_headings[:, 0] < 0, -1.0, 1.0)
# Smooth it over 15 frames so the flip becomes a graceful 3D-like roll
SMOOTH_FLIPS = _smooth(raw_flips, 15) 
# --------------------------

SL_DS = [int(np.argmin(np.abs(idx - s))) for s in SL_STEPS]






ENT_LO, ENT_HI = float(np.percentile(PENT, 5)), float(np.percentile(PENT, 95))

# ----------------------------------------------------------- glyphs
def closed(v):
    v = list(v)
    return Path(v + [v[0]], [Path.MOVETO] + [Path.LINETO] * (len(v) - 1) + [Path.CLOSEPOLY])



import json # Make sure this is at the top of your script with the other imports

def bat_path():
    """Loads a custom traced bat silhouette from a JSON file."""
    try:
        with open("custom_bat.json", "r") as f:
            r = json.load(f)
        # Convert the JSON lists [x, y] into tuples (x, y)
        r_tuples = [(pt[0], pt[1]) for pt in r]
        return closed(r_tuples)
    except FileNotFoundError:
        print("Warning: custom_bat.json not found! Falling back to a simple triangle.")
        return closed([(0.0, 1.0), (0.5, -0.5), (-0.5, -0.5)])





def arrow_path():
    return closed([(0.0, 1.0), (0.60, -0.35), (0.0, -0.02), (-0.60, -0.35)])

GLYPH = bat_path() if MARKER == "bat" else arrow_path()
GSIZE = 0.03 * min(W, H)                      # marker scale relative to map

# def place(patch, xy, heading, ax, scale=1.0):
#     h = heading if np.linalg.norm(heading) > 1e-6 else np.array([0.0, 1.0])
#     th = np.arctan2(h[1], h[0]) - np.pi / 2
#     patch.set_transform(Affine2D().scale(GSIZE * scale).rotate(th).translate(*xy) + ax.transData)


# def place(patch, xy, heading, ax, scale=1.0):
#     h = heading if np.linalg.norm(heading) > 1e-6 else np.array([0.0, 1.0])
#     th = np.arctan2(h[1], h[0])
    
#     # If the bat is heading left (negative X), flip its Y-axis so it isn't upside down
#     flip_y = -1 if h[0] < 0 else 1
    
#     # Apply the flip as part of the scale, then rotate, then translate
#     patch.set_transform(
#         Affine2D().scale(GSIZE * scale, GSIZE * scale * flip_y).rotate(th).translate(*xy) + ax.transData
#     )



def place(patch, xy, heading, flip_y, ax, scale=1.0):
    h = heading if np.linalg.norm(heading) > 1e-6 else np.array([0.0, 1.0])
    th = np.arctan2(h[1], h[0])
    # Apply the smoothed flip_y to the scale
    patch.set_transform(Affine2D().scale(GSIZE * scale, GSIZE * scale * flip_y).rotate(th).translate(*xy) + ax.transData)

# ----------------------------------------------------------- inside-SL schematic (variable, one '!')
def inside_path():
    rp = np.random.default_rng(21)
    n = int(rp.integers(6, 10))
    pos = np.array([rp.uniform(-.5, -.2), rp.uniform(-.5, -.2)]); pts = [pos.copy()]
    ang = rp.uniform(0, 2 * np.pi)
    for _ in range(n - 1):
        ang += rp.uniform(-1.1, 1.1)
        nxt = pos + rp.uniform(.2, .34) * np.array([np.cos(ang), np.sin(ang)])
        if np.linalg.norm(nxt) > .66:
            ang += np.pi; nxt = pos + .26 * np.array([np.cos(ang), np.sin(ang)])
        pos = nxt; pts.append(pos.copy())
    return np.array(pts)

INUV = inside_path(); NST = len(INUV)
CONF = list(np.linspace(0.16, 0.46, NST - 1)) + [0.72]     # only last >= 0.5 -> single '!'

# ----------------------------------------------------------- choreography
PING_FR, HOLD = 7, 3
R_PING = 0.09 * min(W, H)
STEP_FR, STEP_HOLD = 6, 5

# def base(f, **kw):
#     hd = PDS[f] - PDS[f - 1] if f > 0 else np.array([0.0, 1.0])
#     s = dict(xy=PDS[f], heading=hd, trail=f, ent=EHS[f],
#              ping_r=0.0, ping_a=0.0, bang=False, fail=False,
#              mag=False, mstep=0, reveal=False, mconf=0.0, mring=0.0, status="en route")
#     s.update(kw); return s


def base(f, **kw):
    hd = PDS[f] - PDS[f - 1] if f > 0 else np.array([0.0, 1.0])
    s = dict(xy=PDS[f], heading=hd, trail=f, ent=EHS[f],
             ping_r=0.0, ping_a=0.0, bang=False, fail=False,
             mag=False, mstep=0, reveal=False, mconf=0.0, mring=0.0, status="en route",
             flip=SMOOTH_FLIPS[f]) # <--- ADD THIS LINE
    s.update(kw); return s



def ping(f, e0, ok=True):
    fr = []; hd = PDS[f] - PDS[f - 1]
    for j in range(PING_FR):
        t = j / (PING_FR - 1)
        fr.append(base(f, heading=hd, ping_r=R_PING * t, ping_a=0.55 * (1 - t),
                       ent=e0, bang=(ok and t > .45), status="recalibrating"))
    for _ in range(HOLD):
        fr.append(base(f, heading=hd, ent=e0, bang=ok, status="recalibrating"))
    return fr

def magnifier(f):
    fr = []; hd = PDS[f] - PDS[f - 1]
    fr += ping(f, EHS[f], ok=True)[:PING_FR]
    for step in range(NST):
        for j in range(STEP_FR):
            t = j / (STEP_FR - 1)
            fr.append(base(f, heading=hd, ent=EHS[f], mag=True, mstep=step, mconf=CONF[step],
                           mring=t, reveal=(t > .5), status="Echo check at sensory location"))
        for _ in range(STEP_HOLD):
            fr.append(base(f, heading=hd, ent=EHS[f], mag=True, mstep=step, mconf=CONF[step],
                           reveal=True, status="Echo check at sensory location"))
    for _ in range(HOLD + 3):
        fr.append(base(f, heading=hd, ent=EHS[f], mag=True, mstep=NST - 1, mconf=CONF[-1],
                       reveal=True, bang=True, status="localized"))
    return fr

frames = []
mag_sl = SL_DS[-1] if SL_DS else -1        # last SL gets the magnifier
for f in range(NF):
    frames.append(base(f))
    if f in SL_DS:
        if f == mag_sl:
            frames += magnifier(f)
        else:
            frames += ping(f, EHS[f], ok=True)
for _ in range(12):
    frames.append(base(NF - 1, ent=EHS[-1], status="ARRIVED at roost"))

# ================================================================ figure
FW = 5.9; fig, ax = plt.subplots(figsize=(FW, FW * H / W * 1.02), dpi=100)
fig.patch.set_facecolor("#101216")
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#565952")

pc = np.random.default_rng(3)
for _ in range(int(W * H / 1.6e6) + 20):
    x, y = pc.uniform(0, W * .9), pc.uniform(0, H * .9)
    w, h = pc.uniform(.10, .22) * W, pc.uniform(.06, .14) * H; g = pc.uniform(.30, .42)
    ax.add_patch(Rectangle((x, y), w, h, facecolor=(g, g, g - .02),
                           edgecolor="#3f423d", lw=.6, alpha=.5, zorder=0))

# rural settlement, below the roost (top-left), fully inside the map
rs = DEST + np.array([0.0, -H * .08])
ax.add_patch(Rectangle((rs[0] - W * .10, rs[1] - H * .04), W * .22, H * .08,
                        facecolor="#6f6a63", edgecolor="#4a4640", lw=.8, alpha=.7, zorder=0))

ax.text(rs[0], rs[1] + H * .01, "Rural settlement", color="#e9e4da",
        fontsize=13, alpha=.85, ha="center")

# water canal — straight line to SL, arc, then straight north
start_x = ORIG[0] + 0.03 * W
start_y = ORIG[1] - 0.05 * H

bend_idx = SL_DS[-1] if len(SL_DS) > 0 else len(PDS) // 2
bend_x = PDS[bend_idx][0] + 0.08 * W
bend_y = PDS[bend_idx][1]

end_x = bend_x
end_y = DEST[1] + 0.1 * H

r = 0.08 * H  # Radius of the arc bend

dx, dy = bend_x - start_x, bend_y - start_y
dist = np.hypot(dx, dy)
ux, uy = dx / dist, dy / dist

arc_start_x = bend_x - ux * r
arc_start_y = bend_y - uy * r
arc_end_x = bend_x
arc_end_y = bend_y + r

t = np.linspace(0, 1, 40)
arc_x = (1 - t)**2 * arc_start_x + 2 * (1 - t) * t * bend_x + t**2 * arc_end_x
arc_y = (1 - t)**2 * arc_start_y + 2 * (1 - t) * t * bend_y + t**2 * arc_end_y

cxw = np.concatenate(([start_x], arc_x, [end_x]))
cy = np.concatenate(([start_y], arc_y, [end_y]))

ax.plot(cxw, cy, color="white", lw=5, alpha=.9, solid_capstyle="round", zorder=1)

ax.text(end_x - 0.035 * W, bend_y + 0.15 * H, "water canal", color="white", fontsize=14,
        alpha=.85, rotation=90, ha="center", va="center")

# --- ADD CROP FIELD LABELS ---
seg0_idx = SL_DS[0] // 2 if len(SL_DS) > 0 else len(PDS) // 4
ax.text(PDS[seg0_idx][0] + 0.12 * W, PDS[seg0_idx][1], "crop field", 
        color="#b0c4a3", fontsize=14, alpha=0.6, rotation=15, ha="center")

def sonar(xy, s=1.0, z=3):
    for rr, a in [(.9, .9), (1.6, .55), (2.3, .3)]:
        ax.add_patch(Circle(xy, rr * .012 * min(W, H), fill=False, ec="#5ad1ff",
                            lw=1.3, alpha=a, zorder=z))
    ax.add_patch(Circle(xy, .003 * min(W, H), color="#5ad1ff", zorder=z))

for f in SL_DS:
    sonar(PDS[f])

ax.scatter(*ORIG, marker="^", s=210, color="#33cc55", edgecolor="white", lw=1.0, zorder=6)
ax.text(ORIG[0] - W * .05, ORIG[1], "origin", color="#bff0c6", fontsize=16, va="center", ha="right")
ax.scatter(*DEST, marker="v", s=220, color="#ff4d4d", edgecolor="white", lw=1.0, zorder=6)
ax.text(DEST[0] + W * .05, DEST[1], "Roost", color="#ffc2c2", fontsize=16, va="center")
ax.set_title("Bat-inspired GPT — a homing flight",
             color="#eef1f5", fontsize=16, pad=6)

trail = LineCollection([], cmap="turbo", linewidth=3.0, zorder=4)
trail.set_clim(ENT_LO, ENT_HI); ax.add_collection(trail)
halo  = Circle((0, 0), 0, color="#ff8a3d", alpha=0.16, zorder=3); ax.add_patch(halo)
ring1 = Circle((0, 0), 0, fill=False, ec="#5ad1ff", lw=2.0, zorder=6); ax.add_patch(ring1)
ring2 = Circle((0, 0), 0, fill=False, ec="#5ad1ff", lw=1.2, zorder=6); ax.add_patch(ring2)
glyph = PathPatch(GLYPH, facecolor="#0f1116", edgecolor="white", lw=1.0, zorder=8); ax.add_patch(glyph)
bang  = ax.text(0, 0, "", fontsize=22, fontweight="bold", ha="center", va="center", zorder=9)
hud   = ax.text(0.5, 0.022, "", transform=ax.transAxes, ha="center", va="bottom",
                color="#e7eaef", fontsize=12, family="monospace", linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="#1b1e24", ec="none", alpha=.8))
cb = fig.colorbar(trail, ax=ax, fraction=0.036, pad=0.015)
cb.set_label("uncertainty (next-token entropy, bits)", color="#aeb6c2", fontsize=16)
cb.ax.tick_params(colors="#aeb6c2", labelsize=10); cb.outline.set_edgecolor("#2a2f38")

# magnifier in the open corner farthest from the path centroid
cen = PDS.mean(0)
corners = np.array([[W*.74, H*.74], [W*.26, H*.26], [W*.74, H*.26], [W*.26, H*.74]])
INS_C = corners[np.argmax(np.linalg.norm(corners - cen, axis=1))]
INS_R = 0.22 * min(W, H)
def Lc(u, v): return INS_C + np.array([u, v]) * INS_R

mag_bg = Circle(tuple(INS_C), INS_R, facecolor="#0e1117", edgecolor="#5ad1ff", lw=2.2, alpha=0, zorder=10)
ax.add_patch(mag_bg)
conn = [ax.plot([], [], color="#5ad1ff", lw=.9, alpha=0, zorder=9)[0] for _ in range(2)]
mag_zone = Circle(tuple(Lc(0, 0)), INS_R * .82, fill=False, ec="#9fb2c9", ls="--", lw=1.2,
                  alpha=0, zorder=11); ax.add_patch(mag_zone)
INXY = np.array([Lc(u, v) for u, v in INUV])
inside_line, = ax.plot([], [], color="#cfd7e3", lw=1.4, ls=":", alpha=0, zorder=11)
mag_ring = Circle((0, 0), 0, fill=False, ec="#5ad1ff", lw=1.6, alpha=0, zorder=12); ax.add_patch(mag_ring)
mag_glyph = PathPatch(GLYPH, facecolor="#0f1116", edgecolor="white", lw=.8, alpha=0, zorder=13)
ax.add_patch(mag_glyph)
verdicts = [ax.text(*INXY[i], "", fontsize=23, fontweight="bold", ha="right",
                    va="center", zorder=13, alpha=0) for i in range(NST)]

# ADJUSTED Y-OFFSETS BELOW
mag_title = ax.text(INS_C[0], INS_C[1] + INS_R * 1.15, "", color="#5ad1ff",
                    fontsize=10, fontweight="bold", ha="center", alpha=0, zorder=13)
mag_txt = ax.text(INS_C[0], INS_C[1] - INS_R * 1.10, "", color="#e7eaef",
                  fontsize=12, ha="center", va="top", family="monospace", alpha=0, zorder=13)

cbar_bg = Rectangle((INS_C[0] - INS_R * .5, INS_C[1] - INS_R * .7), INS_R, INS_R * .1,
                    facecolor="#2a2f38", alpha=0, zorder=13)
cbar_fg = Rectangle((INS_C[0] - INS_R * .5, INS_C[1] - INS_R * .7), 0, INS_R * .1,
                    facecolor="#3ddc84", alpha=0, zorder=14)
ax.add_patch(cbar_bg); ax.add_patch(cbar_fg)
MAG = [mag_bg, mag_zone, inside_line, mag_ring, mag_glyph, mag_title, mag_txt,
       cbar_bg, cbar_fg] + conn + verdicts

def mag_alpha(a):
    mag_bg.set_alpha(min(a, .95))
    for art in [mag_zone, mag_title, mag_txt, cbar_bg]: art.set_alpha(a)
    for c in conn: c.set_alpha(a * .7)

def animate(i):
    s = frames[i]; k = s["trail"]
    if k > 1:
        pts = PDS[:k + 1].reshape(-1, 1, 2)
        trail.set_segments(np.concatenate([pts[:-1], pts[1:]], axis=1))
        trail.set_array(EDS[:k])
    # place(glyph, s["xy"], s["heading"], ax)
    place(glyph, s["xy"], s["heading"], s["flip"], ax)



    halo.center = tuple(s["xy"])
    halo.set_radius(.02 * min(W, H) + .10 * min(W, H) * (s["ent"] - ENT_LO) / (ENT_HI - ENT_LO + 1e-9))
    halo.set_color("#ff7a33")
    ring1.center = ring2.center = tuple(s["xy"])
    ring1.set_radius(s["ping_r"]); ring1.set_alpha(s["ping_a"])
    ring2.set_radius(s["ping_r"] * .6); ring2.set_alpha(s["ping_a"])
    if s["bang"]:
        bang.set_position((s["xy"][0], s["xy"][1] + .05 * H)); bang.set_text("!"); bang.set_color("#3ddc84")
    else:
        bang.set_text("")
    d = np.linalg.norm(s["xy"] - DEST)
    hud.set_text(f"{s['status']}\ndist to roost: {d:5.0f} m")

    if s["mag"]:
        mag_alpha(.95)
        base_xy = PDS[mag_sl]
        for c, ang in zip(conn, [1.05, 1.35]):
            p = Lc(np.cos(np.pi * ang) * .85, np.sin(np.pi * ang) * .85)
            c.set_data([base_xy[0], p[0]], [base_xy[1], p[1]])
        nshow = s["mstep"] + (1 if s["reveal"] else 0)
        inside_line.set_data(INXY[:max(1, nshow), 0], INXY[:max(1, nshow), 1]); inside_line.set_alpha(.8)
        # bxy = INXY[s["mstep"]]
        # # place(mag_glyph, bxy, [0, 1], ax, scale=.85); mag_glyph.set_alpha(1)


        # place(mag_glyph, bxy, [0, 1], 1.0, ax, scale=.85); mag_glyph.set_alpha(1)



        bxy = INXY[s["mstep"]]
        # Use the actual trajectory heading and flip instead of [0, 1]
        place(mag_glyph, bxy, [1.0, 0.0], 1.0, ax, scale=.85); mag_glyph.set_alpha(1)




        mag_ring.center = tuple(bxy); mag_ring.set_radius(INS_R * .33 * s["mring"])
        mag_ring.set_alpha(.55 * (1 - s["mring"]))
        for i2, vt in enumerate(verdicts):
            if i2 < nshow:
                ok = CONF[i2] >= .5
                vt.set_text("!" if ok else "?"); vt.set_color("#3ddc84" if ok else "#ffb03a")
                vt.set_alpha(1); vt.set_position((INXY[i2][0] + INS_R * .07, INXY[i2][1] + INS_R * .07))
            else:
                vt.set_alpha(0)
        mag_title.set_text("Inside the sensory location"); mag_title.set_alpha(1)
        mag_txt.set_text(f"confidence {s['mconf']:.2f}"); mag_txt.set_alpha(1)
        cbar_bg.set_alpha(1); cbar_fg.set_alpha(1)
        cbar_fg.set_width(INS_R * s["mconf"]); cbar_fg.set_color("#3ddc84" if s["mconf"] >= .5 else "#ffb03a")
    else:
        mag_alpha(0); mag_glyph.set_alpha(0); mag_ring.set_alpha(0); inside_line.set_alpha(0)
        cbar_fg.set_alpha(0)
        for vt in verdicts: vt.set_alpha(0)
    return [trail, glyph, halo, ring1, ring2, bang, hud] + MAG

anim = FuncAnimation(fig, animate, frames=len(frames), interval=55, blit=False)

if os.environ.get("PREVIEW") == "1":
    print(f"W={W:.0f} H={H:.0f} NF={NF} frames={len(frames)} SL_DS={SL_DS} magSL={mag_sl}")
    print(f"INS_C={INS_C.round(0)} INS_R={INS_R:.0f} ENT[{ENT_LO:.2f},{ENT_HI:.2f}]")
    for tag, i in [("fly", NF // 2), ("mag", next(j for j, s in enumerate(frames)
                                                  if s['mag'] and s['mstep'] == NST - 1 and s['reveal']))]:
        animate(i); fig.savefig(f"./preview_{tag}.png", dpi=110, facecolor=fig.get_facecolor())
        print("preview", tag, i)
else:
    out = "./bat_nav_real.gif"
    anim.save(out, writer=PillowWriter(fps=18)); print("frames:", len(frames), "->", out)