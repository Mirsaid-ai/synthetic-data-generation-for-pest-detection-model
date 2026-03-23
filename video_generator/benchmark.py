"""
benchmark.py
============
Benchmarks video generation performance for pest animation.
Measures frame render time, path generation time, and projects
the compute needed to generate videos at scale.

Does NOT benchmark depth map or floor mask generation.

Usage:
    python benchmark.py

    # Custom benchmark:
    python benchmark.py --image kitchen1.png --mask kitchen1_mask.png \
                        --duration 10 --fps 25 --runs 3

    # Quick estimate only:
    python benchmark.py --quick

Arguments:
    --image       Kitchen image (uses synthetic if not provided)
    --mask        Floor mask PNG (optional)
    --depth       Depth map PNG (optional)
    --duration    Duration of each test video in seconds (default: 10)
    --fps         Frames per second (default: 25)
    --runs        Number of timed runs to average (default: 3)
    --quick       Single short run for a fast estimate
    --output      Save a sample rendered video (optional path)
"""

import argparse
import time
import sys
import os
import math
import random
import platform
import subprocess

import cv2
import numpy as np


# ── Inline the core rendering logic so benchmark is self-contained ────────────
# (Mirrors add_pests_to_kitchen.py — keep in sync if sprites change)

def _draw_mouse(base_size, frame_idx, angle_deg, scale=1.0):
    c = max(16, int(base_size * scale))
    img = np.zeros((c, c, 4), dtype=np.uint8)
    cx, cy = c // 2, c // 2
    r = int(c * 0.22)
    if r < 2: return img
    FUR=(80,80,100,255); BELLY=(130,130,160,255); EAR=(100,60,120,255)
    PINK=(140,90,200,255); EYE=(10,10,10,255); NOSE=(100,80,200,255); TAIL=(90,70,110,255)
    def fc(fx,fy,rad,col):
        if rad<1: return
        cv2.circle(img,(int(fx),int(fy)),int(rad),col[:3],-1,cv2.LINE_AA)
        m=np.zeros(img.shape[:2],dtype=np.uint8)
        cv2.circle(m,(int(fx),int(fy)),int(rad),255,-1,cv2.LINE_AA)
        img[:,:,3]=np.where(m>0,col[3],img[:,:,3])
    def la(p1,p2,col,t): cv2.line(img,p1,p2,col[:3],max(1,t),cv2.LINE_AA)
    ar=math.radians(angle_deg); ca,sa=math.cos(ar),math.sin(ar)
    def rot(dx,dy): return int(cx+dx*ca-dy*sa),int(cy+dx*sa+dy*ca)
    sway=math.sin(frame_idx*0.25)*r*0.8
    tp=[rot(-r*1.1-(i/12)*r*2.0,sway*(i/12)**2) for i in range(13)]
    for i in range(12): la(tp[i],tp[i+1],TAIL,max(1,int((1-i/12)*r*0.18)))
    for t in np.linspace(-0.45,0.45,10):
        bx,by=rot(t*r*1.6,0); fc(bx,by,int(r*(1.0-0.3*abs(t/0.45))),FUR)
    for t in np.linspace(-0.2,0.2,5):
        bx,by=rot(t*r*0.9,0); fc(bx,by,int(r*0.45*(1-0.4*abs(t/0.2))),BELLY)
    for s in (-1,1):
        ex,ey=rot(r*0.55,s*r*0.85); fc(ex,ey,int(r*0.38),EAR); fc(ex,ey,int(r*0.22),PINK)
    fc(*rot(r*0.95,0),int(r*0.68),FUR)
    for s in (-1,1):
        ex,ey=rot(r*1.05,s*r*0.32); fc(ex,ey,int(r*0.13),EYE)
        fc(ex-1,ey-1,max(1,int(r*0.05)),(255,255,255,255))
    fc(*rot(r*1.55,0),int(r*0.12),NOSE)
    wig=math.sin(frame_idx*0.5)*3
    for lx,ly,w_ in [(0.3,0.9,wig),(0.3,-0.9,-wig),(-0.3,0.9,-wig),(-0.3,-0.9,wig)]:
        la(rot(lx*r,ly*r),rot(lx*r,(ly+w_*0.1)*r*1.45),FUR,max(1,int(r*0.18)))
    return img

def _draw_cockroach(base_size, frame_idx, angle_deg, scale=1.0):
    c = max(16, int(base_size * scale))
    img = np.zeros((c, c, 4), dtype=np.uint8)
    cx, cy = c // 2, c // 2
    r = int(c * 0.20)
    if r < 2: return img
    SHELL=(20,45,60,255); LEGS=(15,35,50,255); HEAD=(10,30,45,255)
    STRIPE=(30,60,80,255); EYE=(200,220,255,255); ANT=(10,30,45,255)
    def fc(fx,fy,rad,col):
        if rad<1: return
        cv2.circle(img,(int(fx),int(fy)),int(rad),col[:3],-1,cv2.LINE_AA)
        m=np.zeros(img.shape[:2],dtype=np.uint8)
        cv2.circle(m,(int(fx),int(fy)),int(rad),255,-1,cv2.LINE_AA)
        img[:,:,3]=np.where(m>0,col[3],img[:,:,3])
    def la(p1,p2,col,t): cv2.line(img,p1,p2,col[:3],max(1,t),cv2.LINE_AA)
    ar=math.radians(angle_deg); ca,sa=math.cos(ar),math.sin(ar)
    def rot(dx,dy): return int(cx+dx*ca-dy*sa),int(cy+dx*sa+dy*ca)
    lp=frame_idx*0.6
    for side in (-1,1):
        for lx,ly,po in [(0.1,1.0,0.2),(-0.2,0.9,0.0),(-0.5,0.8,-0.2)]:
            sw=math.sin(lp+po)*0.25*side
            p1=rot(lx*r,side*ly*r); p2=rot((lx-0.5)*r,side*(ly+0.6+sw)*r)
            la(p1,p2,LEGS,max(1,int(r*0.12))); la(p2,rot((lx-0.8)*r,side*(ly+0.9+sw)*r),LEGS,max(1,int(r*0.09)))
    for t in np.linspace(-0.5,0.5,12):
        bx,by=rot(t*r*1.8,0); fc(bx,by,int(r*(1.0-0.35*abs(t/0.5))),SHELL)
    for t in np.linspace(-0.35,0.35,5):
        bx,by=rot(t*r*1.4,0); fc(bx,by,int(r*0.18*(1-0.5*abs(t/0.35))),STRIPE)
    fc(*rot(r*1.1,0),int(r*0.45),HEAD)
    for s in (-1,1): fc(*rot(r*1.25,s*r*0.28),max(1,int(r*0.10)),EYE)
    asw=math.sin(frame_idx*0.3)*r*0.3
    for s in (-1,1):
        la(rot(r*1.4,s*r*0.2),rot(r*2.2,s*(r*0.4+asw*0.5)),ANT,max(1,int(r*0.08)))
        la(rot(r*2.2,s*(r*0.4+asw*0.5)),rot(r*3.0,s*(r*0.5+asw)),ANT,max(1,int(r*0.06)))
    return img

def _overlay(bg, sprite, cx, cy):
    h,w=sprite.shape[:2]; bh,bw=bg.shape[:2]
    x0,y0=cx-w//2,cy-h//2; x1,y1=x0+w,y0+h
    sx0=max(0,-x0); sy0=max(0,-y0)
    sx1=w-max(0,x1-bw); sy1=h-max(0,y1-bh)
    bx0=max(0,x0); by0=max(0,y0)
    bx1=bx0+(sx1-sx0); by1=by0+(sy1-sy0)
    if bx1<=bx0 or by1<=by0: return bg
    roi=bg[by0:by1,bx0:bx1].astype(np.float32)
    sp=sprite[sy0:sy1,sx0:sx1]
    a=sp[:,:,3:4].astype(np.float32)/255.0
    bg[by0:by1,bx0:bx1]=np.clip(roi*(1-a)+sp[:,:,:3].astype(np.float32)*a,0,255).astype(np.uint8)
    return bg

def _shadow(frame, cx, cy, size):
    r=max(2,int(size*0.18)); hr=max(1,int(r*0.35))
    ov=frame.copy()
    cv2.ellipse(ov,(int(cx),int(cy)+int(size*0.08)),(r,hr),0,0,360,(10,10,10),-1,cv2.LINE_AA)
    cv2.addWeighted(ov,0.28,frame,0.72,0,frame)
    return frame


# ─────────────────────────────────────────────
#  SYSTEM INFO
# ─────────────────────────────────────────────

def get_system_info():
    info = {
        "platform": platform.platform(),
        "python":   platform.python_version(),
        "cpu":      platform.processor() or "unknown",
        "cores":    os.cpu_count(),
        "cv2":      cv2.__version__,
        "numpy":    np.__version__,
    }
    # RAM
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        info["ram_gb"] = "unknown (pip install psutil)"
    return info


def print_system_info(info):
    print("\n┌─ System ──────────────────────────────────────────────")
    for k, v in info.items():
        print(f"│  {k:<12} {v}")
    print("└───────────────────────────────────────────────────────")


# ─────────────────────────────────────────────
#  BENCHMARK SCENARIOS
#  Each scenario defines a pest configuration
#  representative of a real use case.
# ─────────────────────────────────────────────

SCENARIOS = [
    {
        "name":    "1 mouse",
        "pests":   [("mouse", 1, 50)],
    },
    {
        "name":    "3 cockroaches",
        "pests":   [("cockroach", 3, 35)],
    },
    {
        "name":    "2 mice + 3 cockroaches",
        "pests":   [("mouse", 2, 50), ("cockroach", 3, 35)],
    },
    {
        "name":    "5 cockroaches (heavy)",
        "pests":   [("cockroach", 5, 35)],
    },
]

DRAW_FNS = {"mouse": _draw_mouse, "cockroach": _draw_cockroach}


def make_dummy_path(n_frames, width, height, floor_mask=None, margin=30):
    """Generate a simple valid path without the full steering logic (for speed isolation)."""
    if floor_mask is not None:
        ys, xs = np.where(floor_mask)
        if len(xs) > 0:
            idx = random.randint(0, len(xs)-1)
            x, y = float(xs[idx]), float(ys[idx])
        else:
            x, y = width/2, height/2
    else:
        x, y = width/2, height*0.8

    path = []
    angle = random.uniform(0, 2*math.pi)
    speed = 6.0
    for _ in range(n_frames):
        path.append((x, y))
        angle += random.uniform(-0.15, 0.15)
        nx = x + math.cos(angle)*speed
        ny = y + math.sin(angle)*speed
        nx = float(np.clip(nx, margin, width-margin))
        ny = float(np.clip(ny, margin, height-margin))
        if floor_mask is not None:
            xi,yi = int(np.clip(nx,0,width-1)), int(np.clip(ny,0,height-1))
            if not floor_mask[yi, xi]:
                angle += math.pi + random.uniform(-0.5, 0.5)
                nx, ny = x, y
        x, y = nx, ny
    return path


def run_scenario(bg, floor_mask, depth_map, scenario, n_frames, fps, save_path=None):
    """
    Time a single scenario. Returns dict with timing breakdown.
    """
    h, w = bg.shape[:2]
    pests = scenario["pests"]  # list of (type, count, size)

    # ── Time: path generation ─────────────────────────────────────────
    t0 = time.perf_counter()
    all_paths = []
    for ptype, count, size in pests:
        for _ in range(count):
            all_paths.append((ptype, size, make_dummy_path(n_frames, w, h, floor_mask)))
    path_time = time.perf_counter() - t0

    # ── Time: frame rendering ─────────────────────────────────────────
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    sprite_times  = []
    overlay_times = []
    shadow_times  = []
    write_times   = []

    for i in range(n_frames):
        frame = bg.copy()
        for ptype, size, path in all_paths:
            px, py = path[i]
            d = float(depth_map[int(np.clip(py,0,h-1)), int(np.clip(px,0,w-1))]) \
                if depth_map is not None else py/h
            scale = float(np.clip(0.35 + 0.65*d, 0.15, 1.0))
            actual = max(8, int(size*scale))

            t = time.perf_counter()
            _shadow(frame, int(px), int(py), actual)
            shadow_times.append(time.perf_counter()-t)

            t = time.perf_counter()
            sprite = DRAW_FNS[ptype](size, i, 45.0, scale=scale)
            sprite_times.append(time.perf_counter()-t)

            t = time.perf_counter()
            frame = _overlay(frame, sprite, int(px), int(py))
            overlay_times.append(time.perf_counter()-t)

        if writer:
            t = time.perf_counter()
            writer.write(frame)
            write_times.append(time.perf_counter()-t)

    if writer:
        writer.release()

    total_render = sum(sprite_times) + sum(overlay_times) + sum(shadow_times)
    total_write  = sum(write_times)
    total        = path_time + total_render + total_write

    return {
        "n_frames":        n_frames,
        "n_pests":         sum(c for _,c,_ in pests),
        "path_gen_s":      path_time,
        "sprite_draw_s":   sum(sprite_times),
        "overlay_s":       sum(overlay_times),
        "shadow_s":        sum(shadow_times),
        "encode_write_s":  total_write if write_times else 0.0,
        "total_render_s":  total_render + total_write,
        "total_s":         total,
        "fps_achieved":    n_frames / max(total, 0.001),
        "ms_per_frame":    (total_render + total_write) / max(n_frames, 1) * 1000,
    }


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}m {s:.0f}s"


def print_scenario_result(name, results):
    avg = {k: sum(r[k] for r in results)/len(results)
           for k in results[0] if isinstance(results[0][k], float)}
    avg["n_frames"] = results[0]["n_frames"]
    avg["n_pests"]  = results[0]["n_pests"]

    print(f"\n  Scenario : {name}")
    print(f"  Pests    : {avg['n_pests']}")
    print(f"  ┌─ Timing breakdown (avg over {len(results)} run(s)) ─────────────")
    print(f"  │  Path generation   {avg['path_gen_s']*1000:8.1f} ms")
    print(f"  │  Sprite drawing    {avg['sprite_draw_s']*1000:8.1f} ms")
    print(f"  │  Alpha overlay     {avg['overlay_s']*1000:8.1f} ms")
    print(f"  │  Shadow            {avg['shadow_s']*1000:8.1f} ms")
    print(f"  │  Video encode/write{avg['encode_write_s']*1000:8.1f} ms")
    print(f"  │  ─────────────────────────────────────")
    print(f"  │  Total             {avg['total_s']*1000:8.1f} ms  ({format_time(avg['total_s'])} per video)")
    print(f"  │  ms / frame        {avg['ms_per_frame']:8.1f} ms")
    print(f"  │  Achieved fps      {avg['fps_achieved']:8.1f}")
    print(f"  └──────────────────────────────────────────────────────")
    return avg


def print_scale_projection(scenario_avgs, video_duration, fps):
    """Project compute needed at 100, 1000, 10000 videos."""
    n_frames = int(video_duration * fps)

    print(f"\n{'═'*62}")
    print(f"  SCALE PROJECTIONS  ({video_duration}s video @ {fps}fps = {n_frames} frames)")
    print(f"{'═'*62}")
    print(f"  {'Scenario':<30} {'100 vids':>10} {'1K vids':>10} {'10K vids':>12}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*12}")

    for name, avg in scenario_avgs.items():
        t = avg["total_s"]
        print(f"  {name:<30} {format_time(t*100):>10} {format_time(t*1000):>10} {format_time(t*10000):>12}")

    print(f"\n  {'─'*60}")
    print(f"  Parallelism speedup estimates (based on CPU cores):")
    cores = os.cpu_count() or 1

    # Use the most common scenario (mice + cockroaches) for projection
    ref_name = "2 mice + 3 cockroaches"
    if ref_name in scenario_avgs:
        ref_t = scenario_avgs[ref_name]["total_s"]
        for n_videos in [100, 1000, 10000]:
            wall_1core  = ref_t * n_videos
            wall_ncores = wall_1core / cores
            print(f"\n  {n_videos:>6} videos  ({ref_name})")
            print(f"           1 core  : {format_time(wall_1core)}")
            print(f"           {cores} cores : {format_time(wall_ncores)}  "
                  f"({'your machine' if cores > 1 else 'single-threaded'})")

    print(f"\n  NOTE: Parallelism via --jobs in batch_render.py")
    print(f"        Mask+depth generation is NOT included here — run once per image.")
    print(f"{'═'*62}\n")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark pest video generation.")
    parser.add_argument("--image",    default=None,
                        help="Kitchen image (synthetic if not provided)")
    parser.add_argument("--mask",     default=None,
                        help="Floor mask PNG")
    parser.add_argument("--depth",    default=None,
                        help="Depth map PNG")
    parser.add_argument("--duration", type=float, default=10,
                        help="Video duration in seconds (default: 10)")
    parser.add_argument("--fps",      type=int,   default=25,
                        help="Frames per second (default: 25)")
    parser.add_argument("--runs",     type=int,   default=3,
                        help="Timing runs to average (default: 3)")
    parser.add_argument("--quick",    action="store_true",
                        help="Single 5-second run per scenario for quick estimate")
    parser.add_argument("--output",   default=None,
                        help="Save a sample rendered video to this path")
    args = parser.parse_args()

    if args.quick:
        args.duration = 5
        args.runs     = 1
        print("[INFO] Quick mode: 5s video, 1 run per scenario")

    n_frames = int(args.duration * args.fps)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║          PEST VIDEO GENERATION BENCHMARK                ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # System info
    sysinfo = get_system_info()
    print_system_info(sysinfo)

    # Load or synthesise background
    if args.image and os.path.exists(args.image):
        bg = cv2.imread(args.image)
        print(f"\n[INFO] Image: {args.image}  {bg.shape[1]}×{bg.shape[0]}")
    else:
        if args.image:
            print(f"[WARN] Image not found: {args.image} — using 1920×1080 synthetic")
        else:
            print("[INFO] No image provided — using 1920×1080 synthetic background")
        bg = np.ones((1080, 1920, 3), dtype=np.uint8) * 180

    h, w = bg.shape[:2]

    # Load masks
    floor_mask = None
    if args.mask and os.path.exists(args.mask):
        m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        floor_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 127
        print(f"[INFO] Floor mask: {int(floor_mask.sum())} walkable px "
              f"({100*floor_mask.sum()/(h*w):.1f}%)")
    else:
        # Synthetic floor: bottom 40% of image
        floor_mask = np.zeros((h, w), dtype=bool)
        floor_mask[int(h*0.6):, int(w*0.05):int(w*0.95)] = True
        print(f"[INFO] No mask provided — using synthetic floor (bottom 40%)")

    depth_map = None
    if args.depth and os.path.exists(args.depth):
        d = cv2.imread(args.depth, cv2.IMREAD_GRAYSCALE)
        depth_map = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
        print(f"[INFO] Depth map loaded")
    else:
        # Synthetic depth: linear gradient (bottom=near)
        depth_map = np.tile(np.linspace(0, 1, h)[:, None], (1, w)).astype(np.float32)
        print(f"[INFO] No depth map — using synthetic gradient")

    print(f"\n[INFO] Benchmark: {len(SCENARIOS)} scenarios × "
          f"{args.runs} run(s) × {n_frames} frames "
          f"({args.duration}s @ {args.fps}fps)")

    scenario_avgs = {}

    for s_idx, scenario in enumerate(SCENARIOS):
        name = scenario["name"]
        print(f"\n[{s_idx+1}/{len(SCENARIOS)}] Running: {name} ─────────────────────")

        save_path = args.output if (s_idx == 0 and args.output) else None
        run_results = []

        for run_i in range(args.runs):
            print(f"  run {run_i+1}/{args.runs}...", end=" ", flush=True)
            t_start = time.perf_counter()
            result  = run_scenario(bg, floor_mask, depth_map, scenario,
                                   n_frames, args.fps,
                                   save_path=save_path if run_i == 0 else None)
            print(f"{time.perf_counter()-t_start:.2f}s")
            run_results.append(result)
            save_path = None  # only save on first run of first scenario

        avg = print_scenario_result(name, run_results)
        scenario_avgs[name] = avg

    if args.output and os.path.exists(args.output):
        print(f"\n[INFO] Sample video saved: {args.output}")

    print_scale_projection(scenario_avgs, args.duration, args.fps)


if __name__ == "__main__":
    main()