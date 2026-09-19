"""
Temporary Taichi Rendering Test: Central Star and Orbiting Planet
Resolution: 1280x800
Closes only when the window is closed by the user.
"""

import math
import sys
import taichi as ti

def initialize_best_backend():
    backends = [
        (ti.cuda, "NVIDIA CUDA"),
        (ti.vulkan, "Vulkan"),
        (ti.cpu, "CPU (x64)"),
    ]
    for arch, name in backends:
        try:
            ti.init(arch=arch, log_level=ti.WARN)
            print(f"[Backend Selected] Successfully initialized {name} backend.", flush=True)
            return arch, name
        except Exception as e:
            print(f"[Backend Notice] Failed initializing {name}: {e}", flush=True)
            try:
                ti.reset()
            except Exception:
                pass
    raise RuntimeError("Failed to initialize any Taichi backend.")

def main():
    arch, backend_name = initialize_best_backend()

    width, height = 1280, 800
    window = ti.ui.Window(
        name="Astrophysics Render Test - Star & Orbiting Planet",
        res=(width, height),
        vsync=True,
    )
    canvas = window.get_canvas()

    # Aspect ratio correction so circular orbits appear circular on a 16:10 window
    aspect_ratio = height / width  # 800 / 1280 = 0.625

    # Star and Planet position fields for ti.ui.Canvas
    star_field = ti.Vector.field(2, dtype=ti.f32, shape=1)
    star_glow_field = ti.Vector.field(2, dtype=ti.f32, shape=1)
    planet_field = ti.Vector.field(2, dtype=ti.f32, shape=1)

    # Place star at canvas center (0.5, 0.5)
    star_field[0] = [0.5, 0.5]
    star_glow_field[0] = [0.5, 0.5]

    # Orbit parameters
    orbit_radius = 0.32
    angular_speed = 1.2  # radians per second
    time_elapsed = 0.0
    dt = 1.0 / 60.0

    print("Render window is open. Close the window to exit.", flush=True)

    while window.running:
        # Update planet orbit position
        time_elapsed += dt
        theta = angular_speed * time_elapsed

        # Aspect-adjusted coordinates in [0, 1] range
        px = 0.5 + orbit_radius * math.cos(theta) * aspect_ratio
        py = 0.5 + orbit_radius * math.sin(theta)
        planet_field[0] = [px, py]

        # Black astrophysical space background
        canvas.set_background_color((0.0, 0.0, 0.0))

        # Render bright central star (outer corona glow + bright core)
        canvas.circles(star_glow_field, radius=0.035, color=(1.0, 0.85, 0.3))
        canvas.circles(star_field, radius=0.022, color=(1.0, 1.0, 0.9))

        # Render orbiting planet
        canvas.circles(planet_field, radius=0.012, color=(0.25, 0.75, 1.0))

        window.show()

    print("Window closed. Exiting temporary render test cleanly.", flush=True)

if __name__ == "__main__":
    main()
