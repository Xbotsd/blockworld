import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockworld import animals, hydrology, materials, vegetation
from blockworld import verifiers as V
from blockworld.world import Region, WorldConfig, build

cfg = WorldConfig()
w = build(cfg)
print("extent", {k: round(v, 2) if isinstance(v, float) else v for k, v in w.extent().items()})
print("meta", {k: v for k, v in w.meta.items() if k in ("dem_rmse_m", "stratum_depth_mae_m")},
      "seed_pool", len(w.meta.get("seed_pool", [])))

slopes = [w.slope_deg(t) for t in w.tiles]
print("slope deg: mean %.1f p95 %.1f max %.1f" % (
    statistics.mean(slopes), sorted(slopes)[int(0.95 * len(slopes))], max(slopes)))
print("temp range %.1f..%.1f  precip %.0f..%.0f" % (
    min(t.temp for t in w.tiles), max(t.temp for t in w.tiles),
    min(t.precip for t in w.tiles), max(t.precip for t in w.tiles)))

hydrology.step(w)
print("hydro", {k: round(v, 1) for k, v in hydrology.step(w).items()})
vegetation.spin_up(w, years=int(sys.argv[1]) if len(sys.argv) > 1 else 55)

print("biomes", vegetation.biome_counts(w))
print("biomass mean %.2f  npp mean %.2f  lai mean %.2f" % (
    statistics.mean([t.biomass for t in w.tiles]),
    statistics.mean([t.npp for t in w.tiles]),
    statistics.mean([t.lai for t in w.tiles])))
print("top materials:", {m: sum(1 for t in w.tiles if t.col.layers[0][1].key == m)
                         for m in sorted({t.col.layers[0][1].key for t in w.tiles})})
print("layer1 materials:", {m: sum(1 for t in w.tiles if t.col.material_at(1.6).key == m)
                            for m in sorted({t.col.material_at(1.6).key for t in w.tiles})})

eco = animals.run(w, years=4, seed=7, initial=200)
print("animals", {k: eco[k] for k in ("population", "by_species", "occupancy", "largest_survivor_kg")})

soft = min(w.tiles, key=lambda t: (t.col.material_at(1.0).E, t.col.material_at(1.0).c))
valley = Region.circle(soft.x, soft.y, 200)
f = V.Foundation(B=1.5, L=1.5, D=1.5, load_kN=1200.0)
vi = V.verify_infrastructure(w, valley, f)
print("infra", vi.passed, vi.metrics["worst_fs"], round(vi.metrics["worst_settlement_mm"], 1),
      round(vi.metrics["qu_kpa"], 0), vi.reasons)

lander = V.Lander(mass_kg=3000.0, n_pads=3, pad_diameter=0.25)
print("slope verdict", V.verify_slope(w, valley).metrics)
print("landing@earth", V.verify_landing(w, valley, lander).passed,
      V.verify_landing(w, valley, lander).metrics["worst_fs"])
print("ecology", V.verify_ecology(w, Region.whole(), animal_stats=eco).reasons)
