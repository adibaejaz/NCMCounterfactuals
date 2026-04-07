import argparse
import json
import math
import re
from pathlib import Path


TRIAL_RE = re.compile(r"trial_index=(\d+)")
RERUN_RE = re.compile(r"/(\d+)/results\.json$")
QUERY_LABEL_RE = re.compile(r"P\(Y=([01]) \| do\(X=([01])\)\)")
DIM_RE = re.compile(r"-dim=(\d+)-trial_index=")


def normalize_graph_name(graph_name: str):
    lowered = graph_name.lower()
    if lowered == "double_bow":
        return "doublebow"
    return lowered


def infer_query_label(payload):
    max_keys = [key for key in payload if key.startswith("max_ncm_")]
    min_keys = [key for key in payload if key.startswith("min_ncm_")]
    max_suffixes = {key[len("max_ncm_"):] for key in max_keys}
    min_suffixes = {key[len("min_ncm_"):] for key in min_keys}
    shared = sorted(max_suffixes.intersection(min_suffixes))
    if len(shared) != 1:
        raise ValueError("Could not infer a unique tracked query from results.json.")
    return shared[0]


def parse_atomic_query_name(query_label: str):
    match = QUERY_LABEL_RE.fullmatch(query_label)
    if not match:
        raise ValueError("Expected an atomic query label like 'P(Y=1 | do(X=1))'.")
    y_value, x_value = match.groups()
    return "y{}_dox{}".format(y_value, x_value)


def _load_torch_object(path):
    import torch as T

    try:
        return T.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return T.load(path, map_location="cpu")


def _extract_ncm_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        raise ValueError("Expected a state_dict dictionary.")
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    if any(key.startswith("ncm.") for key in state_dict):
        return {key[len("ncm."):]: value for key, value in state_dict.items() if key.startswith("ncm.")}
    return state_dict


def _model_hyperparams_from_json(hyperparams):
    typed = dict(hyperparams)
    for key in ("h-layers", "h-size", "u-size"):
        if key in typed:
            typed[key] = int(typed[key])
    return typed


def compute_empirical_bound_from_model(model_path: Path, graph_name: str, query_label: str, sample_size: int):
    from src.metric import evaluation
    from src.metric.queries import get_atomic_query
    from src.ds.causal_graph import CausalGraph
    from src.scm.ncm.mle_ncm import MLE_NCM

    trial_dir = model_path.parent.parent
    hyperparams_path = trial_dir / "hyperparams.json"
    if not hyperparams_path.exists():
        return None

    with hyperparams_path.open() as fp:
        hyperparams = _model_hyperparams_from_json(json.load(fp))

    dim_match = DIM_RE.search(trial_dir.name)
    if not dim_match:
        raise ValueError("Could not infer dimension from '{}'.".format(trial_dir.name))
    dim = int(dim_match.group(1))

    cg_path = Path(__file__).resolve().parents[1] / "dat" / "cg" / "{}.cg".format(graph_name)
    cg = CausalGraph.read(cg_path)
    v_size = {k: 1 if k in {"X", "Y", "M", "W"} else dim for k in cg}
    ncm = MLE_NCM(cg, v_size=v_size, default_u_size=hyperparams.get("u-size", 1), hyperparams=hyperparams)

    state_dict = _extract_ncm_state_dict(_load_torch_object(model_path))
    ncm.load_state_dict(state_dict)
    ncm.eval()

    query = get_atomic_query(graph_name, parse_atomic_query_name(query_label))
    prob_table = evaluation.probability_table(m=ncm, n=sample_size)
    return evaluation.atomic_query_bounds_from_probability_table(graph_name, query, prob_table)


def attach_debug_empirical_bounds(records, base_dir: Path, graph_name: str, query_label: str, sample_size: int):
    for record in records:
        rerun_dir = base_dir / Path(record["path"]).parent
        max_model_path = rerun_dir / "best_max.th"
        min_model_path = rerun_dir / "best_min.th"

        record["debug_empirical_upper_bound"] = None
        record["debug_empirical_lower_bound"] = None

        if max_model_path.exists():
            _, empirical_upper_bound = compute_empirical_bound_from_model(
                max_model_path,
                graph_name,
                query_label,
                sample_size,
            )
            record["debug_empirical_upper_bound"] = float(empirical_upper_bound)

        if min_model_path.exists():
            empirical_lower_bound, _ = compute_empirical_bound_from_model(
                min_model_path,
                graph_name,
                query_label,
                sample_size,
            )
            record["debug_empirical_lower_bound"] = float(empirical_lower_bound)


def load_results(base_dir: Path, graph_name: str):
    graph_name = normalize_graph_name(graph_name)
    result_glob = "{}/gen=CTM-graph={}-n_samples=*-dim=*-trial_index=*/*/results.json".format(
        graph_name,
        graph_name,
    )
    records = []
    query_label = None
    for path in sorted(base_dir.glob(result_glob)):
        trial_match = TRIAL_RE.search(str(path))
        rerun_match = RERUN_RE.search(str(path))
        if not trial_match or not rerun_match:
            continue
        with path.open() as fp:
            payload = json.load(fp)
        if query_label is None:
            query_label = infer_query_label(payload)
        records.append(
            {
                "trial": int(trial_match.group(1)),
                "rerun": int(rerun_match.group(1)),
                "upper_bound": float(payload["upper_bound"]),
                "lower_bound": float(payload["lower_bound"]),
                "estimated_upper": float(payload["max_ncm_{}".format(query_label)]),
                "estimated_lower": float(payload["min_ncm_{}".format(query_label)]),
                "path": str(path.relative_to(base_dir)),
            }
        )
    return records, query_label


def summarize_trials(records):
    trials = {}
    for record in records:
        trial = record["trial"]
        trials.setdefault(
            trial,
            {
                "trial": trial,
                "upper_bound": record["upper_bound"],
                "lower_bound": record["lower_bound"],
                "estimated_upper": [],
                "estimated_lower": [],
                "debug_upper_bounds": [],
                "debug_lower_bounds": [],
                "upper_paths": [],
                "lower_paths": [],
            },
        )
        trials[trial]["estimated_upper"].append(record["estimated_upper"])
        trials[trial]["estimated_lower"].append(record["estimated_lower"])
        trials[trial]["upper_paths"].append(
            {
                "rerun": record["rerun"],
                "value": record["estimated_upper"],
                "path": record["path"],
            }
        )
        trials[trial]["lower_paths"].append(
            {
                "rerun": record["rerun"],
                "value": record["estimated_lower"],
                "path": record["path"],
                "debug_empirical_bound": record.get("debug_empirical_lower_bound"),
            }
        )
        if record.get("debug_empirical_upper_bound") is not None:
            trials[trial]["debug_upper_bounds"].append(record["debug_empirical_upper_bound"])
        if record.get("debug_empirical_lower_bound") is not None:
            trials[trial]["debug_lower_bounds"].append(record["debug_empirical_lower_bound"])
        trials[trial]["upper_paths"][-1]["debug_empirical_bound"] = record.get("debug_empirical_upper_bound")

    summarized = []
    for trial in sorted(trials):
        row = trials[trial]
        upper_mean = sum(row["estimated_upper"]) / len(row["estimated_upper"])
        lower_mean = sum(row["estimated_lower"]) / len(row["estimated_lower"])
        upper_var = sum((value - upper_mean) ** 2 for value in row["estimated_upper"]) / len(
            row["estimated_upper"]
        )
        lower_var = sum((value - lower_mean) ** 2 for value in row["estimated_lower"]) / len(
            row["estimated_lower"]
        )
        row["estimated_upper_mean"] = upper_mean
        row["estimated_lower_mean"] = lower_mean
        row["estimated_upper_std"] = math.sqrt(upper_var)
        row["estimated_lower_std"] = math.sqrt(lower_var)
        row["estimated_upper_max"] = max(row["estimated_upper"])
        row["estimated_lower_min"] = min(row["estimated_lower"])
        summarized.append(row)
    return summarized


def build_svg(trials, query_label, graph_name):
    width = 980
    height = 640
    margin_top = 34
    margin_right = 30
    margin_bottom = 56
    margin_left = 76
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom
    band = inner_w / max(len(trials), 1)
    slot_width = min(92.0, band * 0.62)
    upper_offset = -slot_width / 2.0
    lower_offset = slot_width / 2.0

    def y(value):
        return margin_top + (1.0 - value) * inner_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]

    for i in range(11):
        value = i / 10.0
        py = y(value)
        parts.append(
            f'<line x1="{margin_left}" x2="{width - margin_right}" y1="{py:.2f}" y2="{py:.2f}" stroke="rgba(0,0,0,0.10)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 12}" y="{py + 4:.2f}" text-anchor="end" fill="#666666" font-family="IBM Plex Mono, monospace" font-size="11">{value:.1f}</text>'
        )

    parts.append(
        f'<text x="32" y="{margin_top + inner_h / 2:.2f}" text-anchor="middle" transform="rotate(-90 32 {margin_top + inner_h / 2:.2f})" fill="#666666" font-family="IBM Plex Mono, monospace" font-size="11">{query_label} value under {graph_name} graph</text>'
    )

    for idx, trial in enumerate(trials):
        cx = margin_left + band * idx + band / 2.0
        upper_x = cx + upper_offset
        lower_x = cx + lower_offset

        if idx < len(trials) - 1:
            divider_x = cx + band / 2.0
            parts.append(
                f'<line x1="{divider_x:.2f}" x2="{divider_x:.2f}" y1="{margin_top - 6}" y2="{margin_top + inner_h + 10}" stroke="rgba(0,0,0,0.05)" stroke-width="1"/>'
            )

        parts.append(
            f'<text x="{cx:.2f}" y="{height - 8}" text-anchor="middle" fill="#111111" font-family="IBM Plex Mono, monospace" font-size="12">trial {trial["trial"]}</text>'
        )
        parts.append(
            f'<text x="{upper_x:.2f}" y="{height - 24}" text-anchor="middle" fill="#666666" font-family="IBM Plex Mono, monospace" font-size="10">upper</text>'
        )
        parts.append(
            f'<text x="{lower_x:.2f}" y="{height - 24}" text-anchor="middle" fill="#666666" font-family="IBM Plex Mono, monospace" font-size="10">lower</text>'
        )

        upper_bound_y = y(trial["upper_bound"])
        lower_bound_y = y(trial["lower_bound"])

        parts.append(
            f'<line x1="{upper_x - slot_width / 2:.2f}" x2="{upper_x + slot_width / 2:.2f}" y1="{upper_bound_y:.2f}" y2="{upper_bound_y:.2f}" stroke="#ff5d73" stroke-width="4" stroke-linecap="round"/>'
        )
        parts.append(
            f'<line x1="{lower_x - slot_width / 2:.2f}" x2="{lower_x + slot_width / 2:.2f}" y1="{lower_bound_y:.2f}" y2="{lower_bound_y:.2f}" stroke="#5da9ff" stroke-width="4" stroke-linecap="round"/>'
        )

        for row in trial["upper_paths"]:
            if row.get("debug_empirical_bound") is not None:
                empirical_y = y(row["debug_empirical_bound"])
                parts.append(
                    f'<line x1="{upper_x - slot_width / 2:.2f}" x2="{upper_x + slot_width / 2:.2f}" y1="{empirical_y:.2f}" y2="{empirical_y:.2f}" stroke="rgba(90,90,90,0.55)" stroke-width="1.4" stroke-linecap="round"/>'
                )
            jitter = (row["rerun"] - (len(trial["upper_paths"]) - 1) / 2.0) * 10.0
            parts.append(
                f'<circle cx="{upper_x + jitter:.2f}" cy="{y(row["value"]):.2f}" r="3.8" fill="rgba(124,124,124,0.72)"/>'
            )
        for row in trial["lower_paths"]:
            if row.get("debug_empirical_bound") is not None:
                empirical_y = y(row["debug_empirical_bound"])
                parts.append(
                    f'<line x1="{lower_x - slot_width / 2:.2f}" x2="{lower_x + slot_width / 2:.2f}" y1="{empirical_y:.2f}" y2="{empirical_y:.2f}" stroke="rgba(90,90,90,0.55)" stroke-width="1.4" stroke-linecap="round"/>'
                )
            jitter = (row["rerun"] - (len(trial["lower_paths"]) - 1) / 2.0) * 10.0
            parts.append(
                f'<circle cx="{lower_x + jitter:.2f}" cy="{y(row["value"]):.2f}" r="3.8" fill="rgba(124,124,124,0.72)"/>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def build_html(trials, query_label, graph_name, debug_empirical_bounds=False):
    data_json = json.dumps(trials)
    upper_detail_prefix = "Upper slots use x = 2 * trial_index.\n\n"
    lower_detail_prefix = "Lower slots use x = 2 * trial_index + 1 (drawn visually adjacent).\n\n"
    if debug_empirical_bounds:
        upper_detail_prefix = (
            "Upper slots use x = 2 * trial_index.\n"
            "Thin gray lines show empirical upper bounds from saved max-model samples.\n\n"
        )
        lower_detail_prefix = (
            "Lower slots use x = 2 * trial_index + 1 (drawn visually adjacent).\n"
            "Thin gray lines show empirical lower bounds from saved min-model samples.\n\n"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>atomic bound visualization</title>
  <style>
    :root {{
      --bg: #ffffff;
      --grid: rgba(0, 0, 0, 0.10);
      --text: #111111;
      --muted: #666666;
      --red: #ff5d73;
      --blue: #5da9ff;
      --estimate: #7c7c7c;
      --gray: rgba(0, 0, 0, 0.22);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: 'IBM Plex Mono', monospace;
      padding: 16px;
    }}
    .chart-wrap {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
      background: #ffffff;
    }}
    .detail {{
      max-width: 1280px;
      margin: 10px auto 0;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .detail pre {{
      margin: 0;
      white-space: pre-wrap;
      line-height: 1.6;
      color: var(--muted);
      font-size: 12px;
      background: #f6f6f6;
      border: 1px solid rgba(0,0,0,0.08);
      padding: 14px;
      min-height: 190px;
    }}
    .axis-label {{
      font-size: 11px;
      fill: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .tick-text {{
      font-size: 11px;
      fill: var(--muted);
    }}
    .slot-label {{
      font-size: 10px;
      fill: var(--muted);
    }}
    .trial-label {{
      font-size: 12px;
      fill: var(--text);
      letter-spacing: 0.06em;
    }}
    .tooltip {{
      position: fixed;
      pointer-events: none;
      background: rgba(255, 255, 255, 0.97);
      color: var(--text);
      border: 1px solid rgba(0,0,0,0.12);
      padding: 10px 12px;
      font-size: 12px;
      line-height: 1.5;
      max-width: 280px;
      transform: translate(14px, 14px);
      opacity: 0;
      transition: opacity 120ms ease;
      z-index: 10;
      box-shadow: 0 12px 30px rgba(0,0,0,0.12);
    }}
    @media (max-width: 960px) {{
      .detail {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="chart-wrap">
    <svg id="chart" viewBox="0 0 1240 640" preserveAspectRatio="xMidYMid meet"></svg>
  </div>
  <div class="detail">
    <pre id="upper-details"></pre>
    <pre id="lower-details"></pre>
  </div>
  <div class="tooltip" id="tooltip"></div>
  <script>
    const trials = {data_json};
    const svg = document.getElementById('chart');
    const tooltip = document.getElementById('tooltip');
    const upperDetails = document.getElementById('upper-details');
    const lowerDetails = document.getElementById('lower-details');
    const W = 980;
    const H = 640;
    const margin = {{ top: 34, right: 30, bottom: 56, left: 76 }};
    const innerW = W - margin.left - margin.right;
    const innerH = H - margin.top - margin.bottom;
    const band = innerW / Math.max(trials.length, 1);
    const slotWidth = Math.min(92, band * 0.62);
    const upperOffset = -slotWidth / 2;
    const lowerOffset = slotWidth / 2;
    const y = (value) => margin.top + (1 - value) * innerH;

    function el(name, attrs = {{}}, parent = svg) {{
      const node = document.createElementNS('http://www.w3.org/2000/svg', name);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
      parent.appendChild(node);
      return node;
    }}

    function tooltipOn(evt, html) {{
      tooltip.innerHTML = html;
      tooltip.style.opacity = 1;
      tooltip.style.left = evt.clientX + 'px';
      tooltip.style.top = evt.clientY + 'px';
    }}
    function tooltipOff() {{
      tooltip.style.opacity = 0;
    }}

    el('rect', {{ x: 0, y: 0, width: W, height: H, fill: 'transparent' }});

    for (let i = 0; i <= 10; i++) {{
      const value = i / 10;
      const py = y(value);
      el('line', {{
        x1: margin.left,
        x2: W - margin.right,
        y1: py,
        y2: py,
        stroke: 'var(--grid)',
        'stroke-width': 1
      }});
      const tick = el('text', {{
        x: margin.left - 12,
        y: py + 4,
        'text-anchor': 'end',
        class: 'tick-text'
      }});
      tick.textContent = value.toFixed(1);
    }}

    const yLabel = el('text', {{
      x: 32,
      y: margin.top + innerH / 2,
      'text-anchor': 'middle',
      transform: `rotate(-90 32 ${{margin.top + innerH / 2}})`,
      class: 'axis-label'
    }});
    yLabel.textContent = {json.dumps("{} value under {} graph".format(query_label, graph_name))};

    trials.forEach((trial, idx) => {{
      const cx = margin.left + band * idx + band / 2;
      const upperX = cx + upperOffset;
      const lowerX = cx + lowerOffset;

      el('line', {{
        x1: cx + band / 2,
        x2: cx + band / 2,
        y1: margin.top - 6,
        y2: margin.top + innerH + 10,
        stroke: idx === trials.length - 1 ? 'transparent' : 'rgba(0,0,0,0.05)',
        'stroke-width': 1
      }});

      const trialText = el('text', {{
        x: cx,
        y: H - 8,
        'text-anchor': 'middle',
        class: 'trial-label'
      }});
      trialText.textContent = `trial ${{trial.trial}}`;

      const upperSlot = el('text', {{
        x: upperX,
        y: H - 24,
        'text-anchor': 'middle',
        class: 'slot-label'
      }});
      upperSlot.textContent = 'upper';

      const lowerSlot = el('text', {{
        x: lowerX,
        y: H - 24,
        'text-anchor': 'middle',
        class: 'slot-label'
      }});
      lowerSlot.textContent = 'lower';

      const upperBoundY = y(trial.upper_bound);
      const lowerBoundY = y(trial.lower_bound);

      const upperBound = el('line', {{
        x1: upperX - slotWidth / 2,
        x2: upperX + slotWidth / 2,
        y1: upperBoundY,
        y2: upperBoundY,
        stroke: 'var(--red)',
        'stroke-width': 4,
        'stroke-linecap': 'round'
      }});
      upperBound.addEventListener('mousemove', (evt) => tooltipOn(evt,
        `Trial ${{trial.trial}}<br>Upper bound: <b>${{trial.upper_bound.toFixed(4)}}</b>`));
      upperBound.addEventListener('mouseleave', tooltipOff);

      const lowerBound = el('line', {{
        x1: lowerX - slotWidth / 2,
        x2: lowerX + slotWidth / 2,
        y1: lowerBoundY,
        y2: lowerBoundY,
        stroke: 'var(--blue)',
        'stroke-width': 4,
        'stroke-linecap': 'round'
      }});
      lowerBound.addEventListener('mousemove', (evt) => tooltipOn(evt,
        `Trial ${{trial.trial}}<br>Lower bound: <b>${{trial.lower_bound.toFixed(4)}}</b>`));
      lowerBound.addEventListener('mouseleave', tooltipOff);

      trial.upper_paths.forEach((row) => {{
        if (row.debug_empirical_bound !== null && row.debug_empirical_bound !== undefined) {{
          const empirical = el('line', {{
            x1: upperX - slotWidth / 2,
            x2: upperX + slotWidth / 2,
            y1: y(row.debug_empirical_bound),
            y2: y(row.debug_empirical_bound),
            stroke: 'rgba(90,90,90,0.55)',
            'stroke-width': 1.4,
            'stroke-linecap': 'round'
          }});
          empirical.addEventListener('mousemove', (evt) => tooltipOn(evt,
            `Trial ${{trial.trial}}, rerun ${{row.rerun}}<br>Empirical upper bound from max model: <b>${{row.debug_empirical_bound.toFixed(4)}}</b>`));
          empirical.addEventListener('mouseleave', tooltipOff);
        }}
        const jitter = (row.rerun - (trial.upper_paths.length - 1) / 2) * 10;
        const dot = el('circle', {{
          cx: upperX + jitter,
          cy: y(row.value),
          r: 3.8,
          fill: 'rgba(124,124,124,0.72)'
        }});
        dot.addEventListener('mousemove', (evt) => tooltipOn(evt,
          `Trial ${{trial.trial}}, rerun ${{row.rerun}}<br>Max query: <b>${{row.value.toFixed(4)}}</b><br>${{row.path}}`));
        dot.addEventListener('mouseleave', tooltipOff);
      }});

      trial.lower_paths.forEach((row) => {{
        if (row.debug_empirical_bound !== null && row.debug_empirical_bound !== undefined) {{
          const empirical = el('line', {{
            x1: lowerX - slotWidth / 2,
            x2: lowerX + slotWidth / 2,
            y1: y(row.debug_empirical_bound),
            y2: y(row.debug_empirical_bound),
            stroke: 'rgba(90,90,90,0.55)',
            'stroke-width': 1.4,
            'stroke-linecap': 'round'
          }});
          empirical.addEventListener('mousemove', (evt) => tooltipOn(evt,
            `Trial ${{trial.trial}}, rerun ${{row.rerun}}<br>Empirical lower bound from min model: <b>${{row.debug_empirical_bound.toFixed(4)}}</b>`));
          empirical.addEventListener('mouseleave', tooltipOff);
        }}
        const jitter = (row.rerun - (trial.lower_paths.length - 1) / 2) * 10;
        const dot = el('circle', {{
          cx: lowerX + jitter,
          cy: y(row.value),
          r: 3.8,
          fill: 'rgba(124,124,124,0.72)'
        }});
        dot.addEventListener('mousemove', (evt) => tooltipOn(evt,
          `Trial ${{trial.trial}}, rerun ${{row.rerun}}<br>Min query: <b>${{row.value.toFixed(4)}}</b><br>${{row.path}}`));
        dot.addEventListener('mouseleave', tooltipOff);
      }});

    }});

    upperDetails.textContent =
      {json.dumps(upper_detail_prefix)} +
      trials.map((trial) =>
        `trial ${{trial.trial}}: bound=${{trial.upper_bound.toFixed(4)}}, max across reruns=${{trial.estimated_upper_max.toFixed(4)}}`
      ).join('\\n');

    lowerDetails.textContent =
      {json.dumps(lower_detail_prefix)} +
      trials.map((trial) =>
        `trial ${{trial.trial}}: bound=${{trial.lower_bound.toFixed(4)}}, min across reruns=${{trial.estimated_lower_min.toFixed(4)}}`
      ).join('\\n');
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Plot atomic query bounds across trials.")
    parser.add_argument(
        "--exp",
        default="out/bow_y1_dox1",
        help="experiment folder, relative to repo root or absolute",
    )
    parser.add_argument("--graph", default="bow", help="graph name inside the experiment folder")
    parser.add_argument(
        "--debug-empirical-bounds",
        action="store_true",
        help="overlay empirical bounds computed from saved rerun models and save to *_debug outputs",
    )
    parser.add_argument(
        "--debug-sample-size",
        type=int,
        default=100000,
        help="sample size used to estimate empirical model bounds for debug plots (default: 100000)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_dir = Path(args.exp)
    if not base_dir.is_absolute():
        base_dir = repo_root / base_dir
    graph_name = normalize_graph_name(args.graph)

    records, query_label = load_results(base_dir, graph_name)
    if not records:
        raise SystemExit(
            "No results.json files found under {} for graph '{}'".format(base_dir, graph_name)
        )
    if args.debug_empirical_bounds:
        attach_debug_empirical_bounds(records, base_dir, graph_name, query_label, args.debug_sample_size)
    trials = summarize_trials(records)
    output_stem = "{}_bounds".format(base_dir.name)
    if args.debug_empirical_bounds:
        output_stem = "{}_debug".format(output_stem)
    html_path = base_dir / "{}.html".format(output_stem)
    svg_path = base_dir / "{}.svg".format(output_stem)
    html_path.write_text(build_html(trials, query_label, graph_name, debug_empirical_bounds=args.debug_empirical_bounds))
    svg_path.write_text(build_svg(trials, query_label, graph_name))
    print("wrote", html_path)
    print("wrote", svg_path)


if __name__ == "__main__":
    main()
