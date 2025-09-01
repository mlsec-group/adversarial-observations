import os
import re
import json
import argparse
import statistics

from pathlib import Path

import matplotlib
import numpy as np

from scipy import stats

def load_original_data(target):
    paths = Path("data/results/").glob(f"{target}*.json")
    results = {}
    for path in paths:
        pattern = f"{target}_(\\d+.\\d+)eps_(\\d+)steps"
        epsilon, steps = re.fullmatch(pattern, path.stem).groups()
        epsilon = float(epsilon)
        if epsilon > 0.07:
            continue
        with open(path, 'r') as f:
            contents = json.load(f)
            results[epsilon] = contents
    return results


def load_ablation_data(target):
    paths = Path("data/results/").glob(f"ablation_{target}*.json")
    results = {}
    for path in paths:
        pattern = f"ablation_{target}_(\\d+.\\d+)eps_(\\d+)steps"
        epsilon, steps = re.fullmatch(pattern, path.stem).groups()
        epsilon = float(epsilon)
        with open(path, 'r') as f:
            contents = json.load(f)
            results[epsilon] = contents
    return results


def match_ablation_and_original(target, offset):
    original = load_original_data(target)
    ablation = load_ablation_data(target)
    combined = {}
    for epsilon in ablation.keys():
        combined[epsilon] = {}
        for attack, values in ablation[epsilon].items():
            combined[epsilon]["ablation "+attack] = values
        length = min(len(v) for v in combined[epsilon].values())
        for attack, values in original[epsilon].items():
            combined[epsilon][attack] = values[offset:(offset+length)]
    return combined


def summarize_ablation(target, offset):
    combined = match_ablation_and_original(target, offset)
    results = {"w/o steps": [], "w/o approx": [], "w/o both": []}
    for epsilon in combined.keys():
        unperturbed = -np.asarray(combined[epsilon]["unperturbed"])
        abl_unperturbed = -np.asarray(combined[epsilon]["ablation unperturbed"])
        assert np.allclose(unperturbed, abl_unperturbed, atol=1e-2, rtol=1e-4), np.max(np.abs(unperturbed - abl_unperturbed))

        our_deviations = np.median(-np.asarray(combined[epsilon]["Ours"]) - unperturbed, axis=-1)
        baseline_deviations = np.median(-np.asarray(combined[epsilon]["DP-Attacker"]) - unperturbed, axis=-1)
        wo_approx_deviations = np.median(-np.asarray(combined[epsilon]["ablation w/o approx"]) - abl_unperturbed, axis=-1)
        wo_steps_deviations = np.median(-np.asarray(combined[epsilon]["ablation w/o steps"]) - abl_unperturbed, axis=-1)
        
        results["w/o both"].extend((baseline_deviations / our_deviations).tolist())
        results["w/o approx"].extend((wo_approx_deviations / our_deviations).tolist())
        results["w/o steps"].extend((wo_steps_deviations / our_deviations).tolist())
    aggregated = {}
    for k, vs in results.items():
        mean = round(sum(vs)/len(vs)*100, 2)
        ci = 1.644850 * statistics.stdev(vs) / len(vs)**0.5
        ci = round(ci*100, 2)
        aggregated[k] = mean
    return aggregated


def get_intersection_per_attack(target, target_value, targets, attacks):
    results = load_original_data(target)
    results_per_attack = {attack: [] for attack in attacks}
    for i, target in enumerate(targets):
        unperturbed = -np.asarray([results[epsilon]["unperturbed"][i] for epsilon in results.keys()])
        lat, lon = target["location"]["latitude"], target["location"]["longitude"]
        lat = round(lat)
        lon = round(lon)
        for attack in attacks:
            epsilons = list(results.keys())
            epsilons = np.sqrt(1 + np.square(epsilons)) - 1
            values = -np.asarray([results[epsilon][attack][i] for epsilon in results.keys()])
            mean = np.median(values - unperturbed, axis=-1)
            order = np.argsort(epsilons)
            mean = mean[order]
            epsilons = epsilons[order]
            
            # find intersection
            # 1. find segment in which it will lie
            i = 0
            while i < len(mean) and mean[i] < target_value:
                i += 1
            # 2. linearly interpolate
            if i < len(mean):
                m = (mean[max(i, 1)] - mean[max(i-1, 0)]) / (epsilons[max(i, 1)] - epsilons[max(i-1, 0)])
                intersection_epsilon = epsilons[i] + (target_value - mean[i]) / m
            else:
                m = (mean[-1] - mean[-2]) / (epsilons[-1] - epsilons[-2])
                intersection_epsilon = epsilons[-1] + (target_value - mean[-1]) / m
            results_per_attack[attack].append((lon, lat, intersection_epsilon))
    return results_per_attack


def save_results_to_dats(target):
    results = load_original_data(target)
    aggregated_results = {}
    for epsilon in sorted(results.keys()):
        unperturbed_results = np.asarray(results[epsilon]["unperturbed"])
        for method, result in results[epsilon].items():
            if method == "unperturbed":
                continue
            # result is list with one list per target
            # each target contains multiple measurements for the single target
            medians = -np.median(np.asarray(result) - unperturbed_results, axis=1)
            mean = statistics.mean(medians)
            if len(medians) > 1:
                ci = 1.644850 * statistics.stdev(medians) / len(medians)**0.5
            else:
                ci = np.nan
            if method not in aggregated_results:
                aggregated_results[method] = []
            aggregated_results[method].append((float(epsilon), mean, ci))

    for method, values in aggregated_results.items():
        values = np.asarray(values)
        epsilons = values[:, 0]
        epsilons = np.sqrt(1 + np.square(epsilons)) - 1
        mean = values[:, 1]
        ci = values[:, 2]
        ci = np.nan_to_num(ci)
        with open(f"data/report/{target}_{method}.dat", "w") as f:
            f.write("\n".join(" ".join(map(str, t)) for t in zip(epsilons,mean,ci)))


var_count = 181 * 360 * (5 + 13*6)
def compute_detectability(target, extreme_weather_threshold):
    results = load_original_data(target)
    aggregated_results = {}
    for epsilon in sorted(results.keys()):
        unperturbed_results = np.asarray(results[epsilon]["unperturbed"])
        for method, result in results[epsilon].items():
            if method == "unperturbed":
                continue
            # result is list with one list per target
            # each target contains multiple measurements for the single target
            medians = -np.median(np.asarray(result) - unperturbed_results, axis=1)
            mean = statistics.mean(medians)
            if len(medians) > 1:
                ci = 1.644850 * statistics.stdev(medians) / len(medians)**0.5
            else:
                ci = np.nan
            if method not in aggregated_results:
                aggregated_results[method] = []
            aggregated_results[method].append((float(epsilon), mean, ci))

    ret = {}
    for method, values in aggregated_results.items():
        values = np.asarray(values)
        epsilons = values[:, 0]
        epsilons = np.sqrt(1 + np.square(epsilons)) - 1
        mean = values[:, 1]
        
        # find intersection
        # 1. find segment in which it will lie
        i = 0
        while i < len(mean) and mean[i] < extreme_weather_threshold:
            i += 1
        # 2. linearly interpolate
        # print(method, epsilons[i], mean[i], epsilons[i-1], mean[i-1], value)
        if i < len(mean):
            m = (mean[max(i, 1)] - mean[max(i-1, 0)]) / (epsilons[max(i, 1)] - epsilons[max(i-1, 0)])
            intersection_epsilon = epsilons[i] + (extreme_weather_threshold - mean[i]) / m
        else:
            m = (mean[-1] - mean[-2]) / (epsilons[-1] - epsilons[-2])
            intersection_epsilon = epsilons[-1] + (extreme_weather_threshold - mean[-1]) / m
        p = 1 - stats.chi2.cdf(stats.chi2.ppf(0.99, var_count-1, var_count-1), var_count - 1, var_count - 1, scale=intersection_epsilon + 1)
        ret[method] = p**2
    return ret


def matplotlib_to_hex_color(cmap, offset):
        r,g,b,a = cmap(offset)
        r = int(r*255)
        g = int(g*255)
        b = int(b*255)
        return f"#{r:02x}{g:02x}{b:02x}"


def heat_gradient(cmap):
    vmin = -15
    vmax = 35
    start = (5 - vmin) / (vmax - vmin)
    color_strings = []
    offsets = np.linspace(start, 1, 10)
    for i, offset in enumerate(offsets):
        color_strings.append(f'<stop offset="{100*(i/(len(offsets) - 1))}%" stop-color="{matplotlib_to_hex_color(cmap, offset)}" />')
    NEWLINE = "\n" # Hack to get \n to work in an f-string
    return f'''<svg width="120" height="16" version="1.1" xmlns="http://www.w3.org/2000/svg" style="stroke-width: 0.2%">
  <defs>
    <linearGradient id="gradient" x1="0" x2="1" y1="0" y2="0">
      {NEWLINE.join(color_strings)}
    </linearGradient>
  </defs>
  <rect
    x="10"
    y="7"
    width="100"
    height="4.75"
    fill="url(#gradient)"
    fill-opacity="0.8"
    rx="1" ry="1"/>
  <line x1="10.3" x2="10.3" y1="7" y2="6" stroke="black" />
  <text x="8" y="5" style="font: 5px Computer Modern Roman;text-anchor: start;">5°C</text>
  <line x1="43" x2="43" y1="6.8" y2="6" stroke="black" />
  <text x="43" y="5" style="font: 5px Computer Modern Roman;text-anchor: middle;">15°C</text>
  <line x1="76" x2="76" y1="6.8" y2="6" stroke="black" />
  <text x="76" y="5" style="font: 5px Computer Modern Roman;text-anchor: middle;">25°C</text>
  <line x1="109.7" x2="109.7" y1="7" y2="6" stroke="black" />
  <text x="118" y="5" style="font: 5px Computer Modern Roman;text-anchor: end;">35°C</text>
</svg>'''


def rain_gradient(cmap):
    color_strings = []
    offsets = np.linspace(0, 1, 10)
    for i, offset in enumerate(offsets):
        color_strings.append(f'<stop offset="{100*(i/(len(offsets) - 1))}%" stop-color="{matplotlib_to_hex_color(cmap, offset)}" />')
    NEWLINE = "\n"
    return f'''<svg width="13" height="74" version="1.1" xmlns="http://www.w3.org/2000/svg" style="stroke-width: 0.3%">
  <defs>
    <linearGradient id="gradient" x1="0" x2="0" y1="1" y2="0">
      {NEWLINE.join(color_strings)}
    </linearGradient>
  </defs>
  <rect
    x="1"
    y="2"
    width="4"
    height="70"
    fill="url(#gradient)"
    fill-opacity="0.9"
    rx="1" ry="1"/>
    <line x1="5" x2="6.5" y1="71.7" y2="71.7" stroke="black" />
    <text x="7.2" y="73.3" style="font: 3px Computer Modern Roman;text-anchor: start;">0</text>
    <line x1="5.2" x2="6.5" y1="48.5" y2="48.5" stroke="black" />
    <text x="7.2" y="50" style="font: 3px Computer Modern Roman;text-anchor: start;">20</text>
    <line x1="5.2" x2="6.5" y1="25" y2="25" stroke="black" />
    <text x="7.2" y="26.5" style="font: 3px Computer Modern Roman;text-anchor: start;">40</text>
    <line x1="5" x2="6.5" y1="2.3" y2="2.3" stroke="black" />
    <text x="7.2" y="3.6" style="font: 3px Computer Modern Roman;text-anchor: start;">60+</text>
</svg>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", action="store_true")
    args = parser.parse_args()
    os.makedirs("data/report", exist_ok=True)

    ## main eval
    save_results_to_dats("wind")
    save_results_to_dats("temperature")
    save_results_to_dats("precipitation")

    ## lat susceptibility
    if not args.short:
        with open("data/weather_evaluation_targets.json", "r") as f:
            targets = json.load(f)
        ATTACKS = ["Ours", "DP-Attacker", "AdvDM"]

        temperature_per_attack = get_intersection_per_attack("temperature", 11.75064901, targets, ATTACKS)
        wind_per_attack = get_intersection_per_attack("wind", 12.56660729, targets, ATTACKS)
        precipitation_per_attack = get_intersection_per_attack("precipitation", 0.06293304, targets, ATTACKS)
        mean_ours = (
            np.asarray(temperature_per_attack["Ours"])[:, 2] +
            np.asarray(wind_per_attack["Ours"])[:, 2] +
            np.asarray(precipitation_per_attack["Ours"])[:, 2]
            ) / 3
        lats = np.asarray(temperature_per_attack["Ours"])[:, 1]


    ## ablation studies
    if not args.short:
        wind = summarize_ablation("wind", 0)
        temperature = summarize_ablation("temperature", 34 if not args.short else 0)
        precipitation = summarize_ablation("precipitation", 68 if not args.short else 0)

    ## detectability
    p_temperature = compute_detectability("temperature", 11.75064901)
    p_wind = compute_detectability("wind", 12.56660729)
    p_precipitation = compute_detectability("precipitation", 0.06293304)

    result = """\\documentclass[sigconf,nonacm]{acmart}
\\usepackage{xcolor}
\\usepackage{booktabs}
\\usepackage{subcaption}

\\definecolor{myred}{HTML}{bf616a}
\\definecolor{poscolor}{RGB}{128, 180, 230}
\\definecolor{negcolor}{RGB}{252, 90, 40}
\\definecolor{neucolor}{RGB}{1, 74, 26}
\\newcommand{\\red}[1]{\\textcolor{myred}{#1}}

\\usepackage{pgfplots}
\\pgfplotsset{compat=1.18}
\\usepgfplotslibrary{fillbetween}
\\usetikzlibrary{intersections}
\\usetikzlibrary{positioning}
\\usepgfplotslibrary{statistics}
\\usetikzlibrary{matrix}
\\usepgfplotslibrary{groupplots}
\\usetikzlibrary{plotmarks}

\\begin{document}
"""
    
    if not args.short:
        result += f"""
\\begin{{table}}
\\centering
\\caption{{Mean relative deviation achieved by different ablations (c.f. Table 1 in the paper).}}
\\begin{{tabular}}{{l rrr}}
    \\toprule
    \\textbf{{Method}} & \\multicolumn{{1}}{{c}}{{\\textbf{{Wind Speed}}}} & \\multicolumn{{1}}{{c}}{{\\textbf{{Temperature}}}} & \\textbf{{Precipitation}} \\\\
    \\midrule
    Ours       & $100.0\\,\\%$ & $100.0\\,\\%$ & $100.0\\,\\%$ \\\\
    w/o steps  & ${round(wind['w/o steps'], 1)}\\,\\%$ ~(\\red{{{round(wind['w/o steps'] - 100, 1)}}})
               & ${round(temperature['w/o steps'], 1)}\\,\\%$ ~(\\red{{{round(temperature['w/o steps'] - 100, 1)}}})
               & ${round(precipitation['w/o steps'], 1)}\\,\\%$ ~(\\red{{{round(precipitation['w/o steps'] - 100, 1)}}}) \\\\
    w/o approx & ${round(wind['w/o approx'], 1)}\\,\\%$ ~(\\red{{{round(wind['w/o approx'] - 100, 1)}}})
               & ${round(temperature['w/o approx'], 1)}\\,\\%$ ~(\\red{{{round(temperature['w/o approx'] - 100, 1)}}})
               & ${round(precipitation['w/o approx'], 1)}\\,\\%$ ~(\\red{{{round(precipitation['w/o approx'] - 100, 1)}}}) \\\\
    w/o both   & ${round(wind['w/o both'], 1)}\\,\\%$ ~(\\red{{{round(wind['w/o both'] - 100, 1)}}})
               & ${round(temperature['w/o both'], 1)}\\,\\%$ ~(\\red{{{round(temperature['w/o both'] - 100, 1)}}})
               & ${round(precipitation['w/o both'], 1)}\\,\\%$ ~(\\red{{{round(precipitation['w/o both'] - 100, 1)}}}) \\\\
    \\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    result += f"""
\\begin{{table}}
\\caption{{Detectability of different approaches used to fabricate extreme weather deviations (c.f. Table 2 in the paper).}}
\\begin{{tabular}}{{l rrr}}
    \\toprule
    \\textbf{{Method}} & \\multicolumn{{1}}{{c}}{{\\textbf{{Wind Speed}}}} & \\multicolumn{{1}}{{c}}{{\\textbf{{Temperature}}}} & \\textbf{{Precipitation}} \\\\
    \\midrule
    AdvDM  & ${round(100 * p_wind['AdvDM'], 2)}\\,$\\%
           & ${round(100 * p_temperature['AdvDM'], 2)}\\,$\\%
           & ${round(100 * p_precipitation['AdvDM'], 2)}\\,$\\% \\\\
    DP-Attacker & ${round(100 * p_wind['DP-Attacker'], 2)}\\,$\\%
                & ${round(100 * p_temperature['DP-Attacker'], 2)}\\,$\\%
                & ${round(100 * p_precipitation['DP-Attacker'], 2)}\\,$\\% \\\\
    Ours & ${round(100 * p_wind['Ours'], 2)}\\,$\\%
         & ${round(100 * p_temperature['Ours'], 2)}\\,$\\%
         & ${round(100 * p_precipitation['Ours'], 2)}\\,$\\% \\\\
    \\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    result += """
\\begin{figure}
\\centering
\\begin{tikzpicture}
\\pgfplotsset{every tick label/.append style={font=\\tiny}}
\\begin{groupplot}[
    group style={
        group size= 3 by 1,
        horizontal sep=1.0cm,
    },
    xlabel={Increase in noise},
    ylabel={},
    axis x line*=bottom,
    axis y line*=left,
    xmin=0.0,
    xmax=0.0025,
    width=0.33\\columnwidth,
    height=0.25\\columnwidth,
    ymajorgrids,
    legend cell align={left},
    xtick={0,0.0005,0.001,0.0015,0.002,0.0025},
    xticklabels={$\\pm$0.0\\%,+0.05\\%,+0.1\\%,+0.15\\%,+0.2\\%,+0.25\\%},
    xtick scale label code/.code={},
]
    \\nextgroupplot[
        title={Wind speed [m/s]},
        ylabel={Induced deviation},
        ymin=0,
        ytick={0,5,10,15,20,25,30},
        yticklabels={0,5,10,15,20,25,30},
    ]
    \\addplot [mark=x, poscolor] table [x index=0,y index=1] {wind_Ours.dat};
    \\addplot [name path=lolower, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} - \\thisrowno{2}]{wind_Ours.dat};
    \\addplot [name path=loupper, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} + \\thisrowno{2}]{wind_Ours.dat};
    \\addplot[poscolor, opacity=0.2, forget plot] fill between[of=lolower and loupper];

    \\addplot [mark=x, negcolor] table [x index=0,y index=1] {wind_AdvDM.dat};
    \\addplot [name path=lolower, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} - \\thisrowno{2}]{wind_AdvDM.dat};
    \\addplot [name path=loupper, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} + \\thisrowno{2}]{wind_AdvDM.dat};
    \\addplot[negcolor, opacity=0.2, forget plot] fill between[of=lolower and loupper];

    \\addplot [mark=x, neucolor] table [x index=0,y index=1] {wind_DP-Attacker.dat};
    \\addplot [name path=lolower, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} - \\thisrowno{2}]{wind_DP-Attacker.dat};
    \\addplot [name path=loupper, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} + \\thisrowno{2}]{wind_DP-Attacker.dat};
    \\addplot[neucolor, opacity=0.2, forget plot] fill between[of=lolower and loupper];
    
    \\addplot[mark=none, black, dashed, samples=2, domain=0:0.0025] {12.57};

    \\nextgroupplot[
        title={Temperature [K]},
        ymin=0,
        ytick={0,5,10,15,20,25},
        yticklabels={0,5,10,15,20,25},
    ]
    \\addplot [mark=x, poscolor] table [x index=0,y index=1] {temperature_Ours.dat};
    \\addplot [name path=lolower, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} - \\thisrowno{2}]{temperature_Ours.dat};
    \\addplot [name path=loupper, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} + \\thisrowno{2}]{temperature_Ours.dat};
    \\addplot[poscolor, opacity=0.2, forget plot] fill between[of=lolower and loupper];

    \\addplot [mark=x, negcolor] table [x index=0,y index=1] {temperature_AdvDM.dat};
    \\addplot [name path=lolower, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} - \\thisrowno{2}]{temperature_AdvDM.dat};
    \\addplot [name path=loupper, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} + \\thisrowno{2}]{temperature_AdvDM.dat};
    \\addplot[negcolor, opacity=0.2, forget plot] fill between[of=lolower and loupper];

    \\addplot [mark=x, neucolor] table [x index=0,y index=1] {temperature_DP-Attacker.dat};
    \\addplot [name path=lolower, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} - \\thisrowno{2}]{temperature_DP-Attacker.dat};
    \\addplot [name path=loupper, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} + \\thisrowno{2}]{temperature_DP-Attacker.dat};
    \\addplot[neucolor, opacity=0.2, forget plot] fill between[of=lolower and loupper];
    
    \\addplot[mark=none, black, dashed, samples=2, domain=0:0.0025] {11.75};

    \\nextgroupplot[
        title={Precipitation [mm]},
        ymin=0,
        ytick={0,0.05,0.1,0.15,0.2},
        yticklabels={0,50,100,150,200},
        legend style={legend columns=4,column sep=0.2cm},legend to name={commonLegend},
    ]
    \\addplot [mark=x, poscolor] table [x index=0,y index=1] {precipitation_Ours.dat};
    \\addlegendentry{Ours}
    \\addplot [name path=lolower, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} - \\thisrowno{2}]{precipitation_Ours.dat};
    \\addplot [name path=loupper, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} + \\thisrowno{2}]{precipitation_Ours.dat};
    \\addplot[poscolor, opacity=0.2, forget plot] fill between[of=lolower and loupper];

    \\addplot [mark=x, negcolor] table [x index=0,y index=1] {precipitation_AdvDM.dat};
    \\addlegendentry{AdvDM}
    \\addplot [name path=lolower, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} - \\thisrowno{2}]{precipitation_AdvDM.dat};
    \\addplot [name path=loupper, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} + \\thisrowno{2}]{precipitation_AdvDM.dat};
    \\addplot[negcolor, opacity=0.2, forget plot] fill between[of=lolower and loupper];

    \\addplot [mark=x, neucolor] table [x index=0,y index=1] {precipitation_DP-Attacker.dat};
    \\addlegendentry{DP-Attacker}
    \\addplot [name path=lolower, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} - \\thisrowno{2}]{precipitation_DP-Attacker.dat};
    \\addplot [name path=loupper, fill=none, draw=none, forget plot] table [
       x index=0,
       y expr=\\thisrowno{1} + \\thisrowno{2}]{precipitation_DP-Attacker.dat};
    \\addplot[neucolor, opacity=0.2, forget plot] fill between[of=lolower and loupper];

    \\addplot[mark=none, black, dashed, samples=2, domain=0:0.0025] {0.063};
    \\addlegendentry{99\\% extreme weather}

\\end{groupplot}
    \\node at (8.5, -1.5) {\\pgfplotslegendfromname{commonLegend}};
\\end{tikzpicture}
\\caption{Resulting mean deviation induced by adversarial observations of different sizes (c.f. Figure 2 in the paper).}
\\end{figure}
"""

    ### ablation studies
    if not args.short:
        result += """
\\begin{figure}
\\centering
\\begin{tikzpicture}
    \\pgfplotsset{every tick label/.append style={font=\\small}}
    \\begin{axis}[
        title={},
        ylabel={Mean noise increase},
        xlabel={Absolute latitude},
        ymin=0,
        ymax=0.0025,
        ytick={0,0.0005,0.001,0.0015,0.002,0.0025},
        yticklabels={$\\pm$0.0\\%,+0.05\\%,+0.1\\%,+0.15\\%,+0.2\\%,+0.25\\%},
        ytick scale label code/.code={},
        axis x line*=bottom,
        axis y line*=left,
        xmin=0,
        xmax=60,
        % xtick={0,20,40,60},
        width=0.95\\columnwidth,
        height=0.65\\columnwidth,
        ymajorgrids,
    ]
        \\addplot [mark size=1.2,poscolor,fill opacity=0.1,only marks] table [x index=0,y index=1] {lat_susceptibility.dat};
        \\addplot[mark=none, poscolor,dashed,line width=1pt, samples=2, domain=0:60] {5.466218016266582e-06*x + 0.0007134301027012592};
    \\end{axis}
\\end{tikzpicture}
\\caption{Mean required noise increase at different locations (c.f. Figure 3 in the paper).}
\\end{figure}
"""

    ### case studies
    result += """
\\begin{figure}[b]
    \\centering
    % IMPORTANT! height/width ratio is chosen to adjust for map distortion!
    \\begin{subfigure}[t]{0.4\\columnwidth}
        \\centering
        \\includegraphics[width=0.95\\linewidth,height=1.3\\linewidth]{rain_case_study_before}
        \\caption{Original prediction}
    \\end{subfigure}
    \\begin{subfigure}[t]{0.4\\columnwidth}
        \\centering
        \\includegraphics[width=0.95\\linewidth,height=1.3\\linewidth]{rain_case_study_after}
        \\caption{Perturbed prediction}
    \\end{subfigure}
    \\begin{subfigure}[t]{0.1\\columnwidth}
        \\centering
        \\includegraphics[width=0.95\\linewidth,height=5.2\\linewidth]{rain_colorbar}
    \\end{subfigure}
    \\caption{Predicted precipitation at the peak of Cyclone Amphan (c.f. Figure 4 in the paper).}
\\end{figure}

\\begin{figure}
    \\centering
    \\includegraphics[width=0.8\\columnwidth]{heat_colorbar}
    % IMPORTANT! height/width ratio is chosen to adjust for map distortion!
    \\begin{subfigure}{\\columnwidth}
        \\centering 
        \\includegraphics[width=0.855\\columnwidth,height=0.54\\columnwidth]{heat_case_study_before}
        \\caption{Original prediction}\\vspace{0.25cm}
    \\end{subfigure}
    \\begin{subfigure}{\\columnwidth}
        \\centering
        \\includegraphics[width=0.855\\columnwidth,height=0.54\\columnwidth]{heat_case_study_after}
        \\caption{Perturbed prediction}
    \\end{subfigure}
    \\caption{Predicted temperature at the peak of the European Heat Wave 2006 (c.f. Figure 5 in the paper).}
\\end{figure}

\\begin{figure}
    \\centering
    {
        \\fontsize{8}{10}\\selectfont
        \\includegraphics[width=0.855\\columnwidth,height=0.4455\\columnwidth]{wind_case_study}
    }
    \\caption{Predicted storm path of Hurricane Katrina (c.f. Figure 6 in the paper).}
\\end{figure}
"""
    

    result += f"""\\end{{document}}"""

    with open("data/report/report.tex", "w") as f:
        f.write(result)
    if not args.short:
        with open("data/report/lat_susceptibility.dat", "w") as f:
            f.write("\n".join(f"{lat} {mean}" for lat, mean in zip(np.abs(lats), mean_ours)))

    with open("data/report/heat_colorbar.svg", "w") as f:
        f.write(heat_gradient(matplotlib.colormaps["coolwarm"]))
    with open("data/report/rain_colorbar.svg", "w") as f:
        f.write(rain_gradient(matplotlib.colormaps["Blues"]))
    
    os.system("inkscape data/report/rain_colorbar.svg --export-area-drawing --batch-process --export-type=pdf --export-filename=data/report/rain_colorbar.pdf")
    os.system("inkscape data/report/heat_colorbar.svg --export-area-drawing --batch-process --export-type=pdf --export-filename=data/report/heat_colorbar.pdf")
    for filename in [
        "heat_case_study_after", "heat_case_study_before", "rain_case_study_after", "rain_case_study_before", "wind_case_study"
    ]:
        os.system(f"inkscape data/results/{filename}.svg --export-area-drawing --batch-process --export-type=pdf --export-filename=data/report/{filename}.pdf")
    os.system("pdflatex -interaction=nonstopmode -output-directory=data/report data/report/report.tex")
    os.system("pdflatex -interaction=nonstopmode -output-directory=data/report data/report/report.tex")