
import ast
from typing import Literal

import colorsys
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import rich.table
from rich import print


def close_plt() -> None:
    plt.cla()
    plt.clf()
    plt.close()


def series_to_array(series: pl.Series) -> np.ndarray:
    try:
        return np.array(ast.literal_eval(series[0]))
    except SyntaxError:
        return np.array(ast.literal_eval(series))


def format_num_params(num_params: int, round_to_digits: int = 1) -> str:
    if num_params < 1_000:
        pnum = str(round(num_params, max(0, round_to_digits)))
        scalar = ""
    elif num_params < 1_000_000:
        pnum = f"{round(num_params/1_000, max(0, round_to_digits))}"
        scalar = "k"
    elif num_params < 1_000_000_000:
        pnum = f"{round(num_params/1_000_000, max(0, round_to_digits))}"
        scalar = "M"
    else:
        pnum = f"{round(num_params/1_000_000_000, max(0, round_to_digits))}"
        scalar = "B"

    before_dot = pnum.split(".")[0]
    after_dot = pnum.split(".")[1] if "." in pnum else ""
    after_dot = "" if after_dot and (round_to_digits <= 0) else after_dot
    after_dot = "" if after_dot and (int(after_dot) == 0) else after_dot
    after_dot = "." + after_dot if after_dot else ""

    return f"{before_dot}{after_dot}{scalar}"


def format_num_params(num_params: int, round_to_digits: int = 1) -> str:
    if num_params < 1_000:
        pnum = str(round(num_params, max(0, round_to_digits)))
        scalar = ""
    elif num_params < 1_000_000:
        pnum = f"{round(num_params/1_000, max(0, round_to_digits))}"
        scalar = "k"
    elif num_params < 1_000_000_000:
        pnum = f"{round(num_params/1_000_000, max(0, round_to_digits))}"
        scalar = "M"
    else:
        pnum = f"{round(num_params/1_000_000_000, max(0, round_to_digits))}"
        scalar = "B"

    before_dot = pnum.split(".")[0]
    after_dot = pnum.split(".")[1] if "." in pnum else ""
    after_dot = "" if after_dot and (round_to_digits <= 0) else after_dot
    after_dot = "" if after_dot and (int(after_dot) == 0) else after_dot
    after_dot = "." + after_dot if after_dot else ""

    return f"{before_dot}{after_dot}{scalar}"


def load_xs_ys_avg_y(
        file: str,
        model_scale: float | None = None,
        depth: int | None = None,
        width: int | None = None,
        num_params: int | None = None,
        linear_value: bool | None = None,
        num_heads: int | None = None,
        run_num: int | None = None,
        seed: int | None = None,
        grokfast: bool | None = None,
        alpha: float | None = None,
        gain: float | None = None,
        to_plot: Literal["val_loss", "train_losses", "val_accs", "train_accs", "val_pplxs", "train_pplxs"] = "val_loss",
        plot_over: Literal["step", "epoch", "token", "time_sec"] = "step",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x, y, and average y from a CSV file."""
    filters = (pl.col("last_val_loss").ge(0))  # initial condition -> always true

    if model_scale is not None:
        filters &= (pl.col("model_scale") == model_scale)
    if depth is not None:
        filters &= (pl.col("depth") == depth)
    if width is not None:
        filters &= (pl.col("width") == width)
    if num_params is not None:
        filters &= (pl.col("num_params") == num_params)
    if linear_value is not None:
        filters &= (pl.col("linear_value") == linear_value)
    if num_heads is not None:
        filters &= (pl.col("num_heads") == num_heads)
    if run_num is not None:
        filters &= (pl.col("run_num") == run_num)
    if seed is not None:
        filters &= (pl.col("seed") == seed)
    if grokfast is not None:
        filters &= (pl.col("grokfast").eq(grokfast))
    if alpha is not None:
        filters &= (pl.col("alpha") == alpha)
    if gain is not None:
        filters &= (pl.col("gain") == gain)

    df = pl.scan_csv(file).filter(filters).collect()
    df.sort("run_num")
    arrays = [series_to_array(df[to_plot][i]) for i in range(len(df[to_plot]))]

    if plot_over == "step":
        return load_steps_ys_avg_ys(df, arrays)
    elif plot_over == "epoch":
        return load_epochs_ys_avg_ys(df, arrays)
    elif plot_over == "token":
        return load_tokens_ys_avg_ys(df, arrays)
    elif plot_over == "time_sec":
        return load_time_ys_avg_ys(df, arrays)
    else:
        raise ValueError(f"{plot_over} not a valid x-value")


def load_steps_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_len = min([len(a) for a in arrays])
    ys = np.array([list(a[:min_len]) for a in arrays])
    num_datapoints = len(ys[0])
    xs = ((np.arange(num_datapoints) + 1) * 12.5).astype(int)
    avg_ys = np.mean(ys, axis=0)
    return xs, ys, avg_ys


def load_epochs_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = [series_to_array(df["epoch"][i]) for i in range(len(df["epoch"]))]
    return interpolate_linearly(xs, arrays)


def load_tokens_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = [series_to_array(df["tokens_seen"][i]) for i in range(len(df["tokens_seen"]))]
    return interpolate_linearly(xs, arrays)


def load_time_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = [series_to_array(df["cumulative_time"][i]) for i in range(len(df["cumulative_time"]))]
    return interpolate_linearly(xs, arrays)


def interpolate_linearly(
        xs: list[np.ndarray], ys: list[np.ndarray], num_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Determine the maximum x value across all datasets
    max_x = max(x_vals.max() for x_vals in xs)
    
    # Generate a single set of new x values for all datasets
    new_x_vals = np.linspace(0, max_x, num_samples)

    new_ys = []
    for x_vals, y_vals in zip(xs, ys):
        # Interpolate y to the common set of new x values
        new_y_vals = np.interp(new_x_vals, x_vals, y_vals)
        new_ys.append(new_y_vals)

    # Convert new_ys to a 2D numpy array for easy manipulation
    new_ys = np.array(new_ys)
    
    # Calculate the average y values across all datasets
    avg_ys = np.nanmean(new_ys, axis=0)

    return new_x_vals, new_ys, avg_ys


def get_unique_settings(file: str, targets: list[str]) -> list[str | int | float | bool]:
    settings = []
    
    # Load the unique combinations of the targets
    combinations = (
        pl.scan_csv(file)
        .select(*[pl.col(target) for target in targets])
        .collect()
        .unique()
    )
    # Sort combinations alphabetically by content, target by target (for consistency in plotting)
    for target in targets:
        combinations = combinations.sort(target)
    # Create a list of settings
    for features in zip(
            *[combinations[target] for target in targets]
    ):
        settings.append(tuple(features))

    return settings


def generate_distinct_colors(n):
    """
    Generates n visually distinct colors.

    Parameters:
        n (int): The number of distinct colors to generate.

    Returns:
        list: A list of n visually distinct colors in hex format.
    """
    colors = []
    for i in range(n):
        hue = i / n
        # Fixing saturation and lightness/value to 0.9 for bright colors
        # You can adjust these values for different color variations
        lightness = 0.5
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    
    return colors


def unique_num_params(file: str) -> np.ndarray:
    return (
        pl.scan_csv(file)
        .select("num_params")
        .collect()
        ["num_params"]
        .unique()
        .to_numpy()
    )


def unique_widths(file: str) -> np.ndarray:
    return (
        pl.scan_csv(file)
        .select("width")
        .collect()
        ["width"]
        .unique()
        .to_numpy()
    )


def unique_depths(file: str) -> np.ndarray:
    return (
        pl.scan_csv(file)
        .select("depth")
        .collect()
        ["depth"]
        .unique()
        .to_numpy()
    )


def plot_line(
        color,
        use_unique_colors: bool,
        plot_all: bool,
        loglog: bool,
        to_plot: Literal["val_loss", "train_losses", "val_accs", "train_accs", "val_pplxs"],
        plot_over: Literal["step", "epoch", "token", "time_sec"],
        num_heads: int,
        linear_value: bool,
        depth: int,
        width: int,
        alpha: float,
        gain: float,
        grokfast: bool,
        from_sample: int | None = None,
        to_sample: int | None = None,
):
    xs, ys, avg_ys = load_xs_ys_avg_y(
        file,
        depth=depth,
        width=width,
        num_heads=num_heads,
        linear_value=linear_value,
        alpha=alpha,
        gain=gain,
        grokfast=grokfast,
        to_plot=to_plot,
        plot_over=plot_over,
    )
    xs = xs[from_sample:to_sample]
    ys = ys[:, from_sample:to_sample]
    avg_ys = avg_ys[from_sample:to_sample]
    color = color if use_unique_colors else None
    if plot_all:
        for y in ys:
            if loglog:
                plt.loglog(xs, y, color=color, alpha=0.2)
            else:
                plt.plot(xs, y, color=color, alpha=0.2)

    num_params = pl.scan_csv(file).filter(
        (pl.col("num_heads") == num_heads)
        & (pl.col("linear_value") == linear_value)
        & (pl.col("depth") == depth)
        & (pl.col("width") == width)
        & (pl.col("alpha") == alpha)
        & (pl.col("gain") == gain)
    ).collect()["num_params"][0]
    
    if grokfast:
        label = f"grokfast (alpha={alpha}, gain={gain})"
    else:
        label = "standard training"
    if loglog:
        plt.loglog(xs, avg_ys, color=color if plot_all else None, label=label)
    else:
        plt.plot(xs, avg_ys, color=color if plot_all else None, label=label)

    return num_params


def example_plot_fct(
        file: str,
        depth: int | None = 8,
        width: int | None = 384,
        num_heads: int | None = None,
        linear_value: bool | None = False,
        alpha: float | None = 0.8,
        gain: float | None = 0.1,
        to_plot: Literal["val_loss", "train_losses", "val_accs", "train_accs", "val_pplxs"] = "val_loss",
        plot_over: Literal["step", "epoch", "token", "time_sec"] = "epoch",
        show: bool = True,
        loglog: bool = False,
        plot_all: bool = False,
        use_unique_colors: bool = False,
        from_sample: int | None = None,
        to_sample: int | None = None,
) -> None:
    settings = get_unique_settings(file, ["num_heads", "linear_value", "depth", "width", "alpha", "gain", "grokfast"])
    if num_heads is not None:
        settings = [(nh, lv, d, w, a, g, gf) for nh, lv, d, w, a, g, gf in settings if nh == num_heads]
    if linear_value is not None:
        settings = [(nh, lv, d, w, a, g, gf) for nh, lv, d, w, a, g, gf in settings if lv == linear_value]
    if depth is not None:
        settings = [(nh, lv, d, w, a, g, gf) for nh, lv, d, w, a, g, gf in settings if d == depth]
    if width is not None:
        settings = [(nh, lv, d, w, a, g, gf) for nh, lv, d, w, a, g, gf in settings if w == width]
    if alpha is not None:
        settings = [(nh, lv, d, w, a, g, gf) for nh, lv, d, w, a, g, gf in settings if a == alpha or not gf]
    if gain is not None:
        settings = [(nh, lv, d, w, a, g, gf) for nh, lv, d, w, a, g, gf in settings if g == gain or not gf]

    colors = generate_distinct_colors(len(settings))

    for color, (num_heads_, linear_value_, depth_, width_, alpha_, gain_, grokfast) in zip(colors, settings):
        num_params = plot_line(
            color=color,
            use_unique_colors=use_unique_colors,
            plot_all=plot_all,
            loglog=loglog,
            to_plot=to_plot,
            plot_over=plot_over,
            num_heads=num_heads_,
            linear_value=linear_value_,
            depth=depth_,
            width=width_,
            alpha=alpha_,
            gain=gain_,
            grokfast=grokfast,
            from_sample=from_sample,
            to_sample=to_sample,
        )

    fig = plt.gcf()
    fig.set_size_inches(12, 7)

    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
    plt.legend()
    plt.grid()
    plt.title(f"{to_plot} vs {plot_over} (depth={depth_}, width={width_}, #params={format_num_params(num_params)})")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        filename = f"{to_plot}_vs_{plot_over}"
        if depth is not None:
            filename += f"_depth_{depth}"
        if width is not None:
            filename += f"_width_{width}"
        if alpha is not None:
            filename += f"_alpha_{alpha}"
        if gain is not None:
            filename += f"_gain_{gain}"
        if num_heads is not None:
            filename += f"_num_heads_{num_heads}"
        if linear_value is not None:
            filename += f"_linear_value_{linear_value}"
        if from_sample is not None:
            filename += f"_from_{from_sample}"
        if to_sample is not None:
            filename += f"_to_{to_sample}"

        plt.savefig(f"results/images/{filename}.png", dpi=300)
    close_plt()  # in case you call this function multiple times with different settings


def n_best_vals(
        file: str,
        n: int,
        best_is: Literal["min", "max"] = "min",
        metric: Literal["val_loss", "train_losses", "val_accs", "train_accs", "val_pplxs"] = "val_loss",
        alpha: float | None = None,
        gain: float | None = None,
        grokfast: bool | None = None,
        from_sample: int | None = None,
        to_sample: int | None = None,
) -> pl.DataFrame:
    settings = get_unique_settings(file, ["alpha", "gain", "grokfast"])
    if alpha is not None:
        settings = [(a, g, gf) for a, g, gf in settings if a == alpha or not gf]
    if gain is not None:
        settings = [(a, g, gf) for a, g, gf in settings if g == gain or not gf]

    table = rich.table.Table("alpha", "gain", "grokfast", f"Mean of {best_is} {n} {metric}", f"Median of {best_is} {n} {metric}")
    rows = []
    for alpha_, gain_, grokfast_ in settings:
        xs, ys, avg_ys = load_xs_ys_avg_y(
            file,
            alpha=alpha_,
            gain=gain_,
            grokfast=grokfast_,
            to_plot=metric,
            plot_over="epoch",
        )
        xs = xs[from_sample:to_sample]
        avg_ys = avg_ys[from_sample:to_sample]
        if best_is == "min":
            best_vals = np.sort(avg_ys)[:n]
        else:
            best_vals = np.sort(avg_ys)[-n:]

        rows.append((str(alpha_), str(gain_), str(grokfast_), f"{np.mean(best_vals).item():.2f}", f"{np.median(best_vals).item():.2f}"))

    rows = sorted(rows, key=lambda x: float(x[3]), reverse=best_is == "max")
    for row in rows:
        table.add_row(*row)
    print(table)


if __name__ == "__main__":
    file = "results/results_many_epochs.csv"
    example_plot_fct(
        file=file,
        depth=None,
        width=None,
        num_heads=None,
        linear_value=None,
        alpha=0.8,
        gain=0.1,
        to_plot="val_loss",
        plot_over="epoch",
        show=False,
        loglog=False,
        plot_all=False,
        from_sample=None,
        to_sample=None,
    )
    # n_best_vals(
    #     file=file,
    #     n=5,
    #     best_is="min",
    #     metric="val_loss",
    #     alpha=None,
    #     gain=None,
    #     grokfast=None,
    #     from_sample=None,
    #     to_sample=50,
    # )
