import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(text="# Non-equilibrium dynamics visualised")
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from pca import pca
    import rasterio
    import rioxarray as rxr
    import matplotlib.pyplot as plt

    return alt, mo, pca, pl, plt, rasterio, rxr


@app.cell
def _(alt):
    alt.data_transformers.enable("vegafusion")
    return


@app.cell
def _(plt):
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    return


@app.cell
def _(mo):
    mo.md(text="## Data")
    return


@app.cell
def _(pl):
    df = pl.read_parquet("/Users/Wanja/Documents/non-equilibrium_data/cv_new/table_wgs84_2002_with_cv.parquet")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(rasterio, rxr):
    # ---------- Load and fix HVW's CV raster ----------

    cv_hvw_path = "/Users/Wanja/Documents/non-equilibrium_data/prec_cv_hvw/Prec_cv.tif"

    # Fix nodata metadata so it's respected on read
    with rasterio.open(cv_hvw_path, "r+") as src:
        src.nodata = -3.4028235e+38

    # Load with masking now applied correctly
    cv_hvw = rxr.open_rasterio(cv_hvw_path, masked=True).squeeze(drop=True)

    # Sanity check the range now makes sense
    print("Min:", float(cv_hvw.min()), "Max:", float(cv_hvw.max()))
    return (cv_hvw,)


@app.cell
def _(rxr):
    # ---------- Load my CV raster ----------
    cv_path = "/Users/Wanja/Documents/non-equilibrium_data/cv_new/cv.tif"

    cv = rxr.open_rasterio(cv_path, masked=True)
    return (cv,)


@app.cell
def _(mo):
    mo.md(text="## Distributions")
    return


@app.cell
def _(alt, df, mo, pl):
    numeric_cols = [c for c, dt in df.schema.items() if dt.is_numeric()]

    # Bin each column independently, then combine into one long dataframe
    frames = []
    for col in numeric_cols:
        col_min, col_max = df[col].min(), df[col].max()
        n_bins = 30
        width = (col_max - col_min) / n_bins if col_max > col_min else 1

        binned = (
            df.select(pl.col(col))
            .with_columns(
                (((pl.col(col) - col_min) / width).floor() * width + col_min).cast(pl.Float64).alias("bin_start")
            )
            .group_by("bin_start")
            .agg(pl.len().alias("count"))
            .with_columns(pl.lit(col).alias("variable"))
        )
        frames.append(binned)

    long_binned = pl.concat(frames)

    chart = (
        alt.Chart(long_binned.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("bin_start:Q", title=None),
            y=alt.Y("count:Q", title="Count"),
        )
        .properties(width=200, height=150)
        .facet(facet="variable:N", columns=3)
        .resolve_scale(x="independent", y="independent")
    )

    mo.ui.altair_chart(chart)
    return


@app.cell
def _(alt, df, mo, pl):
    continent_counts = (
        df.group_by("continent")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    region_counts = (
        df.group_by("region")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    continent_chart = (
        alt.Chart(continent_counts.to_pandas())
        .mark_bar()
        .encode(
            y=alt.Y("continent:N", sort="-x", title=None),
            x=alt.X("count:Q", title="Count"),
        )
        .properties(width=200, height=300, title="Count by Continent")
    )

    region_chart = (
        alt.Chart(region_counts.to_pandas())
        .mark_bar()
        .encode(
            y=alt.Y("region:N", sort="-x", title=None),
            x=alt.X("count:Q", title="Count"),
        )
        .properties(width=400, height=300, title="Count by Region")
    )

    combined = continent_chart | region_chart

    mo.ui.altair_chart(combined)
    return


@app.cell
def _(mo):
    mo.md(text="## Relations")
    return


@app.cell
def _(alt, pl):
    # ---------- Scatterplot (20000 points sampled, 0.99 percent of predictor var) ----------
    def make_pair_scatter(df: pl.DataFrame, response_col: str, predictor_col: str, chart_title: str, sample_size: int = 20000, percentage_to_viz: float = 0.99) -> alt.Chart:
        """Build a scatterplot of a response variable against a predictor, using a sample.

        Extreme predictor outliers are excluded first via `percentage_to_viz`
        (computed on the full dataset, so the cutoff reflects the true
        population rather than a sample), then a random sample is drawn from
        the filtered data for plotting — since a scatter embeds raw points
        rather than aggregating, unlike the binned histogram/boxplot
        functions.

        Args:
            df: Source dataframe containing `response_col` and `predictor_col`.
                Can be the full, unsampled dataset — filtering and sampling
                both happen inside this function.
            response_col: Numeric column shown on the y-axis.
            predictor_col: Numeric column shown on the x-axis.
            chart_title: Title displayed above the chart.
            sample_size: Number of rows to randomly sample (with a fixed seed
                of 42 for reproducibility) after filtering. If fewer rows
                remain after filtering than `sample_size`, all remaining rows
                are used instead. Defaults to 20,000.
            percentage_to_viz: Quantile cutoff (0-1) applied to `predictor_col`
                before sampling, used to exclude extreme outliers from the
                visualized range. Defaults to 0.99 (top 1% excluded).

        Returns:
            An Altair scatter chart, ready to be combined with other charts
            (e.g. via `|` or `&`) or passed to `mo.ui.altair_chart`.
        """

        filtered_df = (
            df.select([response_col, predictor_col])
            .drop_nulls()
            .filter(pl.col(predictor_col) <= pl.col(predictor_col).quantile(percentage_to_viz))
        )
        local_df = filtered_df.sample(n=min(sample_size, filtered_df.height), seed=42).to_pandas()
    
        return (
            alt.Chart(local_df)
            .mark_circle(size=10, opacity=0.25)
            .encode(
                x=alt.X(f"{predictor_col}:Q", title=predictor_col),
                y=alt.Y(f"{response_col}:Q", title=response_col),
            )
            .properties(width=280, height=280, title=chart_title)
        )

    return (make_pair_scatter,)


@app.cell
def _(alt, pl):
    def make_grid_scatter(
        full_df: pl.DataFrame,
        predictor_cols: list,
        target_col: str = "Npp",
        sample_size: int = 5_000,
        facet_columns: int = 3,
        seed: int = 42,
    ) -> alt.Chart:
        """Build a faceted grid of scatterplots, one per predictor column.

        A single random sample is drawn once (before reshaping), then reshaped
        into long format so all predictors share the same sampled rows —
        keeping the sample consistent across panels rather than resampling
        per predictor. Sampling happens because a scatter embeds raw points
        rather than aggregating, unlike the binned histogram/boxplot grid
        functions, so it doesn't scale to full-size datasets on its own.

        Args:
            full_df: Source dataframe containing `target_col` and every column
                in `predictor_cols`. Can be the full, unsampled dataset —
                sampling happens inside this function.
            predictor_cols: List of numeric columns to facet over, one panel
                per column, each plotted on the x-axis against `target_col`.
            target_col: Numeric column shown on the y-axis of every panel.
                Defaults to "Npp".
            sample_size: Number of rows to randomly sample (with a fixed seed
                for reproducibility) before reshaping and plotting. Defaults
                to 5,000. Note this is the *total* row count sampled, not
                per-predictor — after unpivoting, each panel will show up to
                this many points (fewer if some predictors have nulls).
            facet_columns: Number of panels per row in the grid. Defaults to 3.
            seed: Random seed used for sampling. Defaults to 42.

        Returns:
            An Altair faceted scatter chart, ready for `mo.ui.altair_chart`.
        """

        # Sample first, since a scatter plots raw points rather than aggregating
        df_scatter_sample = full_df.sample(n=sample_size, seed=seed)

        scatter_long_df = (
            df_scatter_sample.select([target_col] + predictor_cols)
            .drop_nulls()
            .unpivot(
                index=target_col,
                on=predictor_cols,
                variable_name="scatter_facet_label",
                value_name="scatter_x_value",
            )
        )

        return (
            alt.Chart(scatter_long_df.to_pandas())
            .mark_circle(size=8, opacity=0.2)
            .encode(
                x=alt.X("scatter_x_value:Q", title=None),
                y=alt.Y(f"{target_col}:Q", title=target_col),
            )
            .properties(width=200, height=180)
             .facet(
                facet=alt.Facet("scatter_facet_label:N", header=alt.Header(title=None)),
                columns=facet_columns
            )
            .resolve_scale(x="independent", y="independent")
        )

    return (make_grid_scatter,)


@app.cell
def _(alt, pl):
    # ---------- Boxplot (binned showing bin starts, full data, 0.99 of predictor var) ----------
    def make_pair_boxplot(full_df: pl.DataFrame, response_col: str, predictor_col: str, chart_title: str, percentage_to_viz: float = 0.99, bin_count : int = 16, px_per_bin: int = 20) -> alt.Chart:
        """Build a binned boxplot of a response variable against a predictor.

        The predictor is bucketed into evenly-spaced bins (computed in polars
        so this scales to full datasets without embedding raw rows in the
        chart), and each bin's box is drawn from the response variable's
        quartiles within that bin. Whiskers follow the standard Tukey
        convention (1.5x IQR), clipped to the bin's observed min/max so they
        never extend past real data. Extreme predictor outliers are excluded
        via `percentage_to_viz` so a long tail doesn't compress the bins that
        contain most of the data.

        Args:
            full_df: Source dataframe containing at least `response_col` and
                `predictor_col`. Can be the full, unsampled dataset.
            response_col: Name of the numeric column shown on the y-axis
                (box quartiles, median, and whiskers are computed on this).
            predictor_col: Name of the numeric column to bin along the x-axis.
            chart_title: Title displayed above the chart.
            percentage_to_viz: Quantile cutoff (0-1) applied to `predictor_col`
                before binning, used to exclude extreme outliers from the
                visualized range. Defaults to 0.99 (top 1% excluded).
            bin_count: Number of bins to divide the (filtered) predictor range
                into. Defaults to 16.
            px_per_bin: Bin width defined as number of pixels per bin. Defaults to 22.

        Returns:
            An Altair chart layering whiskers, IQR box, and median tick per
            bin, ready to be combined with other charts (e.g. via `|` or `&`)
            or passed to `mo.ui.altair_chart`.
        """

        filtered_df = full_df.select([response_col, predictor_col]).drop_nulls().filter(pl.col(predictor_col) <= pl.col(predictor_col).quantile(percentage_to_viz))
        x_min, x_max = float(filtered_df[predictor_col].min()), float(filtered_df[predictor_col].max())
        x_step = (x_max - x_min) / bin_count if x_max > x_min else 1.0

        summary = (
            filtered_df.select([response_col, predictor_col])
            .drop_nulls()
            .with_columns(
                (((pl.col(predictor_col) - x_min) / x_step).floor() * x_step + x_min)
                .round(1)
                .alias("pair_bin_start")
            )
            .group_by("pair_bin_start")
            .agg([
                pl.col(response_col).quantile(0.25).alias("pair_q1"),
                pl.col(response_col).quantile(0.50).alias("pair_median"),
                pl.col(response_col).quantile(0.75).alias("pair_q3"),
                pl.col(response_col).min().alias("pair_data_min"),
                pl.col(response_col).max().alias("pair_data_max"),
            ])
            .with_columns((pl.col("pair_q3") - pl.col("pair_q1")).alias("pair_iqr"))
            .with_columns([
                pl.max_horizontal(pl.col("pair_data_min"), pl.col("pair_q1") - 1.5 * pl.col("pair_iqr")).alias("pair_whisker_low"),
                pl.min_horizontal(pl.col("pair_data_max"), pl.col("pair_q3") + 1.5 * pl.col("pair_iqr")).alias("pair_whisker_high"),
            ])
            .sort("pair_bin_start")
            .to_pandas()
        )
    
        chart_width = max(280, bin_count * px_per_bin)

        base = alt.Chart(summary).encode(x=alt.X("pair_bin_start:O", title=predictor_col, axis=alt.Axis(labelAngle=-45)))
        whiskers = base.mark_rule().encode(y=alt.Y("pair_whisker_low:Q", title=response_col), y2="pair_whisker_high:Q")
        iqr_bar = base.mark_bar(size=min(14, px_per_bin * 0.6)).encode(y="pair_q1:Q", y2="pair_q3:Q")
        median_tick = base.mark_tick(color="white", size=min(14, px_per_bin * 0.6), thickness=2).encode(y="pair_median:Q")

        return (whiskers + iqr_bar + median_tick).properties(width=chart_width, height=280, title=chart_title)

    return (make_pair_boxplot,)


@app.cell
def _(alt, pl):
    def make_grid_boxplot(
        full_df: pl.DataFrame,
        response_col: str,
        predictor_cols: list[str],
        bin_count: int = 12,
        percentage_to_viz: float = 1.0,
        facet_columns: int = 3,
        px_per_bin: int = 18,
    ) -> alt.Chart:
        """Build a faceted grid of binned boxplots, one per predictor column.

        Each predictor is independently binned (its own range, own bin edges)
        so variables with very different scales don't distort each other's
        panel, matching the per-column binning approach used elsewhere. Bin
        aggregation happens in polars, so this scales to full-size datasets
        without embedding raw rows into the chart.

        Args:
            full_df: Source dataframe containing `response_col` and every
                column in `predictor_cols`. Can be the full, unsampled dataset.
            response_col: Numeric column shown on the y-axis of every panel.
            predictor_cols: List of numeric columns to facet over, one panel
                per column.
            bin_count: Number of bins per predictor. Defaults to 12.
            percentage_to_viz: Quantile cutoff (0-1) applied to each predictor
                before binning, to exclude extreme outliers. Defaults to 0.99.
            facet_columns: Number of panels per row in the grid. Defaults to 4.
            px_per_bin: Pixels of width allotted per bin, used to size each
                panel so bars don't get squished at high bin counts.

        Returns:
            An Altair faceted chart, ready for `mo.ui.altair_chart`.
        """
        box_frames = []
        for box_col in predictor_cols:
            filtered = (
                full_df.select([response_col, box_col])
                .drop_nulls()
                .filter(pl.col(box_col) <= pl.col(box_col).quantile(percentage_to_viz))
            )
            box_x_min, box_x_max = float(filtered[box_col].min()), float(filtered[box_col].max())
            box_x_step = (box_x_max - box_x_min) / bin_count if box_x_max > box_x_min else 1.0

            box_frame = (
                filtered
                .with_columns(
                    (((pl.col(box_col) - box_x_min) / box_x_step).floor() * box_x_step + box_x_min)
                    .cast(pl.Float64)
                    .round(1)
                    .alias("box_bin_start")
                )
                .group_by("box_bin_start")
                .agg([
                    pl.col(response_col).quantile(0.25).alias("box_q1"),
                    pl.col(response_col).quantile(0.50).alias("box_median"),
                    pl.col(response_col).quantile(0.75).alias("box_q3"),
                    pl.col(response_col).min().alias("box_data_min"),
                    pl.col(response_col).max().alias("box_data_max"),
                ])
                .with_columns([
                    (pl.col("box_q3") - pl.col("box_q1")).alias("box_iqr"),
                    pl.lit(box_col).alias("box_facet_label"),
                ])
                .with_columns([
                    pl.max_horizontal(
                        pl.col("box_data_min"), pl.col("box_q1") - 1.5 * pl.col("box_iqr")
                    ).alias("box_whisker_low"),
                    pl.min_horizontal(
                        pl.col("box_data_max"), pl.col("box_q3") + 1.5 * pl.col("box_iqr")
                    ).alias("box_whisker_high"),
                ])
            )
            box_frames.append(box_frame)

        box_long_df = pl.concat(box_frames).sort(["box_facet_label", "box_bin_start"]).to_pandas()

        panel_width = max(180, bin_count * px_per_bin)
        bar_size = min(14, px_per_bin * 0.6)

        box_base = alt.Chart(box_long_df).encode(
            x=alt.X("box_bin_start:O", title=None, axis=alt.Axis(labelAngle=-45))
        )
        box_whiskers = box_base.mark_rule().encode(
            y=alt.Y("box_whisker_low:Q", title=response_col),
            y2="box_whisker_high:Q",
        )
        box_iqr_bar = box_base.mark_bar(size=bar_size).encode(
            y="box_q1:Q",
            y2="box_q3:Q",
        )
        box_median_tick = box_base.mark_tick(color="white", size=bar_size, thickness=2).encode(
            y="box_median:Q",
        )

        return (
            (box_whiskers + box_iqr_bar + box_median_tick)
            .properties(width=panel_width, height=180)
            .facet(
                facet=alt.Facet("box_facet_label:N", header=alt.Header(title=None)),
                columns=facet_columns,
            )
            .resolve_scale(x="independent", y="independent")
        )

    return (make_grid_boxplot,)


@app.cell
def _(alt, pl):
    def make_threshold_boxplot(
        full_df: pl.DataFrame,
        predictor_col: str,
        response_col: str,
        threshold: float,
        chart_title: str,
    ) -> alt.Chart:
        """Build a two-group boxplot of a response variable, split by a predictor threshold.

        Rows are split into two categories based on whether `predictor_col` is
        less-than-or-equal-to or greater than `threshold`, and quartiles/whiskers
        for `response_col` are computed per group in polars (so this scales to
        full-size datasets without embedding raw rows). Whiskers follow the
        Tukey 1.5x IQR convention, clipped to each group's observed min/max.

        Args:
            full_df: Source dataframe containing `predictor_col` and `response_col`.
            predictor_col: Numeric column used to split rows into two groups.
            response_col: Numeric column shown on the y-axis (quartiles computed on this).
            threshold: Cutoff value; rows with predictor <= threshold form one
                group, rows with predictor > threshold form the other.
            chart_title: Title displayed above the chart.

        Returns:
            An Altair chart with two boxes (one per group), ready to be
            combined with other charts or passed to `mo.ui.altair_chart`.
        """
        summary = (
            full_df.select([predictor_col, response_col])
            .drop_nulls()
            .with_columns(
                pl.when(pl.col(predictor_col) <= threshold)
                .then(pl.lit(f"<= {threshold}"))
                .otherwise(pl.lit(f"> {threshold}"))
                .alias("cv_group")
            )
            .group_by("cv_group")
            .agg([
                pl.col(response_col).quantile(0.25).alias("cv_q1"),
                pl.col(response_col).quantile(0.50).alias("cv_median"),
                pl.col(response_col).quantile(0.75).alias("cv_q3"),
                pl.col(response_col).min().alias("cv_data_min"),
                pl.col(response_col).max().alias("cv_data_max"),
            ])
            .with_columns((pl.col("cv_q3") - pl.col("cv_q1")).alias("cv_iqr"))
            .with_columns([
                pl.max_horizontal(pl.col("cv_data_min"), pl.col("cv_q1") - 1.5 * pl.col("cv_iqr")).alias("cv_whisker_low"),
                pl.min_horizontal(pl.col("cv_data_max"), pl.col("cv_q3") + 1.5 * pl.col("cv_iqr")).alias("cv_whisker_high"),
            ])
            .sort("cv_group")
            .to_pandas()
        )

        base = alt.Chart(summary).encode(x=alt.X("cv_group:N", title=predictor_col, sort=None))
        whiskers = base.mark_rule().encode(y=alt.Y("cv_whisker_low:Q", title=response_col), y2="cv_whisker_high:Q")
        iqr_bar = base.mark_bar(size=40).encode(y="cv_q1:Q", y2="cv_q3:Q")
        median_tick = base.mark_tick(color="white", size=40, thickness=2).encode(y="cv_median:Q")

        return (whiskers + iqr_bar + median_tick).properties(width=180, height=280, title=chart_title)

    return (make_threshold_boxplot,)


@app.cell
def _(df, mo):
    continent_options = ["All"] + sorted(v for v in df["continent"].unique().to_list() if v not in [None, "Antarctica"])

    continent_dropdown = mo.ui.dropdown(
        options=continent_options,
        value="All",
        label="Continent",
    )

    continent_dropdown
    return (continent_dropdown,)


@app.cell
def _(continent_dropdown, df, pl):
    if continent_dropdown.value == "All":
        df_continent = df
    else:
        df_continent = df.filter(pl.col("continent") == continent_dropdown.value)
    return (df_continent,)


@app.cell
def _(mo):
    mo.md(text="### Influence of other covariates on NPP")
    return


@app.cell
def _(df, df_continent, make_grid_scatter, mo):
    scatter_grid_sample_size = min(4_000, df_continent.height)

    scatter_predictor_cols = [c for c in df.columns if c.startswith("tmm") or c.startswith("veg_tmm")]
    scatter_predictor_cols += ["vegetation_length", "elevation_mean"]

    npp_scatter_grid = make_grid_scatter(df_continent, scatter_predictor_cols, sample_size=scatter_grid_sample_size)
    mo.ui.altair_chart(npp_scatter_grid)
    return


@app.cell
def _(df, df_continent, make_grid_boxplot, mo):
    box_predictor_cols1 = [c for c in df.columns if c.startswith("tmm") or c.startswith("veg_tmm")]
    box_predictor_cols1 += ["vegetation_length", "elevation_mean"]

    box_grid1 = make_grid_boxplot(df_continent, "Npp", box_predictor_cols1)
    mo.ui.altair_chart(box_grid1)
    return


@app.cell
def _(mo):
    mo.md(text="### Relation between precipitation and precipitation variability")
    return


@app.cell
def _(df_continent, make_pair_boxplot, make_pair_scatter, mo):
    pair_scatter_pr_vs_cv = make_pair_scatter(df_continent, "pr_sum_cv", "pr_sum", "Yearly precipitation vs. CV precipitation", percentage_to_viz=0.95)
    pair_scatter_vegpr_vs_cv = make_pair_scatter(df_continent, "veg_pr_sum_cv", "veg_pr_sum", "Precipitation vs. CV precipitation in vegetation period", percentage_to_viz=0.95)
    pair_box_pr_vs_cv = make_pair_boxplot(df_continent, "pr_sum_cv", "pr_sum", "Yearly precipitation vs. CV precipitation", bin_count=20, percentage_to_viz=0.95)
    pair_box_vegpr_vs_cv = make_pair_boxplot(df_continent, "veg_pr_sum_cv", "veg_pr_sum", "Precipitation vs. CV precipitation in vegetation period", bin_count=20, percentage_to_viz=0.95)

    pair_scatter_pr_vs_cv_row = pair_scatter_pr_vs_cv | pair_scatter_vegpr_vs_cv 
    pair_box_pr_vs_cv_row = pair_box_pr_vs_cv | pair_box_vegpr_vs_cv

    mo.ui.altair_chart(pair_scatter_pr_vs_cv_row & pair_box_pr_vs_cv_row)
    return


@app.cell
def _(mo):
    mo.md(text="### Influence of precipitation (variability) on NPP")
    return


@app.cell
def _(df_continent, make_pair_scatter, mo):
    pair_scatter_pr = make_pair_scatter(df_continent, "Npp", "pr_sum", "Npp vs. pr_sum")
    pair_scatter_vegpr = make_pair_scatter(df_continent, "Npp", "veg_pr_sum", "Npp vs. veg_pr_sum")
    pair_scatter_prcv = make_pair_scatter(df_continent, "Npp", "pr_sum_cv", "Npp vs. pr_sum_cv")
    pair_scatter_vegprcv = make_pair_scatter(df_continent, "Npp", "veg_pr_sum_cv", "Npp vs. veg_pr_sum_cv")


    pair_scatter_row1 = pair_scatter_pr | pair_scatter_vegpr
    pair_scatter_row2 = pair_scatter_prcv | pair_scatter_vegprcv

    mo.ui.altair_chart(pair_scatter_row1 & pair_scatter_row2)
    return


@app.cell
def _(df_continent, make_pair_boxplot, mo):
    pair_box_pr = make_pair_boxplot(df_continent, "Npp", "pr_sum", "Npp vs. pr_sum", bin_count=20)
    pair_box_vegpr = make_pair_boxplot(df_continent, "Npp", "veg_pr_sum", "Npp vs. veg_pr_sum", bin_count=20)
    pair_box_prcv = make_pair_boxplot(df_continent, "Npp", "pr_sum_cv", "Npp vs. pr_sum_cv", bin_count=20)
    pair_box_vegprcv = make_pair_boxplot(df_continent, "Npp", "veg_pr_sum_cv", "Npp vs. veg_pr_sum_cv", bin_count=20)

    pair_box_row1 = pair_box_pr | pair_box_vegpr
    pair_box_row2 = pair_box_prcv | pair_box_vegprcv

    mo.ui.altair_chart(pair_box_row1 & pair_box_row2)
    return


@app.cell
def _(mo):
    mo.md(text="### Influence of precipitation (variability) on NPP variability")
    return


@app.cell
def _(df_continent, make_pair_scatter, mo):
    pair_scatter_pr_nppcv = make_pair_scatter(df_continent, "Npp_cv", "pr_sum", "Npp_cv vs. pr_sum")
    pair_scatter_vegpr_nppcv = make_pair_scatter(df_continent, "Npp_cv", "veg_pr_sum", "Npp_cv vs. veg_pr_sum")
    pair_scatter_prcv_nppcv = make_pair_scatter(df_continent, "Npp_cv", "pr_sum_cv", "Npp_cv vs. pr_sum_cv")
    pair_scatter_vegprcv_nppcv = make_pair_scatter(df_continent, "Npp_cv", "veg_pr_sum_cv", "Npp_cv vs. veg_pr_sum_cv")


    pair_scatter_nppcv_row1 = pair_scatter_pr_nppcv | pair_scatter_vegpr_nppcv
    pair_scatter_nppcv_row2 = pair_scatter_prcv_nppcv | pair_scatter_vegprcv_nppcv

    mo.ui.altair_chart(pair_scatter_nppcv_row1 & pair_scatter_nppcv_row2)
    return


@app.cell
def _(df_continent, make_pair_boxplot, mo):
    pair_box_pr_nppcv = make_pair_boxplot(df_continent, "Npp_cv", "pr_sum", "Npp_cv vs. pr_sum", bin_count=20)
    pair_box_vegpr_nppcv = make_pair_boxplot(df_continent, "Npp_cv", "veg_pr_sum", "Npp_cv vs. veg_pr_sum", bin_count=20)
    pair_box_prcv_nppcv = make_pair_boxplot(df_continent, "Npp_cv", "pr_sum_cv", "Npp_cv vs. pr_sum_cv", bin_count=20)
    pair_box_vegprcv_nppcv = make_pair_boxplot(df_continent, "Npp_cv", "veg_pr_sum_cv", "Npp_cv vs. veg_pr_sum_cv", bin_count=20)

    pair_box_nppcv_row1 = pair_box_pr_nppcv | pair_box_vegpr_nppcv
    pair_box_nppcv_row2 = pair_box_prcv_nppcv | pair_box_vegprcv_nppcv

    mo.ui.altair_chart(pair_box_nppcv_row1 & pair_box_nppcv_row2)
    return


@app.cell
def _(df_continent, make_threshold_boxplot, mo):
    cv_threshold = 33

    box_pr_npp = make_threshold_boxplot(df_continent, "pr_sum_cv", "Npp", cv_threshold, "Npp by pr_sum_cv")
    box_pr_nppcv = make_threshold_boxplot(df_continent, "pr_sum_cv", "Npp_cv", cv_threshold, "Npp_cv by pr_sum_cv")
    box_vegpr_npp = make_threshold_boxplot(df_continent, "veg_pr_sum_cv", "Npp", cv_threshold, "Npp by veg_pr_sum_cv")
    box_vegpr_nppcv = make_threshold_boxplot(df_continent, "veg_pr_sum_cv", "Npp_cv", cv_threshold, "Npp_cv by veg_pr_sum_cv")

    cv_box_grid = (box_pr_npp | box_vegpr_npp) & (box_pr_nppcv | box_vegpr_nppcv)

    mo.ui.altair_chart(cv_box_grid)
    return


@app.cell
def _(mo):
    mo.md(text="## PCA")
    return


@app.cell
def _(df):
    # ---------- Prepare data (same selection as before: numeric, excluding year and Npp_QC) ----------

    pca_excluded_cols = {"year", "Npp_QC"}
    pca_numeric_cols = [
        c for c, dt in df.schema.items()
        if dt.is_numeric() and c not in pca_excluded_cols
    ]

    pca_input_pd = df.select(pca_numeric_cols).drop_nulls().to_pandas()
    print(f"Using {len(pca_numeric_cols)} numeric columns on {len(pca_input_pd)} complete rows")

    return pca_input_pd, pca_numeric_cols


@app.cell
def _(pca, pca_input_pd):
    # ---------- Fit PCA ----------

    # normalize=True handles standardization internally (equivalent to StandardScaler)
    pca_model = pca(n_components=0.95, normalize=True)  # keep components explaining 95% of variance
    pca_results = pca_model.fit_transform(pca_input_pd)
    return (pca_model,)


@app.cell
def _(pca_model):
    # ---------- Scree plot ----------

    pca_model.plot()
    return


@app.cell
def _(pca_model, pca_numeric_cols):
    # ---------- Biplot with variable arrows ----------

    pca_model.biplot(
        n_feat=len(pca_numeric_cols),
        s=0,  # marker size 0 = points invisible
    )
    return


@app.cell
def _(mo):
    mo.md(text="## Maps - Coefficient of Variance")
    return


@app.cell
def _(cv):
    # ---------- Calculate difference between band 1 and band 2 ----------

    cv_diff = cv.isel(band=0) - cv.isel(band=1)

    print("Difference min:", float(cv_diff.min()), "max:", float(cv_diff.max()))
    return (cv_diff,)


@app.cell
def _(cv, cv_diff, cv_hvw, plt):
    # ---------- Build the 3-panel figure ----------

    fig, axes = plt.subplots(5, 1, figsize=(6, 14))

    # HvW CV precipitation
    cv_hvw.plot(
        ax=axes[0], cmap="viridis",
        vmin=0, vmax=50,
        cbar_kwargs={"label": "CV (%)"}
    )
    axes[0].set_title("CV (%) — HVW (external source)")

    # my CV precipitation in the year
    cv_first_band = cv.isel(band=0)  # first band by position
    cv_first_band.plot(
        ax=axes[1], cmap="viridis",
        vmin=0, vmax=50,
        cbar_kwargs={"label": "CV (%)"}
    )
    axes[1].set_title(f"CV (%) — Precipitation in the year, 2002–2018")

    # my CV precipitation in the vegetation period
    cv_second_band = cv.isel(band=1)  # second band by position
    cv_second_band.plot(
        ax=axes[2], cmap="viridis",
        vmin=0, vmax=50,
        cbar_kwargs={"label": "CV (%)"}
    )
    axes[2].set_title(f"CV (%) — Precipitation in the vegetation period, 2002–2018")

    # my CV NPP
    cv_third_band = cv.isel(band=2)  # third band by position
    cv_third_band.plot(
        ax=axes[3], cmap="viridis",
        vmin=0, vmax=50,
        cbar_kwargs={"label": "CV (%)"}
    )
    axes[3].set_title(f"CV (%) — Net primary productivity, 2002–2018")

    # difference between CV precipitation in the year vs. vegetation period
    # positive values indicate year has higher CV, negative values indicate vegetation period has higher CV
    cv_diff_abs_max = float(max(abs(cv_diff.min()), abs(cv_diff.max())))
    cv_diff.plot(
        ax=axes[4], cmap="RdBu_r",
        vmin=-20, vmax=20,
        cbar_kwargs={"label": "CV difference (%)"}
    )
    axes[4].set_title(f"Difference: CV precipitation in the year vs. in the vegetation period")

    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(text="## Trash")
    return


@app.cell
def _(alt, df, mo, pl):
    density_predictor_cols = [c for c in df.columns if c.startswith("tmm") or c.startswith("veg_tmm")]
    density_predictor_cols += ["vegetation_length", "elevation_mean"]

    bin_count = 60

    y_axis_min, y_axis_max = float(df["Npp"].min()), float(df["Npp"].max())
    y_axis_step = (y_axis_max - y_axis_min) / bin_count if y_axis_max > y_axis_min else 1.0

    density_frames = []
    for predictor_col in density_predictor_cols:
        x_axis_min, x_axis_max = float(df[predictor_col].min()), float(df[predictor_col].max())
        x_axis_step = (x_axis_max - x_axis_min) / bin_count if x_axis_max > x_axis_min else 1.0

        frame = (
            df.select(["Npp", predictor_col])
            .drop_nulls()
            .with_columns(
                (((pl.col(predictor_col) - x_axis_min) / x_axis_step).floor() * x_axis_step + x_axis_min)
                .cast(pl.Float64)
                .alias("x_bin"),
                (((pl.col("Npp") - y_axis_min) / y_axis_step).floor() * y_axis_step + y_axis_min)
                .cast(pl.Float64)
                .alias("y_bin"),
            )
            .group_by(["x_bin", "y_bin"])
            .agg(pl.len().alias("bin_count_val"))
            .with_columns(pl.lit(predictor_col).alias("facet_label"))
        )
        density_frames.append(frame)

    density_long = pl.concat(density_frames)

    density_facet_grid = (
        alt.Chart(density_long.to_pandas())
        .mark_rect()
        .encode(
            x=alt.X("x_bin:Q", title=None),
            y=alt.Y("y_bin:Q", title="Npp"),
            color=alt.Color("bin_count_val:Q", scale=alt.Scale(scheme="viridis", type = "log"), title="Count"),
        )
        .properties(width=200, height=180)
        .facet(facet="facet_label:N", columns=4)
        .resolve_scale(x="independent", y="independent", color="independent")
    )

    mo.ui.altair_chart(density_facet_grid)
    return


if __name__ == "__main__":
    app.run()
