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

    return alt, mo, pl


@app.cell
def _(alt):
    alt.data_transformers.enable("vegafusion")
    return


@app.cell
def _(mo):
    mo.md(text="## Load data")
    return


@app.cell
def _(pl):
    df = pl.read_parquet("/Users/Wanja/Documents/non-equilibrium_data/tables_wgs84_new/table_wgs84_2002.parquet")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md(text="## Inspect distributions")
    return


@app.cell
def _(alt, df, mo):
    chart_one = alt.Chart(df).mark_bar().encode(
        alt.X("pr_sum", bin=alt.Bin(maxbins=50), title="pr_sum"),
        alt.Y("count()", title="Count"),
    ).properties(title="Distribution of pr_sum")

    mo.ui.altair_chart(chart_one)
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
        .facet(facet="variable:N", columns=4)
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
def _(alt, df, mo):
    scatter_predictor_cols = [c for c in df.columns if c.startswith("tmm") or c.startswith("veg_tmm")]
    scatter_predictor_cols += ["vegetation_length", "elevation_mean"]

    # Sample first, since a scatter plots raw points rather than aggregating
    sample_size = 5_000
    df_scatter_sample = df.sample(n=sample_size, seed=42)

    scatter_long_df = (
        df_scatter_sample.select(["Npp"] + scatter_predictor_cols)
        .drop_nulls()
        .unpivot(
            index="Npp",
            on=scatter_predictor_cols,
            variable_name="scatter_facet_label",
            value_name="scatter_x_value",
        )
    )

    npp_scatter_grid = (
        alt.Chart(scatter_long_df.to_pandas())
        .mark_circle(size=8, opacity=0.2)
        .encode(
            x=alt.X("scatter_x_value:Q", title=None),
            y=alt.Y("Npp:Q", title="Npp"),
        )
        .properties(width=200, height=180)
        .facet(facet="scatter_facet_label:N", columns=4)
        .resolve_scale(x="independent", y="independent")
    )

    mo.ui.altair_chart(npp_scatter_grid)
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


@app.cell
def _(alt, df, mo, pl):
    box_predictor_cols = [c for c in df.columns if c.startswith("tmm") or c.startswith("veg_tmm")]
    box_predictor_cols += ["vegetation_length", "elevation_mean"]

    box_bin_count = 12

    box_frames = []
    for box_col in box_predictor_cols:
        box_x_min, box_x_max = float(df[box_col].min()), float(df[box_col].max())
        box_x_step = (box_x_max - box_x_min) / box_bin_count if box_x_max > box_x_min else 1.0

        box_frame = (
            df.select(["Npp", box_col])
            .drop_nulls()
            .with_columns(
                (((pl.col(box_col) - box_x_min) / box_x_step).floor() * box_x_step + box_x_min)
                .cast(pl.Float64)
                .round(1)
                .alias("box_bin_start")
            )
            .group_by("box_bin_start")
            .agg([
                pl.col("Npp").quantile(0.25).alias("box_q1"),
                pl.col("Npp").quantile(0.50).alias("box_median"),
                pl.col("Npp").quantile(0.75).alias("box_q3"),
                pl.col("Npp").min().alias("box_data_min"),
                pl.col("Npp").max().alias("box_data_max"),
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

    box_base = alt.Chart(box_long_df).encode(
        x=alt.X("box_bin_start:O", title=None)
    )

    box_whiskers = box_base.mark_rule().encode(
        y=alt.Y("box_whisker_low:Q", title="Npp"),
        y2="box_whisker_high:Q",
    )

    box_iqr_bar = box_base.mark_bar(size=14).encode(
        y="box_q1:Q",
        y2="box_q3:Q",
    )

    box_median_tick = box_base.mark_tick(color="white", size=14, thickness=2).encode(
        y="box_median:Q",
    )

    box_grid = (
        (box_whiskers + box_iqr_bar + box_median_tick)
        .properties(width=220, height=180)
        .facet(facet="box_facet_label:N", columns=4)
        .resolve_scale(x="independent", y="independent")
    )

    mo.ui.altair_chart(box_grid)
    return


@app.cell
def _(alt, df, mo, pl):
    pair_target_col = "Npp"
    pair_predictor_cols = ["pr_sum", "veg_pr_sum"]


    # ---------- Scatter (sampled) ----------

    pair_sample_size = 20_000  # per predictor, kept modest since we're building 2 panels
    pair_df_sample = df.sample(n=pair_sample_size, seed=42)

    def make_pair_scatter(sample_df: pl.DataFrame, predictor_col: str, chart_title: str) -> alt.Chart:
        local_df = sample_df.select([pair_target_col, predictor_col]).drop_nulls().filter(pl.col(predictor_col) <= pl.col(predictor_col).quantile(0.95)).to_pandas()
        return (
            alt.Chart(local_df)
            .mark_circle(size=10, opacity=0.25)
            .encode(
                x=alt.X(f"{predictor_col}:Q", title=predictor_col),
                y=alt.Y(f"{pair_target_col}:Q", title=pair_target_col),
            )
            .properties(width=280, height=280, title=chart_title)
        )

    pair_scatter_pr = make_pair_scatter(pair_df_sample, "pr_sum", "Npp vs. pr_sum")
    pair_scatter_vegpr = make_pair_scatter(pair_df_sample, "veg_pr_sum", "Npp vs. veg_pr_sum")

    pair_scatter_row = pair_scatter_pr | pair_scatter_vegpr

    mo.ui.altair_chart(pair_scatter_row)
    return (pair_target_col,)


@app.cell
def _(alt, df, mo, pair_target_col, pl):
    # ---------- Boxplot (binned, full data) ----------

    pair_bin_count = 16

    def make_pair_boxplot(full_df: pl.DataFrame, predictor_col: str, chart_title: str) -> alt.Chart:
        filtered_df = full_df.select([pair_target_col, predictor_col]).drop_nulls().filter(pl.col(predictor_col) <= pl.col(predictor_col).quantile(0.95))
        x_min, x_max = float(filtered_df[predictor_col].min()), float(filtered_df[predictor_col].max())
        x_step = (x_max - x_min) / pair_bin_count if x_max > x_min else 1.0

        summary = (
            filtered_df.select([pair_target_col, predictor_col])
            .drop_nulls()
            .with_columns(
                (((pl.col(predictor_col) - x_min) / x_step).floor() * x_step + x_min)
                .round(1)
                .alias("pair_bin_start")
            )
            .group_by("pair_bin_start")
            .agg([
                pl.col(pair_target_col).quantile(0.25).alias("pair_q1"),
                pl.col(pair_target_col).quantile(0.50).alias("pair_median"),
                pl.col(pair_target_col).quantile(0.75).alias("pair_q3"),
                pl.col(pair_target_col).min().alias("pair_data_min"),
                pl.col(pair_target_col).max().alias("pair_data_max"),
            ])
            .with_columns((pl.col("pair_q3") - pl.col("pair_q1")).alias("pair_iqr"))
            .with_columns([
                pl.max_horizontal(pl.col("pair_data_min"), pl.col("pair_q1") - 1.5 * pl.col("pair_iqr")).alias("pair_whisker_low"),
                pl.min_horizontal(pl.col("pair_data_max"), pl.col("pair_q3") + 1.5 * pl.col("pair_iqr")).alias("pair_whisker_high"),
            ])
            .sort("pair_bin_start")
            .to_pandas()
        )

        base = alt.Chart(summary).encode(x=alt.X("pair_bin_start:O", title=predictor_col))
        whiskers = base.mark_rule().encode(y=alt.Y("pair_whisker_low:Q", title=pair_target_col), y2="pair_whisker_high:Q")
        iqr_bar = base.mark_bar(size=14).encode(y="pair_q1:Q", y2="pair_q3:Q")
        median_tick = base.mark_tick(color="white", size=14, thickness=2).encode(y="pair_median:Q")

        return (whiskers + iqr_bar + median_tick).properties(width=280, height=280, title=chart_title)

    pair_box_pr = make_pair_boxplot(df, "pr_sum", "Npp vs. pr_sum")
    pair_box_vegpr = make_pair_boxplot(df, "veg_pr_sum", "Npp vs. veg_pr_sum")

    pair_box_row = pair_box_pr | pair_box_vegpr

    mo.ui.altair_chart(pair_box_row)
    return


if __name__ == "__main__":
    app.run()
