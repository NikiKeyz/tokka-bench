"""
Chart creation utilities for the visualization dashboard.
"""

from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .constants import CHART_HEIGHT, LEGEND_CONFIG


def get_global_tokenizer_order(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> List[str]:
    """Get consistent global ordering of tokenizers for all charts."""
    filtered_df = df[df["tokenizer_key"].isin(selected_tokenizers)]
    return sorted(filtered_df["tokenizer_name"].unique())


def prepare_chart_data(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> pd.DataFrame:
    """Prepare and sort dataframe with consistent tokenizer ordering for charts."""
    filtered_df = df[df["tokenizer_key"].isin(selected_tokenizers)].copy()

    # Apply consistent categorical ordering
    global_order = get_global_tokenizer_order(df, selected_tokenizers)
    filtered_df["tokenizer_name"] = pd.Categorical(
        filtered_df["tokenizer_name"], categories=global_order, ordered=True
    )

    # Sort to ensure consistent color assignment and preserve language presentation order
    sort_cols = ["tokenizer_name"]
    if "language_rank" in filtered_df.columns:
        sort_cols = ["language_rank"] + sort_cols
    return filtered_df.sort_values(sort_cols)


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    y_label: str,
    x_label: str = "Language",
) -> go.Figure:
    """Create a standardized bar chart with consistent styling."""
    # Prepare optional wrapped preview for nicer hover tooltips
    df_plot = df.copy()
    custom_data = None
    has_preview = "sample_tokens_preview" in df_plot.columns
    if has_preview:
        # Manually wrap into multiple lines for reliable hover rendering.
        # Keep no spaces around the '|' separator, per UI preference.
        def _wrap_lines(val: str, per_line: int = 8) -> str:
            if not isinstance(val, str) or not val:
                return ""
            # Do not strip spaces inside tokens; preserve leading/trailing spaces
            parts = val.split("|")
            lines = [
                "|".join(parts[i : i + per_line])
                for i in range(0, len(parts), per_line)
            ]
            return "<br>".join(lines)

        df_plot["_sample_tokens_preview_wrapped"] = df_plot[
            "sample_tokens_preview"
        ].apply(_wrap_lines)
        custom_data = ["_sample_tokens_preview_wrapped"]

    fig = px.bar(
        df_plot,
        x=x,
        y=y,
        color="tokenizer_name",
        title=title,
        labels={y: y_label, x: x_label},
        barmode="group",
        height=CHART_HEIGHT,
        hover_data=None,
        custom_data=custom_data,
    )

    # Improve hover formatting and wrap preview lines using <br>
    if has_preview:
        hovertemplate = (
            f"{x_label}: %{{x}}<br>"
            f"{y_label}: %{{y}}<br>"
            "Sample tokens:<br>%{customdata[0]}<extra>%{fullData.name}</extra>"
        )
        fig.update_traces(hovertemplate=hovertemplate)

    fig.update_layout(
        xaxis_tickangle=-45,
        legend=LEGEND_CONFIG,
        # Create extra room above the plot so the title never overlaps the legend
        margin=dict(t=120),
        title=dict(y=0.995),
    )
    return fig


def create_efficiency_chart(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> go.Figure:
    """Create a bar chart comparing bytes per token across languages."""
    chart_data = prepare_chart_data(df, selected_tokenizers)
    return create_bar_chart(
        chart_data,
        x="lang_label",
        y="bytes_per_token",
        title="Tokenization Efficiency (Bytes per Token)",
        y_label="Bytes per Token (Higher = More Efficient)",
    )


def create_coverage_chart(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> go.Figure:
    """Create a bar chart comparing unique tokens (vocabulary coverage)."""
    chart_data = prepare_chart_data(df, selected_tokenizers)
    return create_bar_chart(
        chart_data,
        x="lang_label",
        y="unique_tokens",
        title="Vocabulary Coverage (Unique Tokens Used)",
        y_label="Unique Tokens (Higher = Better Coverage)",
    )


def create_subword_fertility_chart(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> go.Figure:
    """Create a bar chart comparing subword fertility across languages."""
    chart_data = prepare_chart_data(df, selected_tokenizers)

    # Check if subword_fertility data is available
    if (
        "subword_fertility" not in chart_data.columns
        or chart_data["subword_fertility"].isna().all()
    ):
        fig = go.Figure()
        fig.add_annotation(
            text="Subword fertility data not available<br>Run benchmarks with updated script",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title="Subword Fertility (Subwords per Word)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=CHART_HEIGHT,
        )
        return fig

    return create_bar_chart(
        chart_data,
        x="lang_label",
        y="subword_fertility",
        title="Subword Fertility (Subwords per Word)",
        y_label="Subwords per Word (Higher = More Fragmented)",
    )


def create_word_splitting_rate_chart(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> go.Figure:
    """Create a bar chart comparing word splitting rate across languages."""
    chart_data = prepare_chart_data(df, selected_tokenizers)

    # Check if word_split_pct data is available
    if (
        "word_split_pct" not in chart_data.columns
        or chart_data["word_split_pct"].isna().all()
    ):
        fig = go.Figure()
        fig.add_annotation(
            text="Word splitting rate data not available<br>Run benchmarks with updated script",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title="Word Splitting Rate (% of Units Split)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=CHART_HEIGHT,
        )
        return fig

    return create_bar_chart(
        chart_data,
        x="lang_label",
        y="word_split_pct",
        title="Word Splitting Rate (% of Units Split)",
        y_label="Word Splitting Rate (%)",
    )


def create_script_distribution_chart(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> go.Figure:
    """Create a stacked bar chart showing script distribution across tokenizers."""
    chart_data = prepare_chart_data(df, selected_tokenizers)

    # Script columns to visualize
    script_columns = [
        "tokens_with_latin_unicode_pct",
        "tokens_with_japanese_unicode_pct",
        "tokens_with_chinese_unicode_pct",
        "tokens_with_cyrillic_unicode_pct",
        "tokens_with_arabic_unicode_pct",
        "tokens_with_korean_unicode_pct",
    ]

    # Check if script data is available
    available_cols = [
        col
        for col in script_columns
        if col in chart_data.columns and not chart_data[col].isna().all()
    ]

    if not available_cols:
        fig = go.Figure()
        fig.add_annotation(
            text="Script distribution data not available<br>Run benchmarks with updated script",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title="Script Distribution in Tokenizer Vocabulary",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=CHART_HEIGHT,
        )
        return fig

    # Get unique tokenizers and their script percentages
    tokenizer_data = (
        chart_data.groupby("tokenizer_name")[available_cols].first().reset_index()
    )

    # Create stacked bar chart
    fig = go.Figure()

    script_names = {
        "tokens_with_latin_unicode_pct": "Latin",
        "tokens_with_japanese_unicode_pct": "Japanese",
        "tokens_with_chinese_unicode_pct": "Chinese",
        "tokens_with_cyrillic_unicode_pct": "Cyrillic",
        "tokens_with_arabic_unicode_pct": "Arabic",
        "tokens_with_korean_unicode_pct": "Korean",
    }

    for col in available_cols:
        fig.add_trace(
            go.Bar(
                x=tokenizer_data["tokenizer_name"],
                y=tokenizer_data[col],
                name=script_names.get(col, col),
            )
        )

    fig.update_layout(
        title=dict(text="Script Distribution in Tokenizer Vocabulary", y=0.98),
        xaxis_title="Tokenizer",
        yaxis_title="Percentage of Tokens",
        barmode="stack",
        height=CHART_HEIGHT,
        legend=LEGEND_CONFIG,
        margin=dict(t=110),
    )

    return fig


def create_vocab_metrics_chart(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> go.Figure:
    """Create a bar chart comparing tokenizer vocabulary metrics."""
    chart_data = prepare_chart_data(df, selected_tokenizers)

    # Check if vocab metrics data is available
    if (
        "tokens_without_leading_space_pct" not in chart_data.columns
        or chart_data["tokens_without_leading_space_pct"].isna().all()
    ):
        fig = go.Figure()
        fig.add_annotation(
            text="Vocabulary metrics data not available<br>Run benchmarks with updated script",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title="Vocabulary Metrics (Tokens Without Leading Space)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=CHART_HEIGHT,
        )
        return fig

    # Get unique tokenizers and their vocab metrics
    tokenizer_data = (
        chart_data.groupby("tokenizer_name")["tokens_without_leading_space_pct"]
        .first()
        .reset_index()
    )

    fig = px.bar(
        tokenizer_data,
        x="tokenizer_name",
        y="tokens_without_leading_space_pct",
        title="Vocabulary Metrics (Tokens Without Leading Space)",
        labels={
            "tokens_without_leading_space_pct": "% Tokens Without Leading Space",
            "tokenizer_name": "Tokenizer",
        },
        height=CHART_HEIGHT,
    )

    fig.update_layout(
        legend=LEGEND_CONFIG,
        margin=dict(t=120),
        title=dict(y=0.995),
    )
    return fig


def create_vocab_efficiency_scatter(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> go.Figure:
    """Create a scatter plot showing average efficiency vs vocabulary size."""
    filtered_df = df[df["tokenizer_key"].isin(selected_tokenizers)]

    # Calculate average efficiency per tokenizer
    summary_data = []
    for tokenizer in selected_tokenizers:
        tokenizer_df = filtered_df[filtered_df["tokenizer_key"] == tokenizer]
        if not tokenizer_df.empty:
            vocab_size = tokenizer_df["vocab_size"].iloc[0]
            if pd.notna(vocab_size):  # Only include if vocab_size is available
                avg_efficiency = tokenizer_df["bytes_per_token"].mean()
                tokenizer_name = tokenizer_df["tokenizer_name"].iloc[0]

                summary_data.append(
                    {
                        "tokenizer_name": tokenizer_name,
                        "vocab_size": vocab_size,
                        "avg_efficiency": avg_efficiency,
                        "languages_count": len(tokenizer_df),
                    }
                )

    if not summary_data:
        # Return empty figure if no vocab size data available
        fig = go.Figure()
        fig.add_annotation(
            text="Vocabulary size data not available<br>Run benchmarks with updated script",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title="Average Efficiency vs Vocabulary Size",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=CHART_HEIGHT,
        )
        return fig

    summary_df = pd.DataFrame(summary_data)

    # Apply consistent ordering
    global_order = get_global_tokenizer_order(df, selected_tokenizers)
    summary_df["tokenizer_name"] = pd.Categorical(
        summary_df["tokenizer_name"], categories=global_order, ordered=True
    )
    summary_df = summary_df.sort_values("tokenizer_name")

    fig = px.scatter(
        summary_df,
        x="vocab_size",
        y="avg_efficiency",
        color="tokenizer_name",
        size="languages_count",
        hover_data=["languages_count"],
        title="Average Efficiency vs Vocabulary Size",
        labels={
            "vocab_size": "Vocabulary Size (tokens)",
            "avg_efficiency": "Average Efficiency (bytes/token)",
            "languages_count": "Languages Tested",
        },
        height=CHART_HEIGHT,
    )

    fig.update_layout(
        xaxis=dict(type="log", title="Vocabulary Size (log scale)"),
        yaxis=dict(title="Average Efficiency (bytes/token)"),
        showlegend=True,
        margin=dict(t=120),
        title=dict(y=0.995),
    )

    return fig


def create_summary_table(*args, **kwargs):
    """Deprecated: summary table no longer used in the streamlined dashboard."""
    raise NotImplementedError("create_summary_table is no longer supported")


def compute_script_safety_table(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> pd.DataFrame:
    """Build a script-level worst-case table from per-language rows.

    For each (tokenizer, script) group reports min/median/max bytes/token and
    the language holding the worst case (min bytes/token -> most tokens per
    byte -> highest context-overflow risk). Coding ("code") rows are excluded.
    """
    import statistics

    sub = df[df["tokenizer_key"].isin(selected_tokenizers)].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    grouped = sub[sub["script"] != "code"].groupby(
        ["tokenizer_name", "script"], sort=False
    )
    for (tok_name, script), g in grouped:
        bpt = g["bytes_per_token"].dropna()
        if bpt.empty:
            continue
        min_idx = bpt.idxmin()
        worst_label = str(g.loc[min_idx, "lang_label"])
        rows.append(
            {
                "tokenizer": tok_name,
                "script": script,
                "num_languages": int(g["lang_key"].nunique()),
                "min_bytes_per_token": float(bpt.min()),
                "median_bytes_per_token": float(statistics.median(bpt)),
                "max_bytes_per_token": float(bpt.max()),
                "worst_case_language": worst_label,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Sort by worst case per tokenizer so the riskiest scripts surface first
    out = out.sort_values(
        ["tokenizer", "min_bytes_per_token"], ascending=[True, True]
    ).reset_index(drop=True)
    return out


def create_script_safety_chart(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> go.Figure:
    """Bar chart of the worst-case (min) bytes/token per script per tokenizer.

    The min is the realistic worst case for that script: fewest bytes per token
    means the most tokens per byte, i.e. the script most likely to overflow a
    context window. Use this to size conservative token budgets.
    """
    table = compute_script_safety_table(df, selected_tokenizers)
    if table.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Script safety data not available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title="Worst-Case (Min) Bytes per Token per Script",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=CHART_HEIGHT,
        )
        return fig

    fig = px.bar(
        table,
        x="script",
        y="min_bytes_per_token",
        color="tokenizer",
        barmode="group",
        title="Worst-Case (Min) Bytes per Token per Script",
        labels={
            "script": "Script",
            "min_bytes_per_token": "Min Bytes/Token (Lower = More Tokens/Riskier)",
            "tokenizer": "Tokenizer",
        },
        height=CHART_HEIGHT,
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        legend=LEGEND_CONFIG,
        margin=dict(t=120),
        title=dict(y=0.995),
    )
    return fig


def create_script_efficiency_chart(
    df: pd.DataFrame, selected_tokenizers: List[str]
) -> go.Figure:
    """Script-level analog of the per-language Efficiency chart.

    Aggregates bytes/token to the script level (median across that script's
    languages, with min/max error bars) so tokenizers can be compared on a
    per-script basis rather than per-language. Coding ("code") rows are excluded
    since code is not a script.
    """
    sub = df[
        (df["tokenizer_key"].isin(selected_tokenizers)) & (df["script"] != "code")
    ].copy()
    if sub.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Script efficiency data not available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title="Tokenization Efficiency by Script (Bytes per Token)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=CHART_HEIGHT,
        )
        return fig

    # Per (tokenizer, script): median (bar) plus min/max whiskers.
    grouped = sub.groupby(["tokenizer_name", "script"], sort=False)[
        "bytes_per_token"
    ]
    stats = grouped.agg(["median", "min", "max"]).reset_index()
    stats = stats.rename(
        columns={"median": "med", "min": "mn", "max": "mx"}
    )

    # Stable script order (by overall median, descending) so bars group cleanly
    script_order = (
        stats.groupby("script")["med"].mean().sort_values(ascending=False).index.tolist()
    )
    tokenizer_order = get_global_tokenizer_order(df, selected_tokenizers)

    fig = go.Figure()
    for tok in tokenizer_order:
        t = stats[stats["tokenizer_name"] == tok]
        by_script = {row["script"]: row for _, row in t.iterrows()}
        med = []
        err_plus = []
        err_minus = []
        for s in script_order:
            if s in by_script:
                row = by_script[s]
                med.append(row["med"])
                err_plus.append(row["mx"] - row["med"])
                err_minus.append(row["med"] - row["mn"])
            else:
                med.append(None)
                err_plus.append(None)
                err_minus.append(None)
        fig.add_trace(
            go.Bar(
                name=tok,
                x=script_order,
                y=med,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=err_plus,
                    arrayminus=err_minus,
                    thickness=1.2,
                    width=3,
                ),
            )
        )

    fig.update_layout(
        barmode="group",
        title=dict(text="Tokenization Efficiency by Script (Bytes per Token)", y=0.995),
        xaxis_tickangle=-45,
        xaxis_title="Script",
        yaxis_title="Bytes/Token (median, whiskers = min-max)",
        legend=LEGEND_CONFIG,
        margin=dict(t=120),
        height=CHART_HEIGHT,
    )
    return fig
