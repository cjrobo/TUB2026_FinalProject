from pathlib import Path

import pandas as pd
import seaborn as sns
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION AND VISUAL STYLE
# ---------------------------------------------------------

st.set_page_config(
    page_title="NBA Salaries and Performance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Viridis colours reordered so that a light green is used first.
PLOTLY_VIRIDIS = [
    px.colors.sequential.Viridis[7],
    px.colors.sequential.Viridis[6],
    px.colors.sequential.Viridis[8],
    px.colors.sequential.Viridis[5],
    px.colors.sequential.Viridis[4],
    px.colors.sequential.Viridis[3],
    px.colors.sequential.Viridis[2],
    px.colors.sequential.Viridis[1],
]

SEABORN_VIRIDIS = sns.color_palette("viridis", n_colors=10)
SEABORN_VIRIDIS = [
    SEABORN_VIRIDIS[7],
    SEABORN_VIRIDIS[6],
    SEABORN_VIRIDIS[8],
    SEABORN_VIRIDIS[5],
    SEABORN_VIRIDIS[4],
    SEABORN_VIRIDIS[3],
    SEABORN_VIRIDIS[2],
    SEABORN_VIRIDIS[1],
]

sns.set_theme(style="whitegrid")
sns.set_palette(SEABORN_VIRIDIS)

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PLOTLY_VIRIDIS
px.defaults.color_continuous_scale = px.colors.sequential.Viridis

PROJECT_GREEN = PLOTLY_VIRIDIS[0]

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }

        .hero {
            padding: 2rem 2.2rem;
            border-radius: 18px;
            background: linear-gradient(
                120deg,
                rgba(53, 183, 121, 0.18),
                rgba(122, 209, 81, 0.08)
            );
            border: 1px solid rgba(53, 183, 121, 0.28);
            margin-bottom: 1.3rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.6rem;
        }

        .hero p {
            margin-top: 0.65rem;
            margin-bottom: 0;
            font-size: 1.08rem;
            line-height: 1.65;
        }

        .insight-card {
            padding: 1.15rem 1.3rem;
            border-radius: 14px;
            border-left: 5px solid #35B779;
            background-color: rgba(53, 183, 121, 0.08);
            margin-top: 0.7rem;
            margin-bottom: 1.3rem;
        }

        .insight-card strong {
            color: #238A5A;
        }

        .section-divider {
            margin: 2.4rem 0 1.5rem 0;
            border-top: 1px solid rgba(120, 120, 120, 0.25);
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(53, 183, 121, 0.25);
            padding: 0.8rem 1rem;
            border-radius: 12px;
            background-color: rgba(53, 183, 121, 0.04);
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# DATA LOADING AND PREPARATION
# ---------------------------------------------------------

@st.cache_data
def read_csv_from_path(path_string):
    return pd.read_csv(path_string)


@st.cache_data
def read_csv_from_upload(uploaded_file):
    return pd.read_csv(uploaded_file)


@st.cache_data
def prepare_data(players_raw, salaries_raw):
    players = players_raw.copy()
    salaries = salaries_raw.copy()

    players = players.drop(columns=["index"], errors="ignore")
    salaries = salaries.drop(columns=["index"], errors="ignore")

    players = players.rename(columns={"_id": "player_id"})

    required_player_columns = {
        "player_id", "name", "birthDate", "position", "draft_pick",
        "career_G", "career_PER", "career_PTS", "career_AST",
        "career_TRB", "career_FG%", "career_FG3%",
        "career_FT%", "career_eFG%"
    }

    required_salary_columns = {
        "player_id", "salary", "season", "season_start",
        "season_end", "team"
    }

    missing_players = required_player_columns.difference(players.columns)
    missing_salaries = required_salary_columns.difference(salaries.columns)

    if missing_players:
        raise ValueError(
            "players.csv is missing: " + ", ".join(sorted(missing_players))
        )

    if missing_salaries:
        raise ValueError(
            "salaries_1985to2018.csv is missing: "
            + ", ".join(sorted(missing_salaries))
        )

    salaries["salary"] = pd.to_numeric(
        salaries["salary"], errors="coerce"
    )
    salaries["season_start"] = pd.to_numeric(
        salaries["season_start"], errors="coerce"
    )
    salaries["season_end"] = pd.to_numeric(
        salaries["season_end"], errors="coerce"
    )

    salaries = salaries.dropna(
        subset=["player_id", "salary", "season", "season_start", "season_end"]
    ).copy()

    season_salaries = (
        salaries.groupby(
            ["player_id", "season", "season_start", "season_end"],
            as_index=False
        )
        .agg(
            season_salary_total=("salary", "sum"),
            largest_team_salary=("salary", "max"),
            salary_records=("salary", "size"),
            number_of_teams=("team", "nunique"),
            teams=(
                "team",
                lambda values: ", ".join(
                    sorted(values.dropna().astype(str).unique())
                )
            )
        )
    )

    player_salary_summary = (
        season_salaries.groupby("player_id", as_index=False)
        .agg(
            mean_salary=("season_salary_total", "mean"),
            median_salary=("season_salary_total", "median"),
            max_season_salary=("season_salary_total", "max"),
            max_team_salary=("largest_team_salary", "max"),
            total_salary=("season_salary_total", "sum"),
            salary_seasons=("season", "nunique"),
            first_salary_year=("season_start", "min"),
            last_salary_year=("season_end", "max")
        )
    )

    nba = players.merge(
        player_salary_summary,
        on="player_id",
        how="inner"
    )

    nba["primary_position"] = (
        nba["position"]
        .astype("string")
        .str.split(" and ")
        .str[0]
    )

    return players, salaries, season_salaries, player_salary_summary, nba


def get_data():
    app_directory = Path(__file__).resolve().parent

    default_players = app_directory / "players.csv"
    default_salaries = app_directory / "salaries_1985to2018.csv"

    if default_players.exists() and default_salaries.exists():
        return (
            read_csv_from_path(str(default_players)),
            read_csv_from_path(str(default_salaries))
        )

    st.sidebar.markdown("### Load the datasets")
    st.sidebar.caption(
        "Place both CSV files beside this Python file, or upload them below."
    )

    players_upload = st.sidebar.file_uploader(
        "Upload players.csv",
        type="csv"
    )

    salaries_upload = st.sidebar.file_uploader(
        "Upload salaries_1985to2018.csv",
        type="csv"
    )

    if players_upload is None or salaries_upload is None:
        st.info(
            "To begin, place `players.csv` and "
            "`salaries_1985to2018.csv` in the same folder as this app, "
            "or upload both files using the sidebar."
        )
        st.stop()

    return (
        read_csv_from_upload(players_upload),
        read_csv_from_upload(salaries_upload)
    )


def insight(text):
    st.markdown(
        f"""
        <div class="insight-card">
            <strong>Key insight</strong><br>
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )


def show_chart(fig):
    fig.update_layout(
        margin=dict(l=20, r=20, t=75, b=30),
        font=dict(size=14),
        hoverlabel=dict(font_size=13)
    )
    st.plotly_chart(fig, use_container_width=True)


try:
    players_raw, salaries_raw = get_data()

    (
        players,
        salaries,
        season_salaries,
        player_salary_summary,
        nba
    ) = prepare_data(players_raw, salaries_raw)

except Exception as error:
    st.error(f"Data preparation failed: {error}")
    st.stop()


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------

st.sidebar.title("Dashboard controls")

minimum_games = st.sidebar.slider(
    "Minimum career games",
    min_value=20,
    max_value=500,
    value=50,
    step=10,
    help="Applied to player-level performance visualisations."
)

available_positions = [
    position
    for position in [
        "Point Guard",
        "Shooting Guard",
        "Small Forward",
        "Power Forward",
        "Center"
    ]
    if position in nba["primary_position"].dropna().unique()
]

selected_positions = st.sidebar.multiselect(
    "Positions included",
    options=available_positions,
    default=available_positions
)

# ---------------------------------------------------------
# HEADER AND OVERVIEW
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>NBA Player Salaries and Performance</h1>
    </div>
    """,
    unsafe_allow_html=True
)





def render_overview():
    st.header("Project description")

    st.write(
        """
        This project explores the relationships between NBA player salaries,
        career performance, draft position, age, playing position and team
        payrolls. Historical salary records from 1985–86 to 2017–18 are
        combined with player career statistics to create a consolidated
        dataset for analysis.
        """
    )

    st.write(
        """
        The analysis addresses ten analytical questions through interactive
        visualisations. It begins by examining the distribution and historical
        movement of salaries, then investigates relationships with performance,
        drafting, age and position. The final sections compare individual
        performance metrics and group players into broader performance profiles.
        """
    )


# ---------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------

def render_q1():
    st.header("1. How are NBA player salaries distributed?")

    season = "2017-18"
    salary_data = season_salaries.loc[
        season_salaries["season"] == season
    ].copy()

    salary_data["salary_millions"] = (
        salary_data["season_salary_total"] / 1_000_000
    )

    mean_salary = salary_data["salary_millions"].mean()
    median_salary = salary_data["salary_millions"].median()

    fig = px.histogram(
        salary_data,
        x="salary_millions",
        nbins=35,
        color_discrete_sequence=[PROJECT_GREEN],
        title=f"Distribution of NBA Player Salaries in the {season} Season",
        labels={
            "salary_millions": "Season salary (USD millions)"
        }
    )

    fig.update_traces(
        hovertemplate=(
            "Salary range: %{x}<br>"
            "Number of players: %{y}<extra></extra>"
        )
    )

    fig.add_vline(
        x=mean_salary,
        line_dash="dash",
        annotation_text=f"Mean: ${mean_salary:.2f}M",
        annotation_position="top right"
    )

    fig.add_vline(
        x=median_salary,
        line_dash="dot",
        annotation_text=f"Median: ${median_salary:.0f}M",
        annotation_position="top right"
    )

    fig.update_layout(
        xaxis_title="Season salary (USD millions)",
        yaxis_title="Number of players",
        bargap=0.05
    )

    show_chart(fig)

    insight(
        "The distribution is strongly right-skewed: most players are "
        "concentrated in the lower salary ranges, while a small group of "
        "highly paid players creates a long upper tail. The mean therefore "
        "sits above the median, making the median a better representation "
        "of a typical player's salary."
    )


# ---------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------

def render_q2():
    st.header("2. How have salaries changed across seasons?")

    salary_trend = (
        season_salaries.groupby(
            ["season_start", "season"],
            as_index=False
        )
        .agg(
            mean_salary=("season_salary_total", "mean"),
            median_salary=("season_salary_total", "median"),
            number_of_players=("player_id", "nunique")
        )
        .sort_values("season_start")
    )

    salary_trend["Mean salary"] = (
        salary_trend["mean_salary"] / 1_000_000
    )
    salary_trend["Median salary"] = (
        salary_trend["median_salary"] / 1_000_000
    )

    measures = ["Mean salary", "Median salary"]

    fig = px.line(
        salary_trend,
        x="season",
        y=measures,
        markers=True,
        title="Change in NBA Player Salaries Across Seasons"
    )

    fig.update_layout(
        xaxis={
            "title": "Season",
            "type": "category",
            "categoryorder": "array",
            "categoryarray": salary_trend["season"].tolist(),
            "tickmode": "array",
            "tickvals": salary_trend["season"].iloc[::2],
            "ticktext": salary_trend["season"].iloc[::2],
            "tickangle": 45
        },
        yaxis_title="Salary (USD millions)",
        legend_title="Salary measure",
        hovermode="x unified"
    )

    show_chart(fig)

    insight(
        "Both mean and median salaries show a clear long-term increase. "
        "The mean remains above the median throughout the period, and the "
        "widening gap suggests that top salaries increased faster than the "
        "salary of the typical player. Values are nominal rather than "
        "inflation-adjusted."
    )


# ---------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------

def render_q3():
    st.header("3. What is the relationship between salary and performance?")

    q3 = nba[
        [
            "player_id", "name", "position", "career_PER",
            "career_G", "max_season_salary", "last_salary_year"
        ]
    ].copy()

    q3["career_PER"] = pd.to_numeric(
        q3["career_PER"], errors="coerce"
    )
    q3["career_G"] = pd.to_numeric(
        q3["career_G"], errors="coerce"
    )
    q3["max_salary_millions"] = (
        q3["max_season_salary"] / 1_000_000
    )

    q3 = q3.loc[
        q3["career_G"] >= minimum_games
    ].dropna(
        subset=["career_PER", "max_salary_millions"]
    )

    correlation = q3["career_PER"].corr(
        q3["max_salary_millions"]
    )

    fig = px.scatter(
        q3,
        x="career_PER",
        y="max_salary_millions",
        trendline='ols',
        trendline_color_override='blue',
        hover_name="name",
        hover_data={
            "position": True,
            "career_G": True,
            "career_PER": ":.1f",
            "max_salary_millions": ":.2f",
            "last_salary_year": True
        },
        opacity=0.55,
        color_discrete_sequence=[PROJECT_GREEN],
        title=(
            "Relationship Between Career PER and "
            "Maximum Recorded Season Salary"
        ),
        labels={
            "career_PER": "Career player efficiency rating",
            "max_salary_millions": (
                "Maximum recorded season salary (USD millions)"
            ),
            "career_G": "Career games",
            "last_salary_year": "Last recorded salary year",
            "position": "Position"
        }
    )

    fig.update_layout(
        yaxis={"rangemode": "tozero"}
    )

    show_chart(fig)

    insight(
        f"The correlation of {correlation:.2f} indicates a moderate positive "
        "relationship: higher-career-PER players generally earned higher "
        "maximum salaries. However, the wide vertical spread among players "
        "with similar PER values shows that performance alone does not "
        "fully explain salary."
    )


# ---------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------

def render_q4():
    st.header("4. Which teams had the highest total payroll?")

    selected_payroll_season = "2017-18"

    q4 = salaries.loc[
        salaries["season"] == selected_payroll_season
    ].copy()

    team_payroll = (
        q4.groupby("team", as_index=False)
        .agg(
            total_payroll=("salary", "sum"),
            number_of_players=("player_id", "nunique")
        )
        .sort_values("total_payroll", ascending=False)
    )

    team_payroll["payroll_millions"] = (
        team_payroll["total_payroll"] / 1_000_000
    )

    team_payroll = team_payroll.head(10)

    fig = px.bar(
        team_payroll,
        x="payroll_millions",
        y="team",
        color="payroll_millions",
        color_continuous_scale="Viridis",
        orientation="h",
        title=(
            f"NBA Teams with the Highest Payrolls in "
            f"the {selected_payroll_season} Season"
        ),
        labels={
            "payroll_millions": "Total payroll (USD millions)",
            "team": "Team"
        },
        text="payroll_millions",
        hover_data={
            "total_payroll": False,
            "payroll_millions": ":.2f",
            "number_of_players": True
        }
    )

    fig.update_traces(
        texttemplate="$%{text:.1f}M",
        textposition="outside",
        cliponaxis=False
    )

    fig.update_layout(
        yaxis={
            "categoryorder": "total ascending",
            "title": ""
        },
        xaxis={
            "title": "Total payroll (USD millions)",
            "rangemode": "tozero"
        },
        coloraxis_showscale=False
    )

    show_chart(fig)

    insight(
        f"The Cleveland Cavaliers recorded the highest payroll in 2017-18 "
        "season, at approximately $137.9M, closely followed by the Golden "
        "State Warrirors at around $135.4M. Interestingly, these "
        "were the two teams that reached the finals this season."
    )


# ---------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------

def render_q5():
    st.header("5. Which performance statistic is most related to salary?")

    performance_stats = [
        "career_PER",
        "career_PTS",
        "career_AST",
        "career_TRB",
        "career_FG%",
        "career_FG3%",
        "career_FT%",
        "career_eFG%",
        "career_G"
    ]

    statistic_names = {
        "career_PER": "Player efficiency rating",
        "career_PTS": "Points per game",
        "career_AST": "Assists per game",
        "career_TRB": "Rebounds per game",
        "career_FG%": "Field-goal percentage",
        "career_FG3%": "Three-point percentage",
        "career_FT%": "Free-throw percentage",
        "career_eFG%": "Effective FG percentage",
        "career_G": "Career games played"
    }

    # Create the dataframe for Question 5
    q5 = nba[
        ["player_id", "name", "max_season_salary"] + performance_stats
    ].copy()

    # Convert salary to millions
    q5["max_salary_millions"] = (
        q5["max_season_salary"] / 1_000_000
    )

    # Convert performance columns to numeric
    q5[performance_stats] = q5[performance_stats].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Apply the minimum-games sidebar filter
    q5 = q5.loc[
        q5["career_G"] >= minimum_games
    ].copy()

    # Calculate correlations with maximum salary
    correlation_table = (
        q5[performance_stats]
        .corrwith(q5["max_salary_millions"])
        .rename("correlation")
        .to_frame()
        .sort_values("correlation", ascending=False)
    )

    # Prepare the single-column heatmap
    heatmap_data = correlation_table[["correlation"]].copy()

    heatmap_data = heatmap_data.rename(
        index=statistic_names
    )

    heatmap_data.columns = [
        "Correlation with maximum season salary"
    ]

    # Clear any previously created Seaborn / Matplotlib figure
    sns.set_theme(style="dark")

    # Create a fresh heatmap
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=1,
        linecolor="white",
        annot_kws={
            "color": "white",
            "fontsize": 6
        },
        cbar_kws={
            "label": "Pearson correlation with salary",
            "shrink": 0.85,
            "pad": 0.04
        }
    )

    # Use a wider figure to prevent crowding
    ax.figure.set_size_inches(4.5, 3.5)

    ax.set_title(
        "Correlation Between Career Player Statistics "
        "and Maximum Career Salary",
        pad=14,
        color="white",
        fontsize=9
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
        color="white",
        fontsize=7
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        color="white",
        fontsize=7
    )

    # Style the colour bar
    colorbar = ax.collections[0].colorbar

    colorbar.ax.tick_params(
        colors="white",
        labelsize=6
    )

    colorbar.set_label(
        "Pearson correlation with salary",
        color="white",
        fontsize=7,
        labelpad=8
    )

    # Transparent figure and axes backgrounds
    ax.figure.patch.set_alpha(0)
    ax.set_facecolor("none")

    # Allow enough room for long y-axis labels and the colour bar
    ax.figure.subplots_adjust(
        left=0.32,
        right=0.88,
        top=0.86,
        bottom=0.15
    )

    st.pyplot(
        ax.figure,
        use_container_width=False,
        transparent=True
    )

    # Clear the figure after rendering so it is not reused
    ax.figure.clear()

    insight(
        "Player efficiency rating has the strongest observed relationship "
        "with maximum season salary, followed by points per game. Rebounds, "
        "career longevity and assists also show positive relationships, while "
        "shooting percentages are comparatively weak. This suggests that "
        "overall impact and production are more closely associated with salary "
        "than shooting efficiency alone."
    )

# ---------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------

def render_q6():
    st.header("6. What is the relationship between draft position and salary?")

    q6 = nba[
        ["player_id", "name", "draft_pick", "max_season_salary"]
    ].copy()

    q6["draft_position"] = pd.to_numeric(
        q6["draft_pick"]
        .astype("string")
        .str.extract(r"(\d+)", expand=False),
        errors="coerce"
    )

    q6["max_salary_millions"] = (
        q6["max_season_salary"] / 1_000_000
    )

    q6 = q6.dropna(
        subset=["draft_position", "max_salary_millions"]
    ).copy()

    q6["draft_group"] = pd.cut(
        q6["draft_position"],
        bins=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            20, 30, 60, float("inf")
        ],
        labels=[
            "1", "2", "3", "4", "5",
            "6", "7", "8", "9", "10",
            "11–20", "21–30", "31–60", "61+"
        ]
    )

    draft_salary_summary = (
        q6.groupby(
            "draft_group",
            observed=False,
            as_index=False
        )
        .agg(
            median_salary_millions=(
                "max_salary_millions", "median"
            ),
            mean_salary_millions=(
                "max_salary_millions", "mean"
            ),
            number_of_players=(
                "player_id", "nunique"
            )
        )
    )

    salary_measure = "Median"
    y_column = "median_salary_millions"

    fig = px.bar(
        draft_salary_summary,
        x="draft_group",
        y=y_column,
        text=y_column,
        color_discrete_sequence=[PROJECT_GREEN],
        title=(
            f"{salary_measure} Maximum Season Salary "
            "by NBA Draft Position"
        ),
        labels={
            "draft_group": "Draft position",
            y_column: (
                f"{salary_measure} maximum season salary "
                "(USD millions)"
            )
        },
        hover_data={
            "mean_salary_millions": ":.2f",
            "median_salary_millions": ":.2f",
            "number_of_players": True
        }
    )

    fig.update_traces(
        texttemplate="$%{text:.1f}M",
        textposition="outside",
        cliponaxis=False
    )

    fig.update_layout(
        yaxis={"rangemode": "tozero"},
        showlegend=False
    )

    show_chart(fig)

    insight(
        "Earlier draft selections generally achieved higher maximum "
        "salaries. The relationship becomes weaker and salaries become "
        "more concentrated at lower levels among later selections, although "
        "draft position still leaves considerable variation between players."
    )


# ---------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------

def render_q7():
    st.header("7. How does age relate to salary and performance?")

    highest_salary_rows = season_salaries.loc[
        season_salaries.groupby("player_id")[
            "season_salary_total"
        ].idxmax()
    ].copy()

    q7 = highest_salary_rows[
        [
            "player_id", "season", "season_start",
            "season_end", "season_salary_total"
        ]
    ].merge(
        nba[
            [
                "player_id", "name", "birthDate", "position",
                "career_PER", "career_G"
            ]
        ],
        on="player_id",
        how="left"
    )

    q7["birthDate"] = pd.to_datetime(
        q7["birthDate"], errors="coerce"
    )
    q7["career_PER"] = pd.to_numeric(
        q7["career_PER"], errors="coerce"
    )
    q7["career_G"] = pd.to_numeric(
        q7["career_G"], errors="coerce"
    )

    q7["age_at_max_salary"] = (
        q7["season_start"] - q7["birthDate"].dt.year
    )

    q7["max_salary_millions"] = (
        q7["season_salary_total"] / 1_000_000
    )

    q7 = q7.loc[
        q7["career_G"] >= minimum_games
    ].dropna(
        subset=[
            "age_at_max_salary",
            "max_salary_millions",
            "career_PER"
        ]
    )

    fig = px.scatter(
        q7,
        x="career_PER",
        y="age_at_max_salary",
        size="max_salary_millions",
        color="max_salary_millions",
        hover_name="name",
        hover_data={
            "position": True,
            "career_G": True,
            "season": True,
            "career_PER": ":.1f",
            "age_at_max_salary": True,
            "max_salary_millions": ":.2f"
        },
        size_max=25,
        opacity=0.5,
        color_continuous_scale="Viridis",
        title=(
            "Player Performance, Age and Maximum Season Salary"
        ),
        labels={
            "career_PER": "Career player efficiency rating",
            "age_at_max_salary": "Age during maximum salary season",
            "max_salary_millions": (
                "Maximum season salary (USD millions)"
            ),
            "career_G": "Career games"
        }
    )

    fig.update_traces(
        marker={"line": {"width": 0.4, "color": "white"}}
        )
    fig.update_layout(
        template='plotly_white',
        xaxis={
            'range': [5, 30],
            'dtick': 1
            },
        yaxis={
            'range': [18, 43],
            'dtick': 2
            }
    )

    show_chart(fig)

    insight(
        "Most players reached their maximum recorded salary between their "
        "late twenties and mid-thirties. Career PER appears more closely "
        "associated with salary than age, but players with similar "
        "performance can still receive substantially different salaries."
    )


# ---------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------

def render_q8():
    st.header("8. Do salaries differ by playing position?")

    position_order = [
        "Point Guard",
        "Shooting Guard",
        "Small Forward",
        "Power Forward",
        "Center"
    ]

    q8 = nba[
        [
            "player_id", "name", "primary_position",
            "median_salary", "max_season_salary",
            "salary_seasons"
        ]
    ].copy()

    q8["median_salary_millions"] = (
        q8["median_salary"] / 1_000_000
    )

    q8 = q8.loc[
        q8["primary_position"].isin(selected_positions)
    ].dropna(
        subset=["primary_position", "median_salary_millions"]
    )

    fig = px.box(
        q8,
        x="primary_position",
        y="median_salary_millions",
        color="primary_position",
        category_orders={
            "primary_position": position_order
        },
        points="outliers",
        hover_name="name",
        hover_data={
            "median_salary_millions": ":.2f",
            "salary_seasons": True,
            "primary_position": False
        },
        title="Distribution of Players' Median Season Salary by Position",
        labels={
            "primary_position": "Position",
            "median_salary_millions": (
                "Median season salary (USD millions)"
            ),
            "salary_seasons": "Recorded salary seasons"
        },
        color_discrete_sequence=PLOTLY_VIRIDIS
    )

    fig.update_layout(
        yaxis={
            "rangemode": "tozero",
            "tickprefix": "$",
            "ticksuffix": "M"
        },
        xaxis_title="Primary position",
        showlegend=False
    )

    show_chart(fig)

    insight(
        "Salary distributions overlap substantially across the five primary "
        "positions, indicating that position alone does not produce a large "
        "difference in typical salary. Each position remains strongly "
        "right-skewed because a relatively small group earns much more than "
        "the typical player."
    )


# ---------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------

def render_q9():
    st.header("9. Which performance metrics are most strongly related?")

    q9_stats = [
        "career_PER", "career_PTS", "career_AST", "career_TRB",
        "career_FG%", "career_FG3%", "career_FT%", "career_eFG%"
    ]

    display_names = {
        "career_PER": "PER",
        "career_PTS": "Points",
        "career_AST": "Assists",
        "career_TRB": "Rebounds",
        "career_FG%": "FG%",
        "career_FG3%": "3P%",
        "career_FT%": "FT%",
        "career_eFG%": "eFG%"
    }

    q9 = nba[["player_id", "name", "career_G"] + q9_stats].copy()

    q9["career_G"] = pd.to_numeric(
        q9["career_G"], errors="coerce"
    )

    q9 = q9.loc[q9["career_G"] >= minimum_games].copy()

    q9[q9_stats] = q9[q9_stats].apply(
        pd.to_numeric,
        errors="coerce"
    )

    correlation = q9[q9_stats].corr().rename(
        index=display_names,
        columns=display_names
    )

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="Viridis",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlations Between NBA Player Performance Metrics",
        labels={
            "color": "Pearson correlation"
        }
    )

    fig.update_layout(height=700)

    show_chart(fig)

    correlation_pairs = (
        correlation.where(
            pd.DataFrame(
                [
                    [
                        column_index > row_index
                        for column_index in range(len(correlation.columns))
                    ]
                    for row_index in range(len(correlation.index))
                ],
                index=correlation.index,
                columns=correlation.columns
            )
        )
        .stack()
        .sort_values(ascending=False)
    )

    strongest_pair = correlation_pairs.index[0]
    strongest_value = correlation_pairs.iloc[0]

    insight(
        f"The strongest pair is {strongest_pair[0]} and "
        f"{strongest_pair[1]}, with a correlation of "
        f"{strongest_value:.2f}. Scoring and shooting-efficiency measures "
        "generally move together, while several more specialised skills "
        "remain comparatively independent."
    )


# ---------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------

def render_q10():
    st.header("10. Can players be grouped into performance profiles?")

    q10_stats = [
        "career_G", "career_PTS", "career_AST",
        "career_TRB", "career_PER"
    ]

    q10 = nba[
        ["player_id", "name", "primary_position"] + q10_stats
    ].copy()

    q10[q10_stats] = q10[q10_stats].apply(
        pd.to_numeric,
        errors="coerce"
    )

    q10 = q10.loc[
        (q10["career_G"] >= minimum_games)
        & q10["primary_position"].isin(selected_positions)
    ].dropna(
        subset=["career_PTS", "career_AST", "career_TRB"]
    ).copy()

    median_assists = q10["career_AST"].median()
    median_rebounds = q10["career_TRB"].median()

    def assign_profile(player):
        if (
            player["career_AST"] >= median_assists
            and player["career_TRB"] >= median_rebounds
        ):
            return "All-round contributors"

        if (
            player["career_AST"] >= median_assists
            and player["career_TRB"] < median_rebounds
        ):
            return "Playmakers"

        if (
            player["career_AST"] < median_assists
            and player["career_TRB"] >= median_rebounds
        ):
            return "Rebounders / interior players"

        return "Lower-volume role players"

    q10["performance_profile"] = q10.apply(
        assign_profile,
        axis=1
    )

    fig = px.scatter(
        q10,
        x="career_AST",
        y="career_TRB",
        size="career_PTS",
        color="performance_profile",
        hover_name="name",
        hover_data={
            "primary_position": True,
            "career_PTS": ":.1f",
            "career_AST": ":.1f",
            "career_TRB": ":.1f",
            "career_PER": ":.1f",
            "performance_profile": False
        },
        size_max=20,
        title="NBA Player Performance Profiles",
        labels={
            "career_AST": "Career assists per game",
            "career_TRB": "Career rebounds per game",
            "career_PTS": "Career points per game",
            "career_PER": "Career PER",
            "primary_position": "Primary position",
            "performance_profile": "Performance profile"
        },
        color_discrete_sequence=PLOTLY_VIRIDIS
    )

    fig.update_layout(
        height=700,
        legend_title_text="Performance profile"
    )

    fig.update_traces(
        marker={
            "opacity": 0.5,
            "line": {"width": 0.3}
        }
    )

    fig.add_vline(
        x=median_assists,
        line_dash="dash",
        line_width=2,
        annotation_text="Median assists",
        annotation_position="top"
    )

    fig.add_hline(
        y=median_rebounds,
        line_dash="dash",
        line_width=2,
        annotation_text="Median rebounds",
        annotation_position="right"
    )

    show_chart(fig)

    insight(
        "Players can be separated into four broad profiles according to "
        "whether assists and rebounds are above or below their respective "
        "medians. Playmaking and rebounding often reflect different player "
        "roles, while many of the highest scorers appear among the more "
        "versatile all-round contributors."
    )


# ---------------------------------------------------------
# RENDER THE COMPLETE SINGLE-PAGE DASHBOARD
# ---------------------------------------------------------

render_overview()

question_renderers = [
    render_q1,
    render_q2,
    render_q3,
    render_q4,
    render_q5,
    render_q6,
    render_q7,
    render_q8,
    render_q9,
    render_q10
]

for render_question in question_renderers:
    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True
    )
    render_question()


st.markdown("---")
st.caption(
    "NBA Player Salaries and Performance | "
    "TUBerlin Python Course Project | Connor Waite"
)