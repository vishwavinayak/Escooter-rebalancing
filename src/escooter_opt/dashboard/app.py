"""Main Streamlit Application for Chicago E-Scooter Operations Research.

Phase 6 – Integrated Streamlit Dashboard
=========================================
Multi-page dashboard providing:
1. Data Overview: Cleaned trips, quality audit waterfall, and trip-flow patterns.
2. Transportation Problem: Fleet redistribution Linear Program via HiGHS.
3. Depot Location: P-Median Facility Location Integer Program via HiGHS.
4. Scenario Analysis: Parametric sensitivity suites over depot counts and cost assumptions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from escooter_opt.dashboard.helpers import DEFAULT_EDA_DIR
from escooter_opt.dashboard.views.page1_overview import render_page1
from escooter_opt.dashboard.views.page2_transport import render_page2
from escooter_opt.dashboard.views.page3_pmedian import render_page3
from escooter_opt.dashboard.views.page4_scenarios import render_page4


def main() -> None:
    """Execute the multi-page Streamlit application."""
    st.set_page_config(
        page_title="Chicago E-Scooter Optimization & Rebalancing",
        page_icon="🛴",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # -----------------------------------------------------------------------
    # Sidebar Configuration & Metadata
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.title("🛴 E-Scooter Optimization")
        st.caption("City of Chicago 2020 Pilot Program")
        st.markdown("---")
        st.subheader("⚙️ Workspace Settings")
        eda_path_input = st.text_input(
            "EDA Results Directory:",
            value=str(DEFAULT_EDA_DIR),
            help="Path to precomputed Phase 2 EDA outputs.",
        )
        eda_dir = Path(eda_path_input)
        st.session_state["eda_dir"] = eda_dir

        st.markdown("---")
        st.markdown(
            """
            ### 📚 Project Architecture
            - **Phase 1B**: Data Cleaning (Policy C)
            - **Phase 2**: EDA & Spatial Centroids
            - **Phase 3**: Transportation LP Model
            - **Phase 4**: P-Median Facility IP Model
            - **Phase 5**: Scenario & Sensitivity Engine
            - **Phase 6**: Integrated Streamlit UI
            """
        )

        st.info(
            "⚖️ **Academic Disclaimer**\n\n"
            "All models utilize historical trip records and centroid approximations. "
            "Depot locations are mathematical candidates, and redistribution quantities "
            "are flow-based disequilibrium proxies."
        )

    # -----------------------------------------------------------------------
    # Streamlit Navigation Router
    # -----------------------------------------------------------------------
    def p1() -> None:
        render_page1(eda_dir=st.session_state.get("eda_dir", DEFAULT_EDA_DIR))

    def p2() -> None:
        render_page2(eda_dir=st.session_state.get("eda_dir", DEFAULT_EDA_DIR))

    def p3() -> None:
        render_page3(eda_dir=st.session_state.get("eda_dir", DEFAULT_EDA_DIR))

    def p4() -> None:
        render_page4(eda_dir=st.session_state.get("eda_dir", DEFAULT_EDA_DIR))

    nav = st.navigation(
        [
            st.Page(p1, title="1. Data Overview & EDA", icon="📊", default=True),
            st.Page(p2, title="2. Transportation Problem", icon="🚚"),
            st.Page(p3, title="3. Depot Location (P-Median)", icon="🏢"),
            st.Page(p4, title="4. Scenario & Sensitivity Analysis", icon="📈"),
        ]
    )
    nav.run()


def run_app() -> None:
    """Entry point for the CLI script `escooter-dashboard`."""
    from streamlit.web import cli as stcli

    app_path = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
