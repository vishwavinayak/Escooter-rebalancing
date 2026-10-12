"""Dashboard view renderers for Page 1, Page 2, Page 3, and Page 4."""

from escooter_opt.dashboard.views.page1_overview import render_page1
from escooter_opt.dashboard.views.page2_transport import render_page2
from escooter_opt.dashboard.views.page3_pmedian import render_page3
from escooter_opt.dashboard.views.page4_scenarios import render_page4

__all__ = [
    "render_page1",
    "render_page2",
    "render_page3",
    "render_page4",
]
