"""
📊 Digital Twin T1D Real-Time Dashboard
======================================

Beautiful, responsive dashboard για real-time glucose monitoring.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque
import asyncio
import threading
from typing import Optional  # Import Optional

from .core import DigitalTwinSDK, Prediction  # Import Prediction
from .datasets import load_diabetes_data


class RealTimeDashboard:
    """Stunning real-time monitoring dashboard."""

    def __init__(self, sdk=None, port=8081):
        self.sdk = sdk or DigitalTwinSDK(mode="demo")
        self.port = port

        # Data storage
        self.glucose_history = deque(maxlen=288)  # 24 hours at 5-min intervals
        self.prediction_history = deque(maxlen=12)  # 1 hour of predictions
        self.current_data = {
            "glucose": 120,
            "trend": "stable",
            "risk": "low",
            "last_update": datetime.now(),
        }

        # Initialize Dash app
        self.app = dash.Dash(
            __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
        )
        self.app.title = "Digital Twin T1D Monitor"

        # Setup layout
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Create the dashboard layout."""
        self.app.layout = html.Div(
            [
                # Header
                html.Div(
                    [
                        html.Div(
                            [
                                html.H1("🩺 Digital Twin T1D Monitor", className="title"),
                                html.P(
                                    "Real-time glucose monitoring powered by AI",
                                    className="subtitle",
                                ),
                            ],
                            className="header-content",
                        ),
                        html.Div(
                            [
                                html.Div(id="live-clock", className="clock"),
                                html.Div(
                                    [
                                        html.Span(
                                            "●", id="status-indicator", className="status-indicator"
                                        ),
                                        html.Span("Connected", id="status-text"),
                                    ],
                                    className="status",
                                ),
                            ],
                            className="header-right",
                        ),
                    ],
                    className="header",
                ),
                # Main metrics row
                html.Div(
                    [
                        # Current Glucose Card
                        html.Div(
                            [
                                html.H3("Current Glucose"),
                                html.Div(
                                    [
                                        html.H1(id="current-glucose", children="--"),
                                        html.Span("mg/dL", className="unit"),
                                    ],
                                    className="metric-value",
                                ),
                                html.Div(id="glucose-trend", className="trend"),
                            ],
                            className="metric-card primary",
                        ),
                        # Prediction Card
                        html.Div(
                            [
                                html.H3("30 min Prediction"),
                                html.Div(
                                    [
                                        html.H1(id="predicted-glucose", children="--"),
                                        html.Span("mg/dL", className="unit"),
                                    ],
                                    className="metric-value",
                                ),
                                html.Div(id="confidence-level", className="confidence"),
                            ],
                            className="metric-card",
                        ),
                        # Time in Range Card
                        html.Div(
                            [
                                html.H3("Time in Range"),
                                html.Div(
                                    [
                                        html.H1(id="time-in-range", children="--"),
                                        html.Span("%", className="unit"),
                                    ],
                                    className="metric-value",
                                ),
                                dcc.Graph(id="tir-gauge", config={"displayModeBar": False}),
                            ],
                            className="metric-card",
                        ),
                        # Risk Level Card
                        html.Div(
                            [
                                html.H3("Risk Assessment"),
                                html.Div(id="risk-level", className="risk-display"),
                                html.Div(id="risk-message", className="risk-message"),
                            ],
                            className="metric-card",
                        ),
                    ],
                    className="metrics-row",
                ),
                # Charts row
                html.Div(
                    [
                        # Main glucose chart
                        html.Div(
                            [
                                html.H3("Glucose History & Predictions"),
                                dcc.Graph(id="glucose-chart"),
                            ],
                            className="chart-container large",
                        ),
                        # Pattern analysis
                        html.Div(
                            [html.H3("Daily Patterns"), dcc.Graph(id="pattern-chart")],
                            className="chart-container small",
                        ),
                    ],
                    className="charts-row",
                ),
                # Recommendations row
                html.Div(
                    [
                        html.H3("AI Recommendations"),
                        html.Div(id="recommendations", className="recommendations-list"),
                    ],
                    className="recommendations-container",
                ),
                # Update interval
                dcc.Interval(
                    id="interval-component", interval=5000, n_intervals=0  # Update every 5 seconds
                ),
            ],
            className="dashboard-container",
        )

    def _setup_callbacks(self):
        """Setup all dashboard callbacks."""

        @self.app.callback(
            [
                Output("current-glucose", "children"),
                Output("glucose-trend", "children"),
                Output("predicted-glucose", "children"),
                Output("confidence-level", "children"),
                Output("time-in-range", "children"),
                Output("risk-level", "children"),
                Output("risk-message", "children"),
                Output("recommendations", "children"),
                Output("glucose-chart", "figure"),
                Output("pattern-chart", "figure"),
                Output("tir-gauge", "figure"),
                Output("live-clock", "children"),
                Output("status-indicator", "className"),
            ],
            [Input("interval-component", "n_intervals")],
        )
        def update_dashboard(n):
            """Update all dashboard components."""
            # Get current time
            now = datetime.now()
            clock = now.strftime("%H:%M:%S")

            # Simulate real-time data (in production, this would come from device)
            self._update_glucose_data()

            # Get prediction
            try:
                sdk_prediction_obj = self.sdk.predict_glucose(horizon_minutes=30)
                sdk_recommendations_dict = self.sdk.get_recommendations()
            except Exception:  # Catch more specific exceptions if possible
                sdk_prediction_obj = None
                sdk_recommendations_dict = {}  # Ensure it's a dict

            # Current glucose
            current = self.current_data["glucose"]
            trend = self._get_trend_arrow(self.current_data["trend"])

            # Prediction
            if sdk_prediction_obj and sdk_prediction_obj.values:
                pred_value = f"{sdk_prediction_obj.values[0]:.0f}"
                # Placeholder for confidence, as sdk.core.Prediction has confidence_intervals
                confidence_percent = 95.0  # Default placeholder
                if sdk_prediction_obj.confidence_intervals:
                    # Implement logic to derive a single confidence % if needed
                    pass
                confidence = f"Confidence: {confidence_percent:.0f}%"
            else:
                pred_value = "--"
                confidence = "N/A"

            # Time in range
            tir = self._calculate_tir()

            # Risk assessment
            risk_level, risk_message = self._get_risk_assessment(current)
            risk_class = f"risk-level {risk_level.lower()}"

            # Recommendations
            rec_items = []
            processed_recs_for_display = []
            if isinstance(sdk_recommendations_dict.get("insulin"), dict):
                action = sdk_recommendations_dict["insulin"].get(
                    "action", "Consider insulin adjustment."
                )
                reason = sdk_recommendations_dict["insulin"].get("reason", "")
                processed_recs_for_display.append(
                    {"action": action, "reason": reason, "category": "insulin"}
                )

            if isinstance(sdk_recommendations_dict.get("meals"), list):
                for meal_rec in sdk_recommendations_dict["meals"][:2]:  # Max 2 meal recs
                    action = (
                        meal_rec
                        if isinstance(meal_rec, str)
                        else meal_rec.get("action", "Consider meal adjustment.")
                    )
                    reason = "" if isinstance(meal_rec, str) else meal_rec.get("reason", "")
                    processed_recs_for_display.append(
                        {"action": action, "reason": reason, "category": "food"}
                    )

            if isinstance(sdk_recommendations_dict.get("activity"), list):
                for activity_rec in sdk_recommendations_dict["activity"][:1]:  # Max 1 activity rec
                    action = (
                        activity_rec
                        if isinstance(activity_rec, str)
                        else activity_rec.get("action", "Consider activity level.")
                    )
                    reason = "" if isinstance(activity_rec, str) else activity_rec.get("reason", "")
                    processed_recs_for_display.append(
                        {"action": action, "reason": reason, "category": "exercise"}
                    )

            for rec_data in processed_recs_for_display[:3]:  # Display up to 3 recommendations total
                icon = self._get_recommendation_icon(rec_data["category"])
                rec_items.append(
                    html.Div(
                        [
                            html.Span(icon, className="rec-icon"),
                            html.Div(
                                [
                                    html.Strong(rec_data["action"]),
                                    html.P(rec_data["reason"], className="rec-reason"),
                                ]
                            ),
                        ],
                        className="recommendation-item",
                    )
                )

            # Charts
            glucose_fig = self._create_glucose_chart(sdk_prediction_obj)  # Pass prediction object
            pattern_fig = self._create_pattern_chart()
            tir_gauge = self._create_tir_gauge(tir)

            # Status
            status_class = "status-indicator connected"

            return (
                f"{current:.0f}",
                trend,
                pred_value,
                confidence,
                f"{tir:.0f}",
                risk_level,
                risk_message,
                rec_items,
                glucose_fig,
                pattern_fig,
                tir_gauge,
                clock,
                status_class,
            )

    def _update_glucose_data(self):
        """Simulate glucose data updates."""
        # Add realistic variation
        last = self.glucose_history[-1] if self.glucose_history else 120

        # Simulate meal effects, circadian rhythm, etc.
        hour = datetime.now().hour
        meal_effect = 0
        if 7 <= hour <= 9:  # Breakfast
            meal_effect = np.random.normal(30, 10)
        elif 12 <= hour <= 14:  # Lunch
            meal_effect = np.random.normal(25, 8)
        elif 18 <= hour <= 20:  # Dinner
            meal_effect = np.random.normal(35, 12)

        # Random walk with drift
        change = np.random.normal(0, 3) + meal_effect * 0.1
        new_glucose = max(70, min(250, last + change))

        # Update data
        self.glucose_history.append(new_glucose)
        self.current_data["glucose"] = new_glucose
        self.current_data["trend"] = self._calculate_trend()
        self.current_data["last_update"] = datetime.now()

    def _calculate_trend(self):
        """Calculate glucose trend."""
        if len(self.glucose_history) < 3:
            return "stable"

        recent = list(self.glucose_history)[-6:]  # Last 30 minutes
        slope = np.polyfit(range(len(recent)), recent, 1)[0]

        if slope > 2:
            return "rising_fast"
        elif slope > 1:
            return "rising"
        elif slope < -2:
            return "falling_fast"
        elif slope < -1:
            return "falling"
        else:
            return "stable"

    def _get_trend_arrow(self, trend):
        """Get trend arrow emoji."""
        arrows = {
            "rising_fast": "↑↑",
            "rising": "↑",
            "stable": "→",
            "falling": "↓",
            "falling_fast": "↓↓",
        }
        return arrows.get(trend, "→")

    def _calculate_tir(self):
        """Calculate time in range."""
        if not self.glucose_history:
            return 0

        in_range = sum(1 for g in self.glucose_history if 70 <= g <= 180)
        return (in_range / len(self.glucose_history)) * 100

    def _get_risk_assessment(self, glucose):
        """Get risk level and message."""
        if glucose < 70:
            return "HIGH", "⚠️ Low glucose alert!"
        elif glucose < 80:
            return "MEDIUM", "Glucose trending low"
        elif glucose > 180:
            return "MEDIUM", "Glucose above target"
        elif glucose > 250:
            return "HIGH", "⚠️ High glucose alert!"
        else:
            return "LOW", "✅ Glucose in range"

    def _get_recommendation_icon(self, category):
        """Get icon for recommendation category."""
        icons = {"insulin": "💉", "food": "🍎", "exercise": "🏃", "alert": "⚠️", "general": "💡"}
        return icons.get(category, "💡")

    def _create_glucose_chart(
        self, prediction_obj: Optional[Prediction] = None
    ):  # Use imported Prediction
        """Create main glucose chart."""
        # Generate time axis
        now = datetime.now()
        times = [
            now - timedelta(minutes=5 * i) for i in range(len(self.glucose_history) - 1, -1, -1)
        ]

        # Create figure
        fig = go.Figure()

        # Add glucose trace
        fig.add_trace(
            go.Scatter(
                x=times,
                y=list(self.glucose_history),
                mode="lines+markers",
                name="Glucose",
                line=dict(color="#2E86C1", width=3),
                marker=dict(size=6),
            )
        )

        # Add target range
        fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.1, layer="below", line_width=0)

        # Add prediction
        if self.prediction_history:
            pred_times = [
                now + timedelta(minutes=5 * i) for i in range(1, 7)
            ]  # For 30 min horizon (6 points)
            # Ensure prediction_history has enough points or handle gracefully
            prediction_points_to_plot = list(self.prediction_history)

            # The y-values for prediction start from the current glucose, then the predicted values.
            # If sdk_prediction_obj.values exists, use them. Otherwise, use what's in prediction_history.
            y_pred_values = [self.current_data["glucose"]]
            if prediction_obj and prediction_obj.values:  # Use the passed prediction_obj
                y_pred_values.extend(prediction_obj.values[:5])  # Get up to 5 future points
            elif prediction_points_to_plot:  # Fallback to deque if no fresh prediction
                y_pred_values.extend(prediction_points_to_plot[:5])

            # Ensure y_pred_values matches length of pred_times for plotting
            # If y_pred_values is shorter, pad with last known value or handle appropriately
            # For simplicity, we'll plot what we have, Plotly handles mismatched lengths to some extent.

            fig.add_trace(
                go.Scatter(
                    x=pred_times[
                        : len(y_pred_values) - 1
                    ],  # Match x to y if y_pred_values is shorter than 6
                    y=y_pred_values[1:],  # Plot only future predicted points
                    mode="lines+markers",
                    name="Prediction",
                    line=dict(color="#E74C3C", width=2, dash="dash"),
                    marker=dict(size=6),
                )
            )

        # Layout
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            hovermode="x unified",
            showlegend=True,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
                showline=True,
                linecolor="rgba(128,128,128,0.4)",
            ),
            yaxis=dict(
                range=[40, 300],
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
                showline=True,
                linecolor="rgba(128,128,128,0.4)",
                title="Glucose (mg/dL)",
            ),
        )

        return fig

    def _create_pattern_chart(self):
        """Create daily pattern analysis chart."""
        # Simulate daily patterns
        hours = list(range(24))
        avg_glucose = [
            100,
            95,
            90,
            88,
            87,
            90,
            95,
            110,
            140,
            135,
            125,
            120,
            130,
            145,
            140,
            130,
            125,
            135,
            150,
            145,
            130,
            120,
            110,
            105,
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=avg_glucose,
                mode="lines",
                fill="tozeroy",
                line=dict(color="#3498DB", width=2),
            )
        )

        fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Hour of Day"),
            yaxis=dict(title="Avg Glucose"),
        )

        return fig

    def _create_tir_gauge(self, tir_value):
        """Create time in range gauge."""
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=tir_value,
                number={"suffix": "%"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": self._get_tir_color(tir_value)},
                    "steps": [
                        {"range": [0, 70], "color": "lightgray"},
                        {"range": [70, 100], "color": "gray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 70,
                    },
                },
            )
        )

        fig.update_layout(
            height=150,
            margin=dict(l=20, r=20, t=0, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        return fig

    def _get_tir_color(self, tir):
        """Get color based on TIR value."""
        if tir >= 70:
            return "#27AE60"
        elif tir >= 50:
            return "#F39C12"
        else:
            return "#E74C3C"

    def run(self):
        """Run the dashboard."""
        print(f"🚀 Starting Digital Twin T1D Dashboard on http://localhost:{self.port}")
        self.app.run_server(
            host="0.0.0.0", port=self.port, debug=False
        )  # nosec B104 - Intentionally binding to all interfaces


# CSS Styling
app_css = """
<style>
.dashboard-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f7fa;
    min-height: 100vh;
    padding: 20px;
}

.header {
    background: white;
    border-radius: 12px;
    padding: 20px 30px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.title {
    margin: 0;
    color: #2c3e50;
    font-size: 28px;
}

.subtitle {
    margin: 5px 0 0 0;
    color: #7f8c8d;
}

.clock {
    font-size: 24px;
    font-weight: 600;
    color: #2c3e50;
}

.status {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
}

.status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #27ae60;
    display: inline-block;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.metrics-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.metric-card.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.metric-value {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin: 10px 0;
}

.metric-value h1 {
    font-size: 48px;
    margin: 0;
    font-weight: 700;
}

.unit {
    font-size: 18px;
    opacity: 0.8;
}

.trend {
    font-size: 24px;
}

.risk-level {
    font-size: 24px;
    font-weight: 600;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
}

.risk-level.low {
    background: #d4edda;
    color: #155724;
}

.risk-level.medium {
    background: #fff3cd;
    color: #856404;
}

.risk-level.high {
    background: #f8d7da;
    color: #721c24;
}

.charts-row {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
}

.chart-container {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.recommendations-container {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.recommendation-item {
    display: flex;
    gap: 15px;
    padding: 15px;
    margin: 10px 0;
    background: #f8f9fa;
    border-radius: 8px;
    border-left: 4px solid #3498db;
}

.rec-icon {
    font-size: 24px;
}

.rec-reason {
    margin: 5px 0 0 0;
    color: #6c757d;
    font-size: 14px;
}
</style>
"""


def main():
    """Run the dashboard."""
    dashboard = RealTimeDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
