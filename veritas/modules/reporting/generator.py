import os
from datetime import datetime
from uuid import uuid4
from reportlab.platypus import SimpleDocTemplate, Spacer, Image, Paragraph
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

from .styles import get_veritas_styles, VeritasColors
from .charts import VeritasCharts
from .sections import VeritasSections
from .formatter import VeritasFormatter

class VeritasPDFGenerator:
    """
    Master PDF generation pipeline for VERITAS AI enterprise reports.
    """
    
    def __init__(self):
        self.styles = get_veritas_styles()
        self.charts = VeritasCharts()
        self.formatter = VeritasFormatter()
        self.sections = VeritasSections(self.styles)

    def generate(self, data: dict, output_path: str = None) -> str:
        """
        Generates a complete PDF report from the provided data.
        """
        try:
            if not output_path:
                report_id = f"VR-{datetime.now().strftime('%Y%m%d')}-{str(uuid4())[:8]}"
                output_path = f"reports/report_{report_id}.pdf"

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=40,
                leftMargin=40,
                topMargin=40,
                bottomMargin=60
            )

            elements = []
            
            # 1. Metadata preparation
            metadata = {
                "id": f"VR-{str(uuid4())[:8].upper()}",
                "date": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                "domain": data.get("domain", "General Environment")
            }

            # 2. Header Section
            elements.extend(self.sections.get_header(metadata))

            # 3. Executive Summary
            elements.append(Paragraph("1. EXECUTIVE SUMMARY", self.styles['VeritasSectionHeader']))
            summary_text = (
                f"VERITAS AI has analyzed the current environmental conditions in the {metadata['domain']} using "
                "multi-parameter sensor data, advanced AI models, and rule-based intelligence. The environment shows "
                f"significant risk levels for certain factors that may impact health, comfort, and operational efficiency."
            )
            elements.append(Paragraph(summary_text, self.styles['VeritasNormal']))
            elements.append(Spacer(1, 0.2 * inch))

            # 4. KPI Cards
            metrics = {
                "risk_score": int(data.get("risk_score", 0)),
                "risk_level": self.formatter.get_risk_level(data.get("risk_score", 0)),
                "stability": data.get("stability_index", 62)
            }
            elements.extend(self.sections.get_kpi_section(metrics))

            # 5. Sensor Analysis Table
            elements.extend(self.sections.get_sensor_analysis(data.get("sensor_data", {})))

            # 6. XAI Section
            elements.append(Paragraph("4. RISK CONTRIBUTION (XAI)", self.styles['VeritasSectionHeader']))
            xai_importance = data.get("xai_importance", {
                "CO2 Level": 0.28, "Humidity": 0.21, "Temperature": 0.18, "PM2.5": 0.15, "TVOC": 0.09
            })
            xai_chart_path = self.charts.generate_contribution_bars(xai_importance)
            if xai_chart_path:
                elements.append(Image(xai_chart_path, width=4*inch, height=2.5*inch))
            elements.append(Spacer(1, 0.2 * inch))

            # 7. Synergistic Risk Analysis
            elements.append(Paragraph("5. SYNERGISTIC RISK ANALYSIS", self.styles['VeritasSectionHeader']))
            synergy_text = "<b>High CO2 + High Humidity</b>: This combination increases the probability of microbial growth and poor air quality."
            elements.append(Paragraph(synergy_text, self.styles['VeritasNormal']))
            elements.append(Spacer(1, 0.3 * inch))

            # 8. Action Plan
            elements.extend(self.sections.get_action_plan(data.get("actions", [])))

            # 9. Environmental Trend Graph
            elements.append(Paragraph("8. ENVIRONMENTAL TREND (24H)", self.styles['VeritasSectionHeader']))
            trend_chart_path = self.charts.generate_trend_chart(data.get("history_data"))
            if trend_chart_path:
                elements.append(Image(trend_chart_path, width=7*inch, height=3*inch))

            # Build PDF with Footer
            doc.build(elements, onFirstPage=self.sections.get_footer, onLaterPages=self.sections.get_footer)
            
            # Cleanup temp files
            self._cleanup_temp_files([xai_chart_path, trend_chart_path])
            
            return output_path

        except Exception as e:
            import traceback
            print(f"PDF Generation Failed: {str(e)}")
            print(traceback.format_exc())
            raise

    def _cleanup_temp_files(self, file_paths):
        """
        Safely removes temporary chart images.
        """
        for path in file_paths:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
