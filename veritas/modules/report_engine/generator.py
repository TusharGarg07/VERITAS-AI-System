from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import uuid
import os
from datetime import datetime
from veritas.utils.logger import logger

class ReportGenerator:
    def __init__(self):
        self.output_dir = "reports"
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_pdf(self, report_id: str, data: dict, user: dict = None) -> str:
        try:
            print("PDF DATA:", data)
            print("USER:", user)

            if user is None:
                user = {"name": "System Guest", "domain": data.get("context", "General")}

            # SAFE PATCH & ROUNDING: Extracting values with fallbacks
            user_name = user.get("name") if isinstance(user, dict) else getattr(user, "name", "User")
            domain = user.get("domain") if isinstance(user, dict) else getattr(user, "domain", "General")
            
            sensors = data.get("sensors", {})
            # Rounding for clean human-readable output
            temperature = round(sensors.get("temperature", data.get("temperature", 0)), 1)
            humidity = round(sensors.get("humidity", data.get("humidity", 0)), 1)
            co2 = round(sensors.get("co2", data.get("co2", 0)), 0)
            pm2_5 = round(sensors.get("pm2_5", data.get("pm2_5", 0)), 1)
            risk_score = round(data.get("risk_score", 0), 2)

            file_name = f"report_{report_id}.pdf"
            file_path = os.path.join(self.output_dir, file_name)

            doc = SimpleDocTemplate(file_path, pagesize=A4)
            styles = getSampleStyleSheet()

            # Custom Styles
            title_style = ParagraphStyle(
                'title',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=colors.HexColor("#0B3C5D"),
                alignment=1, # Center
                spaceAfter=20
            )

            section_style = ParagraphStyle(
                'section',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor("#1D3557"),
                spaceBefore=12,
                spaceAfter=10,
                borderPadding=2
            )

            normal_style = styles["Normal"]
            normal_style.fontSize = 10
            normal_style.leading = 14

            elements = []

            # ================= HEADER =================
            elements.append(Paragraph("VERITAS AI ENVIRONMENTAL INTELLIGENCE REPORT", title_style))
            elements.append(Spacer(1, 14))

            elements.append(Paragraph(f"<b>User:</b> {user_name}", normal_style))
            elements.append(Paragraph(f"<b>Domain:</b> {domain}", normal_style))
            elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d %b %Y, %H:%M')}", normal_style))
            elements.append(Spacer(1, 14))

            # ================= RISK LEVEL =================
            if risk_score > 0.7:
                risk_text = "HIGH RISK"
                risk_color = colors.red
            elif risk_score > 0.4:
                risk_text = "MODERATE RISK"
                risk_color = colors.orange
            else:
                risk_text = "LOW RISK"
                risk_color = colors.green

            elements.append(Paragraph(
                f"<b>Overall Risk Level:</b> <font color='{risk_color}' size='12'><b>{risk_text}</b></font> ({risk_score})",
                normal_style
            ))
            elements.append(Spacer(1, 14))

            # ================= EXECUTIVE SUMMARY =================
            elements.append(Paragraph("1. Executive Summary", section_style))
            elements.append(Paragraph(
                "The system has analyzed real-time environmental data to assess potential impacts on human health. "
                "Based on the identified parameters, we have categorized the risk level and provided specific recommendations to maintain optimal air quality and safety.",
                normal_style
            ))
            elements.append(Spacer(1, 14))

            # ================= SENSOR ANALYSIS =================
            elements.append(Paragraph("2. Sensor Analysis", section_style))

            table_data = [["Parameter", "Value", "Status"]]

            def get_status(value, threshold):
                return "HIGH" if value > threshold else "NORMAL"

            table_data += [
                ["Temperature (°C)", f"{temperature} °C", get_status(temperature, 30)],
                ["Humidity (%)", f"{humidity} %", get_status(humidity, 70)],
                ["CO2 (ppm)", f"{int(co2)} ppm", get_status(co2, 1000)],
                ["PM2.5", f"{pm2_5} µg/m³", get_status(pm2_5, 25)],
            ]

            table = Table(table_data, colWidths=[2.2*inch, 2*inch, 1.8*inch])

            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#457B9D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
            ]))

            elements.append(table)
            elements.append(Spacer(1, 14))

            # ================= SYNERGY =================
            elements.append(Paragraph("3. Synergistic Risk Analysis", section_style))

            if humidity > 70 and co2 > 1000:
                elements.append(Paragraph(
                    "<font color='red'><b>CRITICAL SYNERGY DETECTED:</b></font><br/>"
                    f"Combined high CO2 ({int(co2)} ppm) and High Humidity ({humidity}%) detected. "
                    "This synergy significantly increases the risk of microbial growth, reduced cognitive function, and perceived stuffiness.",
                    normal_style
                ))
            else:
                elements.append(Paragraph("No critical synergy risks detected between monitored parameters.", normal_style))

            elements.append(Spacer(1, 14))

            # ================= HUMAN IMPACT =================
            elements.append(Paragraph("4. Human Health Impact", section_style))
            impact_text = "".join([f"• {impact}<br/>" for impact in data.get('impacts', ["Normal environment conditions detected."])])
            elements.append(Paragraph(impact_text, normal_style))
            elements.append(Spacer(1, 14))

            # ================= XAI =================
            elements.append(Paragraph("5. Explainable AI Insights", section_style))
            explanation = data.get("explanation", "All parameters are within normal range; risk is minimal.")
            elements.append(Paragraph(explanation, normal_style))
            elements.append(Spacer(1, 14))

            # ================= ACTIONS =================
            elements.append(Paragraph("6. Recommended Actions", section_style))
            
            # Formulate dynamic clean actions
            clean_actions = []
            if co2 > 1000: clean_actions.append(f"✔ <b>Immediate ventilation required</b> (CO2: {int(co2)} ppm)")
            if temperature > 30: clean_actions.append(f"✔ <b>Activate cooling system</b> (Temp: {temperature} °C)")
            if humidity > 70: clean_actions.append(f"✔ <b>Use dehumidifier</b> (Humidity: {humidity} %)")
            
            if not clean_actions:
                clean_actions = ["✔ Continue routine environmental monitoring", "✔ Maintain current ventilation rates"]

            action_text = "<br/><br/>".join(clean_actions)
            elements.append(Paragraph(action_text, normal_style))

            # ================= BUILD =================
            doc.build(elements)

            logger.info(f"Professional PDF Report generated: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Professional PDF generation failed: {str(e)}")
            return None

report_generator = ReportGenerator()
