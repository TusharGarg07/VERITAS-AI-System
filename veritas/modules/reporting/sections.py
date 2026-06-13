from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
import uuid

from .styles import VeritasColors
from .templates import VeritasTemplates
from .formatter import VeritasFormatter

class VeritasSections:
    """
    Constructs high-level PDF sections for the VERITAS report.
    """
    
    def __init__(self, styles):
        self.styles = styles
        self.templates = VeritasTemplates()
        self.formatter = VeritasFormatter()

    def get_header(self, report_metadata):
        """
        Generates the professional report header with logo and metadata.
        """
        elements = []
        
        # Title and Subtitle
        elements.append(Paragraph("VERITAS AI", self.styles['VeritasTitle']))
        elements.append(Paragraph("Environmental Intelligence & Risk Assessment", self.styles['VeritasSubtitle']))
        
        # Metadata Row
        metadata_data = [
            [
                Paragraph(f"<b>Report ID</b><br/>{report_metadata.get('id', 'N/A')}", self.styles['VeritasNormal']),
                Paragraph(f"<b>Generated On</b><br/>{report_metadata.get('date', 'N/A')}", self.styles['VeritasNormal']),
                Paragraph(f"<b>Location</b><br/>{report_metadata.get('domain', 'N/A')}", self.styles['VeritasNormal']),
                Paragraph(f"<b>Report Type</b><br/>Real-time Assessment", self.styles['VeritasNormal'])
            ]
        ]
        
        meta_table = Table(metadata_data, colWidths=[1.8 * inch] * 4)
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), VeritasColors.SLATE_50),
            ('ROUNDEDCORNERS', [4, 4, 4, 4]),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(meta_table)
        elements.append(Spacer(1, 0.3 * inch))
        
        return elements

    def get_kpi_section(self, metrics):
        """
        Generates the KPI cards section.
        """
        elements = []
        
        kpis = [
            self.templates.create_kpi_card("Overall Risk Score", f"{metrics['risk_score']}/100", metrics['risk_level'], self.styles),
            self.templates.create_kpi_card("Risk Level", metrics['risk_level'], metrics['risk_level'], self.styles),
            self.templates.create_kpi_card("Stability Index", f"{metrics['stability']}/100", "MODERATE", self.styles),
            self.templates.create_kpi_card("Trend", "DETERIORATING", "HIGH", self.styles)
        ]
        
        kpi_table = Table([kpis], colWidths=[1.75 * inch] * 4)
        kpi_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(kpi_table)
        elements.append(Spacer(1, 0.3 * inch))
        return elements

    def get_sensor_analysis(self, sensor_data):
        """
        Generates the detailed sensor analysis table.
        """
        elements = []
        elements.append(Paragraph("2. ENVIRONMENTAL CONDITIONS", self.styles['VeritasSectionHeader']))
        
        data = [["Parameter", "Measured Value", "Status"]]
        
        parameters = [
            ("CO2", "co2"),
            ("Temperature", "temperature"),
            ("Humidity", "humidity"),
            ("PM2.5", "pm2_5"),
            ("TVOC", "tvoc"),
            ("Noise", "noise")
        ]
        
        for label, key in parameters:
            val = sensor_data.get(key, 0)
            data.append([
                label,
                self.formatter.format_value(key, val),
                self.formatter.get_status(key, val)
            ])
            
        table = self.templates.create_styled_table(data, [2.5 * inch, 2.5 * inch, 2 * inch], self.styles)
        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))
        return elements

    def get_action_plan(self, actions):
        """
        Generates the recommended action plan section.
        """
        elements = []
        elements.append(Paragraph("7. RECOMMENDED ACTION PLAN", self.styles['VeritasSectionHeader']))
        
        action_cards = []
        for action in actions[:4]: # Top 4 actions
            card = self.templates.create_action_item(
                None, 
                action.get('title', 'Recommendation'),
                action.get('message', ''),
                action.get('severity', 'MEDIUM'),
                self.styles
            )
            action_cards.append(card)
            
        # Layout cards in a grid
        if action_cards:
            grid_data = [action_cards[i:i+2] for i in range(0, len(action_cards), 2)]
            action_grid = Table(grid_data, colWidths=[3.5 * inch, 3.5 * inch])
            action_grid.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ]))
            elements.append(action_grid)
            
        elements.append(Spacer(1, 0.3 * inch))
        return elements

    def get_footer(self, canvas, doc):
        """
        Draws the footer on each page.
        """
        canvas.saveState()
        canvas.setStrokeColor(VeritasColors.SLATE_200)
        canvas.line(0.5 * inch, 0.75 * inch, 8 * inch, 0.75 * inch)
        
        footer_text = f"VERITAS AI - Confidential Intelligence Report | Generated on {datetime.now().strftime('%Y-%m-%d')} | Page {doc.page}"
        p = Paragraph(footer_text, self.styles['VeritasFooter'])
        w, h = p.wrap(doc.width, doc.bottomMargin)
        p.drawOn(canvas, doc.leftMargin, 0.5 * inch)
        canvas.restoreState()
