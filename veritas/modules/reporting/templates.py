from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from .styles import VeritasColors

class VeritasTemplates:
    """
    Reusable UI blocks and layout templates for the VERITAS report.
    """
    
    @staticmethod
    def create_kpi_card(label, value, status, styles):
        """
        Creates a stylized KPI card.
        """
        status_colors = {
            "CRITICAL": (VeritasColors.CRITICAL_BG, VeritasColors.CRITICAL),
            "HIGH": (VeritasColors.HIGH_BG, VeritasColors.HIGH),
            "MODERATE": (VeritasColors.MODERATE_BG, VeritasColors.MODERATE),
            "SAFE": (VeritasColors.SAFE_BG, VeritasColors.SAFE),
            "GOOD": (VeritasColors.SAFE_BG, VeritasColors.SAFE),
            "NORMAL": (VeritasColors.SLATE_100, VeritasColors.NAVY)
        }
        
        bg_color, text_color = status_colors.get(status, (VeritasColors.SLATE_50, VeritasColors.NAVY))
        
        # Create specialized style for this status
        status_style = styles['KPIStatus'].clone('DynamicStatus')
        status_style.textColor = text_color
        
        data = [
            [Paragraph(label, styles['KPILabel'])],
            [Paragraph(value, styles['KPIValue'])],
            [Paragraph(status, status_style)]
        ]
        
        t = Table(data, colWidths=[1.4 * 72]) # 1.4 inch
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('ROUNDEDCORNERS', [8, 8, 8, 8]),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        return t

    @staticmethod
    def create_styled_table(data, col_widths, styles):
        """
        Creates a professional data table with Veritas styling.
        """
        # Wrap data in Paragraphs for better control
        table_data = []
        for i, row in enumerate(data):
            styled_row = []
            for j, cell in enumerate(row):
                if i == 0:
                    styled_row.append(Paragraph(str(cell), styles['TableHeader']))
                else:
                    styled_row.append(Paragraph(str(cell), styles['TableCell']))
            table_data.append(styled_row)
            
        t = Table(table_data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), VeritasColors.NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, VeritasColors.SLATE_200),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [VeritasColors.WHITE, VeritasColors.SLATE_50]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        return t

    @staticmethod
    def create_action_item(icon, title, description, priority, styles):
        """
        Creates a stylized recommendation card.
        """
        priority_colors = {
            "CRITICAL": VeritasColors.CRITICAL,
            "HIGH": VeritasColors.HIGH,
            "MEDIUM": VeritasColors.MODERATE,
            "LOW": VeritasColors.SAFE
        }
        color = priority_colors.get(priority.upper(), VeritasColors.NAVY)
        
        # Action Item Layout
        header = Paragraph(f"<b>{title}</b>", styles['VeritasNormal'])
        desc = Paragraph(description, styles['VeritasNormal'])
        prio_tag = Paragraph(f"<font color='{color.hexval()}'>PRIORITY: {priority}</font>", styles['VeritasNormal'])
        
        data = [[header], [desc], [prio_tag]]
        t = Table(data, colWidths=[1.6 * 72])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('BOX', (0, 0), (-1, -1), 1, VeritasColors.SLATE_200),
            ('ROUNDEDCORNERS', [4, 4, 4, 4]),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        return t
