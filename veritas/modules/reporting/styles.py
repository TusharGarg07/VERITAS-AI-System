from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.units import inch

def get_veritas_styles():
    """
    Returns a collection of professional enterprise styles for VERITAS AI reports.
    """
    styles = getSampleStyleSheet()
    
    # Custom colors
    NAVY = colors.HexColor("#0F172A")
    EMERALD = colors.HexColor("#065F46")
    SLATE_50 = colors.HexColor("#F8FAFC")
    SLATE_200 = colors.HexColor("#E2E8F0")
    SLATE_400 = colors.HexColor("#94A3B8")
    SLATE_600 = colors.HexColor("#475569")
    WHITE = colors.white
    
    # Risk Colors
    CRITICAL = colors.HexColor("#EF4444")
    HIGH = colors.HexColor("#F97316")
    MODERATE = colors.HexColor("#F59E0B")
    SAFE = colors.HexColor("#10B981")

    # Base Text
    styles.add(ParagraphStyle(
        name='VeritasNormal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=SLATE_600,
        leading=14,
        alignment=TA_JUSTIFY
    ))

    # Header Styles
    styles.add(ParagraphStyle(
        name='VeritasTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=NAVY,
        leading=30,
        alignment=TA_LEFT,
        spaceAfter=12
    ))

    styles.add(ParagraphStyle(
        name='VeritasSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        textColor=EMERALD,
        leading=16,
        alignment=TA_LEFT,
        spaceAfter=20
    ))

    # Section Headers
    styles.add(ParagraphStyle(
        name='VeritasSectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=NAVY,
        leading=18,
        spaceBefore=20,
        spaceAfter=10,
        borderPadding=5,
        alignment=TA_LEFT
    ))

    # KPI Card Styles
    styles.add(ParagraphStyle(
        name='KPILabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=SLATE_600,
        alignment=TA_CENTER,
        spaceAfter=4
    ))

    styles.add(ParagraphStyle(
        name='KPIValue',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=NAVY,
        alignment=TA_CENTER
    ))

    styles.add(ParagraphStyle(
        name='KPIStatus',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        alignment=TA_CENTER,
        spaceBefore=4
    ))

    # Table Styles
    styles.add(ParagraphStyle(
        name='TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=WHITE,
        alignment=TA_LEFT
    ))

    styles.add(ParagraphStyle(
        name='TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=SLATE_600,
        alignment=TA_LEFT
    ))

    # Footer
    styles.add(ParagraphStyle(
        name='VeritasFooter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        textColor=SLATE_400,
        alignment=TA_CENTER
    ))

    return styles

class VeritasColors:
    NAVY = colors.HexColor("#0F172A")
    EMERALD = colors.HexColor("#059669")
    EMERALD_LIGHT = colors.HexColor("#ECFDF5")
    SLATE_50 = colors.HexColor("#F8FAFC")
    SLATE_100 = colors.HexColor("#F1F5F9")
    SLATE_200 = colors.HexColor("#E2E8F0")
    SLATE_300 = colors.HexColor("#CBD5E1")
    SLATE_700 = colors.HexColor("#334155")
    WHITE = colors.white
    
    # Status Colors
    CRITICAL = colors.HexColor("#B91C1C") # Darker red for text
    CRITICAL_BG = colors.HexColor("#FEF2F2")
    HIGH = colors.HexColor("#C2410C") # Darker orange
    HIGH_BG = colors.HexColor("#FFF7ED")
    MODERATE = colors.HexColor("#B45309") # Darker amber
    MODERATE_BG = colors.HexColor("#FFFBEB")
    SAFE = colors.HexColor("#047857") # Darker emerald
    SAFE_BG = colors.HexColor("#ECFDF5")
