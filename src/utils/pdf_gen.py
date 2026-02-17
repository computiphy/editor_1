from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from pathlib import Path
from typing import List, Dict

class PDFReportGenerator:
    def __init__(self, output_path: Path):
        self.output_path = output_path

    def generate(self, data: Dict):
        """
        Generates a summary PDF report.
        """
        c = canvas.Canvas(str(self.output_path), pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 24)
        c.drawString(100, height - 100, "Wedding AI Pipeline Summary")

        # Summary Metrics
        summary = data.get("summary", {})
        c.setFont("Helvetica", 12)
        c.drawString(100, height - 140, f"Total Input: {summary.get('total_input')}")
        c.drawString(100, height - 160, f"Passed Culling: {summary.get('total_passed')}")
        c.drawString(100, height - 180, f"Restored: {summary.get('total_restored')}")
        c.drawString(100, height - 200, f"Elapsed Time: {summary.get('elapsed_seconds', 0):.2f}s")

        # Table Header
        y = height - 250
        c.setFont("Helvetica-Bold", 10)
        c.drawString(100, y, "Image Path")
        c.drawString(400, y, "Blur Score")
        c.drawString(500, y, "Status")
        
        y -= 20
        c.setFont("Helvetica", 8)
        
        # Details (capped for PDF brevity)
        images = data.get("images", [])[:30]
        for img in images:
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 8)
            
            p = Path(img['path']).name
            status = "PASSED" if img['passed'] else "CULLED"
            
            c.drawString(100, y, p[:50])
            c.drawString(400, y, f"{img['blur_score']:.2f}")
            c.drawString(500, y, status)
            y -= 15

        c.save()
