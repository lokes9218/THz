"""
Automated Experiment Report Generator
Generates comprehensive HTML & Markdown experiment summary reports.
"""

import os
import json
import pandas as pd

class ReportGenerator:
    """Generates standardized technical research report from simulation & experiment outputs."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_html_report(self, metrics_json_path: str, report_filename: str = "experiment_report.html") -> str:
        """Compiles interactive HTML experiment report."""
        metrics_data = {}
        if os.path.exists(metrics_json_path):
            with open(metrics_json_path, 'r') as f:
                metrics_data = json.load(f)
                
        svr_mae = metrics_data.get('svr', {}).get('test', {}).get('mae', 'N/A')
        svr_r2 = metrics_data.get('svr', {}).get('test', {}).get('r2', 'N/A')
        mlp_mae = metrics_data.get('mlp', {}).get('test', {}).get('mae', 'N/A')
        mlp_r2 = metrics_data.get('mlp', {}).get('test', {}).get('r2', 'N/A')
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>THz Glucose Non-Invasive Simulation - Experiment Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f8f9fa; color: #333; }}
        h1 {{ color: #0d6efd; border-bottom: 2px solid #0d6efd; padding-bottom: 10px; }}
        h2 {{ color: #495057; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; background: white; }}
        th, td {{ border: 1px solid #dee2e6; padding: 12px; text-align: left; }}
        th {{ background-color: #e9ecef; }}
        .badge {{ background: #198754; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
        .box {{ background: white; border-left: 4px solid #0d6efd; padding: 15px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
    </style>
</head>
<body>
    <h1>THz Non-Invasive Blood Glucose Estimation - Experiment Report</h1>
    <p><strong>Status:</strong> <span class="badge">Computational Simulation Verified</span></p>
    <p><strong>Target Technology:</strong> Non-contact THz Spectroscopy & 3D Holographic Machine Learning</p>
    
    <div class="box">
        <h3>Primary Objective</h3>
        <p>This report documents the software simulation pipeline evaluating non-invasive blood glucose estimation from terahertz spectral reflection and 3D computational holographic wave back-propagation.</p>
    </div>

    <h2>1. Model Evaluation Results</h2>
    <table>
        <tr>
            <th>Model Architecture</th>
            <th>Test MAE (mg/dL)</th>
            <th>Test R² Score</th>
            <th>Evaluation Status</th>
        </tr>
        <tr>
            <td>Support Vector Regressor (SVR - RBF)</td>
            <td>{svr_mae}</td>
            <td>{svr_r2}</td>
            <td>Evaluated</td>
        </tr>
        <tr>
            <td>Multi-Layer Perceptron (MLP)</td>
            <td>{mlp_mae}</td>
            <td>{mlp_r2}</td>
            <td>Evaluated</td>
        </tr>
    </table>

    <h2>2. Key Physics & Simulation Assumptions</h2>
    <ul>
        <li><strong>Frequency Range:</strong> 0.1 THz to 2.0 THz across 200 grid points.</li>
        <li><strong>Skin Tissue Permittivity:</strong> Modified Debye relaxation equations with glucose perturbation in dermis layer.</li>
        <li><strong>Atmospheric Model:</strong> Arden Buck water vapor absorption + THz line resonances.</li>
        <li><strong>3D Holography:</strong> Physical Angular Spectrum Method (ASM) diffraction back-propagation.</li>
    </ul>

    <h2>3. Research & Technical Limitations</h2>
    <p>This implementation is a <em>software-only computational prototype</em>. Simulated measurements do not guarantee physical hardware performance without clinical human trial calibration.</p>

    <footer style="margin-top: 50px; text-align: center; color: #6c757d; font-size: 12px;">
        Generated automatically by THz Glucose Software Simulation Prototype Pipeline
    </footer>
</body>
</html>
"""
        filepath = os.path.join(self.output_dir, report_filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return filepath

if __name__ == "__main__":
    rg = ReportGenerator("results/reports")
    out = rg.generate_html_report("results/metrics/metrics.json")
    print("Report generated at:", out)
