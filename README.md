# XAI-RAS v1.4.0 | Global IME Bank

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?logo=plotly&logoColor=white)
![uv](https://img.shields.io/badge/uv-Environment%20%26%20Packages-6E56CF)

An explainable AI (XAI) risk assessment dashboard for transparent and practical credit decision support at Global IME Bank.

यो प्रणाली Global IME Bank का लागि बनाइएको Explainable AI (XAI) आधारित जोखिम मूल्याङ्कन ड्यासबोर्ड हो, जसले पारदर्शी र व्यवहारिक ऋण निर्णय सहयोग प्रदान गर्छ।

## Key Features

- **Real-time XAI**
  - Live risk score updates as applicant inputs change.
  - Contribution-based risk driver chart for clear model interpretation.
- **NRB 2025 Compliance**
  - Supports transparency and traceability patterns aligned with NRB AI Guidelines 2025.
  - Improves audit-readiness for internal credit review workflows.
- **Nepali Customer View**
  - Includes Sajilo Natija mode for customer-facing Nepali messaging.
  - Provides plain-language guidance and What-If suggestions.
- **Advanced Fraud Prevention**
  - Combines e-KYC identity verification flow with simulated CIB API checks to strengthen applicant validation.
  - Demonstrates layered fraud-screening controls often used in real-world digital banking workflows.

## विशेषताहरू (Nepali)

- **रियल-टाइम XAI**: इनपुट परिवर्तनसँगै जोखिम स्कोर र व्याख्या तुरुन्त अपडेट हुन्छ।
- **NRB 2025 अनुरूपता**: पारदर्शिता र ट्रेसबिलिटीलाई प्राथमिकता दिँदै नियामकीय दिशानिर्देशसँग मेल खान्छ।
- **नेपाली ग्राहक दृश्य**: सजिलो भाषामा नतिजा र सुझाव प्रस्तुत हुन्छ।

## Technologies Used

- Python
- Streamlit
- Plotly
- uv

## Technical Excellence

- **Simulated ML Inference Engine (Sigmoid-based)**
  - The application uses a simulated logistic inference pipeline with sigmoid probability scoring to emulate modern risk prediction behavior.
  - This design helps demonstrate how explainable AI can be applied to credit scoring with clear, interpretable outputs.
- **Simulated CIB API Integration**
  - A simulated CIB Nepal verification flow is included to mirror real-world external credit-record checks in a banking process.
  - The persistent verification state in the interface demonstrates practical workflow continuity from data verification to advisory output.
- **Real-world Workflow and NRB 2025 Alignment**
  - Together, these simulations showcase an end-to-end banking decision-support journey, from applicant profiling to explainable recommendation.
  - The workflow supports transparency, traceability, and human-review readiness aligned with NRB 2025 compliance expectations.

## Compliance

- **NRB AI Guidelines 2025**
  - The application is structured to reflect NRB AI Guidelines 2025 principles, including transparency, traceability, and human-in-the-loop review.
  - Simulated explainability, verification checkpoints, and advisory outputs are included to model compliance-oriented banking operations.

## Project Structure

```text
XAI_RAS_GlobalIME/
|- app.py
|- requirements.txt
|- assets/
`- reports/
```

## Screenshots

Add screenshots to highlight both Bank Officer View and Sajilo Natija View.

Example placeholder paths:

- assets/screenshot-officer-view.png
- assets/screenshot-sajilo-natija.png

## Getting Started

### Prerequisites

- Python 3.10+
- uv

Install uv if needed:

```bash
pip install uv
```

### Setup With uv (Recommended)

Create and activate a virtual environment:

```bash
uv venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

Run the app:

```bash
uv run streamlit run app.py
```

Open the local URL shown in terminal (typically http://localhost:8501).

## Contribution

Contributions are welcome. For changes, please:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes with clear messages.
4. Open a pull request describing the improvement.

## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for details.

## Contact

For project and collaboration inquiries:

- Team: XAI-RAS / Global IME Bank
- Open an issue in this repository for bug reports and feature requests.
- Use repository discussions for architecture and collaboration topics.

## Version

Current release: v1.4.0
