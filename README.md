# Forensic Fingerprint Analysis System

A comprehensive, AI-powered fingerprint analysis system built with modern web technologies and machine learning.

## Setup

1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd "Fingerprint Project"
   ```

2. **Install dependencies** (minimal, fast):
   ```bash
   pip install -r requirements-minimal.txt
   ```
   Or full install (includes TF/PyTorch for experimental modules):
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment** (optional — needed only for OpenAI AI explanations):
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

4. **Train the classifier** (required for classification to work):
   ```bash
   # Place labeled images in: data/Loop/, data/Whorl/, data/Arch/
   python -c "from src.core.classifier import train_classifier; train_classifier('data')"
   ```

5. **Run the app**:
   ```bash
   streamlit run src/ui/app_modern.py
   # Or: python main.py
   # Or on Windows: double-click run_app.bat
   ```

## Team — M.A.H.E
| Member | Module |
|--------|--------|
| A | UI & Integration (`src/ui/`) |
| B | Pattern Classifier (`src/core/classifier.py`) |
| C | Minutiae Detection (`src/core/minutiae_detector.py`) |
| D | Matching & Database (`src/core/fingerprint_matcher.py`) |
| E | AI Assistant (`src/core/ai_assistant.py`) |

> ⚠️ This is educational/research software — not production-grade biometrics.

## 🏗️ Project Structure

```
src/
├── core/           # Core fingerprint analysis modules
│   ├── classifier.py
│   ├── minutiae_detector.py
│   ├── fingerprint_matcher.py
│   └── ai_assistant.py
├── ui/             # User interface components
│   ├── app.py
│   ├── app_modern.py
│   └── design_system.py
├── utils/          # Utility functions and helpers
├── models/         # Machine learning models and training scripts
├── data/           # Data storage and management
├── config/         # Configuration files
├── tests/          # Test suite
└── docs/           # Documentation
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python main.py
   ```

3. **Or use Streamlit directly:**
   ```bash
   streamlit run src/ui/app_modern.py
   ```

## ✨ Features

- **4-Step Analysis Workflow**: Upload → Classify → Minutiae → Match
- **AI-Powered Classification**: Advanced pattern recognition
- **Minutiae Detection**: Precise ridge ending and bifurcation detection
- **Identity Matching**: Secure database matching with spoof detection
- **Responsive Design**: Works on all screen sizes
- **Modern UI**: Professional forensic analysis interface

## 🎯 Core Capabilities

- **Fingerprint Classification**: Arch, Loop, Whorl, and composite patterns
- **Minutiae Extraction**: Automated feature point detection
- **Spoof Detection**: AI-powered liveness detection
- **Database Management**: Secure fingerprint storage and retrieval
- **Report Generation**: Comprehensive analysis reports

## 🔧 Configuration

Edit `src/config/config.yaml` to customize:
- Database settings
- Model parameters
- UI preferences
- Security thresholds

## 📱 Responsive Design

The application automatically adapts to:
- Desktop screens (1920x1080+)
- Tablet devices (768x1024)
- Mobile phones (375x667)
- All modern web browsers

## 🧪 Testing

Run the test suite:
```bash
python -m pytest src/tests/
```

## 📚 Documentation

- `src/docs/` - Technical documentation
- `src/docs/HOW_TO_RUN.md` - Detailed setup guide
- `src/docs/PROJECT_COMPLETION_SUMMARY.md` - Project overview

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the documentation in `src/docs/`
2. Review existing issues
3. Create a new issue with detailed information

---

**Built with ❤️ using Streamlit, Python, and modern AI technologies**
