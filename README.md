# CRAI - Cry Recognition and Analysis Intelligence

> **An AI-powered baby cry analysis system for early detection of genetic diseases and health conditions**  
> Built for **HackMIT 2025** 🚀

## 🌟 Overview

CRAI is an advanced intelligent baby cry monitoring and analysis application that leverages cutting-edge AI technologies, signal processing, and clinical research to detect potential health issues in infants through acoustic analysis. The system can identify genetic disorders, neurological conditions, respiratory distress, and other medical indicators from baby cry patterns.

## 🏆 HackMIT 2025 Project

This project was developed for **HackMIT 2025** with significant contributions from **Arya Manjaramkar**. The system combines real-time audio processing, deep learning models, and clinical heuristics to provide caregivers with early warning indicators for various infant health conditions.

## 🚀 Key Technologies

### **Cerebras AI** 🤖
- **Purpose**: Powers the intelligent AI chatbot that provides contextual medical guidance and explanations
- **Model**: `llama-4-scout-17b-16e-instruct` via Cerebras API
- **Integration**: Real-time conversational AI that helps parents understand cry analysis results and provides evidence-based recommendations
- **Location**: `src/components/AIChatbot.tsx`

### **Modal** ☁️
- **Purpose**: Serverless cloud deployment of the ML backend API
- **Infrastructure**: FastAPI application deployed on Modal's serverless platform
- **Features**: 
  - Scalable audio processing endpoints
  - Real-time cry detection and classification
  - Comprehensive medical pattern analysis
- **Location**: `MLbackend/modal_app.py`
- **Deployment**: `https://aryagm01--baby-cry-web-api-fastapi-app.modal.run`

### **Librosa** 🎵
- **Purpose**: Core audio signal processing and feature extraction
- **Capabilities**:
  - Fundamental frequency (F0) extraction using YIN algorithm
  - Mel-spectrogram generation for deep learning models
  - MFCC (Mel-Frequency Cepstral Coefficients) extraction
  - Spectral feature analysis (centroid, rolloff, flatness)
  - RMS energy calculation for intensity analysis
- **Integration**: Used throughout the audio analysis pipeline in both frontend and backend
- **Location**: `MLbackend/audio_analyzer.py`, `MLbackend/modal_app.py`

## 🔬 How Cry Classification Works

### 1. **Audio Feature Extraction**
The system uses **Librosa** to extract comprehensive acoustic features:

- **Fundamental Frequency (F0)**: Detected using Parselmouth (Praat) and Librosa's YIN algorithm
- **Spectral Features**: Centroid, rolloff, flatness, and MFCCs
- **Voice Quality Metrics**: Harmonics-to-Noise Ratio (HNR), jitter, shimmer
- **Temporal Patterns**: Voiced segment detection, burst analysis, repetition rates
- **Energy Analysis**: RMS intensity, spectral band energy ratios

### 2. **Deep Learning Classification**
- **Model Architecture**: Custom TensorFlow/Keras models trained on 1,126+ baby cry samples
- **Cry Types Classified**: Hunger, discomfort, pain, fatigue, attention, belly pain, burping, cold/hot, scared, lonely
- **Input**: Mel-spectrograms and feature vectors extracted via Librosa
- **Output**: Probability distributions across cry categories with confidence scores

### 3. **Genetic Disease Detection**
The system implements heuristic algorithms based on clinical research to detect:

#### **Cri-du-chat Syndrome** (Genetic Disorder)
- **Pattern**: Mean F0 ≥ 800 Hz, low F0 variability (< 80 Hz SD)
- **Characteristics**: Cat-like cry with burst lengths 100-500ms, repetition rate 1-2 Hz
- **Research Basis**: Clinical studies on 5p- deletion syndrome acoustic markers

#### **Hyperphonation** (Neurological Indicator)
- **Pattern**: Voiced frames with F0 > 1000 Hz for ≥150ms
- **Significance**: May indicate brain injury or neurological distress
- **Detection**: Continuous high-pitch episode analysis

#### **Hoarseness Patterns** (Laryngeal/Endocrine)
- **Metrics**: HNR < 5-7 dB, jitter > 1.5%, shimmer > 3-5%
- **Conditions**: Laryngeal issues, hypothyroidism
- **Analysis**: Voice quality degradation detection

#### **Weak Cry Patterns** (Neuromuscular)
- **Indicators**: Low RMS amplitude, short voiced segments, long pauses
- **Conditions**: Spinal Muscular Atrophy (SMA), botulism, muscle weakness
- **Method**: Baseline comparison and temporal pattern analysis

#### **Respiratory Distress** (Grunting)
- **Pattern**: Repetitive expiratory pulses (150-350ms)
- **Frequency**: Strong low-mid harmonics at breathing cadence (0.5-1.2 Hz)
- **Detection**: Pulse envelope analysis and spectral band energy

## 📚 Research Foundation

This system is built upon extensive research in:

1. **Infant Cry Analysis and Automatic Recognition**
   - Acoustic feature extraction for cry classification
   - Machine learning approaches to pediatric vocalization analysis

2. **Genetic Disease Detection via Acoustic Analysis**
   - Cri-du-chat syndrome acoustic markers
   - Neurological condition indicators in infant cries
   - Early detection methodologies for genetic disorders

3. **Clinical Applications of Infant Vocalization**
   - Voice quality assessment in pediatric populations
   - Acoustic correlates of neurological conditions
   - Respiratory distress pattern recognition

4. **Signal Processing for Medical Diagnosis**
   - Fundamental frequency analysis for pathology detection
   - Spectral analysis for voice quality assessment
   - Temporal pattern recognition for disease classification

## 🏗️ System Architecture

### Frontend (React + TypeScript + Vite)
- **Real-time Audio Capture**: WebRTC MediaRecorder API
- **Audio Visualization**: Waveform display and spectral analysis
- **AI Chatbot**: Cerebras-powered conversational interface
- **Alert Dashboard**: Multi-level severity classification and risk assessment
- **Feature Display**: Detailed acoustic measurements and visualizations

### Backend (Python + FastAPI + Modal)
- **Audio Processing**: Librosa-based feature extraction
- **ML Models**: TensorFlow/Keras cry classification models
- **API Endpoints**: RESTful API for cry detection and analysis
- **Cloud Deployment**: Serverless architecture on Modal

### ML Pipeline
1. **Audio Preprocessing**: Sample rate normalization, noise reduction
2. **Feature Extraction**: Librosa-based acoustic feature computation
3. **Model Inference**: Deep learning classification + heuristic analysis
4. **Pattern Recognition**: Genetic disease and medical condition detection
5. **Alert Generation**: Multi-level severity classification

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/aaravhmodi/crai.git
cd crai/dababy

# Install frontend dependencies
npm install

# Install Python dependencies (for local development)
cd MLbackend
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the frontend development server
npm run dev

# The app will be available at http://localhost:5173
```

### Using Modal Backend

The ML backend is deployed on Modal. API endpoints are available at:
- `https://aryagm01--baby-cry-web-api-fastapi-app.modal.run`

## 📊 Features

### Real-time Monitoring
- Continuous audio monitoring with visual waveform display
- Automatic cry detection and analysis
- Real-time processing with 3-second analysis windows

### Comprehensive Analysis
- **8 Detection Algorithms**: Hyperphonation, hoarseness, cri-du-chat, weak cry, grunting, serious illness, hearing impairment, hypernasality
- **Multi-level Alerts**: Critical, High, Medium, Low severity classification
- **AI-Powered Guidance**: Cerebras chatbot provides contextual explanations

### Medical Indicators Detected
- 🔴 **Critical**: Cri-du-chat syndrome, serious illness (sepsis/meningitis), respiratory distress
- 🟠 **High**: Neurological distress, laryngeal issues
- 🟡 **Medium**: Hearing impairment indicators, weak cry patterns
- 🔵 **Low**: Baseline monitoring and tracking

## ⚠️ Medical Disclaimer

**This system is for monitoring and alerting purposes only:**
- Not a substitute for professional medical diagnosis
- Always consult healthcare providers for medical decisions
- False positives and negatives are possible
- Intended as a supportive tool for caregivers

**For medical emergencies, always contact emergency services immediately.**

## 👥 Credits

### Development Team
- **Primary Developer**: Aarav Modi
- **Major Contributor**: **Arya Manjaramkar** - Significant contributions to ML backend, Modal deployment, and cry classification algorithms

### Acknowledgments
- HackMIT 2025 organizers and sponsors
- Research community in infant cry analysis and pediatric health monitoring

## 📁 Project Structure

```
crai/
├── dababy/
│   ├── src/                    # React frontend
│   │   ├── components/         # UI components
│   │   │   ├── AIChatbot.tsx   # Cerebras AI integration
│   │   │   └── ...
│   │   └── lib/                # Core libraries
│   │       ├── audioAnalysis.ts
│   │       ├── cryDetection.ts
│   │       └── medicalDiagnosis.ts
│   ├── MLbackend/              # Python ML backend
│   │   ├── modal_app.py        # Modal deployment
│   │   ├── baby_cry_detector.py # Detection algorithms
│   │   ├── audio_analyzer.py    # Librosa processing
│   │   └── baby_cry_training.ipynb # Model training
│   └── package.json
└── README.md
```

## 🔮 Future Enhancements

- [ ] Clinical validation studies
- [ ] Healthcare provider integration
- [ ] Mobile application (iOS/Android)
- [ ] Historical trend analysis
- [ ] Multi-language support
- [ ] Regulatory compliance (FDA considerations)

## 📄 License

MIT License - See LICENSE file for details.

---

**Built with ❤️ for HackMIT 2025**

*Empowering parents with AI-driven insights for infant health monitoring*

