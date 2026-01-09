# DaBaby - Intelligent Baby Cry Analysis System

An advanced baby cry monitoring and analysis application that uses heuristic algorithms to detect potential health issues in infants through acoustic analysis.

## Features

### Real-time Monitoring
- Continuous audio monitoring with visual waveform display
- Automatic cry detection and analysis
- Real-time processing with 3-second analysis windows

### Comprehensive Heuristic Analysis

The system implements multiple detection algorithms based on clinical research:

#### 1. **Hyperphonation Detection**
- Detects voiced frames with F0 > 1000 Hz for ≥150ms
- Flags potential neurological distress
- **Clinical Significance**: May indicate brain injury or neurological issues

#### 2. **Hoarseness Pattern Recognition**
- Monitors HNR < 5-7 dB, jitter > 1.5%, shimmer > 3-5%
- **Clinical Significance**: May suggest laryngeal issues or hypothyroidism

#### 3. **Cri-du-chat Syndrome Detection**
- Identifies cat-like cry patterns: mean F0 ≥ 800 Hz, low F0 variability
- Burst lengths 100-500ms, repetition rate 1-2 Hz
- **Clinical Significance**: Genetic disorder requiring immediate evaluation

#### 4. **Weak Cry Pattern Analysis**
- Detects low amplitude cries with short voiced segments
- Compares against infant's baseline patterns
- **Clinical Significance**: May indicate SMA, botulism, or muscle weakness

#### 5. **Respiratory Distress (Grunting)**
- Identifies repetitive expiratory pulses (150-350ms)
- Strong low-mid harmonics at breathing cadence (0.5-1.2 Hz)
- **Clinical Significance**: Respiratory distress requiring immediate care

#### 6. **Serious Illness Alert**
- Continuous crying >5-10 minutes with abnormal spectral characteristics
- High-pitched inconsolable crying patterns
- **Clinical Significance**: Possible sepsis, meningitis - seek immediate care

#### 7. **Hearing Impairment Indicators**
- Higher mean F0, lower intensity, longer cry duration
- Uses lightweight classification on acoustic features
- **Clinical Significance**: Early hearing loss detection

#### 8. **Hypernasality Detection**
- Analyzes nasal vs oral energy ratios (250-300 Hz vs 500-1500 Hz)
- **Clinical Significance**: May indicate palatal or velopharyngeal issues

## Technical Implementation

### Audio Processing
- **Sample Rate**: 44.1 kHz
- **Analysis Windows**: 25ms with 50% overlap
- **Features Extracted**:
  - Fundamental frequency (F0) contours
  - Harmonics-to-noise ratio (HNR)
  - Jitter and shimmer measurements
  - Spectral centroid and flatness
  - Voice activity detection
  - Temporal segmentation

### Real-time Analysis Pipeline
1. **Audio Capture**: WebRTC MediaRecorder API
2. **Feature Extraction**: FFT-based spectral analysis
3. **Pattern Recognition**: Rule-based heuristic algorithms
4. **Alert Generation**: Multi-level severity classification
5. **Dashboard Display**: Real-time visualization

## Usage

### Starting the Application
```bash
npm install
npm run dev
```

### Operating the System
1. **Start Monitoring**: Click "Start Continuous Monitoring"
2. **Real-time Analysis**: System automatically processes detected audio
3. **Alert Review**: Monitor the dashboard for alerts and recommendations
4. **Feature Analysis**: Review detailed acoustic measurements

### Alert Levels
- **🔴 Critical**: Immediate medical attention required
- **🟠 High**: Prompt medical evaluation recommended
- **🟡 Medium**: Monitor and consider consultation
- **🔵 Low**: Awareness and tracking recommended

## Important Medical Disclaimers

⚠️ **This system is for monitoring and alerting purposes only**
- Not a substitute for professional medical diagnosis
- Always consult healthcare providers for medical decisions
- False positives and negatives are possible
- Intended as a supportive tool for caregivers

## Technical Architecture

### Components
- **ContinuousMonitor**: Real-time audio capture and visualization
- **AudioAnalyzer**: Core signal processing and feature extraction
- **AlertDashboard**: Alert management and risk assessment
- **FeatureDisplay**: Detailed acoustic analysis visualization

### Dependencies
- React 19+ with TypeScript
- Web Audio API for real-time processing
- Tailwind CSS for responsive UI
- Lucide React for iconography

## Research Background

This system implements algorithms based on published research in:
- Infant cry analysis and automatic recognition
- Acoustic correlates of neurological conditions
- Voice quality assessment in pediatric populations
- Clinical applications of infant vocalization analysis

## Development Roadmap

### Phase 1 (Current)
- ✅ Real-time monitoring system
- ✅ Heuristic-based analysis
- ✅ Multi-level alert system
- ✅ Feature visualization

### Phase 2 (Planned)
- [ ] Machine learning model integration
- [ ] Historical trend analysis
- [ ] Parent/caregiver profiles
- [ ] Mobile application
- [ ] Cloud-based data storage

### Phase 3 (Future)
- [ ] Clinical validation studies
- [ ] Healthcare provider integration
- [ ] Regulatory compliance
- [ ] Multilingual support

## Contributing

This project is part of ongoing research in computational pediatric health monitoring. Contributions welcome for:
- Signal processing improvements
- Clinical validation
- User interface enhancements
- Documentation and testing

## License

MIT License - See LICENSE file for details.

---

**For medical emergencies, always contact emergency services immediately. This tool is not intended to replace professional medical care.**