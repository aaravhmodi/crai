export interface AudioFeatures {
  f0: number[];           // Fundamental frequency array
  f0Mean: number;         // Mean F0
  f0Std: number;          // F0 standard deviation
  hnr: number;            // Harmonics-to-noise ratio
  jitter: number;         // Frequency jitter
  shimmer: number;        // Amplitude shimmer
  rms: number;            // Root mean square amplitude
  spectralCentroid: number;
  spectralFlatness: number;
  voicedSegmentLengths: number[];
  pauseLengths: number[];
  burstLengths: number[];
  repetitionRate: number;
  nasalEnergyRatio: number;
  lowMidHarmonics: number;
  duration: number;
  sampleRate: number;
}

export interface DetectionResult {
  timestamp: Date;
  features: AudioFeatures;
  alerts: Alert[];
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
}

export interface Alert {
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  description: string;
  recommendation: string;
  confidence: number;
}

export class AudioAnalyzer {
  private sampleRate: number = 44100;
  private baseline: { rms: number; count: number } = { rms: 0, count: 0 };
  
  constructor(sampleRate: number = 44100) {
    this.sampleRate = sampleRate;
  }

  // Main analysis function
  async analyzeAudio(audioBuffer: AudioBuffer, detectionCount: number = 1): Promise<DetectionResult> {
    const features = await this.extractFeatures(audioBuffer);
    const alerts = this.detectPatterns(features);
    const riskLevel = this.calculateRiskLevel(alerts, detectionCount);

    return {
      timestamp: new Date(),
      features,
      alerts,
      riskLevel
    };
  }

  // Extract audio features from buffer
  private async extractFeatures(audioBuffer: AudioBuffer): Promise<AudioFeatures> {
    const channelData = audioBuffer.getChannelData(0);
    const sampleRate = audioBuffer.sampleRate;
    
    // Basic signal properties
    const rms = this.calculateRMS(channelData);
    const duration = audioBuffer.duration;
    
    // Frequency analysis
    const windowSize = Math.floor(sampleRate * 0.025); // 25ms windows
    const hopSize = Math.floor(windowSize / 2);
    
    const f0Array: number[] = [];
    const spectralData: number[][] = [];
    
    // Windowed analysis
    for (let i = 0; i < channelData.length - windowSize; i += hopSize) {
      const window = channelData.slice(i, i + windowSize);
      const f0 = this.estimateF0(window, sampleRate);
      if (f0 > 0) {
        f0Array.push(f0);
      }
      
      const spectrum = this.fft(this.applyWindow(window));
      spectralData.push(spectrum);
    }

    const f0Mean = f0Array.length > 0 ? f0Array.reduce((a, b) => a + b) / f0Array.length : 0;
    const f0Std = this.calculateStandardDeviation(f0Array);
    
    // Advanced features
    const hnr = this.calculateHNR(channelData, f0Mean, sampleRate);
    const jitter = this.calculateJitter(f0Array);
    const shimmer = this.calculateShimmer(channelData, sampleRate);
    const spectralCentroid = this.calculateSpectralCentroid(spectralData, sampleRate);
    const spectralFlatness = this.calculateSpectralFlatness(spectralData);
    
    // Segment analysis
    const voicedSegments = this.detectVoicedSegments(channelData, sampleRate);
    const pauseSegments = this.detectPauses(channelData, sampleRate);
    const burstSegments = this.detectBursts(channelData, sampleRate);
    
    const repetitionRate = this.calculateRepetitionRate(burstSegments, duration);
    const nasalEnergyRatio = this.calculateNasalEnergyRatio(spectralData, sampleRate);
    const lowMidHarmonics = this.calculateLowMidHarmonics(spectralData, sampleRate);

    return {
      f0: f0Array,
      f0Mean,
      f0Std,
      hnr,
      jitter,
      shimmer,
      rms,
      spectralCentroid,
      spectralFlatness,
      voicedSegmentLengths: voicedSegments.map(s => s.duration),
      pauseLengths: pauseSegments.map(s => s.duration),
      burstLengths: burstSegments.map(s => s.duration),
      repetitionRate,
      nasalEnergyRatio,
      lowMidHarmonics,
      duration,
      sampleRate
    };
  }

  // Detect patterns and generate alerts
  private detectPatterns(features: AudioFeatures): Alert[] {
    const alerts: Alert[] = [];

    // 1. Hyperphonation detection
    const hyperphonationFrames = features.f0.filter(f0 => f0 > 1000);
    if (hyperphonationFrames.length >= 6) { // ~150ms at 25ms windows
      alerts.push({
        type: 'hyperphonation',
        severity: 'high',
        message: 'Hyperphonation detected',
        description: 'Voiced frames with F0 > 1000 Hz detected for ≥150ms',
        recommendation: 'Consider neurological distress evaluation',
        confidence: Math.min(0.9, hyperphonationFrames.length / 10)
      });
    }

    // 2. Hoarseness detection
    if (features.hnr < 6 || features.jitter > 1.5 || features.shimmer > 4) {
      alerts.push({
        type: 'hoarseness',
        severity: 'medium',
        message: 'Hoarseness-like pattern detected',
        description: `HNR: ${features.hnr.toFixed(1)}dB, Jitter: ${features.jitter.toFixed(2)}%, Shimmer: ${features.shimmer.toFixed(2)}%`,
        recommendation: 'Consider laryngeal issues or hypothyroidism check',
        confidence: 0.7
      });
    }

    // 3. Cri-du-chat pattern
    if (features.f0Mean >= 800 && features.f0Std < 80 && 
        features.burstLengths.some(b => b >= 0.1 && b <= 0.5) &&
        features.repetitionRate >= 1 && features.repetitionRate <= 2) {
      alerts.push({
        type: 'cri_du_chat',
        severity: 'critical',
        message: 'Cat-like cry pattern detected',
        description: `High-pitched monotonous cry (F0: ${features.f0Mean.toFixed(0)}Hz, SD: ${features.f0Std.toFixed(0)}Hz)`,
        recommendation: 'Immediate genetic evaluation recommended',
        confidence: 0.85
      });
    }

    // 4. Weak cry pattern
    const baselineRms = this.baseline.rms;
    const isWeakCry = baselineRms > 0 && features.rms < (baselineRms - 2 * 0.1) && // Simplified baseline calculation
                      features.voicedSegmentLengths.some(v => v < 0.3) &&
                      features.pauseLengths.some(p => p > 1.0);
    
    if (isWeakCry) {
      alerts.push({
        type: 'weak_cry',
        severity: 'high',
        message: 'Weak cry pattern detected',
        description: 'Low amplitude cry with short voiced segments and long pauses',
        recommendation: 'Consider SMA/botulism screening',
        confidence: 0.75
      });
    }

    // 5. Grunting detector
    const gruntingBursts = features.burstLengths.filter(b => b >= 0.15 && b <= 0.35);
    if (gruntingBursts.length >= 3 && features.repetitionRate >= 0.5 && features.repetitionRate <= 1.2 &&
        features.lowMidHarmonics > 0.6) {
      alerts.push({
        type: 'grunting',
        severity: 'high',
        message: 'Grunting pattern detected',
        description: 'Repetitive expiratory pulses with strong low-mid harmonics',
        recommendation: 'Respiratory distress - seek immediate care',
        confidence: 0.8
      });
    }

    // 6. Sepsis/meningitis alert
    const isContinuousCry = features.duration > 300; // 5 minutes
    const hasMoaningTimbre = features.spectralCentroid < 600 && features.spectralFlatness > 0.7;
    const isHighPitchedInconsolable = features.f0Mean > 1200 && features.duration > 180;
    
    if (isContinuousCry && (hasMoaningTimbre || isHighPitchedInconsolable)) {
      alerts.push({
        type: 'serious_illness',
        severity: 'critical',
        message: 'Serious illness alert pattern',
        description: 'Continuous cry with concerning spectral characteristics',
        recommendation: 'SEEK IMMEDIATE MEDICAL CARE',
        confidence: 0.9
      });
    }

    // 7. Hearing impairment cue (simplified)
    if (features.f0Mean > 600 && features.rms < 0.3 && features.duration > 5) {
      alerts.push({
        type: 'hearing_impairment',
        severity: 'low',
        message: 'Hearing impairment indicators',
        description: 'Higher F0, lower intensity, longer cries detected',
        recommendation: 'Consider hearing evaluation',
        confidence: 0.6
      });
    }

    // 8. Hypernasality proxy
    if (features.nasalEnergyRatio > 0.4) {
      alerts.push({
        type: 'hypernasality',
        severity: 'low',
        message: 'Hypernasality indicators',
        description: 'Elevated nasal resonance detected',
        recommendation: 'Consider palatal or velopharyngeal evaluation',
        confidence: 0.5
      });
    }

    return alerts;
  }

  private calculateRiskLevel(alerts: Alert[], detectionCount: number = 1): 'low' | 'medium' | 'high' | 'critical' {
    if (alerts.length === 0) return 'low';
    
    // Count alerts by severity
    const criticalCount = alerts.filter(a => a.severity === 'critical').length;
    const highCount = alerts.filter(a => a.severity === 'high').length;
    const mediumCount = alerts.filter(a => a.severity === 'medium').length;
    
    // Apply detection count-based risk reduction
    // First few detections should be treated with lower confidence
    let riskReduction = 0;
    if (detectionCount <= 2) {
      riskReduction = 2; // Reduce by 2 levels for first 2 detections
    } else if (detectionCount <= 5) {
      riskReduction = 1; // Reduce by 1 level for detections 3-5
    } else if (detectionCount <= 10) {
      riskReduction = 0; // No reduction for detections 6-10
    }
    // After 10 detections, patterns are more reliable
    
    // Determine base risk level (more conservative thresholds)
    let baseRisk: 'low' | 'medium' | 'high' | 'critical' = 'low';
    
    if (criticalCount >= 2) {
      baseRisk = 'critical';
    } else if (criticalCount >= 1 && (highCount >= 1 || mediumCount >= 2)) {
      baseRisk = 'high';
    } else if (criticalCount >= 1) {
      baseRisk = 'medium'; // Single critical alert becomes medium
    } else if (highCount >= 2) {
      baseRisk = 'high';
    } else if (highCount >= 1 && mediumCount >= 1) {
      baseRisk = 'medium';
    } else if (highCount >= 1) {
      baseRisk = 'low'; // Single high alert becomes low
    } else if (mediumCount >= 2) {
      baseRisk = 'medium';
    } else if (mediumCount >= 1) {
      baseRisk = 'low'; // Single medium alert becomes low
    }
    
    // Apply risk reduction based on detection count
    const riskLevels: ('low' | 'medium' | 'high' | 'critical')[] = ['low', 'medium', 'high', 'critical'];
    const currentIndex = riskLevels.indexOf(baseRisk);
    const adjustedIndex = Math.max(0, currentIndex - riskReduction);
    
    return riskLevels[adjustedIndex];
  }

  // Helper methods for audio analysis
  private calculateRMS(signal: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < signal.length; i++) {
      sum += signal[i] * signal[i];
    }
    return Math.sqrt(sum / signal.length);
  }

  private calculateStandardDeviation(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = values.reduce((a, b) => a + b) / values.length;
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  private estimateF0(window: Float32Array, sampleRate: number): number {
    // Simplified autocorrelation-based F0 estimation
    const minPeriod = Math.floor(sampleRate / 800); // Min 800 Hz
    const maxPeriod = Math.floor(sampleRate / 50);  // Max 50 Hz
    
    let maxCorrelation = 0;
    let bestPeriod = 0;
    
    for (let period = minPeriod; period <= maxPeriod && period < window.length / 2; period++) {
      let correlation = 0;
      for (let i = 0; i < window.length - period; i++) {
        correlation += window[i] * window[i + period];
      }
      
      if (correlation > maxCorrelation) {
        maxCorrelation = correlation;
        bestPeriod = period;
      }
    }
    
    return bestPeriod > 0 ? sampleRate / bestPeriod : 0;
  }

  private fft(signal: Float32Array): number[] {
    // Simplified FFT - in real implementation, use a proper FFT library
    const N = signal.length;
    const spectrum: number[] = new Array(N / 2);
    
    for (let k = 0; k < N / 2; k++) {
      let real = 0, imag = 0;
      for (let n = 0; n < N; n++) {
        const angle = -2 * Math.PI * k * n / N;
        real += signal[n] * Math.cos(angle);
        imag += signal[n] * Math.sin(angle);
      }
      spectrum[k] = Math.sqrt(real * real + imag * imag);
    }
    
    return spectrum;
  }

  private applyWindow(signal: Float32Array): Float32Array {
    // Apply Hamming window
    const windowed = new Float32Array(signal.length);
    for (let i = 0; i < signal.length; i++) {
      const w = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (signal.length - 1));
      windowed[i] = signal[i] * w;
    }
    return windowed;
  }

  private calculateHNR(signal: Float32Array, f0: number, sampleRate: number): number {
    // Simplified HNR calculation
    if (f0 <= 0) return 0;
    
    const period = sampleRate / f0;
    const numPeriods = Math.floor(signal.length / period);
    
    if (numPeriods < 3) return 0;
    
    let harmonicEnergy = 0;
    let noiseEnergy = 0;
    
    // This is a simplified version - real HNR calculation is more complex
    for (let i = 0; i < signal.length - period; i++) {
      const diff = signal[i] - signal[i + Math.floor(period)];
      noiseEnergy += diff * diff;
      harmonicEnergy += signal[i] * signal[i];
    }
    
    return harmonicEnergy > 0 ? 10 * Math.log10(harmonicEnergy / Math.max(noiseEnergy, 1e-10)) : 0;
  }

  private calculateJitter(f0Array: number[]): number {
    if (f0Array.length < 3) return 0;
    
    let periodDiffs = 0;
    let totalPeriods = 0;
    
    for (let i = 1; i < f0Array.length; i++) {
      if (f0Array[i] > 0 && f0Array[i-1] > 0) {
        const period1 = 1 / f0Array[i-1];
        const period2 = 1 / f0Array[i];
        periodDiffs += Math.abs(period1 - period2);
        totalPeriods += (period1 + period2) / 2;
      }
    }
    
    return totalPeriods > 0 ? (periodDiffs / totalPeriods) * 100 : 0;
  }

  private calculateShimmer(signal: Float32Array, sampleRate: number): number {
    // Simplified shimmer calculation
    const frameSize = Math.floor(sampleRate * 0.025);
    const hopSize = Math.floor(frameSize / 2);
    const amplitudes: number[] = [];
    
    for (let i = 0; i < signal.length - frameSize; i += hopSize) {
      const frame = signal.slice(i, i + frameSize);
      const rms = this.calculateRMS(frame);
      amplitudes.push(rms);
    }
    
    if (amplitudes.length < 3) return 0;
    
    let ampDiffs = 0;
    let totalAmps = 0;
    
    for (let i = 1; i < amplitudes.length; i++) {
      ampDiffs += Math.abs(amplitudes[i] - amplitudes[i-1]);
      totalAmps += (amplitudes[i] + amplitudes[i-1]) / 2;
    }
    
    return totalAmps > 0 ? (ampDiffs / totalAmps) * 100 : 0;
  }

  private calculateSpectralCentroid(spectralData: number[][], sampleRate: number): number {
    if (spectralData.length === 0) return 0;
    
    let weightedSum = 0;
    let magnitudeSum = 0;
    
    for (const spectrum of spectralData) {
      for (let i = 0; i < spectrum.length; i++) {
        const frequency = (i * sampleRate) / (2 * spectrum.length);
        weightedSum += frequency * spectrum[i];
        magnitudeSum += spectrum[i];
      }
    }
    
    return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0;
  }

  private calculateSpectralFlatness(spectralData: number[][]): number {
    if (spectralData.length === 0) return 0;
    
    let totalFlatness = 0;
    
    for (const spectrum of spectralData) {
      let geometricMean = 1;
      let arithmeticMean = 0;
      let validBins = 0;
      
      for (const magnitude of spectrum) {
        if (magnitude > 0) {
          geometricMean *= Math.pow(magnitude, 1 / spectrum.length);
          arithmeticMean += magnitude;
          validBins++;
        }
      }
      
      if (validBins > 0) {
        arithmeticMean /= validBins;
        totalFlatness += geometricMean / arithmeticMean;
      }
    }
    
    return spectralData.length > 0 ? totalFlatness / spectralData.length : 0;
  }

  private detectVoicedSegments(signal: Float32Array, sampleRate: number): Array<{start: number, end: number, duration: number}> {
    // Simplified voice activity detection
    const frameSize = Math.floor(sampleRate * 0.025);
    const hopSize = Math.floor(frameSize / 2);
    const threshold = 0.01;
    
    const segments: Array<{start: number, end: number, duration: number}> = [];
    let inSegment = false;
    let segmentStart = 0;
    
    for (let i = 0; i < signal.length - frameSize; i += hopSize) {
      const frame = signal.slice(i, i + frameSize);
      const energy = this.calculateRMS(frame);
      
      if (energy > threshold && !inSegment) {
        inSegment = true;
        segmentStart = i / sampleRate;
      } else if (energy <= threshold && inSegment) {
        inSegment = false;
        const segmentEnd = i / sampleRate;
        segments.push({
          start: segmentStart,
          end: segmentEnd,
          duration: segmentEnd - segmentStart
        });
      }
    }
    
    return segments;
  }

  private detectPauses(signal: Float32Array, sampleRate: number): Array<{start: number, end: number, duration: number}> {
    // Detect silent periods
    const frameSize = Math.floor(sampleRate * 0.025);
    const hopSize = Math.floor(frameSize / 2);
    const threshold = 0.005;
    
    const pauses: Array<{start: number, end: number, duration: number}> = [];
    let inPause = false;
    let pauseStart = 0;
    
    for (let i = 0; i < signal.length - frameSize; i += hopSize) {
      const frame = signal.slice(i, i + frameSize);
      const energy = this.calculateRMS(frame);
      
      if (energy <= threshold && !inPause) {
        inPause = true;
        pauseStart = i / sampleRate;
      } else if (energy > threshold && inPause) {
        inPause = false;
        const pauseEnd = i / sampleRate;
        pauses.push({
          start: pauseStart,
          end: pauseEnd,
          duration: pauseEnd - pauseStart
        });
      }
    }
    
    return pauses;
  }

  private detectBursts(signal: Float32Array, sampleRate: number): Array<{start: number, end: number, duration: number}> {
    // Detect burst-like segments (short, intense periods)
    const frameSize = Math.floor(sampleRate * 0.010); // 10ms frames for burst detection
    const hopSize = Math.floor(frameSize / 2);
    const threshold = 0.05;
    
    const bursts: Array<{start: number, end: number, duration: number}> = [];
    let inBurst = false;
    let burstStart = 0;
    
    for (let i = 0; i < signal.length - frameSize; i += hopSize) {
      const frame = signal.slice(i, i + frameSize);
      const energy = this.calculateRMS(frame);
      
      if (energy > threshold && !inBurst) {
        inBurst = true;
        burstStart = i / sampleRate;
      } else if (energy <= threshold && inBurst) {
        inBurst = false;
        const burstEnd = i / sampleRate;
        const duration = burstEnd - burstStart;
        if (duration <= 0.5) { // Only consider short bursts
          bursts.push({
            start: burstStart,
            end: burstEnd,
            duration
          });
        }
      }
    }
    
    return bursts;
  }

  private calculateRepetitionRate(bursts: Array<{start: number, end: number, duration: number}>, totalDuration: number): number {
    if (bursts.length < 2 || totalDuration <= 0) return 0;
    return bursts.length / totalDuration;
  }

  private calculateNasalEnergyRatio(spectralData: number[][], sampleRate: number): number {
    if (spectralData.length === 0) return 0;
    
    let nasalEnergy = 0;
    let oralEnergy = 0;
    
    for (const spectrum of spectralData) {
      for (let i = 0; i < spectrum.length; i++) {
        const frequency = (i * sampleRate) / (2 * spectrum.length);
        
        if (frequency >= 250 && frequency <= 300) {
          nasalEnergy += spectrum[i];
        } else if (frequency >= 500 && frequency <= 1500) {
          oralEnergy += spectrum[i];
        }
      }
    }
    
    return oralEnergy > 0 ? nasalEnergy / oralEnergy : 0;
  }

  private calculateLowMidHarmonics(spectralData: number[][], sampleRate: number): number {
    if (spectralData.length === 0) return 0;
    
    let lowMidEnergy = 0;
    let totalEnergy = 0;
    
    for (const spectrum of spectralData) {
      for (let i = 0; i < spectrum.length; i++) {
        const frequency = (i * sampleRate) / (2 * spectrum.length);
        
        if (frequency >= 150 && frequency <= 800) {
          lowMidEnergy += spectrum[i];
        }
        totalEnergy += spectrum[i];
      }
    }
    
    return totalEnergy > 0 ? lowMidEnergy / totalEnergy : 0;
  }

  // Update baseline for weak cry detection
  updateBaseline(rms: number): void {
    this.baseline.rms = (this.baseline.rms * this.baseline.count + rms) / (this.baseline.count + 1);
    this.baseline.count++;
  }
}