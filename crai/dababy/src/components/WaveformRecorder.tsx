import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { WaveformBar } from '@/components/ui/progress';
import { Mic, Square, Brain, Search, CheckCircle, AlertTriangle } from 'lucide-react';
import { AudioAnalyzer, type AudioFeatures } from '@/lib/audioAnalysis';
import { CryDetectionEngine, CryDiagnosisEngine, CryStorage, type CryInstance } from '@/lib/cryDetection';

interface WaveformRecorderProps {
  onCryDetected?: (cry: CryInstance) => void;
}

type RecordingPhase = 'idle' | 'recording' | 'cry-detected' | 'analyzing' | 'complete';
type DetectionState = { isDetecting: boolean; confidence: number; backgroundNoise: number };

export const WaveformRecorder: React.FC<WaveformRecorderProps> = ({ onCryDetected }) => {
  const [phase, setPhase] = useState<RecordingPhase>('idle');
  const [waveformData, setWaveformData] = useState<number[]>(Array(50).fill(0));
  const [currentCry, setCurrentCry] = useState<CryInstance | null>(null);
  const [detectedCries, setDetectedCries] = useState<CryInstance[]>([]);
  const [detectionState, setDetectionState] = useState<DetectionState>({ isDetecting: false, confidence: 0, backgroundNoise: 0 });
  const [realtimeCryAlert, setRealtimeCryAlert] = useState(false);
  const [recordingStartTime, setRecordingStartTime] = useState<number | null>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const cryDetectionEngineRef = useRef<CryDetectionEngine | null>(null);
  const audioAnalyzerRef = useRef<AudioAnalyzer>(new AudioAnalyzer());
  const recordedAudioRef = useRef<Blob | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);

  const handleCryDetected = async (features: AudioFeatures) => {
    console.log('Cry detected with features:', features);
    setPhase('cry-detected');
    
    // Start analysis phase after brief delay to show detection
    setTimeout(async () => {
      setPhase('analyzing');
      
      try {
        // Get current detection count from localStorage
        const savedState = typeof window !== 'undefined' ? 
          JSON.parse(localStorage.getItem('dababy_app_state') || '{}') : {};
        const detectionCount = (savedState.detectionResults?.length || 0) + 1;
        
        // Create a mock audio buffer for analysis (in real implementation, use actual audio)
        const mockAudioBuffer = createMockAudioBuffer(features);
        const result = await audioAnalyzerRef.current.analyzeAudio(mockAudioBuffer, detectionCount);
        
        // Generate diagnosis
        const diagnosis = CryDiagnosisEngine.generateDiagnosis(result);
        
        // Create cry instance
        const cry: CryInstance = {
          id: `cry-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          timestamp: new Date(),
          duration: features.duration,
          features: result.features,
          alerts: result.alerts,
          diagnosis,
          riskLevel: result.riskLevel,
          confidence: diagnosis.confidence
        };
        
        setCurrentCry(cry);
        setDetectedCries(prev => [...prev, cry]);
        
        // Store cry with audio data
        if (recordedAudioRef.current) {
          CryStorage.saveCryWithAudio(cry, recordedAudioRef.current);
        } else {
          CryStorage.saveCry(cry);
        }
        
        // Notify parent component
        onCryDetected?.(cry);
        
        // Show results
        setTimeout(() => {
          setPhase('complete');
        }, 2000);
        
      } catch (error) {
        console.error('Error analyzing cry:', error);
        setPhase('complete');
      }
    }, 1500);
  };

  // Create a mock audio buffer for demonstration (replace with real audio in production)
  const createMockAudioBuffer = (features: AudioFeatures): AudioBuffer => {
    const audioContext = new AudioContext();
    const duration = features.duration || 1.0;
    const sampleRate = 44100;
    const buffer = audioContext.createBuffer(1, sampleRate * duration, sampleRate);
    const channelData = buffer.getChannelData(0);
    
    // Generate a sine wave at the detected frequency
    const frequency = features.f0Mean || 400;
    for (let i = 0; i < channelData.length; i++) {
      channelData[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * 0.1;
    }
    
    return buffer;
  };

  const updateWaveform = () => {
    if (!analyserRef.current) {
      return;
    }
    
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    analyserRef.current.getByteFrequencyData(dataArray);
    
    // Process data into 50 bars for visualization
    const bars = 50;
    const usableRange = Math.floor(bufferLength * 0.8);
    
    // Simple cry detection based on right-side waveform activity
    if (phase === 'recording') {
      // Check if 10 seconds have passed since recording started
      const currentTime = Date.now();
      const timeSinceStart = recordingStartTime ? (currentTime - recordingStartTime) / 1000 : 0;
      const detectionEnabled = timeSinceStart >= 10;
      // Calculate frequency mapping for waveform bars
      // FFT size: 1024, Sample rate: 44100 Hz
      // Frequency per bin: 44100 / 1024 = ~43 Hz per bin
      // 50 bars use 80% of 512 frequency bins
      // Right side (bars 25-50) = higher frequencies (roughly 540-8600 Hz)
      
      // Check activity in right-side bars (higher frequencies)
      const rightSideStartBar = 25; // Middle point
      let rightSideActivity = 0;
      let rightSideCount = 0;
      
      // Calculate right-side activity from current waveform data
      for (let i = rightSideStartBar; i < bars; i++) {
        const barIndex = i;
        const startFreqBin = Math.floor((barIndex * usableRange) / bars);
        const endFreqBin = Math.floor(((barIndex + 1) * usableRange) / bars);
        
        let barSum = 0;
        let barCount = 0;
        for (let j = startFreqBin; j < endFreqBin && j < dataArray.length; j++) {
          barSum += dataArray[j];
          barCount++;
        }
        
        if (barCount > 0) {
          const barAverage = barSum / barCount;
          rightSideActivity += barAverage;
          rightSideCount++;
        }
      }
      
      // Simple crying detection rule (only after 10 seconds)
      const avgRightSideActivity = rightSideCount > 0 ? rightSideActivity / rightSideCount : 0;
      const activityThreshold = 30; // Adjust based on testing
      const isCrying = detectionEnabled && avgRightSideActivity > activityThreshold;
      
      // Update detection state
      const confidence = detectionEnabled ? Math.min(avgRightSideActivity / 100, 1.0) : 0;
      setDetectionState({
        isDetecting: isCrying,
        confidence: confidence,
        backgroundNoise: avgRightSideActivity / 255
      });
      
      // Show real-time cry alert (only after 10 seconds)
      setRealtimeCryAlert(detectionEnabled && isCrying && confidence > 0.4);
      
      // Trigger cry detection if threshold is met (only after 10 seconds)
      if (detectionEnabled && isCrying && confidence > 0.6) {
        // Create mock audio features for the callback
        const mockFeatures = {
          f0: [],
          f0Mean: 400 + (confidence * 400), // Scale with confidence
          f0Std: 50,
          hnr: 15,
          jitter: 0.02,
          shimmer: 0.05,
          rms: confidence * 0.3,
          spectralCentroid: 800 + (confidence * 1200),
          spectralFlatness: 0.2,
          voicedSegmentLengths: [],
          pauseLengths: [],
          burstLengths: [],
          repetitionRate: 0.5,
          nasalEnergyRatio: 0.1,
          lowMidHarmonics: 0.3,
          duration: 1.0,
          sampleRate: 44100
        };
        
        // Throttle detection to avoid spam
        const now = Date.now();
        if (!detectionState.isDetecting || (now - (window as any).lastCryDetection || 0) > 3000) {
          (window as any).lastCryDetection = now;
          handleCryDetected(mockFeatures);
        }
      }
    }
    
    // Continue with waveform visualization
    const samplesPerBar = Math.floor(usableRange / bars);
    const waveData: number[] = [];
    
    for (let i = 0; i < bars; i++) {
      let sum = 0;
      let count = 0;
      const startIndex = i * samplesPerBar;
      const endIndex = Math.min(startIndex + samplesPerBar, usableRange);
      
      for (let j = startIndex; j < endIndex; j++) {
        sum += dataArray[j];
        count++;
      }
      
      if (count > 0) {
        const average = sum / count;
        const normalized = average / 255;
        const scaled = Math.pow(normalized, 0.7) * 1.2;
        waveData.push(Math.min(scaled, 1));
      } else {
        waveData.push(0);
      }
    }
    
    setWaveformData(waveData);
    
    // Continue animation if recording or processing
    if (phase === 'recording' || phase === 'cry-detected' || phase === 'analyzing') {
      animationRef.current = requestAnimationFrame(updateWaveform);
    }
  };

  const getPhaseDisplay = () => {
    switch (phase) {
      case 'idle':
        return {
          title: 'Ready to Record',
          description: detectedCries.length > 0 
            ? `${detectedCries.length} cries detected this session`
            : 'Click start to begin cry detection',
          badge: { text: 'Idle', variant: 'secondary' as const },
          icon: <Mic className="w-5 h-5" />
        };
      case 'recording':
        const recordingDescription = realtimeCryAlert 
          ? `Potential cry detected! Confidence: ${Math.round(detectionState.confidence * 100)}%`
          : detectionState.isDetecting 
          ? 'Analyzing potential cry pattern...'
          : 'Monitoring audio and filtering for baby cry patterns';
        
        return {
          title: realtimeCryAlert ? '🚨 Potential Cry Detected!' : 'Listening for Crying',
          description: recordingDescription,
          badge: { 
            text: realtimeCryAlert ? 'Cry Alert!' : 'Recording', 
            variant: realtimeCryAlert ? 'warning' as const : 'info' as const 
          },
          icon: realtimeCryAlert 
            ? <AlertTriangle className="w-5 h-5 animate-bounce text-yellow-500" />
            : <Search className="w-5 h-5 animate-pulse" />
        };
      case 'cry-detected':
        return {
          title: 'Cry Detected!',
          description: 'Baby cry pattern identified and captured',
          badge: { text: 'Cry Found', variant: 'warning' as const },
          icon: <AlertTriangle className="w-5 h-5" />
        };
      case 'analyzing':
        return {
          title: 'Analyzing Cry Pattern',
          description: 'Processing acoustic features and generating diagnosis',
          badge: { text: 'Analyzing', variant: 'default' as const },
          icon: <Brain className="w-5 h-5 animate-pulse" />
        };
      case 'complete':
        return {
          title: 'Analysis Complete',
          description: currentCry 
            ? `Diagnosis: ${currentCry.diagnosis.primary}`
            : 'Cry analysis finished',
          badge: { text: 'Complete', variant: 'success' as const },
          icon: <CheckCircle className="w-5 h-5" />
        };
      default:
        return getPhaseDisplay();
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          sampleRate: 44100
        }
      });
      
      streamRef.current = stream;
      
      // Set up audio context
      const AudioContextClass = window.AudioContext || (window as typeof window & { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      audioContextRef.current = new AudioContextClass();
      
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      
      analyserRef.current.fftSize = 1024;
      analyserRef.current.smoothingTimeConstant = 0.3;
      analyserRef.current.minDecibels = -80;
      analyserRef.current.maxDecibels = -20;
      
      source.connect(analyserRef.current);
      
      // Initialize cry detection engine
      cryDetectionEngineRef.current = new CryDetectionEngine(handleCryDetected);
      
      // Set up media recorder for audio capture
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      recordingChunksRef.current = [];
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordingChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(recordingChunksRef.current, { type: 'audio/webm' });
        recordedAudioRef.current = audioBlob;
      };
      
      mediaRecorderRef.current.start(100);
      setPhase('recording');
      setCurrentCry(null);
      setRecordingStartTime(Date.now());
      
      // Start waveform animation
      animationRef.current = requestAnimationFrame(updateWaveform);
      
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Could not access microphone. Please check permissions and try again.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    // Stop animation
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    
    // Clean up
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    
    // Reset to idle after showing completion
    setTimeout(() => {
      setPhase('idle');
      setWaveformData(Array(50).fill(0));
      setDetectionState({ isDetecting: false, confidence: 0, backgroundNoise: 0 });
      setRealtimeCryAlert(false);
      setRecordingStartTime(null);
    }, 3000);
  };

  const handleStopRecording = () => {
    if (phase === 'recording') {
      setPhase('complete');
    }
    stopRecording();
  };

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  const phaseDisplay = getPhaseDisplay();

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6">
      <Card>
        <CardHeader className="text-center">
          <div className="flex items-center justify-center gap-3 mb-2">
            {phaseDisplay.icon}
            <CardTitle className="text-xl">{phaseDisplay.title}</CardTitle>
          </div>
          <Badge variant={phaseDisplay.badge.variant} className="mx-auto w-fit">
            {phaseDisplay.badge.text}
          </Badge>
        </CardHeader>
        
        <CardContent className="space-y-6">
          <p className="text-center text-muted-foreground">
            {phaseDisplay.description}
          </p>
          
          {/* Enhanced Waveform Visualization with Real-time Feedback */}
          <div className={`h-24 flex items-end justify-center gap-1 rounded-lg p-4 transition-all duration-300 ${
            realtimeCryAlert 
              ? 'bg-yellow-100 border-2 border-yellow-400 shadow-lg' 
              : detectionState.isDetecting
              ? 'bg-orange-50 border border-orange-200'
              : 'bg-muted/30'
          }`}>
            {waveformData.map((value, index) => (
              <WaveformBar
                key={index}
                value={value * 100}
                max={100}
                height="3px"
                isActive={phase !== 'idle'}
                className={`transition-all duration-100 ${
                  realtimeCryAlert
                    ? 'bg-yellow-500 animate-pulse'
                    : detectionState.isDetecting
                    ? 'bg-orange-400'
                    : phase === 'cry-detected' 
                    ? 'bg-yellow-500' 
                    : phase === 'analyzing' 
                    ? 'bg-blue-500' 
                    : phase === 'complete'
                    ? 'bg-green-500'
                    : ''
                }`}
              />
            ))}
          </div>
          
          {/* Real-time Detection Status */}
          {phase === 'recording' && (
            <div className="bg-muted/50 rounded-lg p-3 space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span>Detection Status:</span>
                <span className={`font-medium ${
                  realtimeCryAlert ? 'text-yellow-600' : 
                  detectionState.isDetecting ? 'text-orange-600' : 'text-green-600'
                }`}>
                  {realtimeCryAlert ? 'Cry Alert!' : 
                   detectionState.isDetecting ? 'Analyzing...' : 'Monitoring'}
                </span>
              </div>
              
              {detectionState.confidence > 0 && (
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span>Confidence:</span>
                    <span>{Math.round(detectionState.confidence * 100)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-300 ${
                        detectionState.confidence > 0.7 ? 'bg-yellow-500' :
                        detectionState.confidence > 0.4 ? 'bg-orange-400' : 'bg-blue-400'
                      }`}
                      style={{ width: `${detectionState.confidence * 100}%` }}
                    />
                  </div>
                </div>
              )}
              
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>Background Noise:</span>
                <span>{(detectionState.backgroundNoise * 100).toFixed(1)}%</span>
              </div>
            </div>
          )}
          
          {/* Current Cry Info */}
          {currentCry && phase === 'complete' && (
            <div className="bg-muted/50 rounded-lg p-4 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-medium">Latest Detection:</span>
                <Badge variant={
                  currentCry.riskLevel === 'critical' ? 'critical' :
                  currentCry.riskLevel === 'high' ? 'destructive' :
                  currentCry.riskLevel === 'medium' ? 'warning' : 'success'
                }>
                  {currentCry.riskLevel.toUpperCase()}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                <strong>Diagnosis:</strong> {currentCry.diagnosis.primary}
              </p>
              <p className="text-xs text-muted-foreground">
                Duration: {currentCry.duration.toFixed(1)}s • 
                Confidence: {(currentCry.confidence * 100).toFixed(0)}%
              </p>
            </div>
          )}
          
          {/* Control Button */}
          <div className="flex justify-center">
            {phase === 'idle' ? (
              <Button
                onClick={startRecording}
                size="lg"
                className="px-8"
              >
                <Mic className="w-5 h-5 mr-2" />
                Start Cry Detection
              </Button>
            ) : phase === 'complete' ? (
              <Button
                onClick={() => setPhase('idle')}
                size="lg"
                variant="outline"
                className="px-8"
              >
                <CheckCircle className="w-5 h-5 mr-2" />
                Ready for Next Recording
              </Button>
            ) : (
              <Button
                onClick={handleStopRecording}
                size="lg"
                variant="destructive"
                className="px-8"
              >
                <Square className="w-5 h-5 mr-2" />
                Stop Recording
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
