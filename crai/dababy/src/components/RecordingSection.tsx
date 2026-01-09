import { useState, useEffect, useRef } from 'react'
import { ContinuousMonitor } from '@/components/ContinuousMonitor'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Volume2, AlertCircle, Play, Pause, X, RotateCcw, Cloud } from 'lucide-react'
import type { DetectionResult } from '@/lib/audioAnalysis'

interface RecordingSectionProps {
  onDetectionResult: (result: DetectionResult) => void;
  isMonitoring: boolean;
  onMonitoringChange: (monitoring: boolean) => void;
  onDetectionStateChange?: (isDetecting: boolean) => void;
}

interface DetectionState {
  isDetecting: boolean;
  startTime: number | null;
  duration: number;
  reason: string;
  showReason: boolean;
  audioPlaying: boolean;
  audioLoaded: boolean;
  showAudioPlayer: boolean;
  wasMonitoringBeforeAudio: boolean;
  apiCallStatus: 'idle' | 'calling' | 'success' | 'error';
  apiResponse: any;
}

export function RecordingSection({ 
  onDetectionResult, 
  isMonitoring, 
  onMonitoringChange,
  onDetectionStateChange
}: RecordingSectionProps) {
  const [detectionState, setDetectionState] = useState<DetectionState>({
    isDetecting: false,
    startTime: null,
    duration: 0,
    reason: '',
    showReason: false,
    audioPlaying: false,
    audioLoaded: false,
    showAudioPlayer: false,
    wasMonitoringBeforeAudio: false,
    apiCallStatus: 'idle',
    apiResponse: null
  });
  
  const detectionTimerRef = useRef<NodeJS.Timeout | null>(null);
  const reasonTimerRef = useRef<NodeJS.Timeout | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const detectionHistoryRef = useRef<boolean[]>([]);

  // Initialize audio element with event handlers
  useEffect(() => {
    audioRef.current = new Audio('/hunger-sound.mp3');
    audioRef.current.preload = 'auto';
    
    const audio = audioRef.current;
    
    // Audio event handlers
    const handleLoadedData = () => {
      setDetectionState(prev => ({ ...prev, audioLoaded: true }));
    };
    
    const handlePlay = () => {
      setDetectionState(prev => ({ 
        ...prev, 
        audioPlaying: true,
        wasMonitoringBeforeAudio: isMonitoring
      }));
      // Pause monitoring when audio starts playing
      if (isMonitoring) {
        onMonitoringChange(false);
      }
    };
    
    const handlePause = () => {
      setDetectionState(prev => ({ ...prev, audioPlaying: false }));
    };
    
    const handleEnded = () => {
      const wasMonitoring = detectionState.wasMonitoringBeforeAudio;
      setDetectionState(prev => ({ 
        ...prev, 
        audioPlaying: false,
        showAudioPlayer: false,
        showReason: false,
        reason: '',
        wasMonitoringBeforeAudio: false,
        apiCallStatus: 'idle',
        apiResponse: null
      }));
      // Resume monitoring if it was active before audio
      if (wasMonitoring) {
        onMonitoringChange(true);
      }
      // Reset detection history
      detectionHistoryRef.current = [];
    };
    
    const handleError = () => {
      console.warn('Error loading hunger sound file');
      setDetectionState(prev => ({ ...prev, audioLoaded: false }));
    };
    
    audio.addEventListener('loadeddata', handleLoadedData);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('error', handleError);
    
    return () => {
      audio.removeEventListener('loadeddata', handleLoadedData);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('error', handleError);
      
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, []);

  const playHungerSound = () => {
    if (audioRef.current && detectionState.audioLoaded) {
      audioRef.current.currentTime = 0;
      audioRef.current.play().catch(error => {
        console.warn('Could not play hunger sound:', error);
      });
      setDetectionState(prev => ({ 
        ...prev, 
        showAudioPlayer: true,
        wasMonitoringBeforeAudio: isMonitoring
      }));
    }
  };
  
  const toggleAudioPlayback = () => {
    if (audioRef.current) {
      if (detectionState.audioPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play().catch(error => {
          console.warn('Could not play hunger sound:', error);
        });
      }
    }
  };
  
  const closeAudioPlayer = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    const wasMonitoring = detectionState.wasMonitoringBeforeAudio;
    setDetectionState(prev => ({ 
      ...prev, 
      showAudioPlayer: false,
      showReason: false,
      reason: '',
      audioPlaying: false,
      wasMonitoringBeforeAudio: false
    }));
    // Resume monitoring if it was active before audio
    if (wasMonitoring) {
      onMonitoringChange(true);
    }
    // Reset detection history
    detectionHistoryRef.current = [];
  };
  
  const restartAudio = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
      if (!detectionState.audioPlaying) {
        audioRef.current.play().catch(error => {
          console.warn('Could not play hunger sound:', error);
        });
      }
    }
  };

  const sendModalAPIRequest = async () => {
    setDetectionState(prev => ({ ...prev, apiCallStatus: 'calling' }));
    
    try {
      // Send dummy request to Modal API
      const response = await fetch('https://aryagm01--baby-cry-web-api-fastapi-app.modal.run/detect-cry?audio_path=dummy_hunger_detection.wav', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        setDetectionState(prev => ({ 
          ...prev, 
          apiCallStatus: 'success',
          apiResponse: data
        }));
        console.log('Modal API Response:', data);
      } else {
        throw new Error(`API responded with status: ${response.status}`);
      }
    } catch (error) {
      console.error('Error calling Modal API:', error);
      setDetectionState(prev => ({ 
        ...prev, 
        apiCallStatus: 'error',
        apiResponse: { error: error instanceof Error ? error.message : 'Unknown error' }
      }));
    }
  };

  const sendModalClassifyRequest = async () => {
    try {
      // Send classification request
      const response = await fetch('https://aryagm01--baby-cry-web-api-fastapi-app.modal.run/classify-cry', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          audio_path: 'hunger_cry_sample.wav',
          intensity_threshold: 0.1
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Modal Classification Response:', data);
      }
    } catch (error) {
      console.error('Error calling Modal Classification API:', error);
    }
  };

  const handleDetectionChange = (isDetecting: boolean) => {
    const now = Date.now();
    
    // Add to detection history for 80% calculation
    detectionHistoryRef.current.push(isDetecting);
    
    // Keep only last 20 seconds of history (assuming 60fps)
    const maxHistory = 60 * 4; // 20 seconds at 60fps
    if (detectionHistoryRef.current.length > maxHistory) {
      detectionHistoryRef.current.shift();
    }
    
    // Calculate detection percentage over last 20 seconds
    const recentHistory = detectionHistoryRef.current.slice(-maxHistory);
    const detectionCount = recentHistory.filter(Boolean).length;
    const detectionPercentage = recentHistory.length > 0 ? detectionCount / recentHistory.length : 0;
    
    // Check if we have 20 seconds of data and 80% detection rate
    const hasEnoughData = recentHistory.length >= maxHistory * 0.9; // 90% of expected frames
    const meetsThreshold = detectionPercentage >= 0.8; // 80% detection rate
    
    if (hasEnoughData && meetsThreshold && !detectionState.showReason) {
      // Trigger hunger detection
      setDetectionState(prev => ({
        ...prev,
        reason: 'Detecting reason...',
        showReason: true
      }));
      
      // After 2 seconds, show "Hunger" and wait for API response before playing sound
      reasonTimerRef.current = setTimeout(async () => {
        setDetectionState(prev => ({
          ...prev,
          reason: 'Hunger'
        }));
        
        // Send API requests to Modal endpoint first
        await sendModalAPIRequest();
        await sendModalClassifyRequest();
        
        // Only play audio after API calls complete
        playHungerSound();
        
        // Don't auto-hide when audio player is shown - let user control it
        // The audio player will handle cleanup when closed or audio ends
      }, 2000);
    }
    
    // Update detection state for display
    setDetectionState(prev => ({
      ...prev,
      isDetecting,
      startTime: isDetecting && !prev.isDetecting ? now : prev.startTime,
      duration: isDetecting && prev.startTime ? (now - prev.startTime) / 1000 : 0
    }));
    
    // Call original callback
    onDetectionStateChange?.(isDetecting);
  };

  // Cleanup timers
  useEffect(() => {
    return () => {
      if (detectionTimerRef.current) {
        clearTimeout(detectionTimerRef.current);
      }
      if (reasonTimerRef.current) {
        clearTimeout(reasonTimerRef.current);
      }
    };
  }, []);

  return (
    <div className="w-full space-y-4">
      {/* Recording Component */}
      <div className="flex justify-center">
        <ContinuousMonitor 
          onDetectionResult={onDetectionResult}
          isMonitoring={isMonitoring}
          onMonitoringChange={onMonitoringChange}
          onDetectionStateChange={handleDetectionChange}
        />
      </div>
      
      {/* Detection Status Display */}
      {(detectionState.isDetecting || detectionState.showReason) && (
        <Card className="mx-auto max-w-md">
          <CardContent className="pt-6">
            <div className="text-center space-y-3">
              {detectionState.showReason ? (
                <div className="space-y-2">
                  <div className="flex items-center justify-center gap-2">
                    <AlertCircle className="w-5 h-5 text-orange-500" />
                    <Badge variant="outline" className="px-3 py-1">
                      {detectionState.reason}
                    </Badge>
                  </div>
                  {detectionState.reason === 'Hunger' && detectionState.showAudioPlayer && (
                    <div className="space-y-3">
                      <div className="flex items-center justify-center gap-2 text-sm text-gray-600">
                        <Volume2 className="w-4 h-4" />
                        <span>Hunger Sound</span>
                      </div>
                      
                      {/* Audio Player Controls */}
                      <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                        <div className="flex items-center justify-center gap-3">
                          <button
                            onClick={toggleAudioPlayback}
                            className="flex items-center justify-center w-10 h-10 bg-blue-500 hover:bg-blue-600 text-white rounded-full transition-colors"
                            disabled={!detectionState.audioLoaded}
                          >
                            {detectionState.audioPlaying ? (
                              <Pause className="w-5 h-5" />
                            ) : (
                              <Play className="w-5 h-5 ml-0.5" />
                            )}
                          </button>
                          
                          <button
                            onClick={restartAudio}
                            className="flex items-center justify-center w-8 h-8 bg-gray-500 hover:bg-gray-600 text-white rounded-full transition-colors"
                            disabled={!detectionState.audioLoaded}
                          >
                            <RotateCcw className="w-4 h-4" />
                          </button>
                          
                          <button
                            onClick={closeAudioPlayer}
                            className="flex items-center justify-center w-8 h-8 bg-red-500 hover:bg-red-600 text-white rounded-full transition-colors"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                        
                        <div className="text-center">
                          <p className="text-xs text-gray-500">
                            {detectionState.audioPlaying ? 'Playing... (Recording paused)' : 
                             detectionState.audioLoaded ? 'Ready to play' : 'Loading...'}
                          </p>
                          {detectionState.audioPlaying && (
                            <p className="text-xs text-blue-600 mt-1">
                              Recording will resume when audio ends
                            </p>
                          )}
                        </div>
                        
                        {/* API Status Display */}
                        <div className="border-t pt-3">
                          <div className="flex items-center justify-center gap-2 text-xs">
                            <Cloud className="w-3 h-3" />
                            <span className="text-gray-600">Modal API:</span>
                            <Badge 
                              variant={detectionState.apiCallStatus === 'success' ? 'default' : 
                                      detectionState.apiCallStatus === 'error' ? 'destructive' : 
                                      detectionState.apiCallStatus === 'calling' ? 'secondary' : 'outline'}
                              className="text-xs px-2 py-0"
                            >
                              {detectionState.apiCallStatus === 'calling' && '🔄 Calling...'}
                              {detectionState.apiCallStatus === 'success' && '✅ Success'}
                              {detectionState.apiCallStatus === 'error' && '❌ Error'}
                              {detectionState.apiCallStatus === 'idle' && '⏸️ Idle'}
                            </Badge>
                          </div>
                          {detectionState.apiResponse && detectionState.apiCallStatus === 'success' && (
                            <p className="text-xs text-green-600 mt-1 text-center">
                              Confidence: {(detectionState.apiResponse.confidence * 100).toFixed(1)}%
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-2">
                  <Badge variant="destructive" className="px-3 py-1">
                    🔴 Crying Detected
                  </Badge>
                  <p className="text-sm text-gray-600">
                    Duration: {detectionState.duration.toFixed(1)}s
                  </p>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-red-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${Math.min((detectionState.duration / 4) * 100, 100)}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-500">
                    Analyzing pattern... {Math.min(detectionState.duration, 4).toFixed(0)}/4s
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}