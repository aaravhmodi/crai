import React, { useRef, useEffect, useState } from 'react';

interface AudioVisualizerProps {
  analyser: AnalyserNode | null;
  isActive: boolean;
  onDetectionStateChange?: (isDetecting: boolean) => void;
}

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({
  analyser,
  isActive,
  onDetectionStateChange
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | undefined>();
  const [isCurrentlyDetecting, setIsCurrentlyDetecting] = useState<boolean>(false);
  const lastDetectionRef = useRef<number>(0);
  const detectionHistoryRef = useRef<boolean[]>([]);
  
  // === SIMPLE CONFIGURATION PARAMETERS ===
  const FREQUENCY_THRESHOLD = 45;     // Volume level (0-255) to detect crying
  const FREQUENCY_START_BAR = 75;     // Which frequency bar to start detection (0-100)
  const DETECTION_PERCENTAGE = 0.95;   // 80% of recent frames must be positive
  const HISTORY_SECONDS = 5;          // How many seconds of history to check

  useEffect(() => {
    if (!analyser || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      if (!isActive) {
        animationRef.current = requestAnimationFrame(draw);
        return;
      }

      analyser.getByteFrequencyData(dataArray);

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // === SIMPLIFIED DETECTION LOGIC ===
      const bars = 100;
      const usableRange = Math.floor(bufferLength * 0.5);
      
      // Calculate average activity in high frequency range
      let highFreqActivity = 0;
      let barCount = 0;
      
      for (let i = FREQUENCY_START_BAR; i < bars; i++) {
        const startFreqBin = Math.floor((i * usableRange) / bars);
        const endFreqBin = Math.floor(((i + 1) * usableRange) / bars);
        
        let barSum = 0;
        let sampleCount = 0;
        for (let j = startFreqBin; j < endFreqBin && j < dataArray.length; j++) {
          barSum += dataArray[j];
          sampleCount++;
        }
        
        if (sampleCount > 0) {
          highFreqActivity += barSum / sampleCount;
          barCount++;
        }
      }
      
      // Simple detection: is average high frequency activity above threshold?
      const avgActivity = barCount > 0 ? highFreqActivity / barCount : 0;
      const currentlyDetecting = avgActivity > FREQUENCY_THRESHOLD;

      // === SUSTAINED DETECTION LOGIC ===
      const now = Date.now();
      
      // Add current detection to history
      detectionHistoryRef.current.push(currentlyDetecting);
      
      // Keep only recent history (60fps * HISTORY_SECONDS)
      const maxHistory = 60 * HISTORY_SECONDS;
      if (detectionHistoryRef.current.length > maxHistory) {
        detectionHistoryRef.current.shift();
      }
      
      // Calculate what percentage of recent frames detected crying
      const recentFrames = detectionHistoryRef.current;
      const positiveFrames = recentFrames.filter(Boolean).length;
      const detectionRate = recentFrames.length > 0 ? positiveFrames / recentFrames.length : 0;
      
      // Final decision: enough history AND high enough detection rate
      const hasEnoughData = recentFrames.length >= (60 * HISTORY_SECONDS * 0.8); // 80% of expected frames
      const finalDetection = hasEnoughData && detectionRate >= DETECTION_PERCENTAGE;
      
      // Update state
      if (finalDetection !== isCurrentlyDetecting || now - lastDetectionRef.current > 2000) {
        setIsCurrentlyDetecting(finalDetection);
        onDetectionStateChange?.(finalDetection);
        lastDetectionRef.current = now;
      }

      // Enhanced canvas visualization
      const barWidth = canvas.width / bars;
      
      // Draw 50 bars matching the detection algorithm
      for (let i = 0; i < bars; i++) {
        const startFreqBin = Math.floor((i * usableRange) / bars);
        const endFreqBin = Math.floor(((i + 1) * usableRange) / bars);
        
        let barSum = 0;
        let barCount = 0;
        for (let j = startFreqBin; j < endFreqBin && j < dataArray.length; j++) {
          barSum += dataArray[j];
          barCount++;
        }
        
        const barAverage = barCount > 0 ? barSum / barCount : 0;
        const barHeight = (barAverage / 255) * canvas.height;
        const x = i * barWidth;
        
        // Color bars based on detection zones
        let barColor;
        if (i >= FREQUENCY_START_BAR) {
          // Detection zone
          barColor = finalDetection ? '#ef4444' : '#f59e0b'; // Red when detecting, orange otherwise
        } else {
          // Lower frequencies
          barColor = '#3b82f6'; // Blue
        }
        
        ctx.fillStyle = barColor;
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
      }

      // Add detection overlay and indicators
      if (currentlyDetecting) {
        // Red overlay
        ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Detection text
        ctx.fillStyle = '#ef4444';
        ctx.font = 'bold 18px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('CRYING DETECTED', canvas.width / 2, 30);
        
        // Simple status display
        ctx.font = '12px sans-serif';
        ctx.fillText(`Activity: ${avgActivity.toFixed(1)} | Threshold: ${FREQUENCY_THRESHOLD}`, canvas.width / 2, 50);
        ctx.fillText(`Detection Rate: ${(detectionRate * 100).toFixed(1)}% | Need: ${(DETECTION_PERCENTAGE * 100)}%`, canvas.width / 2, 65);
      }
      
      // Draw frequency zone labels
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Low Freq', 5, canvas.height - 5);
      ctx.textAlign = 'right';
      ctx.fillText('High Freq (Detection Zone)', canvas.width - 5, canvas.height - 5);

      animationRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [analyser, isActive, isCurrentlyDetecting, onDetectionStateChange]);

  return (
    <div className="w-full">
      <canvas
        ref={canvasRef}
        width={800}
        height={200}
        className="w-full h-48 bg-gray-900 rounded-lg border"
        style={{ maxHeight: '200px' }}
      />
      <div className="mt-2 text-center text-sm text-gray-600">
        {isActive ? (
          <span className="text-green-600">● Live Audio Visualization</span>
        ) : (
          <span className="text-gray-400">○ Inactive</span>
        )}
        {isCurrentlyDetecting && (
          <span className="ml-4 text-red-600 font-semibold">🔴 CRYING DETECTED</span>
        )}
      </div>
    </div>
  );
};
