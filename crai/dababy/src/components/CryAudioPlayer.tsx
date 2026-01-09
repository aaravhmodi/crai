import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Play, Pause, Volume2, Download, Loader2 } from 'lucide-react';
import { CryStorage } from '@/lib/cryDetection';

interface CryAudioPlayerProps {
  cryId: string;
  className?: string;
}

export const CryAudioPlayer: React.FC<CryAudioPlayerProps> = ({ cryId, className = '' }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    const loadAudio = async () => {
      setIsLoading(true);
      try {
        const audioBlob = await CryStorage.getAudioForCry(cryId);
        if (audioBlob) {
          const url = URL.createObjectURL(audioBlob);
          setAudioUrl(url);
        }
      } catch (error) {
        console.error('Failed to load audio:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadAudio();
    
    return () => {
      // Cleanup audio URL to prevent memory leaks
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [cryId, audioUrl]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateTime = () => setCurrentTime(audio.currentTime);
    const updateDuration = () => setDuration(audio.duration);
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', updateTime);
    audio.addEventListener('loadedmetadata', updateDuration);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', updateTime);
      audio.removeEventListener('loadedmetadata', updateDuration);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [audioUrl]);

  const togglePlayback = () => {
    const audio = audioRef.current;
    if (!audio || !audioUrl) return;

    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      audio.play();
      setIsPlaying(true);
    }
  };

  const handleDownload = async () => {
    if (!audioUrl) return;
    
    try {
      const audioBlob = await CryStorage.getAudioForCry(cryId);
      if (audioBlob) {
        const url = URL.createObjectURL(audioBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `cry-${cryId}.webm`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Failed to download audio:', error);
    }
  };

  const formatTime = (seconds: number): string => {
    if (isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const audio = audioRef.current;
    if (!audio || !duration) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    const newTime = percent * duration;
    audio.currentTime = newTime;
    setCurrentTime(newTime);
  };

  if (isLoading) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-sm text-muted-foreground">Loading audio...</span>
      </div>
    );
  }

  if (!audioUrl) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <Volume2 className="w-4 h-4 text-muted-foreground" />
        <span className="text-sm text-muted-foreground">No audio available</span>
      </div>
    );
  }

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <audio ref={audioRef} src={audioUrl} preload="metadata" />
      
      {/* Play/Pause Button */}
      <Button
        variant="outline"
        size="sm"
        onClick={togglePlayback}
        className="h-8 w-8 p-0"
      >
        {isPlaying ? (
          <Pause className="w-3 h-3" />
        ) : (
          <Play className="w-3 h-3" />
        )}
      </Button>

      {/* Progress Bar */}
      <div className="flex-1 min-w-0">
        <div 
          className="h-2 bg-muted rounded-full cursor-pointer group relative"
          onClick={handleSeek}
        >
          <div 
            className="h-full bg-primary rounded-full transition-all"
            style={{ width: `${duration ? (currentTime / duration) * 100 : 0}%` }}
          />
          <div 
            className="absolute top-1/2 transform -translate-y-1/2 w-3 h-3 bg-primary rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
            style={{ left: `${duration ? (currentTime / duration) * 100 : 0}%`, marginLeft: '-6px' }}
          />
        </div>
        <div className="flex justify-between text-xs text-muted-foreground mt-1">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {/* Download Button */}
      <Button
        variant="outline"
        size="sm"
        onClick={handleDownload}
        className="h-8 w-8 p-0"
        title="Download audio"
      >
        <Download className="w-3 h-3" />
      </Button>
    </div>
  );
};