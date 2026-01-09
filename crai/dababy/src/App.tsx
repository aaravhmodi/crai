import './App.css'
import { useState, useEffect } from 'react'
import { RecordingSection } from '@/components/RecordingSection'
import { ResultsSection } from '@/components/ResultsSection'
import { AIChatbot } from '@/components/AIChatbot'
import { ResultsExplanation } from '@/components/ResultsExplanation'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Trash2, Download, Upload } from 'lucide-react'
import type { DetectionResult } from '@/lib/audioAnalysis'
import type { DiagnosisContext } from '@/lib/medicalDiagnosis'
import { AppStorage } from '@/lib/appStorage'

function App() {
  const [detectionResults, setDetectionResults] = useState<DetectionResult[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [isCurrentlyDetecting, setIsCurrentlyDetecting] = useState(false);
  const [storageInfo, setStorageInfo] = useState(AppStorage.getStorageInfo());
  const [activeSection, setActiveSection] = useState<'recording' | 'results' | 'ai'>('recording');
  const [diagnosisContext, setDiagnosisContext] = useState<DiagnosisContext>({
    babyAge: undefined,
    isJaundiced: false,
    postFeed: false,
    hasFever: false,
    gestationalAge: undefined,
    feedingDifficulty: false,
    constipation: false
  });

  // Load saved state on component mount
  useEffect(() => {
    const savedState = AppStorage.loadAppState();
    setDetectionResults(savedState.detectionResults);
    setDiagnosisContext(savedState.diagnosisContext);
    setActiveSection(savedState.activeSection);
    setStorageInfo(AppStorage.getStorageInfo());
  }, []);

  // Update storage info when detection results change
  useEffect(() => {
    setStorageInfo(AppStorage.getStorageInfo());
  }, [detectionResults, diagnosisContext]);

  const handleDetectionResult = (result: DetectionResult) => {
    const updatedResults = AppStorage.addDetectionResult(result);
    setDetectionResults(updatedResults);
  };

  const handleClearAllData = () => {
    if (window.confirm('Are you sure you want to clear all stored data? This action cannot be undone.')) {
      AppStorage.clearAllData();
      setDetectionResults([]);
      setDiagnosisContext({
        babyAge: undefined,
        isJaundiced: false,
        postFeed: false,
        hasFever: false,
        gestationalAge: undefined,
        feedingDifficulty: false,
        constipation: false
      });
      setStorageInfo(AppStorage.getStorageInfo());
      // Reload the page to reset chat messages
      window.location.reload();
    }
  };

  const handleExportData = () => {
    const data = AppStorage.exportAppState();
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `crai-backup-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleImportData = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const jsonData = e.target?.result as string;
        if (AppStorage.importAppState(jsonData)) {
          alert('Data imported successfully! The page will reload to apply changes.');
          window.location.reload();
        } else {
          alert('Failed to import data. Please check the file format.');
        }
      } catch (error) {
        alert('Failed to import data. Invalid file format.');
      }
    };
    reader.readAsText(file);
    event.target.value = ''; // Reset input
  };

  const latestFeatures = detectionResults.length > 0 
    ? detectionResults[detectionResults.length - 1].features 
    : null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20 fade-in">
      {/* Minimal Header */}
      <div className="border-b bg-background/80 backdrop-blur-sm shadow-sm">
        <div className="container mx-auto px-6 py-12 max-w-5xl">
          <div className="text-center space-y-4">
            <h1 className="text-8xl font-bold tracking-tight drop-shadow-sm">crai.</h1>
            <p className="text-2xl text-muted-foreground font-medium">
              AI-powered infant cry analysis
            </p>
            
            {/* Storage Info */}
            <div className="flex justify-center items-center gap-4 pt-4">
              <Badge variant="outline" className="text-xs">
                {detectionResults.length} recordings stored
              </Badge>
              <Badge variant={storageInfo.percentage > 80 ? 'destructive' : 'secondary'} className="text-xs">
                {storageInfo.percentage.toFixed(1)}% storage used
              </Badge>
              
              {/* Data Management Buttons */}
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleExportData}
                  className="text-xs"
                >
                  <Download className="w-3 h-3 mr-1" />
                  Export
                </Button>
                
                <label className="cursor-pointer">
                  <Button
                    variant="outline"
                    size="sm"
                    className="text-xs"
                    asChild
                  >
                    <span>
                      <Upload className="w-3 h-3 mr-1" />
                      Import
                    </span>
                  </Button>
                  <input
                    type="file"
                    accept=".json"
                    onChange={handleImportData}
                    className="hidden"
                  />
                </label>
                
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={handleClearAllData}
                  className="text-xs"
                >
                  <Trash2 className="w-3 h-3 mr-1" />
                  Clear All
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-16 max-w-5xl">
        {/* Simple Navigation */}
        <div className="flex justify-center mb-16">
          <div className="flex bg-card/80 backdrop-blur-sm border rounded-xl p-1.5 shadow-lg">
            <Button
              variant={activeSection === 'recording' ? 'default' : 'ghost'}
              onClick={() => {
                setActiveSection('recording');
                AppStorage.saveActiveSection('recording');
              }}
              className="px-10 py-4 text-lg font-medium rounded-lg"
            >
              Record
            </Button>
            <Button
              variant={activeSection === 'results' ? 'default' : 'ghost'}
              onClick={() => {
                setActiveSection('results');
                AppStorage.saveActiveSection('results');
              }}
              className="px-10 py-4 text-lg font-medium rounded-lg"
            >
              Results
            </Button>
            <Button
              variant={activeSection === 'ai' ? 'default' : 'ghost'}
              onClick={() => {
                setActiveSection('ai');
                AppStorage.saveActiveSection('ai');
              }}
              className="px-10 py-4 text-lg font-medium rounded-lg"
            >
              AI Chat
            </Button>
          </div>
        </div>

        {/* Results Overview */}
        {detectionResults.length > 0 && activeSection !== 'ai' && (
          <div className="mb-12">
            <ResultsExplanation 
              latestResult={detectionResults[detectionResults.length - 1]}
              totalDetections={detectionResults.length}
              isCurrentlyDetecting={isCurrentlyDetecting}
              isRecording={isMonitoring}
            />
          </div>
        )}

        {/* Main Tool */}
        <div className="w-full">
          {activeSection === 'recording' && (
            <RecordingSection 
              onDetectionResult={handleDetectionResult}
              isMonitoring={isMonitoring}
              onMonitoringChange={setIsMonitoring}
              onDetectionStateChange={setIsCurrentlyDetecting}
            />
          )}

          {activeSection === 'results' && (
            <ResultsSection 
              detectionResults={detectionResults}
              isMonitoring={isMonitoring}
              latestFeatures={latestFeatures}
            />
          )}

          {activeSection === 'ai' && (
            <div className="space-y-6">
              {/* Diagnosis Context Form */}
              <div className="bg-card/50 backdrop-blur-sm border rounded-xl p-6">
                <h3 className="text-lg font-semibold mb-4">Baby Information (Optional)</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Age (weeks)</label>
                    <input
                      type="number"
                      value={diagnosisContext.babyAge || ''}
                      onChange={(e) => {
                        const newContext = {
                          ...diagnosisContext,
                          babyAge: e.target.value ? parseInt(e.target.value) : undefined
                        };
                        setDiagnosisContext(newContext);
                        AppStorage.saveDiagnosisContext(newContext);
                      }}
                      className="w-full px-3 py-2 border rounded-lg text-sm"
                      placeholder="e.g., 8"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Gestational Age</label>
                    <input
                      type="number"
                      value={diagnosisContext.gestationalAge || ''}
                      onChange={(e) => {
                        const newContext = {
                          ...diagnosisContext,
                          gestationalAge: e.target.value ? parseInt(e.target.value) : undefined
                        };
                        setDiagnosisContext(newContext);
                        AppStorage.saveDiagnosisContext(newContext);
                      }}
                      className="w-full px-3 py-2 border rounded-lg text-sm"
                      placeholder="e.g., 37"
                    />
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="jaundiced"
                      checked={diagnosisContext.isJaundiced}
                      onChange={(e) => {
                        const newContext = {
                          ...diagnosisContext,
                          isJaundiced: e.target.checked
                        };
                        setDiagnosisContext(newContext);
                        AppStorage.saveDiagnosisContext(newContext);
                      }}
                      className="rounded"
                    />
                    <label htmlFor="jaundiced" className="text-sm font-medium">Jaundiced</label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="postFeed"
                      checked={diagnosisContext.postFeed}
                      onChange={(e) => {
                        const newContext = {
                          ...diagnosisContext,
                          postFeed: e.target.checked
                        };
                        setDiagnosisContext(newContext);
                        AppStorage.saveDiagnosisContext(newContext);
                      }}
                      className="rounded"
                    />
                    <label htmlFor="postFeed" className="text-sm font-medium">Post-Feed</label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="fever"
                      checked={diagnosisContext.hasFever}
                      onChange={(e) => {
                        const newContext = {
                          ...diagnosisContext,
                          hasFever: e.target.checked
                        };
                        setDiagnosisContext(newContext);
                        AppStorage.saveDiagnosisContext(newContext);
                      }}
                      className="rounded"
                    />
                    <label htmlFor="fever" className="text-sm font-medium">Has Fever</label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="feedingDifficulty"
                      checked={diagnosisContext.feedingDifficulty}
                      onChange={(e) => {
                        const newContext = {
                          ...diagnosisContext,
                          feedingDifficulty: e.target.checked
                        };
                        setDiagnosisContext(newContext);
                        AppStorage.saveDiagnosisContext(newContext);
                      }}
                      className="rounded"
                    />
                    <label htmlFor="feedingDifficulty" className="text-sm font-medium">Feeding Issues</label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="constipation"
                      checked={diagnosisContext.constipation}
                      onChange={(e) => {
                        const newContext = {
                          ...diagnosisContext,
                          constipation: e.target.checked
                        };
                        setDiagnosisContext(newContext);
                        AppStorage.saveDiagnosisContext(newContext);
                      }}
                      className="rounded"
                    />
                    <label htmlFor="constipation" className="text-sm font-medium">Constipation</label>
                  </div>
                </div>
              </div>
              
              <AIChatbot 
                detectionResults={detectionResults}
                diagnosisContext={diagnosisContext}
              />
            </div>
          )}
        </div>
      </div>

      {/* Subtle background texture */}
      <div className="fixed inset-0 pointer-events-none opacity-5">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/10 via-transparent to-primary/10"></div>
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl"></div>
      </div>
    </div>
  )
}

export default App
