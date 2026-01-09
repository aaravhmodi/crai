import { RiskAssessmentTable } from '@/components/RiskAssessmentTable'
import { FeatureDisplay } from '@/components/FeatureDisplay'
import { CryAnalytics } from '@/components/CryAnalytics'
import { Button } from '@/components/ui/button'
import { BarChart3, AlertTriangle, Cpu } from 'lucide-react'
import { useState } from 'react'
import type { DetectionResult, AudioFeatures } from '@/lib/audioAnalysis'

interface ResultsSectionProps {
  detectionResults: DetectionResult[];
  isMonitoring: boolean;
  latestFeatures: AudioFeatures | null;
}

export function ResultsSection({ detectionResults, isMonitoring, latestFeatures }: ResultsSectionProps) {
  const [resultsView, setResultsView] = useState<'dashboard' | 'features' | 'analytics'>('dashboard');

  return (
    <div className="w-full space-y-6">
      {/* Header Section */}
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center gap-3">
          <BarChart3 className="w-6 h-6 text-primary" />
          <h2 className="text-2xl font-semibold tracking-tight">Analysis & Results</h2>
        </div>
        <p className="text-muted-foreground max-w-md mx-auto">
          View detection results, analytics, and feature analysis
        </p>
      </div>

      {/* Navigation Tabs */}
      <div className="flex justify-center">
        <div className="inline-flex p-1 bg-muted rounded-lg">
          <Button
            variant={resultsView === 'dashboard' ? 'default' : 'ghost'}
            onClick={() => setResultsView('dashboard')}
            className="flex items-center gap-2 px-4 py-2"
            size="sm"
          >
            <AlertTriangle className="w-4 h-4" />
            Dashboard
          </Button>
          <Button
            variant={resultsView === 'features' ? 'default' : 'ghost'}
            onClick={() => setResultsView('features')}
            className="flex items-center gap-2 px-4 py-2"
            size="sm"
          >
            <Cpu className="w-4 h-4" />
            Features
          </Button>
          <Button
            variant={resultsView === 'analytics' ? 'default' : 'ghost'}
            onClick={() => setResultsView('analytics')}
            className="flex items-center gap-2 px-4 py-2"
            size="sm"
          >
            <BarChart3 className="w-4 h-4" />
            Analytics
          </Button>
        </div>
      </div>

      {/* Content Area */}
      <div className="min-h-[500px]">
        {resultsView === 'dashboard' && (
          <RiskAssessmentTable 
            detectionResults={detectionResults}
            isMonitoring={isMonitoring}
          />
        )}

        {resultsView === 'features' && (
          <FeatureDisplay features={latestFeatures} />
        )}

        {resultsView === 'analytics' && (
          <CryAnalytics />
        )}
      </div>
    </div>
  );
}