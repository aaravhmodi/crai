import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { CheckCircle, Info, TrendingUp, Clock } from 'lucide-react';
import type { DetectionResult } from '@/lib/audioAnalysis';

interface ResultsExplanationProps {
  latestResult: DetectionResult | null;
  totalDetections: number;
  isCurrentlyDetecting?: boolean;
  isRecording?: boolean;
}

export function ResultsExplanation({ latestResult, totalDetections, isCurrentlyDetecting, isRecording }: ResultsExplanationProps) {
  const getExplanation = () => {
    if (!isRecording && !latestResult) {
      return {
        title: "No Crying Detected",
        description: "Start recording to monitor for baby crying.",
        icon: <Info className="w-5 h-5" />,
        variant: "secondary" as const,
        insights: []
      };
    }

    if (isRecording && !isCurrentlyDetecting) {
      return {
        title: "Monitoring for Crying",
        description: "Listening for baby cry patterns...",
        icon: <Info className="w-5 h-5" />,
        variant: "secondary" as const,
        insights: []
      };
    }

    if (isCurrentlyDetecting) {
      return {
        title: "Crying Detected",
        description: "Baby cry pattern currently being detected.",
        icon: <CheckCircle className="w-5 h-5" />,
        variant: "success" as const,
        insights: []
      };
    }

    // Has previous results but not currently detecting
    return {
      title: "No Current Crying",
      description: "Not currently detecting crying patterns.",
      icon: <Info className="w-5 h-5" />,
      variant: "secondary" as const,
      insights: []
    };
  };

  const explanation = getExplanation();

  return (
    <Card className="w-full">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-3">
          {explanation.icon}
          <span className="text-lg font-semibold">Detection Status</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0 space-y-6">
        {/* Current Status */}
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <h3 className="font-semibold text-base mb-2">{explanation.title}</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">{explanation.description}</p>
          </div>
          <Badge variant={explanation.variant} className="px-3 py-1 text-xs font-medium">
            {isCurrentlyDetecting ? 'DETECTING' : 'NONE'}
          </Badge>
        </div>

        {/* Session Statistics */}
        <div className="grid grid-cols-2 gap-6 p-4 bg-muted/20 rounded-lg">
          <div className="text-center space-y-2">
            <div className="flex items-center justify-center gap-2">
              <TrendingUp className="w-5 h-5 text-primary" />
              <span className="text-2xl font-bold text-foreground">{totalDetections}</span>
            </div>
            <p className="text-sm text-muted-foreground font-medium">Total Detections</p>
          </div>
          <div className="text-center space-y-2">
            <div className="flex items-center justify-center gap-2">
              <Clock className="w-5 h-5 text-primary" />
              <span className="text-2xl font-bold text-foreground">
                {latestResult ? latestResult.features.duration.toFixed(1) : '0.0'}s
              </span>
            </div>
            <p className="text-sm text-muted-foreground font-medium">Last Duration</p>
          </div>
        </div>

        {/* AI Insights */}
        {explanation.insights.length > 0 && (
          <div className="space-y-4">
            <h4 className="font-semibold text-sm text-foreground">AI Insights:</h4>
            <div className="space-y-3">
              {explanation.insights.map((insight, index) => (
                <div key={index} className="flex items-start gap-3 p-3 bg-muted/20 rounded-lg">
                  <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0"></div>
                  <p className="text-sm text-muted-foreground leading-relaxed">{insight}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}