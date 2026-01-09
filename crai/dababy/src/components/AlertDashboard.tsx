import React, { useState, useEffect } from 'react';
import type { Alert, DetectionResult } from '@/lib/audioAnalysis';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert as AlertComponent, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { AlertTriangle, Heart, Brain, Stethoscope, Ear, Volume2, Activity, AlertCircle } from 'lucide-react';

interface AlertDashboardProps {
  detectionResults: DetectionResult[];
  isMonitoring: boolean;
}

export const AlertDashboard: React.FC<AlertDashboardProps> = ({ detectionResults, isMonitoring }) => {
  const [currentAlerts, setCurrentAlerts] = useState<Alert[]>([]);

  useEffect(() => {
    if (detectionResults.length > 0) {
      const latest = detectionResults[detectionResults.length - 1];
      setCurrentAlerts(latest.alerts);
    }
  }, [detectionResults]);

  const currentRiskLevel = detectionResults.length > 0 ? 
    detectionResults[detectionResults.length - 1].riskLevel : 'low';

  const criticalAlerts = currentAlerts.filter(a => a.severity === 'critical');

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'hyperphonation': return <Brain className="w-5 h-5" />;
      case 'hoarseness': return <Volume2 className="w-5 h-5" />;
      case 'cri_du_chat': return <AlertTriangle className="w-5 h-5" />;
      case 'weak_cry': return <Activity className="w-5 h-5" />;
      case 'grunting': return <Stethoscope className="w-5 h-5" />;
      case 'serious_illness': return <Heart className="w-5 h-5" />;
      case 'hearing_impairment': return <Ear className="w-5 h-5" />;
      case 'hypernasality': return <Volume2 className="w-5 h-5" />;
      default: return <AlertCircle className="w-5 h-5" />;
    }
  };

  const getBadgeVariant = (severity: string) => {
    switch (severity) {
      case 'critical': return 'critical';
      case 'high': return 'destructive';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'secondary';
    }
  };

  return (
    <div className="w-full space-y-8">
      {/* Status Overview */}
      <Card>
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg font-semibold">Monitoring Status</CardTitle>
            <Badge 
              variant={isMonitoring ? 'default' : 'secondary'}
              className="px-3 py-1"
            >
              {isMonitoring ? 'Active' : 'Inactive'}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-muted/30 rounded-lg">
              <div className="text-2xl font-bold text-foreground mb-1">{detectionResults.length}</div>
              <div className="text-sm text-muted-foreground">Total Sessions</div>
            </div>
            <div className="text-center p-4 bg-muted/30 rounded-lg">
              <div className="text-2xl font-bold text-orange-600 mb-1">{currentAlerts.length}</div>
              <div className="text-sm text-muted-foreground">Active Alerts</div>
            </div>
            <div className="text-center p-4 bg-muted/30 rounded-lg">
              <div className="text-2xl font-bold text-red-600 mb-1">{criticalAlerts.length}</div>
              <div className="text-sm text-muted-foreground">Critical Events</div>
            </div>
            <div className="text-center p-4 bg-muted/30 rounded-lg">
              <div className={`text-2xl font-bold mb-1 ${
                currentRiskLevel === 'critical' ? 'text-red-600' :
                currentRiskLevel === 'high' ? 'text-orange-600' :
                currentRiskLevel === 'medium' ? 'text-yellow-600' : 'text-green-600'
              }`}>
                {currentRiskLevel.toUpperCase()}
              </div>
              <div className="text-sm text-muted-foreground">Risk Level</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Critical Alerts */}
      {criticalAlerts.length > 0 && (
        <AlertComponent variant="destructive" className="border-red-200 bg-red-50/50">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Critical Alert</AlertTitle>
          <AlertDescription>
            {criticalAlerts.length} critical issue(s) detected. Immediate medical attention may be required.
            <div className="mt-3 space-y-2">
              {criticalAlerts.map((alert, index) => (
                <div key={index} className="text-sm font-medium p-2 bg-red-100/50 rounded border-l-2 border-red-500">
                  • {alert.message}: {alert.recommendation}
                </div>
              ))}
            </div>
          </AlertDescription>
        </AlertComponent>
      )}

      {/* Current Active Alerts */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle className="text-lg font-semibold">Current Alerts</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          {currentAlerts.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Activity className="w-16 h-16 mx-auto mb-4 opacity-40" />
              <h3 className="font-medium mb-2">No Active Alerts</h3>
              <p className="text-sm">Monitoring normal - no issues detected</p>
            </div>
          ) : (
            <div className="space-y-4">
              {currentAlerts.map((alert, index) => (
                <div key={index} className="p-4 bg-muted/30 rounded-lg border border-muted">
                  <div className="flex items-start gap-4">
                    <div className="mt-1 text-muted-foreground">{getAlertIcon(alert.type)}</div>
                    <div className="flex-1 space-y-3">
                      <div className="flex items-center gap-3">
                        <span className="font-semibold text-foreground">{alert.message}</span>
                        <Badge variant={getBadgeVariant(alert.severity)}>
                          {alert.severity.toUpperCase()}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground leading-relaxed">{alert.description}</p>
                      <p className="text-sm font-medium text-foreground">{alert.recommendation}</p>
                      <div className="flex items-center gap-6 text-xs text-muted-foreground pt-2 border-t border-muted">
                        <span>Confidence: {(alert.confidence * 100).toFixed(0)}%</span>
                        <span>Type: {alert.type}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Recent Analysis History */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle className="text-lg font-semibold">Recent Analysis</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          {detectionResults.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Stethoscope className="w-16 h-16 mx-auto mb-4 opacity-40" />
              <h3 className="font-medium mb-2">No Analysis Data</h3>
              <p className="text-sm">Start recording to see analysis results</p>
            </div>
          ) : (
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {detectionResults.slice(-10).reverse().map((result, index) => (
                <div key={index} className="p-4 bg-muted/30 rounded-lg border border-muted">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm text-muted-foreground font-medium">
                      {result.timestamp.toLocaleTimeString()}
                    </span>
                    <Badge variant={getBadgeVariant(result.riskLevel)}>
                      {result.riskLevel}
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
                    <div className="space-y-1">
                      <div className="text-xs text-muted-foreground">F0 (Hz)</div>
                      <div className="font-mono text-foreground">{result.features.f0Mean.toFixed(0)}</div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-xs text-muted-foreground">Duration</div>
                      <div className="font-mono text-foreground">{result.features.duration.toFixed(1)}s</div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-xs text-muted-foreground">HNR (dB)</div>
                      <div className="font-mono text-foreground">{result.features.hnr.toFixed(1)}</div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-xs text-muted-foreground">Alerts</div>
                      <div className="font-mono text-foreground">{result.alerts.length}</div>
                    </div>
                  </div>
                  {result.alerts.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-muted">
                      <div className="text-xs text-muted-foreground">
                        Alert types: {result.alerts.map(a => a.type).join(', ')}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};