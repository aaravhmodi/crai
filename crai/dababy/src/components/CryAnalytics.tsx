import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { BarChart3, TrendingUp, Clock, AlertTriangle, Trash2 } from 'lucide-react';
import { CryStorage, type CryInstance } from '@/lib/cryDetection';
import { CryAudioPlayer } from './CryAudioPlayer';

interface CryAnalyticsProps {
  className?: string;
}

interface AnalyticsData {
  totalCries: number;
  avgDuration: number;
  mostCommonDiagnosis: string;
  criticalAlerts: number;
  criesPerHour: number;
  diagnosisBreakdown: Record<string, number>;
  riskLevelCounts: Record<string, number>;
}

export const CryAnalytics: React.FC<CryAnalyticsProps> = ({ className = '' }) => {
  const [cries, setCries] = useState<CryInstance[]>([]);
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1h' | '6h' | '24h' | 'all'>('24h');

  useEffect(() => {
    const calculateAnalytics = (cryData: CryInstance[]) => {
      const totalCries = cryData.length;
      const avgDuration = cryData.reduce((sum, cry) => sum + cry.duration, 0) / totalCries;
      
      // Count diagnoses
      const diagnosisCount: Record<string, number> = {};
      const riskCount: Record<string, number> = {};
      let criticalAlerts = 0;
      
      cryData.forEach(cry => {
        const diagnosis = cry.diagnosis.primary;
        diagnosisCount[diagnosis] = (diagnosisCount[diagnosis] || 0) + 1;
        
        const risk = cry.riskLevel;
        riskCount[risk] = (riskCount[risk] || 0) + 1;
        
        if (risk === 'critical') criticalAlerts++;
      });
      
      const mostCommonDiagnosis = Object.entries(diagnosisCount)
        .sort(([,a], [,b]) => b - a)[0]?.[0] || 'None';
      
      // Calculate cries per hour
      const timeSpanHours = selectedTimeframe === 'all' 
        ? Math.max(1, (Date.now() - Math.min(...cryData.map(c => c.timestamp.getTime()))) / (1000 * 60 * 60))
        : parseFloat(selectedTimeframe.replace('h', ''));
      
      const criesPerHour = totalCries / timeSpanHours;

      setAnalytics({
        totalCries,
        avgDuration,
        mostCommonDiagnosis,
        criticalAlerts,
        criesPerHour,
        diagnosisBreakdown: diagnosisCount,
        riskLevelCounts: riskCount
      });
    };

    const loadCriesData = () => {
      const allCries = CryStorage.getAllCries();
      
      // Filter by timeframe
      const now = new Date();
      const filtered = allCries.filter((cry: CryInstance) => {
        const ageHours = (now.getTime() - cry.timestamp.getTime()) / (1000 * 60 * 60);
        switch (selectedTimeframe) {
          case '1h': return ageHours <= 1;
          case '6h': return ageHours <= 6;
          case '24h': return ageHours <= 24;
          case 'all': return true;
          default: return true;
        }
      });

      setCries(filtered);
      
      if (filtered.length > 0) {
        calculateAnalytics(filtered);
      } else {
        setAnalytics(null);
      }
    };

    loadCriesData();
  }, [selectedTimeframe]);

  const clearAllData = () => {
    if (confirm('Are you sure you want to clear all cry data? This action cannot be undone.')) {
      CryStorage.clearAllCries();
      setCries([]);
      setAnalytics(null);
    }
  };

  const formatDuration = (seconds: number): string => {
    return `${seconds.toFixed(1)}s`;
  };

  const formatTimeAgo = (timestamp: Date): string => {
    const now = new Date();
    const diffMs = now.getTime() - timestamp.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMins / 60);
    
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${Math.floor(diffHours / 24)}d ago`;
  };

  const getRiskBadgeVariant = (risk: string) => {
    switch (risk) {
      case 'critical': return 'critical' as const;
      case 'high': return 'destructive' as const;
      case 'medium': return 'warning' as const;
      case 'low': return 'success' as const;
      default: return 'secondary' as const;
    }
  };

  return (
    <div className={`w-full space-y-8 ${className}`}>
      {/* Header with Timeframe Selection */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <BarChart3 className="w-6 h-6 text-primary" />
          <h2 className="text-xl font-semibold">Cry Analytics</h2>
        </div>
        <div className="flex gap-1 p-1 bg-muted rounded-lg">
          {(['1h', '6h', '24h', 'all'] as const).map((timeframe) => (
            <Button
              key={timeframe}
              variant={selectedTimeframe === timeframe ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setSelectedTimeframe(timeframe)}
              className="px-3 py-1.5 text-xs"
            >
              {timeframe === 'all' ? 'All Time' : timeframe.toUpperCase()}
            </Button>
          ))}
        </div>
      </div>

      {analytics ? (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-6 pb-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground mb-1">Total Cries</p>
                    <p className="text-2xl font-bold">{analytics.totalCries}</p>
                  </div>
                  <TrendingUp className="w-5 h-5 text-muted-foreground" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6 pb-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground mb-1">Avg Duration</p>
                    <p className="text-2xl font-bold">{formatDuration(analytics.avgDuration)}</p>
                  </div>
                  <Clock className="w-5 h-5 text-muted-foreground" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6 pb-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground mb-1">Cries/Hour</p>
                    <p className="text-2xl font-bold">{analytics.criesPerHour.toFixed(1)}</p>
                  </div>
                  <BarChart3 className="w-5 h-5 text-muted-foreground" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6 pb-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground mb-1">Critical Alerts</p>
                    <p className="text-2xl font-bold">{analytics.criticalAlerts}</p>
                  </div>
                  <AlertTriangle className="w-5 h-5 text-muted-foreground" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Diagnosis Breakdown */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="text-lg font-semibold">Diagnosis Breakdown</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-4">
                  {Object.entries(analytics.diagnosisBreakdown)
                    .sort(([,a], [,b]) => b - a)
                    .map(([diagnosis, count]) => (
                      <div key={diagnosis} className="flex items-center justify-between">
                        <span className="text-sm font-medium">{diagnosis}</span>
                        <div className="flex items-center gap-3">
                          <div className="w-20 bg-muted rounded-full h-2">
                            <div 
                              className="bg-primary h-2 rounded-full transition-all duration-300" 
                              style={{ width: `${(count / analytics.totalCries) * 100}%` }}
                            />
                          </div>
                          <span className="text-sm font-bold w-6 text-right">{count}</span>
                        </div>
                      </div>
                    ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="text-lg font-semibold">Risk Level Distribution</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-4">
                  {Object.entries(analytics.riskLevelCounts)
                    .sort(([,a], [,b]) => b - a)
                    .map(([risk, count]) => (
                      <div key={risk} className="flex items-center justify-between">
                        <Badge variant={getRiskBadgeVariant(risk)} className="capitalize px-3 py-1">
                          {risk}
                        </Badge>
                        <div className="flex items-center gap-3">
                          <div className="w-20 bg-muted rounded-full h-2">
                            <div 
                              className="bg-primary h-2 rounded-full transition-all duration-300" 
                              style={{ width: `${(count / analytics.totalCries) * 100}%` }}
                            />
                          </div>
                          <span className="text-sm font-bold w-6 text-right">{count}</span>
                        </div>
                      </div>
                    ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recent Cries */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-4">
              <CardTitle className="text-lg font-semibold">Recent Cries</CardTitle>
              <Button variant="outline" size="sm" onClick={clearAllData} className="text-xs">
                <Trash2 className="w-3 h-3 mr-2" />
                Clear All Data
              </Button>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="space-y-4">
                {cries.slice(0, 10).map((cry) => (
                  <div key={cry.id} className="p-4 bg-muted/30 rounded-lg border border-muted space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Badge variant={getRiskBadgeVariant(cry.riskLevel)} className="px-2 py-1">
                          {cry.riskLevel.toUpperCase()}
                        </Badge>
                        <div>
                          <p className="font-medium text-sm">{cry.diagnosis.primary}</p>
                          <p className="text-xs text-muted-foreground">
                            Duration: {formatDuration(cry.duration)} • 
                            Confidence: {(cry.confidence * 100).toFixed(0)}%
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-muted-foreground">
                          {formatTimeAgo(cry.timestamp)}
                        </p>
                      </div>
                    </div>
                    
                    {/* Audio Player */}
                    <CryAudioPlayer cryId={cry.id} className="pt-3 border-t border-muted/50" />
                  </div>
                ))}
                {cries.length === 0 && (
                  <div className="text-center py-12 text-muted-foreground">
                    <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-40" />
                    <h3 className="font-medium mb-2">No Cries Recorded</h3>
                    <p className="text-sm">No cries recorded in the selected timeframe</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </>
      ) : (
        <Card>
          <CardContent className="text-center py-16">
            <BarChart3 className="w-16 h-16 mx-auto text-muted-foreground/40 mb-4" />
            <h3 className="text-lg font-medium mb-2">No Data Available</h3>
            <p className="text-muted-foreground">
              Record some cries to see analytics and patterns
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};