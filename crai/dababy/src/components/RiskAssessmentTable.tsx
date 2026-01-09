import React, { useState, useEffect } from 'react';
import type { Alert, DetectionResult } from '@/lib/audioAnalysis';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  AlertTriangle, 
  Heart, 
  Brain, 
  Stethoscope, 
  Ear, 
  Volume2, 
  Activity, 
  ChevronDown,
  ChevronRight,
  Info
} from 'lucide-react';

interface RiskFactor {
  id: string;
  name: string;
  description: string;
  occurrences: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  rules: string[];
  signs: string[];
  recommendations: string[];
  medicalAttention: 'none' | 'monitor' | 'consult' | 'urgent' | 'emergency';
  icon: React.ReactNode;
}

interface RiskAssessmentTableProps {
  detectionResults: DetectionResult[];
  isMonitoring: boolean;
}

const RISK_FACTOR_DEFINITIONS: Record<string, Omit<RiskFactor, 'occurrences' | 'confidence'>> = {
  hyperphonation: {
    id: 'hyperphonation',
    name: 'Hyperphonation',
    description: 'Abnormally high-pitched crying that may indicate neurological issues',
    riskLevel: 'high',
    rules: [
      'F0 > 800 Hz consistently',
      'Spectral centroid > 2000 Hz',
      'Duration > 2 seconds'
    ],
    signs: [
      'Very high-pitched, shrill crying',
      'Piercing quality to the cry',
      'May sound almost like a scream'
    ],
    recommendations: [
      'Monitor for neurological symptoms',
      'Check for signs of increased intracranial pressure',
      'Document frequency and triggers'
    ],
    medicalAttention: 'consult',
    icon: <Brain className="w-4 h-4" />
  },
  hoarseness: {
    id: 'hoarseness',
    name: 'Hoarseness',
    description: 'Rough, breathy cry quality that may indicate vocal cord issues',
    riskLevel: 'medium',
    rules: [
      'HNR < 10 dB',
      'High jitter (> 2%)',
      'Irregular voicing patterns'
    ],
    signs: [
      'Rough, scratchy cry quality',
      'Breathy or weak voice',
      'Inconsistent vocal quality'
    ],
    recommendations: [
      'Monitor for respiratory symptoms',
      'Check for signs of vocal strain',
      'Ensure proper hydration'
    ],
    medicalAttention: 'monitor',
    icon: <Volume2 className="w-4 h-4" />
  },
  cri_du_chat: {
    id: 'cri_du_chat',
    name: 'Cri-du-Chat Syndrome',
    description: 'Cat-like cry that may indicate chromosomal abnormality',
    riskLevel: 'critical',
    rules: [
      'F0 > 1000 Hz with cat-like quality',
      'Specific spectral pattern',
      'Consistent across multiple cries'
    ],
    signs: [
      'High-pitched, cat-like crying',
      'Mewing sound quality',
      'Often accompanied by feeding difficulties'
    ],
    recommendations: [
      'Immediate pediatric evaluation',
      'Genetic counseling referral',
      'Developmental assessment'
    ],
    medicalAttention: 'urgent',
    icon: <AlertTriangle className="w-4 h-4" />
  },
  weak_cry: {
    id: 'weak_cry',
    name: 'Weak Cry',
    description: 'Unusually quiet or weak crying that may indicate illness or weakness',
    riskLevel: 'medium',
    rules: [
      'RMS amplitude < 0.05',
      'Short cry bursts (< 0.5s)',
      'Low energy across frequencies'
    ],
    signs: [
      'Very quiet, barely audible crying',
      'Short, weak cry bursts',
      'Lack of typical cry intensity'
    ],
    recommendations: [
      'Check for signs of illness',
      'Monitor feeding and activity levels',
      'Assess overall energy and responsiveness'
    ],
    medicalAttention: 'consult',
    icon: <Activity className="w-4 h-4" />
  },
  grunting: {
    id: 'grunting',
    name: 'Grunting',
    description: 'Grunting sounds during breathing that may indicate respiratory distress',
    riskLevel: 'high',
    rules: [
      'Low frequency components (< 200 Hz)',
      'Rhythmic pattern with breathing',
      'Occurs during expiration'
    ],
    signs: [
      'Grunting sounds with each breath',
      'Increased work of breathing',
      'May have chest retractions'
    ],
    recommendations: [
      'Monitor respiratory rate and effort',
      'Check oxygen saturation if available',
      'Watch for signs of respiratory distress'
    ],
    medicalAttention: 'urgent',
    icon: <Stethoscope className="w-4 h-4" />
  },
  serious_illness: {
    id: 'serious_illness',
    name: 'Serious Illness Indicators',
    description: 'Cry patterns that may indicate serious underlying illness',
    riskLevel: 'critical',
    rules: [
      'Multiple abnormal acoustic features',
      'Persistent abnormal patterns',
      'Combination of weak cry and other symptoms'
    ],
    signs: [
      'Unusual cry quality or pattern',
      'Decreased responsiveness',
      'Changes in feeding or sleeping patterns'
    ],
    recommendations: [
      'Immediate medical evaluation',
      'Monitor vital signs',
      'Document all symptoms'
    ],
    medicalAttention: 'emergency',
    icon: <Heart className="w-4 h-4" />
  },
  hearing_impairment: {
    id: 'hearing_impairment',
    name: 'Hearing Impairment',
    description: 'Cry patterns that may suggest hearing difficulties',
    riskLevel: 'medium',
    rules: [
      'Monotonous cry pattern',
      'Limited frequency variation',
      'Lack of response to auditory stimuli'
    ],
    signs: [
      'Monotonous, unchanging cry',
      'Limited vocal variety',
      'May not respond to sounds'
    ],
    recommendations: [
      'Hearing assessment referral',
      'Monitor response to sounds',
      'Early intervention evaluation'
    ],
    medicalAttention: 'consult',
    icon: <Ear className="w-4 h-4" />
  },
  hypernasality: {
    id: 'hypernasality',
    name: 'Hypernasality',
    description: 'Excessive nasal resonance that may indicate cleft palate or other issues',
    riskLevel: 'medium',
    rules: [
      'High nasal energy ratio (> 0.3)',
      'Specific formant patterns',
      'Consistent across vocalizations'
    ],
    signs: [
      'Nasal quality to crying',
      'Muffled or congested sound',
      'May have feeding difficulties'
    ],
    recommendations: [
      'Oral examination for cleft palate',
      'Speech pathology consultation',
      'Monitor feeding patterns'
    ],
    medicalAttention: 'consult',
    icon: <Volume2 className="w-4 h-4" />
  }
};

export const RiskAssessmentTable: React.FC<RiskAssessmentTableProps> = ({ 
  detectionResults, 
  isMonitoring 
}) => {
  const [riskFactors, setRiskFactors] = useState<RiskFactor[]>([]);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [aiSummary, setAiSummary] = useState<string>('');
  
  // Note: isMonitoring parameter available for future monitoring status features

  useEffect(() => {
    // Aggregate alerts across all detection results to track occurrences
    const alertCounts: Record<string, { count: number; totalConfidence: number; alerts: Alert[] }> = {};
    
    detectionResults.forEach(result => {
      result.alerts.forEach(alert => {
        if (!alertCounts[alert.type]) {
          alertCounts[alert.type] = { count: 0, totalConfidence: 0, alerts: [] };
        }
        alertCounts[alert.type].count++;
        alertCounts[alert.type].totalConfidence += alert.confidence;
        alertCounts[alert.type].alerts.push(alert);
      });
    });

    // Create risk factors only for conditions with multiple occurrences (conservative approach)
    const factors: RiskFactor[] = [];
    
    Object.entries(alertCounts).forEach(([type, data]) => {
      const definition = RISK_FACTOR_DEFINITIONS[type];
      if (definition && data.count >= 2) { // Only show if 2+ occurrences
        factors.push({
          ...definition,
          occurrences: data.count,
          confidence: data.totalConfidence / data.count
        });
      }
    });

    // Sort by risk level and occurrences
    factors.sort((a, b) => {
      const riskOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      const aRisk = riskOrder[a.riskLevel];
      const bRisk = riskOrder[b.riskLevel];
      
      if (aRisk !== bRisk) return bRisk - aRisk;
      return b.occurrences - a.occurrences;
    });

    setRiskFactors(factors);
    generateAISummary(factors, detectionResults.length);
  }, [detectionResults]);

  const generateAISummary = (factors: RiskFactor[], totalSessions: number) => {
    if (factors.length === 0) {
      setAiSummary(`Based on ${totalSessions} monitoring sessions, no significant risk patterns have been detected. The baby's cry patterns appear to be within normal ranges. Continue regular monitoring and maintain normal care routines.`);
      return;
    }

    const criticalFactors = factors.filter(f => f.riskLevel === 'critical');
    const highFactors = factors.filter(f => f.riskLevel === 'high');
    const mediumFactors = factors.filter(f => f.riskLevel === 'medium');

    let summary = `Analysis of ${totalSessions} monitoring sessions reveals `;

    if (criticalFactors.length > 0) {
      summary += `${criticalFactors.length} critical risk factor(s) requiring immediate attention: ${criticalFactors.map(f => f.name).join(', ')}. `;
      summary += `Immediate medical evaluation is strongly recommended. `;
    }

    if (highFactors.length > 0) {
      summary += `${highFactors.length} high-priority concern(s) detected: ${highFactors.map(f => f.name).join(', ')}. `;
      summary += `These patterns warrant prompt medical consultation. `;
    }

    if (mediumFactors.length > 0) {
      summary += `${mediumFactors.length} moderate concern(s) identified: ${mediumFactors.map(f => f.name).join(', ')}. `;
      summary += `Continue monitoring and consider discussing with your pediatrician. `;
    }

    summary += `All identified patterns are based on multiple occurrences to ensure reliability. Regular monitoring should continue.`;

    setAiSummary(summary);
  };

  const toggleRowExpansion = (id: string) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(id)) {
      newExpanded.delete(id);
    } else {
      newExpanded.add(id);
    }
    setExpandedRows(newExpanded);
  };

  const getRiskBadgeVariant = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical': return 'destructive';
      case 'high': return 'destructive';
      case 'medium': return 'warning';
      case 'low': return 'secondary';
      default: return 'secondary';
    }
  };

  const getAttentionBadgeVariant = (attention: string) => {
    switch (attention) {
      case 'emergency': return 'destructive';
      case 'urgent': return 'destructive';
      case 'consult': return 'warning';
      case 'monitor': return 'secondary';
      case 'none': return 'outline';
      default: return 'secondary';
    }
  };

  return (
    <div className="w-full space-y-6">
      {/* AI Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            AI Risk Assessment Summary
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
              <p className="text-sm text-blue-900 leading-relaxed">{aiSummary}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Risk Assessment Table */}
      <Card>
        <CardHeader>
          <CardTitle>Risk Factor Analysis</CardTitle>
          <p className="text-sm text-muted-foreground">
            Conservative assessment showing only conditions with multiple occurrences (2+)
          </p>
        </CardHeader>
        <CardContent>
          {riskFactors.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Activity className="w-16 h-16 mx-auto mb-4 opacity-40" />
              <h3 className="font-medium mb-2">No Risk Factors Detected</h3>
              <p className="text-sm">All cry patterns appear normal based on current analysis</p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Table Header */}
              <div className="grid grid-cols-6 gap-4 p-4 bg-muted/50 rounded-lg font-medium text-sm">
                <div></div>
                <div>Risk Factor</div>
                <div>Occurrences</div>
                <div>Risk Level</div>
                <div>Confidence</div>
                <div>Medical Attention</div>
              </div>
              
              {/* Table Rows */}
              {riskFactors.map((factor) => (
                <div key={factor.id} className="border rounded-lg">
                  <div className="grid grid-cols-6 gap-4 p-4 hover:bg-muted/50 cursor-pointer" onClick={() => toggleRowExpansion(factor.id)}>
                    <div className="flex items-center">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="p-1 h-8 w-8"
                      >
                        {expandedRows.has(factor.id) ? (
                          <ChevronDown className="w-4 h-4" />
                        ) : (
                          <ChevronRight className="w-4 h-4" />
                        )}
                      </Button>
                    </div>
                    <div className="flex items-center gap-2">
                      {factor.icon}
                      <div>
                        <div className="font-medium">{factor.name}</div>
                        <div className="text-sm text-muted-foreground">
                          {factor.description}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center">
                      <Badge variant="outline">{factor.occurrences}</Badge>
                    </div>
                    <div className="flex items-center">
                      <Badge variant={getRiskBadgeVariant(factor.riskLevel)}>
                        {factor.riskLevel.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="flex items-center">
                      <span className="font-mono text-sm">
                        {(factor.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="flex items-center">
                      <Badge variant={getAttentionBadgeVariant(factor.medicalAttention)}>
                        {factor.medicalAttention.toUpperCase()}
                      </Badge>
                    </div>
                  </div>
                  
                  {expandedRows.has(factor.id) && (
                    <div className="border-t bg-muted/30 p-4">
                      <div className="grid md:grid-cols-3 gap-4">
                        <div>
                          <h4 className="font-medium text-sm mb-2">Detection Rules</h4>
                          <ul className="text-sm text-muted-foreground space-y-1">
                            {factor.rules.map((rule, idx) => (
                              <li key={idx} className="flex items-start gap-2">
                                <span className="text-xs mt-1">•</span>
                                <span>{rule}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                        
                        <div>
                          <h4 className="font-medium text-sm mb-2">Observable Signs</h4>
                          <ul className="text-sm text-muted-foreground space-y-1">
                            {factor.signs.map((sign, idx) => (
                              <li key={idx} className="flex items-start gap-2">
                                <span className="text-xs mt-1">•</span>
                                <span>{sign}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                        
                        <div>
                          <h4 className="font-medium text-sm mb-2">Recommendations</h4>
                          <ul className="text-sm text-muted-foreground space-y-1">
                            {factor.recommendations.map((rec, idx) => (
                              <li key={idx} className="flex items-start gap-2">
                                <span className="text-xs mt-1">•</span>
                                <span>{rec}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
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
