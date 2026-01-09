import type { AudioFeatures } from './audioAnalysis';

export interface MedicalCondition {
  id: string;
  name: string;
  category: 'genetic' | 'neurologic' | 'endocrine' | 'metabolic' | 'infection' | 'respiratory' | 'hearing' | 'neuromuscular' | 'behavioral' | 'developmental';
  severity: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  description: string;
  heuristics: string[];
  recommendations: string[];
  urgency: 'monitor' | 'consult' | 'urgent' | 'emergency';
}

export interface DiagnosisContext {
  babyAge?: number; // in weeks
  isJaundiced?: boolean;
  postFeed?: boolean;
  hasFever?: boolean;
  gestationalAge?: number; // in weeks
  feedingDifficulty?: boolean;
  constipation?: boolean;
}

export class AdvancedMedicalDiagnosis {
  static analyzeConditions(features: AudioFeatures, context: DiagnosisContext = {}): MedicalCondition[] {
    const conditions: MedicalCondition[] = [];
    
    // Genetic/Neurologic Conditions
    conditions.push(...this.checkGeneticConditions(features, context));
    
    // Endocrine/Metabolic
    conditions.push(...this.checkEndocrineConditions(features, context));
    
    // Infection/Systemic
    conditions.push(...this.checkInfectionConditions(features, context));
    
    // Respiratory/Airway
    conditions.push(...this.checkRespiratoryConditions(features, context));
    
    // Hearing & Speech
    conditions.push(...this.checkHearingConditions(features, context));
    
    // Neuromuscular
    conditions.push(...this.checkNeuromuscularConditions(features, context));
    
    // Behavioral/GI
    conditions.push(...this.checkBehavioralConditions(features, context));
    
    // Developmental (ASD/Speech)
    conditions.push(...this.checkDevelopmentalConditions(features, context));
    
    return conditions.filter(c => c.confidence > 0.3).sort((a, b) => b.confidence - a.confidence);
  }

  private static checkGeneticConditions(features: AudioFeatures, _context: DiagnosisContext): MedicalCondition[] {
    const conditions: MedicalCondition[] = [];
    
    // Cri-du-chat syndrome
    const highPitchFrames = features.f0.filter(f => f > 800).length;
    const totalFrames = features.f0.length;
    const highPitchPercentage = totalFrames > 0 ? highPitchFrames / totalFrames : 0;
    
    if (highPitchPercentage > 0.3 && features.f0Mean > 800 && features.f0Std < 80) {
      conditions.push({
        id: 'cri-du-chat',
        name: 'Cri-du-chat Syndrome Pattern',
        category: 'genetic',
        severity: 'high',
        confidence: Math.min(0.9, highPitchPercentage * 2),
        description: 'High-pitched, cat-like cry pattern detected. Characteristic of 5p- deletion syndrome.',
        heuristics: [`High F0: ${features.f0Mean.toFixed(0)}Hz`, `High pitch frames: ${(highPitchPercentage * 100).toFixed(1)}%`],
        recommendations: ['Genetic consultation recommended', 'Karyotype analysis', 'Developmental assessment'],
        urgency: 'consult'
      });
    }

    // Hypoxic-ischemic encephalopathy
    const hyperphonationFrames = features.f0.filter(f => f > 1000).length;
    const hyperphonationRate = totalFrames > 0 ? hyperphonationFrames / totalFrames : 0;
    
    if (hyperphonationRate > 0.1 || features.f0Mean > 1000) {
      conditions.push({
        id: 'hie',
        name: 'Hypoxic-Ischemic Encephalopathy Pattern',
        category: 'neurologic',
        severity: 'critical',
        confidence: Math.min(0.85, hyperphonationRate * 5),
        description: 'Hyperphonated cry pattern suggesting possible neurologic injury.',
        heuristics: [`Hyperphonation rate: ${(hyperphonationRate * 100).toFixed(1)}%`, `Mean F0: ${features.f0Mean.toFixed(0)}Hz`],
        recommendations: ['Immediate neurological evaluation', 'Brain imaging', 'Continuous monitoring'],
        urgency: 'emergency'
      });
    }

    return conditions;
  }

  private static checkEndocrineConditions(features: AudioFeatures, context: DiagnosisContext): MedicalCondition[] {
    const conditions: MedicalCondition[] = [];
    
    // Congenital hypothyroidism - hoarse cry
    if (features.hnr < 7 && features.jitter > 1.5 && features.shimmer > 3) {
      conditions.push({
        id: 'hypothyroidism',
        name: 'Congenital Hypothyroidism Pattern',
        category: 'endocrine',
        severity: 'medium',
        confidence: 0.6,
        description: 'Hoarse cry quality suggesting possible thyroid dysfunction.',
        heuristics: [`Low HNR: ${features.hnr.toFixed(1)}dB`, `High jitter: ${features.jitter.toFixed(2)}%`],
        recommendations: ['Thyroid function tests', 'Pediatric endocrinology consultation'],
        urgency: 'consult'
      });
    }

    // Severe hyperbilirubinemia with jaundice context
    if (context.isJaundiced && features.f0Mean > 600) {
      conditions.push({
        id: 'kernicterus-risk',
        name: 'Acute Bilirubin Encephalopathy Risk',
        category: 'metabolic',
        severity: 'critical',
        confidence: 0.8,
        description: 'High-pitched cry in jaundiced infant - kernicterus risk.',
        heuristics: [`High F0 with jaundice: ${features.f0Mean.toFixed(0)}Hz`],
        recommendations: ['Immediate bilirubin levels', 'Consider phototherapy/exchange transfusion'],
        urgency: 'emergency'
      });
    }

    return conditions;
  }

  private static checkInfectionConditions(features: AudioFeatures, _context: DiagnosisContext): MedicalCondition[] {
    const conditions: MedicalCondition[] = [];
    
    // Sepsis/meningitis pattern - weak/continuous cry
    const continuousCry = features.duration > 300; // 5+ minutes
    const lowIntensity = features.rms < 0.1;
    const moaning = features.spectralCentroid < 600 && features.spectralFlatness > 0.7;
    
    if (continuousCry && (lowIntensity || moaning)) {
      conditions.push({
        id: 'sepsis-meningitis',
        name: 'Serious Illness Pattern (Sepsis/Meningitis)',
        category: 'infection',
        severity: 'critical',
        confidence: 0.75,
        description: 'Continuous weak or moaning cry pattern - serious illness concern.',
        heuristics: [`Duration: ${features.duration.toFixed(1)}s`, `Low intensity: ${lowIntensity}`, `Moaning quality: ${moaning}`],
        recommendations: ['Immediate medical evaluation', 'Blood cultures', 'Lumbar puncture consideration'],
        urgency: 'emergency'
      });
    }

    return conditions;
  }

  private static checkRespiratoryConditions(features: AudioFeatures, _context: DiagnosisContext): MedicalCondition[] {
    const conditions: MedicalCondition[] = [];
    
    // Respiratory distress - grunting pattern
    const gruntingPattern = this.detectGrunting(features);
    if (gruntingPattern.detected) {
      conditions.push({
        id: 'respiratory-distress',
        name: 'Respiratory Distress Syndrome',
        category: 'respiratory',
        severity: 'high',
        confidence: gruntingPattern.confidence,
        description: 'Expiratory grunting pattern detected - respiratory distress.',
        heuristics: [`Grunting rate: ${gruntingPattern.rate.toFixed(1)}/min`],
        recommendations: ['Oxygen saturation monitoring', 'Chest X-ray', 'Respiratory support'],
        urgency: 'urgent'
      });
    }

    return conditions;
  }

  private static checkHearingConditions(features: AudioFeatures, _context: DiagnosisContext): MedicalCondition[] {
    const conditions: MedicalCondition[] = [];
    
    // Hearing impairment indicators
    const hearingRisk = features.f0Mean > 500 && features.rms < 0.15 && features.duration > 8;
    if (hearingRisk) {
      conditions.push({
        id: 'hearing-impairment',
        name: 'Possible Hearing Impairment',
        category: 'hearing',
        severity: 'medium',
        confidence: 0.4,
        description: 'Cry characteristics suggest possible hearing difficulties.',
        heuristics: [`High F0: ${features.f0Mean.toFixed(0)}Hz`, `Low intensity`, `Long duration`],
        recommendations: ['Hearing screening', 'Audiological evaluation'],
        urgency: 'monitor'
      });
    }

    return conditions;
  }

  private static checkNeuromuscularConditions(features: AudioFeatures, context: DiagnosisContext): MedicalCondition[] {
    const conditions: MedicalCondition[] = [];
    
    // Infant botulism/SMA - weak cry
    const weakCry = features.rms < 0.08 && features.duration < 2 && features.f0Mean < 400;
    if (weakCry && (context.constipation || context.feedingDifficulty)) {
      conditions.push({
        id: 'neuromuscular-weakness',
        name: 'Neuromuscular Weakness (Botulism/SMA)',
        category: 'neuromuscular',
        severity: 'high',
        confidence: 0.7,
        description: 'Weak cry with feeding/GI symptoms - neuromuscular concern.',
        heuristics: [`Low RMS: ${features.rms.toFixed(3)}`, `Short duration: ${features.duration.toFixed(1)}s`],
        recommendations: ['Neurological evaluation', 'EMG consideration', 'Stool botulism toxin test'],
        urgency: 'urgent'
      });
    }

    return conditions;
  }

  private static checkBehavioralConditions(features: AudioFeatures, _context: DiagnosisContext): MedicalCondition[] {
    const conditions: MedicalCondition[] = [];
    
    // Colic pattern (temporal detection needed)
    if (features.duration > 180 * 60) { // 3+ hours
      conditions.push({
        id: 'colic',
        name: 'Colic Pattern',
        category: 'behavioral',
        severity: 'low',
        confidence: 0.6,
        description: 'Extended crying duration consistent with colic.',
        heuristics: [`Duration: ${(features.duration / 60).toFixed(1)} minutes`],
        recommendations: ['Comfort measures', 'Rule out other causes', 'Parent support'],
        urgency: 'monitor'
      });
    }

    return conditions;
  }

  private static checkDevelopmentalConditions(features: AudioFeatures, context: DiagnosisContext): MedicalCondition[] {
    const conditions: MedicalCondition[] = [];
    
    // ASD risk - atypical cry patterns
    if (context.babyAge && context.babyAge < 24) { // First 6 months
      const atypicalPattern = features.f0Mean > 600 && features.f0Std > 150;
      if (atypicalPattern) {
        conditions.push({
          id: 'asd-risk',
          name: 'Atypical Development Risk',
          category: 'developmental',
          severity: 'low',
          confidence: 0.3,
          description: 'Atypical cry patterns - monitor development.',
          heuristics: [`Irregular F0 patterns`, `High variability`],
          recommendations: ['Developmental screening', 'Early intervention referral'],
          urgency: 'monitor'
        });
      }
    }

    return conditions;
  }

  private static detectGrunting(features: AudioFeatures): { detected: boolean; confidence: number; rate: number } {
    // Simplified grunting detection
    const lowFreqEnergy = features.lowMidHarmonics > 0.6;
    const repetitivePattern = features.repetitionRate > 0.5 && features.repetitionRate < 1.2;
    
    if (lowFreqEnergy && repetitivePattern) {
      return {
        detected: true,
        confidence: 0.7,
        rate: features.repetitionRate * 60 // Convert to per minute
      };
    }
    
    return { detected: false, confidence: 0, rate: 0 };
  }
}
