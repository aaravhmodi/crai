import type { DetectionResult } from './audioAnalysis';
import type { DiagnosisContext } from './medicalDiagnosis';

export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
}

export interface AppState {
  detectionResults: DetectionResult[];
  diagnosisContext: DiagnosisContext;
  chatMessages: Message[];
  activeSection: 'recording' | 'results' | 'ai';
  lastUpdated: Date;
}

export class AppStorage {
  private static readonly STORAGE_KEY = 'dababy_app_state';
  private static readonly MAX_DETECTION_RESULTS = 100;
  private static readonly MAX_CHAT_MESSAGES = 50;

  /**
   * Save the complete app state to localStorage
   */
  static saveAppState(state: Partial<AppState>): void {
    try {
      const currentState = this.loadAppState();
      const updatedState: AppState = {
        ...currentState,
        ...state,
        lastUpdated: new Date()
      };

      // Limit stored data to prevent localStorage overflow
      if (updatedState.detectionResults.length > this.MAX_DETECTION_RESULTS) {
        updatedState.detectionResults = updatedState.detectionResults.slice(-this.MAX_DETECTION_RESULTS);
      }

      if (updatedState.chatMessages.length > this.MAX_CHAT_MESSAGES) {
        updatedState.chatMessages = updatedState.chatMessages.slice(-this.MAX_CHAT_MESSAGES);
      }

      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(updatedState, this.dateReplacer));
    } catch (error) {
      console.error('Failed to save app state to localStorage:', error);
    }
  }

  /**
   * Load the complete app state from localStorage
   */
  static loadAppState(): AppState {
    try {
      const stored = localStorage.getItem(this.STORAGE_KEY);
      if (!stored) {
        return this.getDefaultState();
      }

      const parsed = JSON.parse(stored, this.dateReviver) as AppState;
      
      // Validate and sanitize loaded data
      return {
        detectionResults: Array.isArray(parsed.detectionResults) ? parsed.detectionResults : [],
        diagnosisContext: this.validateDiagnosisContext(parsed.diagnosisContext),
        chatMessages: Array.isArray(parsed.chatMessages) ? parsed.chatMessages : this.getDefaultChatMessages(),
        activeSection: ['recording', 'results', 'ai'].includes(parsed.activeSection) ? parsed.activeSection : 'recording',
        lastUpdated: parsed.lastUpdated instanceof Date ? parsed.lastUpdated : new Date()
      };
    } catch (error) {
      console.error('Failed to load app state from localStorage:', error);
      return this.getDefaultState();
    }
  }

  /**
   * Save detection results specifically
   */
  static saveDetectionResults(results: DetectionResult[]): void {
    this.saveAppState({ detectionResults: results });
  }

  /**
   * Add a new detection result
   */
  static addDetectionResult(result: DetectionResult): DetectionResult[] {
    const currentState = this.loadAppState();
    const updatedResults = [...currentState.detectionResults, result].slice(-this.MAX_DETECTION_RESULTS);
    this.saveAppState({ detectionResults: updatedResults });
    return updatedResults;
  }

  /**
   * Save diagnosis context (baby information)
   */
  static saveDiagnosisContext(context: DiagnosisContext): void {
    this.saveAppState({ diagnosisContext: context });
  }

  /**
   * Save chat messages
   */
  static saveChatMessages(messages: Message[]): void {
    this.saveAppState({ chatMessages: messages });
  }

  /**
   * Add a new chat message
   */
  static addChatMessage(message: Message): Message[] {
    const currentState = this.loadAppState();
    const updatedMessages = [...currentState.chatMessages, message].slice(-this.MAX_CHAT_MESSAGES);
    this.saveAppState({ chatMessages: updatedMessages });
    return updatedMessages;
  }

  /**
   * Save the active section
   */
  static saveActiveSection(section: 'recording' | 'results' | 'ai'): void {
    this.saveAppState({ activeSection: section });
  }

  /**
   * Clear all stored data
   */
  static clearAllData(): void {
    try {
      localStorage.removeItem(this.STORAGE_KEY);
    } catch (error) {
      console.error('Failed to clear app state:', error);
    }
  }

  /**
   * Get storage usage information
   */
  static getStorageInfo(): { used: number; available: number; percentage: number } {
    try {
      const stored = localStorage.getItem(this.STORAGE_KEY);
      const usedBytes = stored ? new Blob([stored]).size : 0;
      const totalBytes = 5 * 1024 * 1024; // Approximate 5MB localStorage limit
      
      return {
        used: usedBytes,
        available: totalBytes - usedBytes,
        percentage: (usedBytes / totalBytes) * 100
      };
    } catch (error) {
      return { used: 0, available: 0, percentage: 0 };
    }
  }

  /**
   * Get default app state
   */
  private static getDefaultState(): AppState {
    return {
      detectionResults: [],
      diagnosisContext: {
        babyAge: undefined,
        isJaundiced: false,
        postFeed: false,
        hasFever: false,
        gestationalAge: undefined,
        feedingDifficulty: false,
        constipation: false
      },
      chatMessages: this.getDefaultChatMessages(),
      activeSection: 'recording',
      lastUpdated: new Date()
    };
  }

  /**
   * Get default chat messages
   */
  private static getDefaultChatMessages(): Message[] {
    return [
      {
        id: 'default-1',
        content: "Hello! I'm your AI assistant for comprehensive cry analysis. I can analyze medical conditions, explain detection results, and provide insights about baby crying patterns using advanced AI. How can I help you today?",
        sender: 'ai',
        timestamp: new Date()
      }
    ];
  }

  /**
   * Validate diagnosis context object
   */
  private static validateDiagnosisContext(context: any): DiagnosisContext {
    if (!context || typeof context !== 'object') {
      return this.getDefaultState().diagnosisContext;
    }

    return {
      babyAge: typeof context.babyAge === 'number' ? context.babyAge : undefined,
      isJaundiced: Boolean(context.isJaundiced),
      postFeed: Boolean(context.postFeed),
      hasFever: Boolean(context.hasFever),
      gestationalAge: typeof context.gestationalAge === 'number' ? context.gestationalAge : undefined,
      feedingDifficulty: Boolean(context.feedingDifficulty),
      constipation: Boolean(context.constipation)
    };
  }

  /**
   * JSON replacer for Date objects
   */
  private static dateReplacer(_key: string, value: unknown): unknown {
    if (value instanceof Date) {
      return { __type: 'Date', value: value.toISOString() };
    }
    return value;
  }

  /**
   * JSON reviver for Date objects
   */
  private static dateReviver(_key: string, value: unknown): unknown {
    if (value && typeof value === 'object' && 'value' in value && (value as { __type?: string }).__type === 'Date') {
      return new Date((value as { value: string }).value);
    }
    return value;
  }

  /**
   * Export app state for backup/sharing
   */
  static exportAppState(): string {
    const state = this.loadAppState();
    return JSON.stringify(state, this.dateReplacer, 2);
  }

  /**
   * Import app state from backup
   */
  static importAppState(jsonData: string): boolean {
    try {
      const parsed = JSON.parse(jsonData, this.dateReviver);
      
      // Validate the imported data structure
      if (parsed && typeof parsed === 'object') {
        this.saveAppState(parsed);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to import app state:', error);
      return false;
    }
  }
}
