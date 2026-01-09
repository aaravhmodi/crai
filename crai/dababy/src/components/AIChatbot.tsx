import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Send, Bot, User, Sparkles, AlertTriangle, CheckCircle, Clock, MapPin } from 'lucide-react';
import type { DetectionResult } from '@/lib/audioAnalysis';
import { AdvancedMedicalDiagnosis, type MedicalCondition, type DiagnosisContext } from '@/lib/medicalDiagnosis';
import { AppStorage, type Message } from '@/lib/appStorage';


interface AIChatbotProps {
  detectionResults: DetectionResult[];
  className?: string;
  diagnosisContext?: DiagnosisContext;
}

export function AIChatbot({ detectionResults, className = '', diagnosisContext = {} }: AIChatbotProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [medicalConditions, setMedicalConditions] = useState<MedicalCondition[]>([]);
  const [babyLocation, setBabyLocation] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Load saved messages and location on component mount
  useEffect(() => {
    const savedState = AppStorage.loadAppState();
    // Ensure timestamps are Date objects
    const messagesWithDates = savedState.chatMessages.map(msg => ({
      ...msg,
      timestamp: msg.timestamp instanceof Date ? msg.timestamp : new Date(msg.timestamp || Date.now())
    }));
    setMessages(messagesWithDates);
    
    // Load saved baby location
    const savedLocation = localStorage.getItem('dababy_location') || '';
    setBabyLocation(savedLocation);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Update medical conditions when new detection results come in
    if (detectionResults.length > 0) {
      const latestResult = detectionResults[detectionResults.length - 1];
      const conditions = AdvancedMedicalDiagnosis.analyzeConditions(latestResult.features, diagnosisContext);
      setMedicalConditions(conditions);
    }
  }, [detectionResults, diagnosisContext]);


  const generateAIPrompt = (userMessage: string): string => {
    const latestDetection = detectionResults[detectionResults.length - 1];
    const conditionsText = medicalConditions.length > 0 
      ? medicalConditions.map(c => `${c.name} (${c.confidence.toFixed(2)} confidence): ${c.description}`).join('\n')
      : 'No specific medical conditions detected';
    
    const contextText = Object.entries(diagnosisContext)
      .filter(([_, value]) => value !== undefined)
      .map(([key, value]) => `${key}: ${value}`)
      .join(', ');
    
    const locationText = babyLocation ? `Baby Location: ${babyLocation}` : 'Location: Not specified';

    return `You are an expert pediatric AI assistant analyzing baby cry patterns. 

Current Analysis Data:
- Latest Detection: ${latestDetection ? `F0: ${latestDetection.features.f0Mean.toFixed(0)}Hz, Duration: ${latestDetection.features.duration.toFixed(1)}s, Risk: ${latestDetection.riskLevel}` : 'None'}
- Medical Conditions Detected: ${conditionsText}
- Context: ${contextText || 'None provided'}
- ${locationText}
- Total Detections: ${detectionResults.length}

User Question: ${userMessage}

Provide a helpful, accurate response about the baby's cry analysis. Focus on medical insights, pattern interpretation, and actionable recommendations. Be concise but thorough.`;
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date()
    };

    const updatedMessages = AppStorage.addChatMessage(userMessage);
    setMessages(updatedMessages);
    const currentMessage = inputMessage;
    setInputMessage('');
    setIsTyping(true);

    try {
      const response = await axios.post('https://api.cerebras.ai/v1/chat/completions', {
        model: 'llama-4-scout-17b-16e-instruct',
        messages: [{
          role: 'user',
          content: generateAIPrompt(currentMessage)
        }]
      }, {
        headers: {
          'Authorization': `Bearer ${import.meta.env.VITE_CEREBRAS_API_KEY || 'csk-9nd6c92xj6pyjknep4mtc6pe8kfr2knxwjkyxevrm5p2d63h'}`,
          'Content-Type': 'application/json'
        }
      });

      // Convert markdown to plain text
      const rawContent = response.data.choices[0].message.content;
      const plainTextContent = rawContent
        .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
        .replace(/\*(.*?)\*/g, '$1') // Remove italic
        .replace(/#{1,6}\s/g, '') // Remove headers
        .replace(/```[\s\S]*?```/g, '') // Remove code blocks
        .replace(/`([^`]+)`/g, '$1') // Remove inline code
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Remove links, keep text
        .replace(/^[-*+]\s/gm, '') // Remove bullet points
        .replace(/^\d+\.\s/gm, '') // Remove numbered lists
        .replace(/\n{3,}/g, '\n\n') // Normalize line breaks
        .trim();

      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: plainTextContent,
        sender: 'ai',
        timestamp: new Date()
      };

      const updatedMessages = AppStorage.addChatMessage(aiResponse);
      setMessages(updatedMessages);
    } catch (error) {
      console.error('Cerebras API error:', error);
      const errorResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: 'I apologize, but I\'m having trouble connecting to the AI service right now. Please try again later.',
        sender: 'ai',
        timestamp: new Date()
      };
      const updatedMessages = AppStorage.addChatMessage(errorResponse);
      setMessages(updatedMessages);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleLocationChange = (location: string) => {
    setBabyLocation(location);
    localStorage.setItem('dababy_location', location);
  };

  const suggestedQuestions = [
    "What medical conditions were detected?",
    "What does my latest detection mean?",
    "Should I be concerned about these patterns?",
    "What are the next steps I should take?"
  ];

  const getMedicalAlertIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <AlertTriangle className="w-4 h-4 text-red-500" />;
      case 'high': return <AlertTriangle className="w-4 h-4 text-orange-500" />;
      case 'medium': return <Clock className="w-4 h-4 text-yellow-500" />;
      default: return <CheckCircle className="w-4 h-4 text-green-500" />;
    }
  };

  const getMedicalAlertColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'border-red-200 bg-red-50';
      case 'high': return 'border-orange-200 bg-orange-50';
      case 'medium': return 'border-yellow-200 bg-yellow-50';
      default: return 'border-green-200 bg-green-50';
    }
  };

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-3">
          <Bot className="w-6 h-6 text-primary" />
          <span className="text-lg font-semibold">AI Assistant</span>
          <Badge variant="secondary" className="ml-auto px-2 py-1">
            <Sparkles className="w-3 h-3 mr-1" />
            Beta
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0 space-y-6">
        {/* Baby Location Section */}
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
            <MapPin className="w-4 h-4" />
            Baby Location
          </h3>
          <div className="flex gap-2">
            <select
              value={babyLocation}
              onChange={(e) => handleLocationChange(e.target.value)}
              className="flex-1 px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            >
              <option value="">Select location...</option>
              <option value="nursery">Nursery</option>
              <option value="bedroom">Bedroom</option>
              <option value="living_room">Living Room</option>
              <option value="kitchen">Kitchen</option>
              <option value="car">Car</option>
              <option value="stroller">Stroller</option>
              <option value="crib">Crib</option>
              <option value="high_chair">High Chair</option>
              <option value="playpen">Playpen</option>
              <option value="outdoors">Outdoors</option>
              <option value="other">Other</option>
            </select>
          </div>
          {babyLocation && (
            <p className="text-xs text-gray-600">
              Current location: <span className="font-medium">{babyLocation.replace('_', ' ')}</span>
            </p>
          )}
        </div>

        {/* Medical Conditions Alert Cards */}
        {medicalConditions.length > 0 && (
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-gray-700">Medical Analysis</h3>
            <div className="grid gap-2 max-h-40 overflow-y-auto">
              {medicalConditions.slice(0, 3).map((condition) => (
                <div
                  key={condition.id}
                  className={`p-3 rounded-lg border ${getMedicalAlertColor(condition.severity)}`}
                >
                  <div className="flex items-start gap-2">
                    {getMedicalAlertIcon(condition.severity)}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-medium text-gray-900">{condition.name}</p>
                        <Badge variant="outline" className="text-xs">
                          {(condition.confidence * 100).toFixed(0)}%
                        </Badge>
                      </div>
                      <p className="text-xs text-gray-600 mt-1">{condition.description}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        Action: {condition.urgency === 'emergency' ? 'Seek immediate care' : 
                                condition.urgency === 'urgent' ? 'Contact doctor soon' :
                                condition.urgency === 'consult' ? 'Schedule consultation' : 'Monitor'}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Chat Messages */}
        <div className="h-80 overflow-y-auto space-y-4 p-4 border rounded-lg bg-muted/10">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex items-start gap-3 ${
                message.sender === 'user' ? 'flex-row-reverse' : ''
              }`}
            >
              <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                message.sender === 'user' 
                  ? 'bg-primary text-primary-foreground' 
                  : 'bg-muted border text-muted-foreground'
              }`}>
                {message.sender === 'user' ? (
                  <User className="w-4 h-4" />
                ) : (
                  <Bot className="w-4 h-4" />
                )}
              </div>
              <div className={`max-w-[80%] p-3 rounded-lg ${
                message.sender === 'user'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-background border shadow-sm'
              }`}>
                <p className="text-sm leading-relaxed">{message.content}</p>
                <p className="text-xs opacity-70 mt-2">
                  {message.timestamp instanceof Date 
                    ? message.timestamp.toLocaleTimeString() 
                    : new Date(message.timestamp).toLocaleTimeString()}
                </p>
              </div>
            </div>
          ))}
          
          {/* Typing Indicator */}
          {isTyping && (
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-muted border text-muted-foreground flex items-center justify-center">
                <Bot className="w-4 h-4" />
              </div>
              <div className="bg-background border shadow-sm p-3 rounded-lg">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Suggested Questions */}
        {messages.length <= 1 && (
          <div className="space-y-3">
            <p className="text-sm font-medium text-muted-foreground">Try asking:</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {suggestedQuestions.map((question, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  onClick={() => setInputMessage(question)}
                  className="text-xs text-left justify-start h-auto py-2 px-3"
                >
                  {question}
                </Button>
              ))}
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="flex gap-3">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me about cry analysis, patterns, or interpretations..."
            className="flex-1 min-h-[44px] max-h-[120px] px-3 py-2 border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent text-sm"
            disabled={isTyping}
          />
          <Button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isTyping}
            size="sm"
            className="px-4 h-11"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}