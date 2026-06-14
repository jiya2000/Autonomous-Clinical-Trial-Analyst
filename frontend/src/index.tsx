import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

interface ChatEvent {
  type: 'thought' | 'answer' | 'done' | 'error';
  agent?: string;
  content?: string;
}

interface Message {
  role: 'user' | 'agent';
  content: string;
}

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [trace, setTrace] = useState<string[]>([]);
  const [input, setInput] = useState('');
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const clientId = Math.random().toString(36).slice(2);
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const host = window.location.hostname;
    const ws = new WebSocket(`${protocol}://${host}:8000/ws/chat/${clientId}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data: ChatEvent = JSON.parse(event.data);
      if (data.type === 'thought' && data.agent && data.content) {
        setTrace((prev) => [...prev, `${data.agent}: ${data.content}`]);
      } else if (data.type === 'answer' && data.content) {
        setMessages((prev) => [...prev, { role: 'agent', content: data.content ?? '' }]);
      } else if (data.type === 'error' && data.content) {
        setTrace((prev) => [...prev, `❌ ${data.content}`]);
      }
    };

    ws.onclose = () => {
      setTrace((prev) => [...prev, 'Connection closed']);
    };

    return () => {
      ws.close();
    };
  }, []);

  const sendMessage = () => {
    const trimmed = input.trim();
    if (!trimmed || !wsRef.current) return;

    setMessages((prev) => [...prev, { role: 'user', content: trimmed }]);

    wsRef.current.send(JSON.stringify({ query: trimmed }));
    setInput('');
  };

  return (
    <div className="app">
      <div className="chat-panel">
        <h1>Autonomous Clinical Trial Analyst</h1>
        <p className="subtitle">Analyse and interrogate clinical trial protocols with a local, zero-cost Agentic AI pipeline.</p>
        <div className="chat-window">
          {messages.map((m, idx) => (
            <div key={idx} className={`message ${m.role}`}>
              <span>{m.content}</span>
            </div>
          ))}
        </div>
        <div className="input-row">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Ask about a clinical trial protocol..."
          />
          <button onClick={sendMessage}>Send</button>
        </div>
      </div>
      <div className="trace-panel">
        <h2>Agent Thought Trace</h2>
        <div className="trace-window">
          {trace.map((t, idx) => (
            <div key={idx} className="trace-entry">
              {t}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}
