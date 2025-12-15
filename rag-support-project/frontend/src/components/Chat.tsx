// src/components/Chat.tsx
import { useState } from 'react';
import { supabase } from '../lib/supabase';

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    setLoading(true);
    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);

    const token = (await supabase.auth.getSession()).data.session?.access_token;
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ query: input })
    });

    const reader = response.body.getReader();
    let botMsg = { role: 'assistant', content: '' };
    setMessages(prev => [...prev, botMsg]);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const text = new TextDecoder().decode(value);
      botMsg.content += text.replace('data: ', '');
      setMessages(prev => [...prev.slice(0, -1), { ...botMsg }]);
    }
    setLoading(false);
    setInput('');
  };

  return (
    <div className="flex flex-col h-screen">
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map((msg, i) => (
          <div key={i} className={msg.role === 'user' ? 'text-right' : 'text-left'}>
            <span className="inline-block p-2 rounded bg-gray-100">{msg.content}</span>
          </div>
        ))}
      </div>
      <div className="p-4 border-t">
        <input value={input} onChange={e => setInput(e.target.value)} className="w-full p-2 border" />
        <button onClick={sendMessage} disabled={loading}>Send</button>
      </div>
    </div>
  );
}