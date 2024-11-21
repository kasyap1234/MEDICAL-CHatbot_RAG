'use client';

import { useState } from 'react';
import axios from 'axios';

interface ChatMessage {
  type: 'user' | 'bot';
  text: string;
}

export default function MedicalChatbot() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim()) return;

    // Add user message
    const userMessage: ChatMessage = { 
      type: 'user', 
      text: input 
    };
    
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/chat', { 
        query: input 
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      const botMessage: ChatMessage = { 
        type: 'bot', 
        text: response.data 
      };

      setMessages(prevMessages => [...prevMessages, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: ChatMessage = { 
        type: 'bot', 
        text: 'Sorry, something went wrong. Please try again.' 
      };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <div className="flex-grow container mx-auto max-w-2xl px-4 py-8">
        <div className="bg-white shadow-md rounded-lg overflow-hidden flex flex-col h-[600px]">
          {/* Chat Messages */}
          <div className="flex-grow overflow-y-auto p-4 space-y-4">
            {messages.map((message, index) => (
              <div 
                key={index} 
                className={`p-3 rounded-lg max-w-[80%] ${
                  message.type === 'user' 
                    ? 'bg-blue-500 text-white self-end ml-auto' 
                    : 'bg-gray-200 text-black self-start mr-auto'
                }`}
              >
                {message.text}
              </div>
            ))}
            {isLoading && (
              <div className="p-3 bg-gray-200 rounded-lg self-start">
                Typing...
              </div>
            )}
          </div>

          {/* Input Area */}
          <form 
            onSubmit={handleSubmit} 
            className="bg-gray-100 p-4 flex items-center"
          >
            <input 
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a medical question..."
              className="flex-grow p-2 border rounded-l-lg"
              disabled={isLoading}
            />
            <button 
              type="submit" 
              className="bg-blue-500 text-white p-2 rounded-r-lg"
              disabled={isLoading}
            >
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
