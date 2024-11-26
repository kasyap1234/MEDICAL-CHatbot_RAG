<script lang="ts">
    import { onMount } from 'svelte';
    import { goto } from '$app/navigation';
    
    let query = '';
    let messages: Array<{type: 'user' | 'bot', content: string, context?: string[]}> = [];
    let loading = false;
  
    async function handleSubmit() {
      if (!query.trim()) return;
      
      loading = true;
      messages = [...messages, { type: 'user', content: query }];
      
      try {
        const response = await fetch('http://localhost:8000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          },
          body: JSON.stringify({ query })
        });
  
        if (response.status === 401) {
          goto('/login');
          return;
        }
  
        const data = await response.json();
        messages = [...messages, {
          type: 'bot',
          content: data.response,
          context: data.context
        }];
      } catch (error) {
        messages = [...messages, {
          type: 'bot',
          content: 'Sorry, there was an error processing your request.'
        }];
      } finally {
        loading = false;
        query = '';
      }
    }
  </script>
  
  <div class="max-w-4xl mx-auto p-4 h-screen flex flex-col">
    <div class="flex-1 overflow-y-auto mb-4 p-4 border border-gray-200 rounded-lg">
      {#each messages as message}
        <div class="mb-4 {message.type === 'user' ? 'ml-[20%]' : 'mr-[20%]'}">
          <div class="p-4 rounded-lg {message.type === 'user' ? 'bg-blue-50' : 'bg-gray-50'}">
            <p class="text-gray-800">{message.content}</p>
            
            {#if message.context}
              <div class="mt-2">
                <details class="text-sm">
                  <summary class="text-blue-600 cursor-pointer hover:text-blue-700">
                    View Context
                  </summary>
                  <div class="mt-2 pl-4 text-gray-600">
                    {#each message.context as ctx}
                      <p class="mb-2">{ctx}</p>
                    {/each}
                  </div>
                </details>
              </div>
            {/if}
          </div>
        </div>
      {/each}
      
      {#if loading}
        <div class="text-center text-gray-500 italic">
          Thinking...
        </div>
      {/if}
    </div>
  
    <form on:submit|preventDefault={handleSubmit} class="flex gap-4">
      <input
        type="text"
        bind:value={query}
        placeholder="Ask a medical question..."
        disabled={loading}
        class="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      />
      <button
        type="submit"
        disabled={loading}
        class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
      >
        Send
      </button>
    </form>
  </div>
  