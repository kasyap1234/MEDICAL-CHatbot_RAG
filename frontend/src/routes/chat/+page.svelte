<script lang="ts">
    import { onMount } from 'svelte';
    import { page } from '$app/stores';
  
    let messages: Array<{ text: string; isUser: boolean; context?: string[] }> = [];
    let newMessage = '';
    let loading = false;
    let userId = $page.data.profile?.sub; 
    
  
    async function sendMessage() {
      if (!newMessage.trim()) return;
  
      const userMessage = newMessage;
      messages = [...messages, { text: userMessage, isUser: true }];
      newMessage = '';
      loading = true;
  
      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: userMessage,
            user_id: userId
          })
        });
  
        const data = await response.json();
        messages = [...messages, { 
          text: data.response, 
          isUser: false,
          context: data.context 
        }];
      } catch (error) {
        messages = [...messages, { 
          text: 'Sorry, there was an error processing your request.', 
          isUser: false 
        }];
      } finally {
        loading = false;
      }
    }
</script>
  <div class="flex flex-col h-screen bg-gray-100">
    <!-- Chat Header -->
    <div class="bg-white shadow-sm p-4">
      <h1 class="text-xl font-semibold text-gray-800">Medical Assistant</h1>
    </div>
  
    <!-- Chat Messages -->
    <div class="flex-1 overflow-y-auto p-4 space-y-4">
      {#each messages as message}
        <div class={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}>
          <div class={`max-w-[70%] rounded-lg p-4 ${
            message.isUser 
              ? 'bg-blue-600 text-white' 
              : 'bg-white text-gray-800 shadow'
          }`}>
            <p class="text-sm">{message.text}</p>
            
            {#if message.context && message.context.length > 0}
              <div class="mt-2 pt-2 border-t border-gray-200">
                <p class="text-xs text-gray-500">Related Context:</p>
                {#each message.context as contextItem}
                  <p class="text-xs mt-1 text-gray-600">{contextItem}</p>
                {/each}
              </div>
            {/if}
          </div>
        </div>
      {/each}
      
      {#if loading}
        <div class="flex justify-start">
          <div class="bg-white rounded-lg p-4 shadow">
            <div class="flex space-x-2">
              <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
              <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
              <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
            </div>
          </div>
        </div>
      {/if}
    </div>
  
    <!-- Chat Input -->
    <div class="bg-white border-t p-4">
      <div class="flex space-x-4">
        <textarea
          bind:value={newMessage}
          on:keydown={handleKeydown}
          placeholder="Type your medical question..."
          class="flex-1 resize-none rounded-lg border border-gray-300 p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows="2"
        ></textarea>
        <button
          on:click={sendMessage}
          disabled={loading}
          class="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  </div>
  