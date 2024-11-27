export const actions = {
    chat: async ({ request, locals }) => {
      const session = await locals.auth();
      const data = await request.formData();
      const message = data.get('message');
    const 
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: message,
          user_id: profile?.sub  // Using the Google ID
        })
      });
  
      const chatResponse = await response.json();
      return chatResponse;
    }
  };
  
