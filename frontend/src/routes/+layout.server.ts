// import type { LayoutServerLoad } from "./$types"
 
// export const load: LayoutServerLoad = async (event) => {
//   const session = await event.locals.auth()
 
//   return {
//     session,
//   }
// }
import type { LayoutServerLoad } from "./$types"
import { redirect } from '@sveltejs/kit';

export const load: LayoutServerLoad = async (events) => {
  const session = await events.locals.auth()
  console.log("session check in layout server",session)
  // If user is authenticated and on the home page, redirect to chat
  if (session?.userId ) {
    throw redirect(307, '/chat')
  }

  // If user is not authenticated and trying to access protected routes
  if (!session?.user && events.url.pathname.startsWith('/chat')) {
    throw redirect(303, '/signin')
  }

  return {
    session
  }
}

