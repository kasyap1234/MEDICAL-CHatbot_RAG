// // export { handle } from "./auth"
// import { redirect } from '@sveltejs/kit';
// import { sequence } from '@sveltejs/kit/hooks';
// import { handle as authHandle } from '$lib/auth';

// export const handle = sequence(authHandle, async ({ event, resolve }) => {
//   if (event.url.pathname === '/') {
//     throw redirect(307, '/chat');
//   }
//   return resolve(event);
// });
export {handle} from "./auth"; 
