// import { SvelteKitAuth } from '@auth/sveltekit';
// import Google from '@auth/core/providers/google';
// import { env } from '$env/dynamic/private';

// export const { handle,signIn,signOut} = SvelteKitAuth({
//   providers: [
//     Google({
//       clientId: env.GOOGLE_CLIENT_ID,
//       clientSecret: env.GOOGLE_CLIENT_SECRET,
//     })
//   ],
//   secret: env.AUTH_SECRET,
//   trustHost: true
// });
// import { SvelteKitAuth } from '@auth/sveltekit';
// import Google from '@auth/core/providers/google';
// import { db } from '$lib/db';
// import { env } from '$env/dynamic/private';
// export const { handle } = SvelteKitAuth({
//   providers: [
//     Google({
//       clientId: env.GOOGLE_CLIENT_ID,
//       clientSecret: env.GOOGLE_CLIENT_SECRET,
//     })
//   ],
//   callbacks: {
//     async signIn({ profile }) {
//       // Store or update user in your database
//       const user = await db.user.upsert({
//         where: { googleId: profile?.sub },
//         create: {
//           googleId: profile?.sub,
//           email: profile?.email,
//           name: profile?.name
//         },
//         update: {
//           email: profile?.email,
//           name: profile?.name
//         }
//       });
//       return true;
//     },
//     async session({ session, token }) {
//       // Add the database user id to the session
//       if (token.sub) {
//         session.userId = token.sub;
//       }
//       return session;
//     }  },
//   secret: env.AUTH_SECRET
// });
import { SvelteKitAuth } from '@auth/sveltekit';
import Google from '@auth/core/providers/google';
import { env } from '$env/dynamic/private';

export const { handle, signIn, signOut } = SvelteKitAuth({
  providers: [
    Google({
      clientId: env.GOOGLE_CLIENT_ID,
      clientSecret: env.GOOGLE_CLIENT_SECRET,
    })
  ],
  callbacks: {
    async signIn({ profile }) {
      const response = await fetch('http://localhost:8000/api/users/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          googleId: profile?.sub,
          email: profile?.email,
          name: profile?.name
        })
      });
      
      const data = await response.json();
      console.log("googledata", data);
      
      return true;
    },
    async session({ session, token }) {

    const jwtToken=jwt.sign({
        sessionId: session.userId,
    googleId:     })
    }
  },
  secret: env.AUTH_SECRET
});
