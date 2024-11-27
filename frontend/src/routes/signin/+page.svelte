<!-- 
<script lang="ts">
    import { SignIn } from "@auth/sveltekit/components"
 </script>
  
 <div>
   <!- <img src="/img/logo.svg" alt="Company Logo" /> -->
   <!-- <SignIn provider="google" signInPage="signin" /> -->
 <!-- </div>  --> 


 <script lang="ts">
  import { page } from "$app/stores"
  import { SignIn } from "@auth/sveltekit/components"
</script>
 
<header>
  <nav>
    <SignIn
      options={{
        redirectTo: $page.data.redirectTo
          ? `/${decodeURIComponent($page.data.redirectTo).slice(1)}`
          : `/chat`,
      }}
      className="w-full"
    >
      <span slot="submitButton">Sign in</span>
    </SignIn>
  </nav>
</header>