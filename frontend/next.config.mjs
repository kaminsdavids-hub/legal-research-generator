/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Netlify publishes `out` (netlify.toml, `publish = "out"`). Without this the
  // build writes `.next` and the deploy publishes a directory that does not
  // exist -- the build goes green and the site serves nothing. The two have
  // disagreed since netlify.toml was written; it never surfaced because the
  // site was building from a branch whose own frontend/netlify.toml published
  // `.next` with @netlify/plugin-nextjs instead.
  //
  // A static export is the right shape here rather than a workaround: this
  // frontend is a client-side SPA that talks to an external backend over
  // NEXT_PUBLIC_API_URL. It has no server components doing data fetching, no
  // route handlers, and no dynamic routes -- nothing that needs a server at
  // the edge. `next dev` is unaffected.
  output: "export",
};

export default nextConfig;
