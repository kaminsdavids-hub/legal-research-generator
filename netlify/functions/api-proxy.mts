/**
 * Server-side proxy to the research backend.
 *
 * The browser bundle is static and public, so any credential compiled into it
 * is readable by anyone who opens dev tools. This function holds the backend
 * key instead: the browser calls `/api/*` on its own origin, this runs on
 * Netlify's servers, attaches `X-API-Key`, and forwards to the backend. The key
 * never reaches a client.
 *
 * ------------------------------------------------------------------------
 * READ THIS BEFORE DEPLOYING
 *
 * A proxy that adds credentials without requiring any of its own does not
 * secure anything — it moves the open door to a new address. Anyone who can
 * load the site can call this function, and this function can call every route
 * on the backend. That is strictly worse than the previous arrangement, where
 * at least the caller needed a key.
 *
 * So this refuses to run wide open by accident. Either gate the site (Netlify
 * site-level password protection, or an access-control rule in front of it) and
 * set PROXY_ALLOW_PUBLIC=true to record that you did, or set PROXY_PASSWORD and
 * the function will require it as a header. With neither, every request gets a
 * 503 that says which one is missing.
 *
 * PROXY_PASSWORD is a different secret from the backend key, and that is the
 * point of having two: the browser-side one can be rotated without touching the
 * backend, and it never confers direct access to the host.
 * ------------------------------------------------------------------------
 */

import type { Config, Context } from "@netlify/functions";

const BACKEND = process.env.BACKEND_URL ?? "";
const BACKEND_KEY = process.env.BACKEND_API_KEY ?? "";
const PROXY_PASSWORD = process.env.PROXY_PASSWORD ?? "";
const ALLOW_PUBLIC = (process.env.PROXY_ALLOW_PUBLIC ?? "").toLowerCase() === "true";

/** Hop-by-hop headers, plus the ones the platform must set itself. Forwarding a
 *  client's `host` would make the backend generate URLs for the wrong origin,
 *  and forwarding `content-length` after any body handling risks a mismatch. */
const STRIP = new Set([
  "host",
  "connection",
  "keep-alive",
  "transfer-encoding",
  "upgrade",
  "content-length",
  "accept-encoding",
  // Never relay a client's credentials to the backend: the whole design is
  // that authority comes from this function's environment, not from the caller.
  "authorization",
  "x-api-key",
]);

function refuse(status: number, detail: string): Response {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/** Constant-time compare, so a wrong password cannot be found a byte at a time.
 *  Node's timingSafeEqual needs equal lengths, hence the length check first —
 *  which does leak the length, and that is an acceptable trade for a shared
 *  password nobody is brute-forcing character by character. */
function matches(presented: string, expected: string): boolean {
  if (presented.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < presented.length; i++) {
    diff |= presented.charCodeAt(i) ^ expected.charCodeAt(i);
  }
  return diff === 0;
}

export default async (request: Request, _context: Context): Promise<Response> => {
  if (!BACKEND) {
    return refuse(503, "BACKEND_URL is not configured on this site");
  }
  if (!PROXY_PASSWORD && !ALLOW_PUBLIC) {
    return refuse(
      503,
      "this proxy is unconfigured: set PROXY_PASSWORD to require one, or " +
        "PROXY_ALLOW_PUBLIC=true if the site is gated some other way. " +
        "Forwarding anonymous requests with the backend key attached would " +
        "publish the backend, not protect it."
    );
  }
  if (PROXY_PASSWORD) {
    const presented = request.headers.get("x-proxy-password") ?? "";
    if (!matches(presented, PROXY_PASSWORD)) {
      return refuse(401, "missing or invalid proxy password");
    }
  }

  const incoming = new URL(request.url);
  const target = new URL(incoming.pathname + incoming.search, BACKEND);

  const headers = new Headers();
  request.headers.forEach((value, name) => {
    if (!STRIP.has(name.toLowerCase()) && name.toLowerCase() !== "x-proxy-password") {
      headers.set(name, value);
    }
  });
  if (BACKEND_KEY) headers.set("x-api-key", BACKEND_KEY);

  try {
    const response = await fetch(target, {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
      // Required by undici whenever a streaming body is passed through.
      // @ts-expect-error -- duplex is not in the DOM RequestInit type yet.
      duplex: "half",
      redirect: "manual",
    });

    // Passed through verbatim, including the status. Rewriting a backend 401
    // into something friendlier here would hide a misconfigured BACKEND_API_KEY
    // behind a message about the proxy password, which are different faults
    // with different fixes.
    const out = new Headers(response.headers);
    out.delete("content-encoding");
    out.delete("content-length");
    return new Response(response.body, { status: response.status, headers: out });
  } catch (error) {
    // The backend is on a private network. If the site is public and the
    // backend is tailnet-only, every call lands here — say so, because the
    // alternative is a generic 502 that looks like the backend is down.
    return refuse(
      502,
      `cannot reach the backend at ${new URL(BACKEND).origin}: ` +
        `${error instanceof Error ? error.message : String(error)}. ` +
        `Netlify's servers must be able to route to it, which a tailnet-only ` +
        `address does not allow.`
    );
  }
};

export const config: Config = {
  path: "/api/*",
};
