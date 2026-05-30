/**
 * CLIProxyAPIPlus extension for pi-coding-agent.
 *
 * Registers models served by a local/remote CLIProxyAPIPlus instance
 * (https://github.com/router-for-me/CLIProxyAPIPlus) as pi providers.
 *
 * Because pi locks one baseUrl per provider but the Anthropic / OpenAI /
 * Gemini SDKs each expect different path prefixes, this extension registers
 * up to three providers and partitions discovered models by family:
 *
 *   cliproxy        -> Claude/Anthropic models via anthropic-messages  (baseUrl "/")
 *   cliproxy-openai -> OpenAI/Codex/Copilot/etc.  via openai-completions (baseUrl "/v1")
 *   cliproxy-gemini -> Gemini/Google models      via google-generative-ai (baseUrl "/v1beta")
 *
 * Config is read from env vars (CLIPROXY_URL, CLIPROXY_API_KEY) first, then
 * ~/.pi/agent/cliproxy.json ({ "baseUrl": "...", "apiKey": "..." }).
 *
 * A missing API key is tolerated — CLIProxyAPIPlus accepts unauthenticated
 * requests when its own `api-keys:` list is empty. A dummy placeholder key
 * is used internally to satisfy pi's provider validation in that case.
 */

import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type {
	ExtensionAPI,
	ExtensionContext,
	ExtensionCommandContext,
} from "@mariozechner/pi-coding-agent";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface CLIProxyListModel {
	id: string;
	object?: string;
	created?: number;
	owned_by?: string;
}

interface Config {
	baseUrl: string;
	apiKey: string; // may be "" if user hasn't set one
	// Per-model context-window overrides, e.g. { "claude-opus-4-5": 1000000 }.
	// Useful when the proxy doesn't encode long-context variants in the id.
	contextOverrides: Record<string, number>;
	// Per-model maxTokens overrides (optional, same key space as contextOverrides).
	maxTokensOverrides: Record<string, number>;
}

type Family = "anthropic" | "openai" | "gemini";

type Api = "anthropic-messages" | "openai-completions" | "google-generative-ai";

interface FamilySpec {
	family: Family;
	providerName: string;
	api: Api;
	baseSuffix: string; // appended to cfg.baseUrl
}

const FAMILIES: Record<Family, FamilySpec> = {
	anthropic: {
		family: "anthropic",
		providerName: "cliproxy",
		api: "anthropic-messages",
		baseSuffix: "",
	},
	openai: {
		family: "openai",
		providerName: "cliproxy-openai",
		api: "openai-completions",
		baseSuffix: "/v1",
	},
	gemini: {
		family: "gemini",
		providerName: "cliproxy-gemini",
		api: "google-generative-ai",
		baseSuffix: "/v1beta",
	},
};

// pi's validation requires a non-empty apiKey when `models` is set. When the
// user hasn't set one (unauthenticated local proxy), we send this placeholder;
// CLIProxyAPIPlus ignores it when its `api-keys:` list is empty.
const PLACEHOLDER_KEY = "no-key";

// Snapshot of the last-known raw model list; used by /cliproxy-models and
// /cliproxy-status for a nice grouped view.
let lastFetched: CLIProxyListModel[] = [];
let lastCount = 0;

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

function loadConfig(): Config {
	const envUrl = process.env.CLIPROXY_URL?.trim();
	const envKey = process.env.CLIPROXY_API_KEY?.trim();

	let fileBase: string | undefined;
	let fileKey: string | undefined;
	let fileContextOverrides: Record<string, number> = {};
	let fileMaxTokensOverrides: Record<string, number> = {};
	const configPath = join(homedir(), ".pi", "agent", "cliproxy.json");
	if (existsSync(configPath)) {
		try {
			const parsed = JSON.parse(readFileSync(configPath, "utf-8")) as {
				baseUrl?: string;
				apiKey?: string;
				contextOverrides?: Record<string, number>;
				maxTokensOverrides?: Record<string, number>;
			};
			fileBase = parsed.baseUrl?.trim();
			fileKey = parsed.apiKey?.trim();
			if (
				parsed.contextOverrides &&
				typeof parsed.contextOverrides === "object"
			) {
				fileContextOverrides = parsed.contextOverrides;
			}
			if (
				parsed.maxTokensOverrides &&
				typeof parsed.maxTokensOverrides === "object"
			) {
				fileMaxTokensOverrides = parsed.maxTokensOverrides;
			}
		} catch (err) {
			console.warn(
				`[cliproxy] Failed to parse ${configPath}: ${(err as Error).message}`,
			);
		}
	}

	let baseUrl = envUrl || fileBase || "http://localhost:8317";
	// Strip trailing slashes so we can safely append suffixes.
	baseUrl = baseUrl.replace(/\/+$/, "");

	const apiKey = envKey ?? fileKey ?? "";

	// Env-var overrides for quick one-off tweaks:
	//   CLIPROXY_CONTEXT_OVERRIDES="claude-opus-4-5=1000000,claude-sonnet-4-5=1000000"
	const contextOverrides = {
		...fileContextOverrides,
		...parseOverrides(process.env.CLIPROXY_CONTEXT_OVERRIDES),
	};
	const maxTokensOverrides = {
		...fileMaxTokensOverrides,
		...parseOverrides(process.env.CLIPROXY_MAX_TOKENS_OVERRIDES),
	};

	return { baseUrl, apiKey, contextOverrides, maxTokensOverrides };
}

function parseOverrides(raw: string | undefined): Record<string, number> {
	if (!raw) return {};
	const out: Record<string, number> = {};
	for (const pair of raw.split(",")) {
		const [k, v] = pair.split("=").map((s) => s.trim());
		if (!k || !v) continue;
		const n = Number(v);
		if (Number.isFinite(n) && n > 0) out[k] = n;
	}
	return out;
}

// ---------------------------------------------------------------------------
// Model discovery
// ---------------------------------------------------------------------------

async function fetchModels(cfg: Config): Promise<CLIProxyListModel[]> {
	const headers: Record<string, string> = { Accept: "application/json" };
	if (cfg.apiKey) headers.Authorization = `Bearer ${cfg.apiKey}`;

	const res = await fetch(`${cfg.baseUrl}/v1/models`, {
		headers,
		signal: AbortSignal.timeout(10_000),
	});
	if (!res.ok) {
		throw new Error(`HTTP ${res.status} ${res.statusText}`);
	}
	const data = (await res.json()) as { data?: CLIProxyListModel[] };
	if (!data?.data || !Array.isArray(data.data)) {
		throw new Error("Unexpected /v1/models response shape");
	}
	return data.data;
}

// ---------------------------------------------------------------------------
// Model classification + metadata inference
// ---------------------------------------------------------------------------

function classifyFamily(m: CLIProxyListModel): Family {
	const id = m.id.toLowerCase();
	const owner = (m.owned_by ?? "").toLowerCase();

	if (owner.includes("anthropic") || id.includes("claude")) return "anthropic";
	if (
		owner.includes("google") ||
		owner.includes("gemini") ||
		id.includes("gemini")
	)
		return "gemini";
	return "openai";
}

function inferReasoning(id: string): boolean {
	const l = id.toLowerCase();
	return (
		l.includes("claude") ||
		l.includes("gemini") ||
		/\bo1\b|\bo3\b|\bo4\b/.test(l) ||
		l.includes("gpt-5") ||
		l.includes("thinking") ||
		l.includes("reasoning") ||
		l.includes("glm-4") ||
		l.includes("glm-5")
	);
}

function inferImageInput(id: string): boolean {
	const l = id.toLowerCase();
	return (
		l.includes("claude") ||
		l.includes("gemini") ||
		l.includes("gpt-4o") ||
		l.includes("gpt-4.") ||
		l.includes("gpt-5") ||
		l.includes("4o")
	);
}

function inferLimits(id: string): { contextWindow: number; maxTokens: number } {
	const l = id.toLowerCase();

	// Anthropic — Opus 4.5 and earlier have 200k ctx; 4.6+ have 1M ctx and 128k output
	if (/claude.*opus.*4[.-][0-5]/.test(l))
		return { contextWindow: 200_000, maxTokens: 32_000 };
	if (/claude.*opus/.test(l))
		return { contextWindow: 1_000_000, maxTokens: 128_000 };
	if (/claude.*sonnet.*4[.-][0-4]/.test(l))
		return { contextWindow: 200_000, maxTokens: 64_000 };
	if (/claude.*sonnet/.test(l))
		return { contextWindow: 1_000_000, maxTokens: 64_000 };
	if (l.includes("claude"))
		return { contextWindow: 200_000, maxTokens: 64_000 };

	// Google Gemini
	if (/gemini-3\.5/.test(l))
		return { contextWindow: 1_048_576, maxTokens: 65_535 };
	if (/gemini-3\.1.*pro/.test(l))
		return { contextWindow: 1_048_576, maxTokens: 65_536 };
	if (/gemini-3\.1.*flash/.test(l))
		return { contextWindow: 1_048_576, maxTokens: 65_536 };
	if (/gemini-3.*pro/.test(l))
		return { contextWindow: 1_048_576, maxTokens: 65_535 };
	if (/gemini-3.*flash/.test(l))
		return { contextWindow: 1_048_576, maxTokens: 65_535 };
	if (l.includes("gemini-2.5"))
		return { contextWindow: 1_048_576, maxTokens: 65_535 };
	if (l.includes("gemini-2.0-flash"))
		return { contextWindow: 1_048_576, maxTokens: 8_192 };
	if (l.includes("gemini"))
		return { contextWindow: 1_048_576, maxTokens: 8_192 };

	// OpenAI — GPT-5 family
	if (/gpt-5\.5/.test(l))
		return { contextWindow: 1_050_000, maxTokens: 128_000 };
	if (/gpt-5\.4/.test(l))
		return { contextWindow: 1_050_000, maxTokens: 128_000 };
	if (/gpt-5\.2.*pro/.test(l))
		return { contextWindow: 272_000, maxTokens: 128_000 };
	if (/gpt-5\.2/.test(l)) return { contextWindow: 272_000, maxTokens: 128_000 };
	if (l.includes("gpt-5-nano") || l.includes("gpt-5-mini"))
		return { contextWindow: 272_000, maxTokens: 128_000 };
	if (l.includes("gpt-5"))
		return { contextWindow: 272_000, maxTokens: 128_000 };

	// OpenAI — GPT-4
	if (l.includes("gpt-4.1"))
		return { contextWindow: 1_047_576, maxTokens: 32_768 };
	if (l.includes("gpt-4o"))
		return { contextWindow: 128_000, maxTokens: 16_384 };

	// OpenAI — o-series
	if (/\bo[134]/.test(l)) return { contextWindow: 200_000, maxTokens: 100_000 };

	// OpenAI — Codex
	if (l.includes("codex-mini"))
		return { contextWindow: 200_000, maxTokens: 100_000 };
	if (l.includes("codex"))
		return { contextWindow: 272_000, maxTokens: 128_000 };

	// Others
	if (l.includes("kiro")) return { contextWindow: 200_000, maxTokens: 64_000 };
	if (l.includes("glm")) return { contextWindow: 200_000, maxTokens: 16_384 };
	if (l.includes("qwen")) return { contextWindow: 128_000, maxTokens: 8_192 };

	return { contextWindow: 128_000, maxTokens: 8_192 };
}

// Cost per million tokens in USD — matches pi's built-in model definitions.
// Source: LiteLLM model_prices_and_context_window.json (Anthropic direct API prices).
function inferCost(id: string): {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
} {
	const l = id.toLowerCase();

	// ── Anthropic ──────────────────────────────────────────────────────
	// Opus 4.0 / 4.1 (and dated variants like 4-20250514, 4-1-20250805)
	if (/claude.*opus.*4[.-][01]/.test(l))
		return { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 };
	// Opus 4.5 – 4.8+  (same $5/$25 tier)
	if (/claude.*opus/.test(l))
		return { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 };
	// Sonnet 4.x (all variants share $3/$15)
	if (/claude.*sonnet/.test(l))
		return { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 };
	// Haiku 4.5+
	if (/claude.*haiku.*4/.test(l))
		return { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 };
	// Haiku 3.5 and older
	if (/claude.*haiku/.test(l))
		return { input: 0.8, output: 4, cacheRead: 0.08, cacheWrite: 1 };
	// Claude 3 Opus
	if (/claude.*3.*opus/.test(l))
		return { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 };
	// Claude 3/3.5/3.7 Sonnet
	if (/claude.*3/.test(l))
		return { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 };

	// ── OpenAI o-series ────────────────────────────────────────────────
	if (l.includes("o3-pro"))
		return { input: 20, output: 80, cacheRead: 0, cacheWrite: 0 };
	if (/\bo3-mini\b/.test(l))
		return { input: 1.1, output: 4.4, cacheRead: 0.55, cacheWrite: 0 };
	if (/\bo3\b/.test(l))
		return { input: 2, output: 8, cacheRead: 0.5, cacheWrite: 0 };
	if (/\bo4-mini\b/.test(l))
		return { input: 1.1, output: 4.4, cacheRead: 0.275, cacheWrite: 0 };
	if (/\bo1-pro\b/.test(l))
		return { input: 150, output: 600, cacheRead: 0, cacheWrite: 0 };
	if (/\bo1\b/.test(l))
		return { input: 15, output: 60, cacheRead: 7.5, cacheWrite: 0 };

	// ── OpenAI GPT-5 family ────────────────────────────────────────────
	if (/gpt-5\.5-pro/.test(l))
		return { input: 30, output: 180, cacheRead: 0, cacheWrite: 0 };
	if (/gpt-5\.5/.test(l))
		return { input: 5, output: 30, cacheRead: 0, cacheWrite: 0 };
	if (/gpt-5\.4-pro/.test(l))
		return { input: 30, output: 180, cacheRead: 0, cacheWrite: 0 };
	if (/gpt-5\.4-nano/.test(l))
		return { input: 0.2, output: 1.25, cacheRead: 0, cacheWrite: 0 };
	if (/gpt-5\.4-mini/.test(l))
		return { input: 0.75, output: 4.5, cacheRead: 0, cacheWrite: 0 };
	if (/gpt-5\.4/.test(l))
		return { input: 2.5, output: 15, cacheRead: 0, cacheWrite: 0 };
	if (/gpt-5\.3/.test(l))
		return { input: 1.75, output: 14, cacheRead: 0, cacheWrite: 0 };
	if (/gpt-5\.2-pro/.test(l))
		return { input: 21, output: 168, cacheRead: 0, cacheWrite: 0 };
	if (/gpt-5\.2/.test(l))
		return { input: 1.75, output: 14, cacheRead: 0, cacheWrite: 0 };
	if (l.includes("gpt-5-pro"))
		return { input: 15, output: 120, cacheRead: 0, cacheWrite: 0 };
	if (l.includes("gpt-5-nano"))
		return { input: 0.05, output: 0.4, cacheRead: 0, cacheWrite: 0 };
	if (l.includes("gpt-5-mini"))
		return { input: 0.25, output: 2, cacheRead: 0, cacheWrite: 0 };
	if (l.includes("gpt-5"))
		return { input: 1.25, output: 10, cacheRead: 0.125, cacheWrite: 0 };

	// ── OpenAI GPT-4 family ────────────────────────────────────────────
	if (l.includes("gpt-4.1-nano"))
		return { input: 0.1, output: 0.4, cacheRead: 0.025, cacheWrite: 0 };
	if (l.includes("gpt-4.1-mini"))
		return { input: 0.4, output: 1.6, cacheRead: 0.1, cacheWrite: 0 };
	if (l.includes("gpt-4.1"))
		return { input: 2, output: 8, cacheRead: 0.5, cacheWrite: 0 };
	if (l.includes("gpt-4o-mini"))
		return { input: 0.15, output: 0.6, cacheRead: 0.075, cacheWrite: 0 };
	if (l.includes("gpt-4o"))
		return { input: 2.5, output: 10, cacheRead: 1.25, cacheWrite: 0 };
	if (l.includes("gpt-4-turbo"))
		return { input: 10, output: 30, cacheRead: 0, cacheWrite: 0 };
	if (l.includes("gpt-4"))
		return { input: 30, output: 60, cacheRead: 0, cacheWrite: 0 };

	// ── OpenAI Codex ───────────────────────────────────────────────────
	if (l.includes("codex-mini"))
		return { input: 1.5, output: 6, cacheRead: 0, cacheWrite: 0 };
	if (l.includes("codex"))
		return { input: 1.25, output: 10, cacheRead: 0.125, cacheWrite: 0 };

	// ── Google Gemini ──────────────────────────────────────────────────
	if (/gemini-3\.5-flash/.test(l))
		return { input: 1.5, output: 9, cacheRead: 0.15, cacheWrite: 0 };
	if (/gemini-3\.1.*pro/.test(l))
		return { input: 2, output: 12, cacheRead: 0.2, cacheWrite: 0 };
	if (/gemini-3\.1.*flash-lite/.test(l))
		return { input: 0.25, output: 1.5, cacheRead: 0.025, cacheWrite: 0 };
	if (/gemini-3\.1.*flash/.test(l))
		return { input: 0.25, output: 1.5, cacheRead: 0.025, cacheWrite: 0 };
	if (/gemini-3.*pro/.test(l))
		return { input: 2, output: 12, cacheRead: 0.2, cacheWrite: 0 };
	if (/gemini-3.*flash/.test(l))
		return { input: 0.5, output: 3, cacheRead: 0.05, cacheWrite: 0 };
	if (l.includes("gemini-2.5-pro"))
		return { input: 1.25, output: 10, cacheRead: 0.125, cacheWrite: 0 };
	if (l.includes("gemini-2.5-flash-lite"))
		return { input: 0.1, output: 0.4, cacheRead: 0.01, cacheWrite: 0 };
	if (l.includes("gemini-2.5-flash"))
		return { input: 0.3, output: 2.5, cacheRead: 0.03, cacheWrite: 0 };
	if (l.includes("gemini-2.0-flash-lite"))
		return { input: 0.075, output: 0.3, cacheRead: 0.019, cacheWrite: 0 };
	if (l.includes("gemini-2.0-flash"))
		return { input: 0.1, output: 0.4, cacheRead: 0.025, cacheWrite: 0 };
	if (l.includes("gemini"))
		return { input: 0.1, output: 0.4, cacheRead: 0.025, cacheWrite: 0 };

	// ── Others ─────────────────────────────────────────────────────────
	if (l.includes("glm"))
		return { input: 2.25, output: 2.75, cacheRead: 0, cacheWrite: 0 };
	if (l.includes("qwen"))
		return { input: 0.22, output: 0.88, cacheRead: 0, cacheWrite: 0 };

	// Fallback — mid-range estimate
	return { input: 1, output: 4, cacheRead: 0.1, cacheWrite: 0 };
}

interface PiModelConfig {
	id: string;
	name: string;
	reasoning: boolean;
	input: ("text" | "image")[];
	cost: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
	};
	contextWindow: number;
	maxTokens: number;
	compat?: Record<string, unknown>;
	thinkingLevelMap?: Record<string, string | null>;
}

// ---------------------------------------------------------------------------
// Per-family compat / thinkingLevelMap inference
//
// pi's built-in providers carry detailed compat flags that control how
// reasoning is requested and streamed.  Because CLIProxy translates
// upstream Responses-API events into Chat Completions, we must tell pi
// the right flags so reasoning_effort is sent and thinking tokens are
// recognised.
// ---------------------------------------------------------------------------

/** Claude 4.6+ requires thinking.type="adaptive" instead of "enabled". */
function needsAdaptiveThinking(id: string): boolean {
	const l = id.toLowerCase();
	return (
		/claude.*4[.-][6-9]/.test(l) ||
		/claude.*4[.-]\d{2,}/.test(l) ||
		/claude.*5[.-]/.test(l)
	);
}

function inferCompat(
	id: string,
	family: Family,
): Record<string, unknown> | undefined {
	const l = id.toLowerCase();

	// Anthropic family — Claude 4.6+ requires adaptive thinking.
	if (family === "anthropic") {
		if (needsAdaptiveThinking(id)) return { forceAdaptiveThinking: true };
		return undefined;
	}

	// OpenAI family — CLIProxy's /v1/chat/completions translation means pi
	// talks openai-completions.  We need supportsReasoningEffort so pi sends
	// the reasoning_effort param that CLIProxy forwards upstream.
	if (family === "openai") {
		if (/\bo[1-4]\b/.test(l) || l.includes("gpt-5") || l.includes("codex")) {
			return { supportsReasoningEffort: true };
		}
		return undefined;
	}

	// Gemini — no special compat needed; google-generative-ai handles it.
	return undefined;
}

function inferThinkingLevelMap(
	id: string,
	family: Family,
): Record<string, string | null> | undefined {
	const l = id.toLowerCase();

	if (family === "anthropic") {
		// CLIProxy only accepts low/medium/high/xhigh, so hide pi's minimal level.
		// Claude 4.5 and older use "max" for xhigh; 4.6+ uses "xhigh".
		if (/claude.*4[.-][0-5]/.test(l) || /claude.*[1-3][.-]/.test(l))
			return { minimal: null, xhigh: "max" };
		if (/claude.*4[.-][6-9]/.test(l)) return { minimal: null, xhigh: "xhigh" };
		return { minimal: null };
	}

	if (family === "openai") {
		// CLIProxy only accepts low/medium/high/xhigh, so hide pi's minimal level.
		// o-series and GPT-5 (up to 5.3) + Codex: thinking can't be turned off.
		if (/\bo[1-4]\b/.test(l) || /gpt-5[.-][0-3]/.test(l) || l.includes("codex"))
			return { minimal: null, off: null };
		return { minimal: null };
	}

	if (family === "gemini") {
		// CLIProxy only accepts low/medium/high/xhigh, so hide pi's minimal level.
		return { minimal: null };
	}

	return undefined;
}

function toProviderModel(m: CLIProxyListModel, cfg: Config): PiModelConfig {
	const inferred = inferLimits(m.id);
	const family = classifyFamily(m);
	const contextWindow = cfg.contextOverrides[m.id] ?? inferred.contextWindow;
	const maxTokens = cfg.maxTokensOverrides[m.id] ?? inferred.maxTokens;
	const compat = inferCompat(m.id, family);
	const thinkingLevelMap = inferThinkingLevelMap(m.id, family);
	const cost = inferCost(m.id);
	const model: PiModelConfig = {
		id: m.id,
		name: m.owned_by ? `${m.id} (${m.owned_by})` : m.id,
		reasoning: inferReasoning(m.id),
		input: inferImageInput(m.id) ? ["text", "image"] : ["text"],
		cost,
		contextWindow,
		maxTokens,
	};
	if (compat) model.compat = compat;
	if (thinkingLevelMap) model.thinkingLevelMap = thinkingLevelMap;
	return model;
}

// ---------------------------------------------------------------------------
// Fallback model list (used when the proxy is unreachable at startup)
// ---------------------------------------------------------------------------

function fallbackModels(): CLIProxyListModel[] {
	return [
		{ id: "claude-opus-4-5", owned_by: "anthropic" },
		{ id: "claude-sonnet-4-5", owned_by: "anthropic" },
		{ id: "gemini-2.5-pro", owned_by: "google" },
		{ id: "gemini-2.5-flash", owned_by: "google" },
		{ id: "gpt-5-codex", owned_by: "openai" },
		{ id: "gpt-4o", owned_by: "openai" },
		{ id: "gpt-4o-mini", owned_by: "openai" },
	];
}

// ---------------------------------------------------------------------------
// Provider registration
// ---------------------------------------------------------------------------

function registerFamilies(
	pi: ExtensionAPI,
	cfg: Config,
	rawModels: CLIProxyListModel[],
): number {
	// Partition models by family.
	const buckets: Record<Family, PiModelConfig[]> = {
		anthropic: [],
		openai: [],
		gemini: [],
	};
	for (const m of rawModels) {
		buckets[classifyFamily(m)].push(toProviderModel(m, cfg));
	}

	// The apiKey pi receives; we never set authHeader so pi won't add its own
	// Bearer header — the underlying SDK (Anthropic/OpenAI/Google) sends auth
	// natively using this value. CLIProxyAPIPlus accepts any value when its
	// `api-keys:` is empty, so a placeholder works for unauthenticated setups.
	const effectiveKey = cfg.apiKey || PLACEHOLDER_KEY;

	let total = 0;
	for (const family of Object.keys(buckets) as Family[]) {
		const spec = FAMILIES[family];
		const models = buckets[family];
		if (models.length === 0) {
			// Nothing to register for this family. Unregister any stale
			// registration from a previous refresh.
			try {
				pi.unregisterProvider(spec.providerName);
			} catch {
				/* no-op if not registered */
			}
			continue;
		}

		const providerConfig: any = {
			baseUrl: cfg.baseUrl + spec.baseSuffix,
			apiKey: effectiveKey,
			api: spec.api,
			models,
		};

		pi.registerProvider(spec.providerName, providerConfig);
		total += models.length;
	}

	return total;
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

function notify(
	ctx: ExtensionContext | ExtensionCommandContext,
	msg: string,
	kind: "info" | "success" | "error" | "warning" = "info",
) {
	if ((ctx as ExtensionContext).hasUI) {
		(ctx as ExtensionContext).ui.notify(msg, kind as any);
	} else {
		// Headless: map to a sensible stream.
		if (kind === "error") console.error(`[cliproxy] ${msg}`);
		else console.log(`[cliproxy] ${msg}`);
	}
}

function groupByOwner(models: CLIProxyListModel[]): Record<string, string[]> {
	const out: Record<string, string[]> = {};
	for (const m of models) {
		const owner = m.owned_by || "unknown";
		(out[owner] ||= []).push(m.id);
	}
	for (const owner of Object.keys(out)) out[owner].sort();
	return out;
}

function registerCommands(pi: ExtensionAPI, cfg: Config) {
	pi.registerCommand("cliproxy-status", {
		description: "Ping CLIProxyAPIPlus and report model count",
		handler: async (_args, ctx) => {
			try {
				const models = await fetchModels(cfg);
				lastFetched = models;
				lastCount = models.length;
				const auth = cfg.apiKey ? "with API key" : "no API key";
				notify(
					ctx,
					`CLIProxy OK — ${models.length} models @ ${cfg.baseUrl} (${auth})`,
					"success",
				);
				if (!ctx.hasUI) {
					const grouped = groupByOwner(models);
					for (const [owner, ids] of Object.entries(grouped)) {
						console.log(`  ${owner}: ${ids.join(", ")}`);
					}
				}
			} catch (err) {
				notify(ctx, `CLIProxy error: ${(err as Error).message}`, "error");
			}
		},
	});

	pi.registerCommand("cliproxy-models", {
		description: "List all available CLIProxyAPIPlus models grouped by owner",
		handler: async (_args, ctx) => {
			try {
				const models = await fetchModels(cfg);
				lastFetched = models;
				lastCount = models.length;
				const grouped = groupByOwner(models);
				const lines = Object.entries(grouped)
					.map(([owner, ids]) => `${owner}:\n  ${ids.join("\n  ")}`)
					.join("\n\n");
				if (ctx.hasUI) {
					ctx.ui.notify(
						`${models.length} models (see console for full list)`,
						"info",
					);
					console.log(`\nCLIProxy models:\n${lines}\n`);
				} else {
					console.log(`CLIProxy models:\n${lines}`);
				}
			} catch (err) {
				notify(
					ctx,
					`CLIProxy models failed: ${(err as Error).message}`,
					"error",
				);
			}
		},
	});

	pi.registerCommand("cliproxy-refresh", {
		description:
			"Re-fetch the CLIProxyAPIPlus model list and re-register providers",
		handler: async (_args, ctx) => {
			try {
				const models = await fetchModels(cfg);
				lastFetched = models;
				lastCount = models.length;
				const total = registerFamilies(pi, cfg, models);
				notify(
					ctx,
					`CLIProxy: refreshed ${total} models across ${new Set(models.map(classifyFamily)).size} providers`,
					"success",
				);
			} catch (err) {
				notify(
					ctx,
					`CLIProxy refresh failed: ${(err as Error).message}`,
					"error",
				);
			}
		},
	});
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

export default async function (pi: ExtensionAPI): Promise<void> {
	const cfg = loadConfig();
	let initError: string | undefined;

	let models: CLIProxyListModel[];
	try {
		models = await fetchModels(cfg);
	} catch (err) {
		initError = (err as Error).message;
		console.warn(
			`[cliproxy] Could not reach CLIProxyAPIPlus at ${cfg.baseUrl}: ${initError}. ` +
				`Using fallback model list; run /cliproxy-refresh once the proxy is up.`,
		);
		models = fallbackModels();
	}

	lastFetched = models;
	lastCount = models.length;

	registerFamilies(pi, cfg, models);
	registerCommands(pi, cfg);

	pi.on("session_start", async (_event, ctx) => {
		if (!ctx.hasUI) return;
		if (initError) {
			ctx.ui.notify(
				`CLIProxy unreachable (${initError}). Loaded ${lastCount} fallback models — /cliproxy-refresh to retry.`,
				"warning",
			);
		} else {
			ctx.ui.notify(`CLIProxy: ${lastCount} models available`, "info");
		}
	});
}
