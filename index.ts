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
	if (l.includes("claude-opus"))
		return { contextWindow: 200_000, maxTokens: 32_000 };
	if (l.includes("claude"))
		return { contextWindow: 200_000, maxTokens: 64_000 };
	if (l.includes("gemini-2.5") || l.includes("gemini-3"))
		return { contextWindow: 1_000_000, maxTokens: 65_536 };
	if (l.includes("gemini"))
		return { contextWindow: 1_000_000, maxTokens: 8_192 };
	if (l.includes("gpt-5")) return { contextWindow: 400_000, maxTokens: 16_384 };
	if (l.includes("gpt-4.1"))
		return { contextWindow: 1_000_000, maxTokens: 32_768 };
	if (l.includes("gpt-4o"))
		return { contextWindow: 128_000, maxTokens: 16_384 };
	if (l.includes("o1") || l.includes("o3") || l.includes("o4"))
		return { contextWindow: 200_000, maxTokens: 100_000 };
	if (l.includes("kiro")) return { contextWindow: 200_000, maxTokens: 64_000 };
	if (l.includes("glm")) return { contextWindow: 200_000, maxTokens: 16_384 };
	if (l.includes("qwen") || l.includes("codex"))
		return { contextWindow: 128_000, maxTokens: 8_192 };
	return { contextWindow: 128_000, maxTokens: 8_192 };
}

interface PiModelConfig {
	id: string;
	name: string;
	reasoning: boolean;
	input: ("text" | "image")[];
	cost: { input: 0; output: 0; cacheRead: 0; cacheWrite: 0 };
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

	// Anthropic family — only adaptive-thinking flag needed.
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
		// Claude 4.6 and older use "max" for xhigh; 4.7+ uses "xhigh".
		if (/claude.*4[.-][0-6]/.test(l) || /claude.*[1-3][.-]/.test(l))
			return { xhigh: "max" };
		if (/claude.*4[.-][7-9]/.test(l)) return { xhigh: "xhigh" };
		return undefined;
	}

	if (family === "openai") {
		// o-series and GPT-5 (up to 5.3) + Codex: thinking can't be turned off.
		if (/\bo[1-4]\b/.test(l) || /gpt-5[.-][0-3]/.test(l) || l.includes("codex"))
			return { off: null };
		return undefined;
	}

	return undefined;
}

function toProviderModel(m: CLIProxyListModel, cfg: Config): PiModelConfig {
	const inferred = inferLimits(m.id);
	const contextWindow = cfg.contextOverrides[m.id] ?? inferred.contextWindow;
	const maxTokens = cfg.maxTokensOverrides[m.id] ?? inferred.maxTokens;
	const family = classifyFamily(m);
	const model: PiModelConfig = {
		id: m.id,
		name: m.owned_by ? `${m.id} (${m.owned_by})` : m.id,
		reasoning: inferReasoning(m.id),
		input: inferImageInput(m.id) ? ["text", "image"] : ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow,
		maxTokens,
	};
	const compat = inferCompat(m.id, family);
	if (compat) model.compat = compat;
	const thinkingLevelMap = inferThinkingLevelMap(m.id, family);
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


// Optional import for thinking tag parser
let piAi: any;
try {
	piAi = await import("@earendil-works/pi-ai");
} catch {
	try {
		piAi = await import("@mariozechner/pi-ai");
	} catch {
		// Ignore if pi-ai is not available
	}
}

class ThinkingTagParser {
	tagNames: string[];
	emit: (event: any) => void;
	buffer: string;
	inTag: boolean;
	currentTag: string;

	constructor(tagNames: string[], emit: (event: any) => void) {
		this.tagNames = tagNames;
		this.emit = emit;
		this.buffer = "";
		this.inTag = false;
		this.currentTag = "";
	}

	process(text: string) {
		this.buffer += text;
		while (this.buffer.length > 0) {
			if (!this.inTag) {
				const startIdx = this.buffer.indexOf("<");
				if (startIdx === -1) {
					this.emit({ type: "text", content: this.buffer });
					this.buffer = "";
					return;
				}
				if (startIdx > 0) {
					this.emit({ type: "text", content: this.buffer.substring(0, startIdx) });
					this.buffer = this.buffer.substring(startIdx);
				}
				let matchedTag = null;
				let partialMatch = false;
				for (const tag of this.tagNames) {
					const fullTag = `<${tag}>`;
					if (this.buffer.startsWith(fullTag)) {
						matchedTag = tag;
						break;
					}
					if (fullTag.startsWith(this.buffer)) {
						partialMatch = true;
					}
				}
				if (matchedTag) {
					this.inTag = true;
					this.currentTag = matchedTag;
					this.emit({ type: "think_start" });
					this.buffer = this.buffer.substring(matchedTag.length + 2);
					continue;
				}
				if (partialMatch) {
					return;
				}
				this.emit({ type: "text", content: "<" });
				this.buffer = this.buffer.substring(1);
			} else {
				const endTag = `</${this.currentTag}>`;
				const endIdx = this.buffer.indexOf("</");
				if (endIdx === -1) {
					this.emit({ type: "think", content: this.buffer });
					this.buffer = "";
					return;
				}
				if (endIdx > 0) {
					this.emit({ type: "think", content: this.buffer.substring(0, endIdx) });
					this.buffer = this.buffer.substring(endIdx);
				}
				if (this.buffer.startsWith(endTag)) {
					this.inTag = false;
					this.emit({ type: "think_end" });
					this.buffer = this.buffer.substring(endTag.length);
					if (this.buffer.startsWith("\n")) this.buffer = this.buffer.substring(1);
					continue;
				}
				if (endTag.startsWith(this.buffer)) {
					return;
				}
				this.emit({ type: "think", content: "</" });
				this.buffer = this.buffer.substring(2);
			}
		}
	}
}

function wrapStreamWithThinkingParser(piAi: any) {
	return (model: any, context: any, options: any) => {
		const originalStream = piAi.streamSimple(model, context, options);
		const newStream = piAi.createAssistantMessageEventStream();
		
		(async () => {
			try {
				let outputMsg: any = null;
				let textBlockIndex = -1;
				let thinkBlockIndex = -1;

				const parser = new ThinkingTagParser(["think", "thinking"], (e) => {
					if (!outputMsg) return;
					
					if (e.type === "text") {
						if (textBlockIndex === -1) {
							outputMsg.content.push({ type: "text", text: "" });
							textBlockIndex = outputMsg.content.length - 1;
							newStream.push({ type: "text_start", contentIndex: textBlockIndex, partial: outputMsg });
						}
						outputMsg.content[textBlockIndex].text += e.content;
						newStream.push({ type: "text_delta", contentIndex: textBlockIndex, delta: e.content, partial: outputMsg });
					} else if (e.type === "think_start") {
						outputMsg.content.push({ type: "thinking", thinking: "", thinkingSignature: "" });
						thinkBlockIndex = outputMsg.content.length - 1;
						newStream.push({ type: "thinking_start", contentIndex: thinkBlockIndex, partial: outputMsg });
						textBlockIndex = -1; // Force new text block after thinking
					} else if (e.type === "think") {
						if (thinkBlockIndex !== -1) {
							outputMsg.content[thinkBlockIndex].thinking += e.content;
							newStream.push({ type: "thinking_delta", contentIndex: thinkBlockIndex, delta: e.content, partial: outputMsg });
						}
					} else if (e.type === "think_end") {
						if (thinkBlockIndex !== -1) {
							newStream.push({ type: "thinking_end", contentIndex: thinkBlockIndex, content: outputMsg.content[thinkBlockIndex].thinking, partial: outputMsg });
							thinkBlockIndex = -1;
						}
					}
				});

				for await (const event of originalStream) {
					if (event.type === "start") {
						outputMsg = JSON.parse(JSON.stringify(event.partial));
						outputMsg.content = []; // We will rebuild content
						newStream.push({ type: "start", partial: outputMsg });
					} else if (event.type === "text_start" || event.type === "text_end") {
						// Ignore, parser handles block boundaries
					} else if (event.type === "text_delta") {
						parser.process(event.delta);
					} else if (event.type === "done") {
						newStream.push({ type: "done", reason: event.reason, message: outputMsg });
					} else if (event.type === "error") {
						newStream.push({ type: "error", reason: event.reason, error: outputMsg });
					} else if (event.type.startsWith("toolcall")) {
						// Pass through toolcalls
						if (event.type === "toolcall_start") {
							outputMsg.content.push(event.partial.content[event.contentIndex]);
							newStream.push({ type: "toolcall_start", contentIndex: outputMsg.content.length - 1, partial: outputMsg });
						} else if (event.type === "toolcall_delta") {
							const idx = outputMsg.content.length - 1;
							outputMsg.content[idx].partialJson += event.delta;
							newStream.push({ type: "toolcall_delta", contentIndex: idx, delta: event.delta, partial: outputMsg });
						} else if (event.type === "toolcall_end") {
							const idx = outputMsg.content.length - 1;
							outputMsg.content[idx] = event.toolCall;
							newStream.push({ type: "toolcall_end", contentIndex: idx, toolCall: event.toolCall, partial: outputMsg });
							textBlockIndex = -1; // Reset text block index after toolcall
						}
					} else {
						// Pass through native thinking if the model uses the API reasoning field natively
						newStream.push(event as any);
						if (event.partial) {
							outputMsg = event.partial;
						}
					}
				}
				newStream.end();
			} catch (err) {
				newStream.push({ type: "error", reason: "error", error: { errorMessage: String(err) } as any });
				newStream.end();
			}
		})();
		
		return newStream;
	};
}


async function registerFamilies(
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
		
		if (piAi && piAi.streamSimple && piAi.createAssistantMessageEventStream) {
			providerConfig.streamSimple = wrapStreamWithThinkingParser(piAi);
		}
		
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
				const total = await registerFamilies(pi, cfg, models);
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
