"""Frontend design generator — NEXUS synthesis tool.

This tool is a thin synthesis layer in the NEXUS agent pipeline:

  Agent receives task
    → mcp_shadcn_search_items_in_registries     (shadcn MCP tool)
    → mcp_shadcn_get_item_examples_from_registries  (shadcn MCP tool)
    → mcp_shadcn_get_add_command_for_items      (shadcn MCP tool)
    → generate_frontend_design(prompt, component_examples=...) ← THIS TOOL
        · LLM synthesises domain-specific TSX + preview data spec
        · JS sandbox renders self-contained interactive HTML
    → returns tsx_code, html_preview, install_command, file_structure

No hardcoded templates, schemas, or fallback data.
The LLM generates everything from the prompt + real MCP registry content.
"""

from __future__ import annotations

import json
import re
from typing import Any

from nexus.exceptions import ToolError
from nexus.types import ToolDefinition, RiskLevel
from nexus.tools.plugin import _registered_tools

# ── Component routing (configuration, not generated content) ─────────────────
# Maps app-type archetypes → shadcn component names the agent should request.

_APP_COMPONENTS: dict[str, list[str]] = {
    "dashboard": ["card", "chart", "badge", "table", "button", "progress", "sidebar", "avatar"],
    "landing":   ["button", "card", "badge", "separator", "navigation-menu", "avatar", "carousel"],
    "form":      ["form", "input", "button", "label", "select", "checkbox", "card", "alert"],
    "crud":      ["table", "dialog", "button", "badge", "input", "dropdown-menu", "pagination", "card"],
    "auth":      ["card", "input", "button", "label", "form", "separator", "alert"],
    "settings":  ["card", "form", "input", "button", "switch", "select", "separator", "badge"],
    "kanban":    ["card", "badge", "button", "scroll-area", "dialog", "dropdown-menu"],
    "inbox":     ["card", "badge", "button", "avatar", "scroll-area", "separator", "input"],
}

_KEYWORD_HINTS: dict[str, str] = {
    "dashboard": "dashboard", "analytics": "dashboard", "metrics": "dashboard", "chart": "dashboard",
    "stats": "dashboard", "kpi": "dashboard", "revenue": "dashboard",
    "landing": "landing", "marketing": "landing", "homepage": "landing", "hero": "landing",
    "form": "form", "survey": "form", "wizard": "form", "onboarding": "form",
    "crud": "crud", "invoice": "crud", "table": "crud", "list": "crud", "admin": "crud",
    "data": "crud", "manage": "crud", "inventory": "crud",
    "login": "auth", "signup": "auth", "register": "auth", "auth": "auth",
    "settings": "settings", "profile": "settings", "preferences": "settings",
    "kanban": "kanban", "board": "kanban", "task": "kanban", "project": "kanban",
    "inbox": "inbox", "email": "inbox", "messages": "inbox", "chat": "inbox",
}


def _detect_app_type(prompt: str) -> str:
    """Classify the prompt into an archetype using whole-word matching."""
    words = set(re.findall(r'\b\w+\b', prompt.lower()))
    for keyword, app_type in _KEYWORD_HINTS.items():
        if keyword in words:
            return app_type
    return "dashboard"


def _resolve_components(prompt: str, app_type: str, extra: list[str] | None) -> list[str]:
    """Return the component list for this archetype, prepending any caller-specified extras."""
    base = list(_APP_COMPONENTS.get(app_type, _APP_COMPONENTS["dashboard"]))
    if extra:
        for c in extra:
            if c not in base:
                base.insert(0, c)
    return base[:8]


# ── TSX post-processing ───────────────────────────────────────────────────────

def _ensure_shadcn_imports(tsx: str, components: list[str]) -> str:
    """Inject missing shadcn/ui import statements.

    Small local models often omit imports entirely. We know the component list,
    so we deterministically add any imports that are missing.
    """
    if "@/components/ui/" in tsx:
        return tsx  # already correct
    imports = []
    for comp in components:
        pascal = "".join(w.capitalize() for w in comp.split("-"))
        if pascal in tsx:
            imports.append(f"import {{ {pascal} }} from '@/components/ui/{comp}';")
    return ("\n".join(imports) + "\n\n" + tsx) if imports else tsx


# ── LLM synthesis ─────────────────────────────────────────────────────────────

async def _llm_generate_design(
    prompt: str,
    app_type: str,
    components: list[str],
    component_examples: str,
    model: str,
) -> tuple[str, dict]:
    """Call the LLM to generate TSX code + a data schema for the HTML renderer.

    Returns (tsx_code: str, schema: dict).
    Raises ToolError on auth/rate-limit failures, or if the response is unparseable.
    The schema is the single source of truth for all preview content — no defaults
    or hardcoded data are injected here.
    """
    import litellm  # type: ignore

    words = [w.capitalize() for w in prompt.split()[:3] if w.isalpha()]
    component_name = "".join(words) + "Page" if words else "GeneratedPage"

    system = (
        "You are an expert React/TypeScript developer and UI designer.\n"
        "Return a JSON object with exactly two keys:\n\n"
        f'  "tsx": A complete, runnable TSX component. '
        f"MUST be declared as: `export default function {component_name}() {{` "
        "Use realistic domain-specific data — no lorem ipsum, no placeholder text.\n\n"
        '  "schema": A JSON object that drives the interactive HTML preview:\n'
        f"    title          — string, page title (must include words from: {prompt!r})\n"
        "    subtitle       — string, one-line description\n"
        "    primary_action — string, main CTA button label\n"
        "    columns        — array of 4-6 column header strings, domain-relevant\n"
        "    rows           — array of 5-6 objects:\n"
        '                       { cells: string[], status: "active"|"pending"|"completed"|"failed" }\n'
        "                     The status field MUST match one of the filter_options values exactly.\n"
        "    stats          — array of 3-4 objects: { title, value, change, positive: bool }\n"
        "    filter_options — array of status strings matching the status values used in rows\n"
        "    sidebar_items  — array of 5 objects: { icon: emoji, label: string }\n\n"
        "ALL content must be specific to the user's request domain.\n"
        "Return ONLY valid JSON. No markdown fences. No explanation."
    )

    user = (
        f"Component name: {component_name}\n"
        f"App type: {app_type}\n"
        f"shadcn/ui components: {', '.join(components)}\n"
        f"Request: {prompt}"
    )
    if component_examples:
        user += (
            f"\n\nReal shadcn component examples from the MCP registry "
            f"(use for import structure and component API):\n{component_examples[:1500]}"
        )

    from nexus.llm.client import is_local_model
    from nexus.config import config as _cfg
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": 2500,
    }
    if is_local_model(model):
        kwargs["api_base"] = _cfg.ollama_base_url

    last_err: Exception | None = None
    for attempt in range(2):
        raw = ""
        try:
            resp = await litellm.acompletion(**kwargs)
            raw = resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            continue  # retry once on LLM call failure

        # Strip markdown fences if the model wrapped the JSON
        candidate = raw
        if "```" in raw:
            for part in raw.split("```")[1::2]:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    data = json.loads(part)
                    tsx = _ensure_shadcn_imports(data.get("tsx", ""), components)
                    return tsx, data.get("schema", {})
                except json.JSONDecodeError:
                    continue
        try:
            data = json.loads(candidate)
            tsx = _ensure_shadcn_imports(data.get("tsx", ""), components)
            return tsx, data.get("schema", {})
        except json.JSONDecodeError as e:
            last_err = e
            continue  # retry once on malformed JSON

    raise ToolError(
        f"LLM failed after 2 attempts ({type(last_err).__name__}): {last_err}. "
        "Check your LLM: cloud (ANTHROPIC_API_KEY/OPENAI_API_KEY) "
        "or local (NEXUS_DEFAULT_LLM_MODEL=ollama/qwen2.5-coder:7b with Ollama running)."
    ) from last_err


# ── HTML preview renderer (data-driven JS, executed in the Phase 21 sandbox) ──
# Content is 100% driven by the schema from _llm_generate_design.
# Layout structure (sidebar/full-page/auth) is determined by app_type.

_HTML_GEN_JS = r"""
const spec = JSON.parse(process.env.NEXUS_INPUT);
const { prompt, appType, colorScheme, schema } = spec;
const {
  title = prompt,
  subtitle = '',
  primary_action: action = 'New',
  columns = [],
  rows = [],
  stats = [],
  filter_options: filters = [],
  sidebar_items: nav = [],
} = schema;

const colors = {
  zinc:   { primary: '#18181b', accent: '#3f3f46', bg: '#fafafa', card: '#ffffff', border: '#e4e4e7', text: '#09090b', muted: '#71717a' },
  slate:  { primary: '#0f172a', accent: '#334155', bg: '#f8fafc', card: '#ffffff', border: '#e2e8f0', text: '#0f172a', muted: '#64748b' },
  blue:   { primary: '#1d4ed8', accent: '#3b82f6', bg: '#f0f9ff', card: '#ffffff', border: '#bae6fd', text: '#0c4a6e', muted: '#6b7280' },
  green:  { primary: '#15803d', accent: '#22c55e', bg: '#f0fdf4', card: '#ffffff', border: '#bbf7d0', text: '#14532d', muted: '#6b7280' },
  violet: { primary: '#7c3aed', accent: '#8b5cf6', bg: '#faf5ff', card: '#ffffff', border: '#ddd6fe', text: '#2e1065', muted: '#6b7280' },
};
const theme = colors[colorScheme] || colors.zinc;

function badge(text, variant = 'default') {
  const bg    = variant === 'success' ? '#dcfce7' : variant === 'warning' ? '#fef9c3' : variant === 'destructive' ? '#fee2e2' : '#f4f4f5';
  const color = variant === 'success' ? '#15803d' : variant === 'warning' ? '#854d0e' : variant === 'destructive' ? '#b91c1c' : '#3f3f46';
  return `<span style="display:inline-flex;align-items:center;border-radius:9999px;padding:2px 10px;font-size:12px;font-weight:500;background:${bg};color:${color}">${text}</span>`;
}

function card(content, cardTitle = '') {
  return `<div style="background:${theme.card};border:1px solid ${theme.border};border-radius:12px;padding:24px;box-shadow:0 1px 3px rgba(0,0,0,0.05)">
    ${cardTitle ? `<h3 style="font-size:14px;font-weight:600;color:${theme.muted};text-transform:uppercase;letter-spacing:0.05em;margin:0 0 16px">${cardTitle}</h3>` : ''}
    ${content}
  </div>`;
}

function statCard(t, value, change, changeType) {
  const arrow = changeType === 'positive' ? '↑' : '↓';
  const clr   = changeType === 'positive' ? '#15803d' : '#b91c1c';
  return card(`
    <div style="font-size:28px;font-weight:700;color:${theme.text};margin-bottom:8px">${value}</div>
    <div style="font-size:13px;color:${clr}">${arrow} ${change}</div>
  `, t);
}

function sidebarItem(icon, label, active) {
  const bg    = active ? theme.accent + '20' : 'transparent';
  const color = active ? theme.primary : theme.muted;
  return `<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:8px;background:${bg};cursor:pointer;font-size:14px;color:${color};font-weight:${active?'600':'400'}">
    <span style="font-size:18px">${icon}</span>${label}
  </div>`;
}

// All status values the LLM puts in active/active-like positions → green badge.
// Everything else: pending → yellow, anything else → red.
const statusVariant = s => {
  const s_ = (s || '').toLowerCase();
  if (['completed','active','paid','approved','resolved','done','open','published'].includes(s_)) return 'success';
  if (['pending','processing','in_progress','review','draft'].includes(s_)) return 'warning';
  return 'destructive';
};

function buildTable() {
  if (!columns.length) return '';
  const headerRow = `<tr style="background:${theme.bg}">${columns.map(c =>
    `<td style="padding:12px 16px;font-size:14px;font-weight:600;color:${theme.muted};border-bottom:1px solid ${theme.border}">${c}</td>`
  ).join('')}</tr>`;
  const dataRows = rows.map(row => {
    const cells = row.cells.map(c =>
      c.toLowerCase() === (row.status || '').toLowerCase()
        ? badge(c, statusVariant(row.status))
        : c
    );
    return `<tr data-nxs="${row.cells.join(' ').toLowerCase()}" data-status="${row.status}" style="background:${theme.card}">
      ${cells.map(c => `<td style="padding:12px 16px;font-size:14px;border-bottom:1px solid ${theme.border}">${c}</td>`).join('')}
    </tr>`;
  }).join('');
  return card(`<table style="width:100%;border-collapse:collapse">${headerRow}${dataRows}</table>`);
}

function buildFilters() {
  if (!filters.length) return '';
  const opts = filters.map(f => `<option value="${f}">${f.charAt(0).toUpperCase()+f.slice(1)}</option>`).join('');
  return `<div style="display:flex;gap:12px;margin-bottom:20px">
    <input id="nxs-q" placeholder="Search..." oninput="nxsFilter()" style="padding:8px 12px;border:1px solid ${theme.border};border-radius:8px;font-size:14px;width:260px;outline:none;background:${theme.card}"/>
    <select id="nxs-f" onchange="nxsFilter()" style="padding:8px 12px;border:1px solid ${theme.border};border-radius:8px;font-size:14px;background:${theme.card}">
      <option value="">All</option>${opts}
    </select>
  </div>`;
}

function buildStats() {
  if (!stats.length) return '';
  return `<div style="display:grid;grid-template-columns:repeat(${Math.min(stats.length,4)},1fr);gap:16px;margin-bottom:24px">
    ${stats.map(s => statCard(s.title, s.value, s.change, s.positive ? 'positive' : 'negative')).join('')}
  </div>`;
}

function buildHeader() {
  return `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px">
    <div>
      <h1 style="font-size:24px;font-weight:700;color:${theme.text};margin:0">${title}</h1>
      ${subtitle ? `<p style="font-size:14px;color:${theme.muted};margin:4px 0 0">${subtitle}</p>` : ''}
    </div>
    <button style="background:${theme.primary};color:#fff;border:none;border-radius:8px;padding:10px 20px;font-size:14px;font-weight:500;cursor:pointer">${action}</button>
  </div>`;
}

let body;
if (appType === 'auth') {
  body = `<div style="min-height:100vh;display:flex;align-items:center;justify-content:center;background:${theme.bg};font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
    <div style="width:400px">
      <div style="text-align:center;margin-bottom:32px">
        <div style="width:48px;height:48px;background:${theme.primary};border-radius:12px;margin:0 auto 16px;display:flex;align-items:center;justify-content:center;font-size:24px">⚡</div>
        <h1 style="font-size:24px;font-weight:700;color:${theme.text};margin:0 0 8px">${title}</h1>
        <p style="font-size:14px;color:${theme.muted};margin:0">${subtitle || 'Sign in to your account to continue'}</p>
      </div>
      ${card(`
        <div style="margin-bottom:16px">
          <label style="display:block;font-size:14px;font-weight:500;color:${theme.text};margin-bottom:6px">Email</label>
          <input type="email" placeholder="you@company.com" style="width:100%;padding:10px 12px;border:1px solid ${theme.border};border-radius:8px;font-size:14px;outline:none;box-sizing:border-box"/>
        </div>
        <div style="margin-bottom:20px">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <label style="font-size:14px;font-weight:500;color:${theme.text}">Password</label>
            <a href="#" style="font-size:13px;color:${theme.primary};text-decoration:none">Forgot password?</a>
          </div>
          <input type="password" placeholder="••••••••" style="width:100%;padding:10px 12px;border:1px solid ${theme.border};border-radius:8px;font-size:14px;outline:none;box-sizing:border-box"/>
        </div>
        <button style="width:100%;padding:11px;background:${theme.primary};color:#fff;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;margin-bottom:16px">Sign In</button>
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
          <div style="flex:1;height:1px;background:${theme.border}"></div>
          <span style="font-size:12px;color:${theme.muted}">or</span>
          <div style="flex:1;height:1px;background:${theme.border}"></div>
        </div>
        <button style="width:100%;padding:11px;background:${theme.card};color:${theme.text};border:1px solid ${theme.border};border-radius:8px;font-size:14px;font-weight:500;cursor:pointer">🔵 Continue with Google</button>
      `)}
    </div>
  </div>`;
} else if (appType === 'dashboard' && nav.length) {
  const sidebarHtml = `<div style="width:240px;background:${theme.card};border-right:1px solid ${theme.border};padding:20px;display:flex;flex-direction:column;gap:4px;flex-shrink:0">
    <div style="font-size:18px;font-weight:700;color:${theme.text};padding:8px 12px;margin-bottom:12px">${title}</div>
    ${nav.map((item, i) => sidebarItem(item.icon, item.label, i === 0)).join('')}
  </div>`;
  body = `<div style="display:flex;height:100vh;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:${theme.bg}">
    ${sidebarHtml}
    <div style="flex:1;overflow:auto;padding:32px">
      ${buildHeader()}${buildStats()}${buildFilters()}${buildTable()}
    </div>
  </div>`;
} else {
  body = `<div style="padding:32px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:${theme.bg};min-height:100vh">
    ${buildHeader()}${buildStats()}${buildFilters()}${buildTable()}
  </div>`;
}

const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>${title} — shadcn/ui Preview</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}</style>
</head>
<body>${body}<script>
function nxsFilter(){
  const q=(document.getElementById('nxs-q')||{value:''}).value.toLowerCase();
  const f=(document.getElementById('nxs-f')||{value:''}).value;
  document.querySelectorAll('[data-nxs]').forEach(r=>{
    r.style.display=(r.dataset.nxs.includes(q)&&(!f||r.dataset.status===f))?'':'none';
  });
}
</script></body>
</html>`;

console.log(JSON.stringify({
  html,
  components_used: spec.components,
  app_type: appType,
  install_command: `npx shadcn@latest add ${spec.components.join(' ')}`,
}));
"""


# ── Main tool function ─────────────────────────────────────────────────────────

async def generate_frontend_design(
    prompt: str,
    components: list[str] | None = None,
    app_type: str | None = None,
    color_scheme: str = "zinc",
    add_command: str | None = None,
    component_examples: str | None = None,
    output_type: str = "html",
) -> dict[str, Any]:
    """Generate a frontend UI from a natural language prompt.

    Intended agent pipeline (call shadcn MCP tools FIRST, then pass results here):
      1. mcp_shadcn_search_items_in_registries(query=...) → component names
      2. mcp_shadcn_get_item_examples_from_registries(...)  → pass as component_examples
      3. mcp_shadcn_get_add_command_for_items(...)           → pass as add_command
      4. generate_frontend_design(prompt, component_examples=..., add_command=...)

    The LLM synthesises domain-specific TSX code and a data schema from the prompt
    + MCP examples.  Requires a configured LLM: cloud API key or local Ollama model.

    Args:
        prompt: Natural language description of the UI (e.g. "SaaS invoice dashboard").
        components: shadcn/ui component names — auto-selected from app_type if omitted.
        app_type: dashboard | crud | auth | landing | form | settings | kanban | inbox.
        color_scheme: zinc | slate | blue | green | violet.
        add_command: Install command from mcp_shadcn_get_add_command_for_items.
        component_examples: TSX examples from mcp_shadcn_get_item_examples_from_registries.
        output_type: "html" (interactive preview) | "nextjs" (Next.js 15 scaffold).

    Returns:
        tsx_code, html_preview, components_used, install_command, file_structure
    """
    import os
    from nexus.tools.sandbox_v2 import CodeSandbox
    from nexus.config import config as _cfg

    if not prompt or not prompt.strip():
        raise ToolError("prompt is required")

    resolved_type = app_type or _detect_app_type(prompt)
    resolved_components = _resolve_components(prompt, resolved_type, components)

    # Require an LLM — cloud API key or a local model (e.g. Ollama)
    from nexus.llm.client import is_local_model, select_model, TASK_CODE
    has_llm = bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or getattr(_cfg, "llm_api_key", None)
        or is_local_model(select_model(TASK_CODE))
    )
    if not has_llm:
        raise ToolError(
            "generate_frontend_design requires an LLM to synthesise domain-specific code and "
            "preview data. Options:\n"
            "  Cloud: set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env\n"
            "  Local: set NEXUS_DEFAULT_LLM_MODEL=ollama/qwen2.5-coder:7b (Ollama must be running)\n"
            "The agent pipeline: call shadcn MCP tools first, then pass component_examples here."
        )

    from nexus.llm.client import select_model, TASK_CODE
    tsx_code, schema = await _llm_generate_design(
        prompt, resolved_type, resolved_components,
        component_examples or "", select_model(TASK_CODE),
    )

    # Guarantee schema has a title (LLM rarely omits it, but be defensive)
    schema.setdefault("title", prompt.title()[:50])

    # Generate interactive HTML preview via the Phase 21 JS sandbox
    sandbox = CodeSandbox(_cfg)
    js_result = await sandbox.execute_javascript(
        _HTML_GEN_JS,
        input_data={
            "prompt": prompt,
            "appType": resolved_type,
            "components": resolved_components,
            "colorScheme": color_scheme,
            "schema": schema,
        },
        output_format="json",
        timeout=15,
    )

    if "error" in js_result:
        raise ToolError(
            f"HTML preview generation failed: {js_result['error']}\n"
            f"{js_result.get('stdout', '')}"
        )

    preview_data = js_result.get("result", {})
    html_preview = preview_data.get("html", "<p>Preview unavailable</p>")
    install_cmd = (
        add_command
        or preview_data.get("install_command")
        or f"npx shadcn@latest add {' '.join(resolved_components)}"
    )

    safe_name = "_".join(prompt.lower().split()[:3])
    component_file = f"{safe_name.replace('-', '_').title().replace('_', '')}.tsx"

    components_json = json.dumps({
        "$schema": "https://ui.shadcn.com/schema.json",
        "style": "new-york", "rsc": True, "tsx": True,
        "tailwind": {
            "config": "tailwind.config.ts", "css": "app/globals.css",
            "baseColor": color_scheme,
        },
        "aliases": {"components": "@/components", "utils": "@/lib/utils"},
    }, indent=2)

    result: dict[str, Any] = {
        "tsx_code": tsx_code,
        "html_preview": html_preview,
        "components_used": resolved_components,
        "app_type": resolved_type,
        "install_command": install_cmd,
        "file_structure": {
            f"app/components/{component_file}": tsx_code,
            "public/preview.html": html_preview,
            "components.json": components_json,
        },
        "shadcn_add_command": install_cmd,
    }

    if output_type == "nextjs":
        result["nextjs_files"] = _build_nextjs_scaffold(
            prompt, component_file, tsx_code, components_json,
            resolved_components, color_scheme, install_cmd,
        )
        result["nextjs_dev_command"] = (
            "npm run dev   # connect next-devtools MCP for live inspection"
        )

    return result


def _build_nextjs_scaffold(
    prompt: str,
    component_file: str,
    tsx_code: str,
    components_json: str,
    components: list[str],
    color_scheme: str,
    install_cmd: str = "",
) -> dict[str, str]:
    """Minimal Next.js 15 App Router scaffold with shadcn/ui pre-wired."""
    name = "_".join(prompt.lower().split()[:3]).replace(" ", "-")
    comp_import = component_file.replace(".tsx", "")

    return {
        "package.json": json.dumps({
            "name": name, "version": "0.1.0", "private": True,
            "scripts": {"dev": "next dev", "build": "next build", "start": "next start"},
            "dependencies": {
                "next": "^15.0.0", "react": "^18.3.0", "react-dom": "^18.3.0",
                "@radix-ui/react-icons": "^1.3.0", "class-variance-authority": "^0.7.0",
                "clsx": "^2.1.0", "lucide-react": "^0.400.0", "tailwind-merge": "^2.3.0",
                "tailwindcss-animate": "^1.0.7",
            },
            "devDependencies": {
                "@types/node": "^20", "@types/react": "^18", "typescript": "^5",
                "tailwindcss": "^3.4.0", "postcss": "^8", "autoprefixer": "^10",
            },
        }, indent=2),
        "app/layout.tsx": (
            'import type { Metadata } from "next"\n'
            'import { Inter } from "next/font/google"\n'
            'import "./globals.css"\n\n'
            'const inter = Inter({ subsets: ["latin"] })\n\n'
            f'export const metadata: Metadata = {{\n'
            f'  title: "{prompt}",\n'
            f'  description: "Generated by NEXUS with shadcn/ui",\n'
            f'}}\n\n'
            'export default function RootLayout({ children }: { children: React.ReactNode }) {\n'
            '  return (\n'
            '    <html lang="en">\n'
            '      <body className={inter.className}>{children}</body>\n'
            '    </html>\n'
            '  )\n'
            '}\n'
        ),
        "app/page.tsx": (
            f'import {comp_import} from "@/components/{comp_import}"\n\n'
            f'export default function Home() {{\n'
            f'  return <{comp_import} />\n'
            f'}}\n'
        ),
        f"app/components/{component_file}": tsx_code,
        "app/globals.css": (
            "@tailwind base;\n@tailwind components;\n@tailwind utilities;\n\n"
            ":root {\n"
            "  --background: 0 0% 100%;\n  --foreground: 222.2 84% 4.9%;\n"
            "  --card: 0 0% 100%;\n  --card-foreground: 222.2 84% 4.9%;\n"
            "  --primary: 222.2 47.4% 11.2%;\n  --primary-foreground: 210 40% 98%;\n"
            "  --muted: 210 40% 96.1%;\n  --muted-foreground: 215.4 16.3% 46.9%;\n"
            "  --border: 214.3 31.8% 91.4%;\n  --radius: 0.5rem;\n"
            "  --chart-1: 221 83% 53%;\n}\n"
        ),
        "tailwind.config.ts": (
            'import type { Config } from "tailwindcss"\n\n'
            'const config: Config = {\n'
            '  darkMode: ["class"],\n'
            '  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],\n'
            '  theme: { extend: {\n'
            '    colors: { border: "hsl(var(--border))", background: "hsl(var(--background))", foreground: "hsl(var(--foreground))" },\n'
            '  } },\n'
            '  plugins: [require("tailwindcss-animate")],\n'
            '}\nexport default config\n'
        ),
        "components.json": components_json,
        "README.md": "\n".join([
            f"# {prompt}", "",
            "Generated by NEXUS frontend design tool.", "",
            "## Quick start", "", "```bash", "npm install",
            install_cmd, "npm run dev", "```", "",
            "## Components used", *[f"- {c}" for c in components], "",
            "## next-devtools MCP", "",
            "Once dev server is running on port 3000:", "", "```python",
            "from nexus.mcp import MCPClient",
            "from nexus.mcp.known_servers import nextjs_server",
            "client = MCPClient()",
            'await client.connect(nextjs_server(tenant_id="demo"))',
            "# Tools: get_errors, get_logs, get_page_metadata, get_project_metadata",
            "```",
        ]),
    }


# ── Tool registration ─────────────────────────────────────────────────────────

_registered_tools["generate_frontend_design"] = (
    ToolDefinition(
        name="generate_frontend_design",
        description=(
            "Synthesise a frontend UI from a natural language prompt using shadcn/ui components. "
            "Call the shadcn MCP tools FIRST to get real component examples, then pass them here. "
            "The LLM generates domain-specific TSX code and preview data from the prompt + examples. "
            "Returns tsx_code, an interactive filterable html_preview, install_command, "
            "and file_structure. Requires a configured LLM (cloud API key or local Ollama)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Natural language UI description (e.g. 'SaaS invoice management dashboard')",
                },
                "components": {
                    "type": "array", "items": {"type": "string"},
                    "description": "shadcn/ui component names — auto-selected from app_type if omitted",
                },
                "app_type": {
                    "type": "string",
                    "enum": ["dashboard", "crud", "auth", "landing", "form", "settings", "kanban", "inbox"],
                    "description": "Layout archetype",
                },
                "color_scheme": {
                    "type": "string",
                    "enum": ["zinc", "slate", "blue", "green", "violet"],
                    "description": "Tailwind color theme (default: zinc)",
                },
                "add_command": {
                    "type": "string",
                    "description": "Install command from mcp_shadcn_get_add_command_for_items",
                },
                "component_examples": {
                    "type": "string",
                    "description": (
                        "TSX examples from mcp_shadcn_get_item_examples_from_registries. "
                        "Passing real MCP output here produces domain-accurate code."
                    ),
                },
                "output_type": {
                    "type": "string", "enum": ["html", "nextjs"],
                    "description": "html = interactive static preview; nextjs = Next.js 15 App Router scaffold",
                },
            },
            "required": ["prompt"],
        },
        risk_level=RiskLevel.LOW,
        resource_pattern="code:*",
        timeout_seconds=60,
    ),
    lambda **p: generate_frontend_design(**p),
)
