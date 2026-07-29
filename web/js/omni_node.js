// Omni node frontend: "Update" button that rebuilds inputs/outputs from pasted code

const { app } = window.comfyAPI.app;

const OMNI_NODE_NAME = "OmniNode";

function findAssignments(source) {
    const assigns = {};
    const lines = source.split("\n");
    const targets = ["FUNCTION", "RETURN_TYPES", "RETURN_NAMES", "CATEGORY", "DESCRIPTION"];
    for (const target of targets) {
        const re = new RegExp("^\\s*" + target + "\\s*=\\s*(.+)$");
        for (let i = 0; i < lines.length; i++) {
            const m = lines[i].match(re);
            if (!m) continue;
            let raw = m[1];
            if (/[([{]/.test(raw)) {
                let depth = 0;
                for (const ch of raw) {
                    if ("([{".includes(ch)) depth++;
                    else if (")]}".includes(ch)) depth--;
                }
                let j = i + 1;
                while (depth > 0 && j < lines.length) {
                    const ln = lines[j];
                    raw += "\n" + ln;
                    for (const ch of ln) {
                        if ("([{".includes(ch)) depth++;
                        else if (")]}".includes(ch)) depth--;
                    }
                    j++;
                }
            }
            assigns[target] = raw.trim();
            break;
        }
    }
    return assigns;
}

function evalPyLiteral(raw) {
    raw = raw.trim();
    if (!raw) return undefined;
    raw = raw.replace(/#.*$/, "").trim();
    try {
        const jsonish = raw
            .replace(/'/g, '"')
            .replace(/\(/g, '[')
            .replace(/\)/g, ']')
            .replace(/,\s*\]/g, ']')
            .replace(/,\s*([^\]\s])/g, ', $1');
        return JSON.parse(jsonish);
    } catch (e) {
        return undefined;
    }
}

function parseInputTypes(source) {
    const out = { required: {}, optional: {}, hidden: {} };
    const defIdx = source.search(/def\s+INPUT_TYPES\s*\(/);
    if (defIdx === -1) return out;
    const body = source.slice(defIdx);
    const retIdx = body.search(/return\s*\{/);
    if (retIdx === -1) return out;
    let depth = 0;
    let start = body.indexOf("{", retIdx);
    let end = start;
    for (let i = start; i < body.length; i++) {
        const ch = body[i];
        if (ch === "{") depth++;
        else if (ch === "}") {
            depth--;
            if (depth === 0) { end = i + 1; break; }
        }
    }
    const dictText = body.slice(start, end);
    for (const section of ["required", "optional", "hidden"]) {
        const secRe = new RegExp('"' + section + '"\\s*:\\s*\\{', 'g');
        const m = secRe.exec(dictText);
        if (!m) continue;
        let d = 0;
        let s = dictText.indexOf("{", m.index);
        let e = s;
        for (let i = s; i < dictText.length; i++) {
            const ch = dictText[i];
            if (ch === "{") d++;
            else if (ch === "}") { d--; if (d === 0) { e = i + 1; break; } }
        }
        const sectionText = dictText.slice(s, e);
        out[section] = parseInputSection(sectionText);
    }
    return out;
}

function parseInputSection(text) {
    const entries = {};
    let body = text.trim();
    if (body.startsWith("{") && body.endsWith("}")) body = body.slice(1, -1);
    let i = 0;
    while (i < body.length) {
        while (i < body.length && /[\s,]/.test(body[i])) i++;
        if (i >= body.length) break;
        if (body[i] !== '"' && body[i] !== "'") break;
        const quote = body[i];
        let j = i + 1;
        let key = "";
        while (j < body.length && body[j] !== quote) {
            if (body[j] === "\\") { key += body[j + 1]; j += 2; }
            else { key += body[j]; j++; }
        }
        j++;
        while (j < body.length && /[\s:]/.test(body[j])) j++;
        if (body[j] !== "(" && body[j] !== "[") break;
        const open = body[j];
        const close = open === "(" ? ")" : "]";
        let d = 0;
        let vs = j;
        let ve = vs;
        for (let k = vs; k < body.length; k++) {
            if (body[k] === open) d++;
            else if (body[k] === close) { d--; if (d === 0) { ve = k + 1; break; } }
        }
        const valueText = body.slice(vs, ve);
        entries[key] = parseInputValue(valueText);
        i = ve;
    }
    return entries;
}

function parseInputValue(text) {
    let inner = text.trim();
    if (inner.startsWith("(") && inner.endsWith(")")) inner = inner.slice(1, -1);
    else if (inner.startsWith("[") && inner.endsWith("]")) {
        const opts = evalPyLiteral(inner);
        return { type: Array.isArray(opts) ? opts : ["*"], options: {} };
    }
    inner = inner.trim();
    let typePart = inner;
    let optsPart = "";
    let depth = 0;
    for (let k = 0; k < inner.length; k++) {
        const ch = inner[k];
        if ("([{".includes(ch)) depth++;
        else if (")]}".includes(ch)) depth--;
        else if (ch === "," && depth === 0) {
            typePart = inner.slice(0, k);
            optsPart = inner.slice(k + 1);
            break;
        }
    }
    typePart = typePart.trim();
    optsPart = optsPart.trim();

    let type;
    if (typePart.startsWith("[") || typePart.startsWith("(")) {
        const arr = evalPyLiteral(typePart);
        type = Array.isArray(arr) ? arr : ["*"];
    } else if (typePart.startsWith('"') || typePart.startsWith("'")) {
        type = evalPyLiteral(typePart);
    } else if (typePart) {
        type = typePart;
    } else {
        type = "*";
    }

    let options = {};
    if (optsPart.startsWith("{")) {
        const ml = optsPart.match(/["']?multiline["']?\s*:\s*(True|False)/);
        if (ml) options.multiline = (ml[1] === "True");
        const def = optsPart.match(/["']?default["']?\s*:\s*("[^"]*"|'[^']*'|-?\d+(\.\d+)?|True|False)/);
        if (def) {
            let dv = def[1];
            if (dv.startsWith('"') || dv.startsWith("'")) dv = dv.slice(1, -1);
            else if (dv === "True") dv = true;
            else if (dv === "False") dv = false;
            else dv = Number(dv);
            options.default = dv;
        }
        const fi = optsPart.match(/["']?forceInput["']?\s*:\s*(True|False)/);
        if (fi) options.forceInput = (fi[1] === "True");
    }
    return { type, options };
}

function parseOmniSource(source) {
    const spec = {
        class_name: "",
        function: "execute",
        required: {},
        optional: {},
        hidden: {},
        return_types: [],
        return_names: [],
        category: "omni",
        error: null,
    };
    if (!source || !source.trim()) {
        spec.error = "No code pasted.";
        return spec;
    }
    const cm = source.match(/class\s+(\w+)\s*(\([^)]*\))?\s*:/);
    if (cm) spec.class_name = cm[1];

    const assigns = findAssignments(source);
    if (assigns.FUNCTION) {
        const f = evalPyLiteral(assigns.FUNCTION);
        if (typeof f === "string") spec.function = f;
    }
    if (assigns.RETURN_TYPES) {
        const rt = evalPyLiteral(assigns.RETURN_TYPES);
        if (Array.isArray(rt)) spec.return_types = rt;
        else if (typeof rt === "string") spec.return_types = [rt];
    }
    if (assigns.RETURN_NAMES) {
        const rn = evalPyLiteral(assigns.RETURN_NAMES);
        if (Array.isArray(rn)) spec.return_names = rn;
    }
    if (assigns.CATEGORY) {
        const c = evalPyLiteral(assigns.CATEGORY);
        if (typeof c === "string") spec.category = c;
    }

    const it = parseInputTypes(source);
    spec.required = it.required;
    spec.optional = it.optional;
    spec.hidden = it.hidden;
    return spec;
}

function snapshotConnections(node) {
    const graph = node.graph;
    const snap = { inputs: {}, outputs: {} };

    if (node.inputs) {
        for (let i = 0; i < node.inputs.length; i++) {
            const inp = node.inputs[i];
            if (!inp || inp.link == null || inp.link < 0) continue;
            const link = graph?.links?.[inp.link];
            if (!link) continue;
            const originNode = graph.getNodeById(link.origin_id);
            if (!originNode) continue;
            snap.inputs[inp.name] = {
                originNode: originNode,
                originSlot: link.origin_slot,
            };
        }
    }

    if (node.outputs) {
        for (let i = 0; i < node.outputs.length; i++) {
            const out = node.outputs[i];
            if (!out || !out.links || out.links.length === 0) continue;
            const targets = [];
            for (const linkId of out.links) {
                const link = graph?.links?.[linkId];
                if (!link) continue;
                const targetNode = graph.getNodeById(link.target_id);
                if (!targetNode) continue;
                targets.push({
                    targetNode: targetNode,
                    targetSlot: link.target_slot,
                });
            }
            if (targets.length > 0) snap.outputs[out.name] = targets;
        }
    }
    return snap;
}

function restoreConnections(node, snap) {
    const inputByName = {};
    if (node.inputs) {
        for (let i = 0; i < node.inputs.length; i++) {
            if (node.inputs[i] && node.inputs[i].name) {
                inputByName[node.inputs[i].name] = i;
            }
        }
    }
    const outputByName = {};
    if (node.outputs) {
        for (let i = 0; i < node.outputs.length; i++) {
            if (node.outputs[i] && node.outputs[i].name) {
                outputByName[node.outputs[i].name] = i;
            }
        }
    }

    for (const name in snap.inputs) {
        const idx = inputByName[name];
        if (idx == null) continue;
        const c = snap.inputs[name];
        try {
            c.originNode.connect(c.originSlot, node, idx);
        } catch (e) { }
    }

    for (const name in snap.outputs) {
        const idx = outputByName[name];
        if (idx == null) continue;
        for (const c of snap.outputs[name]) {
            try {
                node.connect(idx, c.targetNode, c.targetSlot);
            } catch (e) { }
        }
    }
}

function rebuildOmniIO(node) {
    const codeWidget = node.widgets?.find(w => w.name === "code");
    if (!codeWidget) return;
    const source = codeWidget.value || "";
    const spec = parseOmniSource(source);

    const snap = snapshotConnections(node);

    const staticWidgetNames = new Set(["code", "omni_spec", "Update"]);
    const prevWidgetValues = {};
    if (node.widgets) {
        for (const w of node.widgets) {
            if (w.name && !staticWidgetNames.has(w.name)) {
                prevWidgetValues[w.name] = w.value;
            }
        }
    }

    const staticInputNames = new Set(["code", "omni_spec"]);
    if (node.inputs) {
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const inp = node.inputs[i];
            if (inp.name && !staticInputNames.has(inp.name)) {
                node.removeInput(i);
            }
        }
    }
    if (node.widgets) {
        node.widgets = node.widgets.filter(w => w.name === "code" || w.name === "Update");
    }
    const WIDGET_TYPES = new Set(["INT", "FLOAT", "STRING", "BOOLEAN"]);
    const addFromSection = (section, isOptional) => {
        for (const name in section) {
            const entry = section[name];
            const type = entry.type;
            const opts = entry.options || {};
            const forceInput = opts.forceInput === true;

            if (Array.isArray(type)) {
                if (forceInput) {
                    node.addInput(name, "*");
                } else {
                    const defVal = opts.default != null ? opts.default : type[0];
                    node.addWidget("combo", name, defVal, () => {}, {
                        values: type,
                    });
                }
                continue;
            }

            if (WIDGET_TYPES.has(type) && !forceInput) {
                if (type === "INT" || type === "FLOAT") {
                    const defVal = opts.default != null ? opts.default : 0;
                    const w = node.addWidget("number", name, defVal, () => {}, {
                        min: opts.min != null ? opts.min : -Infinity,
                        max: opts.max != null ? opts.max : Infinity,
                        step: opts.step != null ? opts.step : (type === "INT" ? 1 : 0.01),
                        precision: type === "INT" ? 0 : 3,
                    });
                    if (w && typeof w.serializeValue === "undefined") {
                        w.type = type;
                    }
                } else if (type === "BOOLEAN") {
                    const defVal = opts.default != null ? opts.default : false;
                    node.addWidget("toggle", name, defVal, () => {});
                } else if (type === "STRING") {
                    const defVal = opts.default != null ? opts.default : "";
                    if (opts.multiline) {
                        node.addWidget("text", name, defVal, () => {}, { multiline: true });
                    } else {
                        node.addWidget("text", name, defVal, () => {});
                    }
                }
                continue;
            }

            node.addInput(name, type || "*");
        }
    };
    addFromSection(spec.required, false);
    addFromSection(spec.optional, true);

    if (node.widgets) {
        for (const w of node.widgets) {
            if (w.name && w.name in prevWidgetValues) {
                w.value = prevWidgetValues[w.name];
            }
        }
    }

    if (node.outputs) {
        while (node.outputs.length > 0) {
            node.removeOutput(0);
        }
    }
    for (let i = 0; i < spec.return_types.length; i++) {
        const t = spec.return_types[i];
        const type = Array.isArray(t) ? "*" : t;
        const name = spec.return_names[i] || `output_${i + 1}`;
        node.addOutput(name, type || "*");
    }

    restoreConnections(node, snap);

    if (spec.class_name) {
        node.title = spec.class_name;
    }

    if (typeof node.computeSize === "function") {
        const sz = node.computeSize();
        node.setSize([Math.max(node.size[0], sz[0]), Math.max(node.size[1], sz[1])]);
    }

    node.setDirtyCanvas(true, true);
}


// overlay animation for OmniNode

const OMNI_COLOR_STYLE_ID = "omni-node-holo-style";
const OMNI_COLOR_SAMPLER_ID = "omni-node-holo-sampler";
const OMNI_HOLO_FLAG = "__omniHolo";

function ensureOmniColorAnimation() {
    if (document.getElementById(OMNI_COLOR_STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = OMNI_COLOR_STYLE_ID;
    style.textContent = `
@keyframes omniHoloHue {
    0%   { background-color: hsl(0, 70%, 65%); }
    25%  { background-color: hsl(90, 70%, 65%); }
    50%  { background-color: hsl(180, 70%, 65%); }
    75%  { background-color: hsl(270, 70%, 65%); }
    100% { background-color: hsl(0, 70%, 65%); }
}
#${OMNI_COLOR_SAMPLER_ID} {
    animation: omniHoloHue 10s linear infinite;
}
`;
    document.head.appendChild(style);

    const sampler = document.createElement("div");
    sampler.id = OMNI_COLOR_SAMPLER_ID;
    sampler.style.cssText = "position:fixed;left:-9999px;top:-9999px;width:1px;height:1px;pointer-events:none;";
    document.body.appendChild(sampler);
}

const omniAnimatedNodes = new Set();
let omniColorLoopRunning = false;

function rgbToHsl(rgbStr) {
    const m = rgbStr.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
    if (!m) return null;
    const r = +m[1] / 255, g = +m[2] / 255, b = +m[3] / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h = 0;
    const l = (max + min) / 2;
    const d = max - min;
    if (d !== 0) {
        if (max === r) h = ((g - b) / d) % 6;
        else if (max === g) h = (b - r) / d + 2;
        else h = (r - g) / d + 4;
        h *= 60;
        if (h < 0) h += 360;
    }
    return { h, l };
}

function hsl(h, s, l, a) {
    return `hsla(${(h % 360 + 360) % 360}, ${s}%, ${l}%, ${a})`;
}

// Trace the node's rounded-rect silhouette (body + title bar) for clipping.
// Insets by 0.6px to counter antialiasing bleed at the edges.
function omniNodePath(ctx, w, h, headerH) {
    const r = typeof globalThis.LiteGraph?.ROUND_RADIUS === "number"
        ? LiteGraph.ROUND_RADIUS
        : 4;
    const inset = 0.6;
    const rr = Math.max(0, Math.min(r - inset, w / 2, h / 2));
    const x0 = inset, x1 = w - inset;
    const top = -headerH + inset, bottom = h - inset;
    ctx.beginPath();
    ctx.moveTo(x0 + rr, top);
    ctx.lineTo(x1 - rr, top);
    ctx.quadraticCurveTo(x1, top, x1, top + rr);        // top-right corner
    ctx.lineTo(x1, bottom - rr);
    ctx.quadraticCurveTo(x1, bottom, x1 - rr, bottom);  // bottom-right corner
    ctx.lineTo(x0 + rr, bottom);
    ctx.quadraticCurveTo(x0, bottom, x0, bottom - rr);  // bottom-left corner
    ctx.lineTo(x0, top + rr);
    ctx.quadraticCurveTo(x0, top, x0 + rr, top);        // top-left corner
    ctx.closePath();
}

// Collapsed-node path: only the visible title bar (-headerH..0) with all
// four corners rounded so the bottom matches the top.
function omniCollapsedPath(ctx, w, headerH) {
    const r = typeof globalThis.LiteGraph?.ROUND_RADIUS === "number"
        ? LiteGraph.ROUND_RADIUS
        : 4;
    const inset = 0.6;
    const rr = Math.max(0, Math.min(r - inset, w / 2, headerH / 2));
    const x0 = inset, x1 = w - inset;
    const top = -headerH + inset, bottom = -inset;
    ctx.beginPath();
    ctx.moveTo(x0 + rr, top);
    ctx.lineTo(x1 - rr, top);
    ctx.quadraticCurveTo(x1, top, x1, top + rr);        // top-right
    ctx.lineTo(x1, bottom - rr);
    ctx.quadraticCurveTo(x1, bottom, x1 - rr, bottom);  // bottom-right
    ctx.lineTo(x0 + rr, bottom);
    ctx.quadraticCurveTo(x0, bottom, x0, bottom - rr);  // bottom-left
    ctx.lineTo(x0, top + rr);
    ctx.quadraticCurveTo(x0, top, x0 + rr, top);        // top-left
    ctx.closePath();
}

function paintOmniHolo(node, ctx, canvas) {
    let [w, h] = node.size;
    const collapsed = !!node.flags?.collapsed;

    const headerH = node.title_height ?? 30;
    const sampler = document.getElementById(OMNI_COLOR_SAMPLER_ID);
    if (!sampler) return;
    const base = rgbToHsl(getComputedStyle(sampler).backgroundColor);
    if (!base) return;
    const h0 = base.h;

    if (collapsed) {
        if (typeof node._collapsed_width === "number" && node._collapsed_width > 0) {
            w = node._collapsed_width;
        } else {
            const title = node.getTitle ? node.getTitle() : (node.title || node.type || "");
            ctx.save();
            ctx.font = node.title_font ?? "12px Arial";
            const m = ctx.measureText(title);
            ctx.restore();
            w = m.width + 20;
        }
    }


    const stops = [h0, h0 + 105, h0 + 325, h0 + 245, h0];
    const sats = [72, 95, 85, 85, 72];
    const g = ctx.createLinearGradient(0, 0, w, h);
    for (let i = 0; i < stops.length; i++) {
        g.addColorStop(i / (stops.length - 1), hsl(stops[i], sats[i], 26 + i * 1.5, 0.92));
    }

    ctx.save();
    // Clip everything to the node's rounded silhouette
    if (collapsed) {
        // Collapsed: only the title bar is visible.
        omniCollapsedPath(ctx, w, headerH);
        ctx.clip();
        ctx.fillStyle = g;
        ctx.fillRect(0, -headerH, w, headerH);
        ctx.restore();
        return;
    }

    omniNodePath(ctx, w, h, headerH);
    ctx.clip();

    ctx.fillStyle = g;
    ctx.fillRect(0, -headerH, w, h + headerH);

    const t = (Math.sin((performance.now() / 6000) * Math.PI * 2 - Math.PI / 2) + 1) / 2;
    const sheenX = -100 + t * (w + 200);
    const sheen = ctx.createLinearGradient(sheenX - 60, -headerH, sheenX + 60, h);
    sheen.addColorStop(0, hsl(h0 + 180, 100, 90, 0));
    sheen.addColorStop(0.5, hsl(h0 + 180, 100, 92, 0.35));
    sheen.addColorStop(1, hsl(h0 + 180, 100, 90, 0));
    ctx.fillStyle = sheen;
    ctx.fillRect(0, -headerH, w, h + headerH);

    const hg = ctx.createLinearGradient(0, 0, w, 0);
    for (let i = 0; i < stops.length; i++) {
        hg.addColorStop(i / (stops.length - 1), hsl(stops[i] + 20, sats[i], 34, 0.45));
    }
    ctx.fillStyle = hg;
    ctx.fillRect(0, -headerH, w, headerH);

    ctx.restore();
}

function startOmniColorLoop() {
    if (omniColorLoopRunning) return;
    omniColorLoopRunning = true;
    const loop = () => {
        if (omniAnimatedNodes.size === 0) {
            omniColorLoopRunning = false;
            return;
        }
        for (const node of omniAnimatedNodes) {
            if (node.graph) {
                node.graph.setDirtyCanvas(true, false);
            } else {
                omniAnimatedNodes.delete(node);
            }
        }
        requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
}

function makeOmniHolo(node) {
    if (node[OMNI_HOLO_FLAG]) return;
    node[OMNI_HOLO_FLAG] = true;

    const prevDrawBg = node.onDrawBackground;
    node.onDrawBackground = function (ctx, canvas) {
        paintOmniHolo(this, ctx, canvas);
        if (prevDrawBg) prevDrawBg.call(this, ctx, canvas);
    };
}

app.registerExtension({
    name: "ComfyUI-OmniNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name !== OMNI_NODE_NAME) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
            this.color = "#1e1e1f80";
            this.bgcolor = "#1e1e1f";
            ensureOmniColorAnimation();
            makeOmniHolo(this);
            omniAnimatedNodes.add(this);
            startOmniColorLoop();
            if (this.outputs) {
                while (this.outputs.length > 0) {
                    this.removeOutput(0);
                }
            }
            if (this.widgets) {
                this.widgets = this.widgets.filter(w => w.name !== "omni_spec");
            }
            this.addWidget("button", "Update", null, () => {
                rebuildOmniIO(this);
            });
            return r;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
            setTimeout(() => rebuildOmniIO(this), 0);
            return r;
        };
    },
});
